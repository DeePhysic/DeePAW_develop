import ase
import torch
from e3nn import o3
from e3nn.o3 import Linear
from torch import nn
from .irreps_tools import get_irreps, InteractionBlock, RadialBasis
from typing import List
import torch
from torch import nn

def unpad_and_cat(stacked_seq: torch.Tensor, seq_len: torch.Tensor):
    """
    Unpad and concatenate by removing batch dimension

    Args:
        stacked_seq: (batch_size, max_length, *) Tensor
        seq_len: (batch_size) Tensor with length of each sequence

    Returns:
        (prod(seq_len), *) Tensor

    """
    unstacked = stacked_seq.unbind(0)
    unpadded = [
        torch.narrow(t, 0, 0, l) for (t, l) in zip(unstacked, seq_len.unbind(0))
    ]
    return torch.cat(unpadded, dim=0)

def pad_and_stack(tensors: List[torch.Tensor]):
    """Pad list of tensors if tensors are arrays and stack if they are scalars"""
    if tensors[0].shape:
        return torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=0
        )
    return torch.stack(tensors)
class AtomicConfigurationModel(nn.Module):
    """
    Atomic Configuration Model

    Learns atomic representations through E3-equivariant message passing.
    This model processes atomic positions and species to generate
    configuration-dependent atomic features.
    """
    def __init__(
        self,
        num_interactions,
        num_neighbors,
        mul=500,
        lmax=4,
        cutoff=4.0,
        basis="gaussian",
        num_basis=10,
    ):
        super().__init__()
        self.lmax = lmax
        self.cutoff = cutoff
        self.number_of_basis = num_basis
        self.basis = RadialBasis(
            start=0.0, 
            end=cutoff,
            number=self.number_of_basis,
            basis=basis,
            cutoff=False,
            normalize=True
        )

        self.convolutions = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

        # store irreps of each output (mostly so the probe model can use)
        self.atom_irreps_sequence = []

        self.num_species = len(ase.data.atomic_numbers)
        #########################################################################
        self.spherical_harmonics = o3.SphericalHarmonics(
            range(self.lmax + 1), normalize=True, normalization="component"
        )        

        # scalar inputs (one-hot atomic numbers) with even parity
        irreps_node_input = f"{self.num_species}x 0e" 
        irreps_node_hidden = o3.Irreps(get_irreps(mul, lmax))
        irreps_node_attr = "0e"
        irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        fc_neurons = [self.number_of_basis, 100]

        # activation to use with even (1) or odd (-1) parities
        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        # irreps_node = irreps_node_input
        ######################################################################

        

        self.interaction_block = InteractionBlock(
            num_interactions,
            irreps_node_input,
            irreps_node_hidden,
            irreps_node_attr,
            irreps_edge_attr,
            mul,
            lmax,
            fc_neurons,
            num_neighbors,
            act,
            act_gates,
            atom_irreps_sequence=None,
        )





        # for _ in range(num_interactions):
        #     # scalar irreps that exist in the tensor product between node and edge irreps
        #     irreps_scalars = o3.Irreps(
        #         [
        #             (mul, ir)
        #             for mul, ir in irreps_node_hidden
        #             if ir.l == 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
        #         ]
        #     ).simplify()
        #     irreps_gated = o3.Irreps(
        #         [
        #             (mul, ir)
        #             for mul, ir in irreps_node_hidden
        #             if ir.l > 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
        #         ]
        #     )
        #     ir = "0e" if tp_path_exists(irreps_node, irreps_edge_attr, "0e") else "0o"
        #     irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

        #     # Gate activation function, see https://docs.e3nn.org/en/stable/api/nn/nn_gate.html
        #     gate = Gate(
        #         irreps_scalars,
        #         [act[ir.p] for _, ir in irreps_scalars],  # scalar
        #         irreps_gates,
        #         [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
        #         irreps_gated,  # gated tensors
        #     )
        #     conv = Convolution(
        #         irreps_node,
        #         irreps_node_attr,
        #         irreps_edge_attr,
        #         gate.irreps_in,
        #         fc_neurons,
        #         num_neighbors,
        #     )
        #     irreps_node = gate.irreps_out
        #     self.convolutions.append(conv)
        #     self.gates.append(gate)

        #     # store output node irreps for each layer
        #     self.atom_irreps_sequence.append(irreps_node)  

    def forward(self, input_dict):
        # Unpad and concatenate edges into batch (0th) dimension
        # incrementing by offset to keep graphs separate
        edges_displacement = unpad_and_cat(
            input_dict["atom_edges_displacement"], input_dict["num_atom_edges"]
        )

        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = unpad_and_cat(edges, input_dict["num_atom_edges"])

        edge_src = edges[:, 0]
        edge_dst = edges[:, 1]

        # Unpad and concatenate all nodes into batch (0th) dimension
        atom_xyz = unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        nodes = unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])

        # one-hot encode atoms
        nodes = torch.nn.functional.one_hot(nodes, num_classes=self.num_species)

        # Node attributes are not used here
        node_attr = nodes.new_ones(nodes.shape[0], 1)


        unitcell_repeat = torch.repeat_interleave(input_dict["cell"], input_dict["num_atom_edges"], dim=0)  # num_edges, 3, 3
        displacement = torch.matmul(
            torch.unsqueeze(edges_displacement, 1), unitcell_repeat
        )  # num_edges, 1, 3
        displacement = torch.squeeze(displacement, dim=1)
        neigh_pos = atom_xyz[edges[:, 0]]  # num_edges, 3
        neigh_abs_pos = neigh_pos + displacement  # num_edges, 3
        this_pos = atom_xyz[edges[:, 1]]  # num_edges, 3
        edge_vec = this_pos - neigh_abs_pos  # num_edges, 3


        # edge_attr = o3.spherical_harmonics(
        #     range(self.lmax + 1), edge_vec, True, normalization="component"
        # )

        edge_attr = self.spherical_harmonics(edge_vec)
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.basis(edge_length)

############################################################################################
        nodes_list = self.interaction_block(
            nodes, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding,
            atom_representation = None,
            atom_node_attr=None,
            probes=None,
            probe_attr=None,
            probe_edge_src=None,
            probe_edge_dst=None,
            probe_edge_attr=None,
            probe_edge_length_embedding=None,      
        )        


        # nodes_list = []
        # # Apply interaction layers
        # for conv, gate in zip(self.convolutions, self.gates):
        #     nodes = conv(
        #         nodes, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding
        #     )
        #     nodes = gate(nodes)
        #     nodes_list.append(nodes)

        return nodes_list
class AtomicPotentialModel(torch.nn.Module):
    """
    Atomic Potential Model

    Predicts charge density at probe points based on atomic representations.
    This model computes the electrostatic potential-like field from atoms
    to probe points using E3-equivariant message passing.
    """
    def __init__(
        self,
        num_interactions,
        num_neighbors,
        atom_irreps_sequence,
        mul=500,
        lmax=4,
        cutoff=4.0,
        basis="gaussian",
        num_basis=10,   # here
        spin=False
    ):
        super().__init__()
        self.lmax = lmax
        self.cutoff = cutoff
        self.number_of_basis = num_basis
        self.basis = RadialBasis(
            start=0.0, 
            end=cutoff,
            number=self.number_of_basis,
            basis=basis,
            cutoff=False,
            normalize=True
        )

        self.convolutions = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

        # scalar inputs with even parity (for probes its just 0s)
        irreps_node_input = "0e"
        irreps_node_hidden = o3.Irreps(get_irreps(mul, lmax))
        irreps_node_attr = "0e"
        irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        fc_neurons = [self.number_of_basis, 100]

        #########################################################################
        self.spherical_harmonics = o3.SphericalHarmonics(
            range(self.lmax + 1), normalize=True, normalization="component"
        )        




        # activation to use with even (1) or odd (-1) parities
        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        irreps_node = irreps_node_input
###########################################################################

        self.interaction_block = InteractionBlock(
            num_interactions,
            irreps_node_input,
            irreps_node_hidden,
            irreps_node_attr,
            irreps_edge_attr,
            mul,
            lmax,
            fc_neurons,
            num_neighbors,
            act,
            act_gates,
            atom_irreps_sequence=atom_irreps_sequence,
        )




        # for i in range(num_interactions):
        #     irreps_scalars = o3.Irreps(
        #         [
        #             (mul, ir)
        #             for mul, ir in irreps_node_hidden
        #             if ir.l == 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
        #         ]
        #     ).simplify()
        #     irreps_gated = o3.Irreps(
        #         [
        #             (mul, ir)
        #             for mul, ir in irreps_node_hidden
        #             if ir.l > 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
        #         ]
        #     )
        #     ir = "0e" if tp_path_exists(irreps_node, irreps_edge_attr, "0e") else "0o"
        #     irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

        #     # Gate activation function, see https://docs.e3nn.org/en/stable/api/nn/nn_gate.html
        #     gate = Gate(
        #         irreps_scalars,
        #         [act[ir.p] for _, ir in irreps_scalars],  # scalar
        #         irreps_gates,
        #         [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
        #         irreps_gated,  # gated tensors
        #     )

        #     conv = ConvolutionOneWay(
        #         irreps_sender_input=atom_irreps_sequence[i],
        #         irreps_sender_attr=irreps_node_attr,
        #         irreps_receiver_input=irreps_node,
        #         irreps_receiver_attr=irreps_node_attr,
        #         irreps_edge_attr=irreps_edge_attr,
        #         irreps_node_output=gate.irreps_in,
        #         fc_neurons=fc_neurons,
        #         num_neighbors=num_neighbors,
        #     )
        #     irreps_node = gate.irreps_out
        #     self.convolutions.append(conv)
        #     self.gates.append(gate)


        # last layer, scalar output
        out = "0e"
        self.readout = Linear(self.interaction_block.irreps_node_sequence[-1], out)

    def forward(self, input_dict, atom_representation):
        atom_xyz = unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        probe_xyz = unpad_and_cat(
            input_dict["probe_xyz"], input_dict["num_probes"]
        )
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        probe_edges_displacement = unpad_and_cat(
            input_dict["probe_edges_displacement"], input_dict["num_probe_edges"]
        )
        edge_probe_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_probes"].device),
                    input_dict["num_probes"][:-1],
                )
            ),
            dim=0,
        )
        edge_probe_offset = edge_probe_offset[:, None, None]
        edge_probe_offset = torch.cat((edge_offset, edge_probe_offset), dim=2)
        probe_edges = input_dict["probe_edges"] + edge_probe_offset
        probe_edges = unpad_and_cat(probe_edges, input_dict["num_probe_edges"])

        unitcell_repeat = torch.repeat_interleave(input_dict["cell"], input_dict["num_probe_edges"], dim=0)  # num_edges, 3, 3
        displacement = torch.matmul(
            torch.unsqueeze(probe_edges_displacement, 1), unitcell_repeat
        )  # num_edges, 1, 3
        displacement = torch.squeeze(displacement, dim=1)
        neigh_pos = atom_xyz[probe_edges[:, 0]]  # num_edges, 3
        neigh_abs_pos = neigh_pos + displacement  # num_edges, 3
        this_pos = probe_xyz[probe_edges[:, 1]]  # num_edges, 3
        probe_edge_vec = this_pos - neigh_abs_pos  # num_edges, 3


        # probe_edge_vec = calc_edge_vec_to_probe(
        #     atom_xyz,
        #     probe_xyz,
        #     input_dict["cell"],
        #     probe_edges,
        #     probe_edges_displacement,
        #     input_dict["num_probe_edges"],
        # )
        # probe_edge_attr = o3.spherical_harmonics(
        #     range(self.lmax + 1), probe_edge_vec, True, normalization="component"
        # )
        probe_edge_attr = self.spherical_harmonics(probe_edge_vec)
        ########################################################################################
        probe_edge_length = probe_edge_vec.norm(dim=1)
        probe_edge_length_embedding = self.basis(probe_edge_length)

        probe_edge_src = probe_edges[:, 0]
        probe_edge_dst = probe_edges[:, 1]

        # initialize probes
        probes = torch.zeros(
            (torch.sum(input_dict["num_probes"]), 1),
            device=atom_representation[0].device,
        )

        # Probe attributes are not used here
        probe_attr = probes.new_ones(probes.shape[0], 1)

        # Node attributes are not used here
        atom_node_attr = probes.new_ones(atom_xyz.shape[0], 1)

        # Apply interaction layers
        probes = self.interaction_block(
            nodes=None,
            node_attr=None,
            edge_src=None, 
            edge_dst=None, 
            edge_attr=None, 
            edge_length_embedding=None,
            atom_representation = atom_representation,
            atom_node_attr=atom_node_attr,
            probes=probes,
            probe_attr=probe_attr,
            probe_edge_src=probe_edge_src,
            probe_edge_dst=probe_edge_dst,
            probe_edge_attr=probe_edge_attr,
            probe_edge_length_embedding=probe_edge_length_embedding,            
        )    

        prob_rep = probes.clone()
        probes = self.readout(probes).squeeze()

        # rebatch
        probes = pad_and_stack(
            torch.split(
                probes,
                list(input_dict["num_probes"].detach().cpu().numpy()),
                dim=0,
            )
        )
        return probes,prob_rep
class F_nonlocal(nn.Module):
    """
    F_nonlocal: Non-local charge density prediction model

    This model uses E3-equivariant neural networks to predict charge density
    based on atomic structure. It consists of two main components:
    1. AtomicConfigurationModel: Learns atomic representations
    2. AtomicPotentialModel: Predicts charge density at probe points

    Args:
        num_interactions (int): Number of message passing layers (default: 3)
        num_neighbors (int): Maximum number of neighbors (default: 20)
        mul (int): Multiplicity for irreps (default: 500)
        lmax (int): Maximum angular momentum (default: 4)
        cutoff (float): Cutoff radius in Angstroms (default: 4.0)
        basis (str): Type of radial basis function (default: "gaussian")
        num_basis (int): Number of basis functions (default: 20)
        spin (bool): Whether to include spin (default: False)
    """
    def __init__(
        self,
        num_interactions=3,
        num_neighbors=20,
        mul=500,
        lmax=4,
        cutoff=4.0,
        basis="gaussian",
        num_basis=20,
        spin=False
    ):
        super().__init__()
        self.spin = spin

        self.atom_model = AtomicConfigurationModel(
            num_interactions,
            num_neighbors,
            mul=mul,
            lmax=lmax,
            cutoff=cutoff,
            basis=basis,
            num_basis=num_basis,
        )

        self.probe_model = AtomicPotentialModel(
            num_interactions,
            num_neighbors,
            self.atom_model.interaction_block.irreps_node_sequence,
            mul=mul,
            lmax=lmax,
            cutoff=cutoff,
            basis=basis,
            num_basis=num_basis,
            spin=spin
        )

    def forward(self, input_dict):
        atom_representation = self.atom_model(input_dict)
        probe_result, prob_pep = self.probe_model(input_dict, atom_representation)
        return probe_result, prob_pep


# Legacy aliases for backward compatibility
E3DensityModel = F_nonlocal
E3AtomRepresentationModel = AtomicConfigurationModel
E3ProbeMessageModel = AtomicPotentialModel


