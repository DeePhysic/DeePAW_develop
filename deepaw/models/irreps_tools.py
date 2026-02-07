import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct
from torch import nn


def get_irreps(total_mul, lmax):
    """
    Get irreps up to lmax, all with roughly the same multiplicity with a total multiplicity of total_mul
    Example:
        get_irreps(500, lmax=2) = 167x0o + 167x0e + 56x1o + 56x1e + 33x2o + 33x2e
    """
    return [
        (round(total_mul / (lmax + 1) / (l * 2 + 1)), (l, p))
        for l in range(lmax + 1)
        for p in [-1, 1]
    ]

def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # special case of torch_scatter.scatter with dim=0
    out = src.new_zeros(dim_size, src.shape[1])
    index = index.reshape(-1, 1).expand_as(src)
    return out.scatter_add_(0, index, src)


class RadialBasis(nn.Module):
    r"""
    Wrapper for e3nn.math.soft_one_hot_linspace, with option for normalization
    Args:
        start (float): mininum value of basis
        end (float): maximum value of basis
        number (int): number of basis functions
        basis ({'gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel'}): basis family
        cutoff (bool): all x outside interval \approx 0
        normalize (bool): normalize function to have a mean of 0, std of 1
        samples (int): number of samples to use to find mean/std
    """
    def __init__(
        self,
        start,
        end,
        number,
        basis="gaussian",
        cutoff=False,
        normalize=True,
        samples=4000
    ):
        super().__init__()
        self.start = start
        self.end = end
        self.number = number
        self.basis = basis
        self.cutoff = cutoff
        self.normalize = normalize

        if normalize:
            with torch.no_grad():
                rs = torch.linspace(start, end, samples+1)[1:]
                bs = soft_one_hot_linspace(rs, start, end, number, basis, cutoff)
                assert bs.ndim == 2 and len(bs) == samples
                std, mean = torch.std_mean(bs, dim=0)
            self.register_buffer("mean", mean)
            self.register_buffer("inv_std", torch.reciprocal(std))
        
    def forward(self, x):
        x = soft_one_hot_linspace(x, self.start, self.end, self.number, self.basis, self.cutoff)
        if self.normalize:
            x = (x - self.mean) * self.inv_std
        return x
    

class InteractionBlock(nn.Module):
    def __init__(
        self,
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
        atom_irreps_sequence,
    ):
        super().__init__()
        self.num_interactions = num_interactions

        self.convolutions = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.irreps_node_sequence = []

        irreps_node = irreps_node_input
        for i in range(num_interactions):
            conv, gate, irreps_out = self._build_layer(
                irreps_node,
                irreps_node_hidden,
                irreps_node_attr,
                irreps_edge_attr,
                mul,
                lmax,
                fc_neurons,
                num_neighbors,
                act,
                act_gates,
                atom_irreps_sequence,
                i,
            )
            self.convolutions.append(conv)
            self.gates.append(gate)
            self.irreps_node_sequence.append(irreps_out)
            irreps_node = irreps_out

    def _build_layer(
        self,
        irreps_node,
        irreps_node_hidden,
        irreps_node_attr,
        irreps_edge_attr,
        mul,
        lmax,
        fc_neurons,
        num_neighbors,
        act,
        act_gates,
        atom_irreps_sequence,
        i,
    ):

        irreps_scalars = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in irreps_node_hidden
                if ir.l == 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
            ]
        ).simplify()
        irreps_gated = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in irreps_node_hidden
                if ir.l > 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
            ]
        )
        ir = "0e" if tp_path_exists(irreps_node, irreps_edge_attr, "0e") else "0o"
        irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()


        gate = Gate(
            irreps_scalars,
            [act[ir.p] for _, ir in irreps_scalars],  # scalar activations
            irreps_gates,
            [act_gates[ir.p] for _, ir in irreps_gates],  # gates activations
            irreps_gated,  # gated tensors
        )

        if atom_irreps_sequence is None:
            conv = Convolution(
                irreps_node,
                irreps_node_attr,
                irreps_edge_attr,
                gate.irreps_in,
                fc_neurons,
                num_neighbors,
            )
        else:
            conv = ConvolutionOneWay(
                irreps_sender_input=atom_irreps_sequence[i],
                irreps_sender_attr=irreps_node_attr,
                irreps_receiver_input=irreps_node,
                irreps_receiver_attr=irreps_node_attr,
                irreps_edge_attr=irreps_edge_attr,
                irreps_node_output=gate.irreps_in,
                fc_neurons=fc_neurons,
                num_neighbors=num_neighbors,
            )            
        return conv, gate, gate.irreps_out

    def forward(self, nodes, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, atom_representation, atom_node_attr, probes, probe_attr,
                probe_edge_src,
                probe_edge_dst,
                probe_edge_attr,
                probe_edge_length_embedding,):

        nodes_list = []

        if atom_representation is None:
            for conv, gate in zip(self.convolutions, self.gates):
                # Convolution
                nodes = conv(
                    nodes, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding
                )
                # Gate
                nodes = gate(nodes)

                nodes_list.append(nodes)

            return nodes_list
        else:
            for conv, gate, atom_nodes in zip(
                self.convolutions, self.gates, atom_representation
            ):
                probes = conv(
                    atom_nodes,
                    atom_node_attr,
                    probes,
                    probe_attr,
                    probe_edge_src,
                    probe_edge_dst,
                    probe_edge_attr,
                    probe_edge_length_embedding,
                )
                probes = gate(probes)
            return probes




class Convolution(torch.nn.Module):
    """
    Equivariant Convolution
    Args:
        irreps_node_input (e3nn.o3.Irreps): representation of the input node features
        irreps_node_attr (e3nn.o3.Irreps): representation of the node attributes
        irreps_edge_attr (e3nn.o3.Irreps): representation of the edge attributes
        irreps_node_output (e3nn.o3.Irreps or None): representation of the output node features
        fc_neurons (list[int]): number of neurons per layers in the fully connected network
            first layer and hidden layers but not the output layer
        num_neighbors (float): typical number of nodes convolved over
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        num_neighbors,
    ) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.num_neighbors = num_neighbors

        self.sc = FullyConnectedTensorProduct(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output
        )

        self.lin1 = FullyConnectedTensorProduct(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_node_input
        )
        
        irreps_mid, instructions = self._compute_intermediate_irreps_and_instructions()

        self.tp = TensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            fc_neurons + [self.tp.weight_numel], torch.nn.functional.silu
        )
        self.lin2 = FullyConnectedTensorProduct(
            irreps_mid, self.irreps_node_attr, self.irreps_node_output
        )
        self.lin3 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, "0e")

    def _compute_intermediate_irreps_and_instructions(self):
        """
        Compute intermediate irreps and instructions for the tensor product.
        """
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_node_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        return irreps_mid, instructions

    
    
        # irreps_mid = []
        # instructions = []
        # for i, (mul, ir_in) in enumerate(self.irreps_node_input):
        #     for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
        #         for ir_out in ir_in * ir_edge:
        #             if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
        #                 k = len(irreps_mid)
        #                 irreps_mid.append((mul, ir_out))
        #                 instructions.append((i, j, k, "uvu", True))
        # irreps_mid = o3.Irreps(irreps_mid)
        # irreps_mid, p, _ = irreps_mid.sort()

        # instructions = [
        #     (i_1, i_2, p[i_out], mode, train)
        #     for i_1, i_2, i_out, mode, train in instructions
        # ]

        # tp = TensorProduct(
        #     self.irreps_node_input,
        #     self.irreps_edge_attr,
        #     irreps_mid,
        #     instructions,
        #     internal_weights=False,
        #     shared_weights=False,
        # )
        # self.fc = FullyConnectedNet(
        #     fc_neurons + [tp.weight_numel], torch.nn.functional.silu
        # )
        # self.tp = tp

        # self.lin2 = FullyConnectedTensorProduct(
        #     irreps_mid, self.irreps_node_attr, self.irreps_node_output
        # )
        # self.lin3 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, "0e")

    def forward(
        self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars
    ) -> torch.Tensor:
        weight = self.fc(edge_scalars)

        node_self_connection = self.sc(node_input, node_attr)
        node_features = self.lin1(node_input, node_attr)

        edge_features = self.tp(node_features[edge_src], edge_attr, weight)
        node_features = scatter(
            edge_features, edge_dst, dim_size=node_input.shape[0]
        ).div(self.num_neighbors**0.5)

        node_conv_out = self.lin2(node_features, node_attr)
        node_angle = 0.1 * self.lin3(node_features, node_attr)

        cos, sin = node_angle.cos(), node_angle.sin()
        m = self.sc.output_mask
        sin = (1 - m) + sin * m
        return cos * node_self_connection + sin * node_conv_out
    

class ConvolutionOneWay(torch.nn.Module):
    """
    Equivariant Convolution, but receiving nodes are differently indexed from sending nodes.
    Additionally, sender and receiver nodes can have different irreps.

    Args:
        irreps_sender_input (e3nn.o3.Irreps): representation of the input sender nodes
        irreps_sender_attr (e3nn.o3.Irreps): representation of the sender attributes
        irreps_receiver_input(e3nn.o3.Irreps): representation of the input receiver nodes
        irreps_receiver_attr (e3nn.o3.Irreps): representation of the receiver attributes
        irreps_edge_attr (e3nn.o3.Irreps): representation of the edge attributes
        irreps_node_output (e3nn.o3.Irreps or None): representation of the output node features
        fc_neurons (list[int]): number of neurons per layers in the fully connected network
            first layer and hidden layers but not the output layer
        num_neighbors (float): typical number of nodes convolved over
    """

    def __init__(
        self,
        irreps_sender_input,
        irreps_sender_attr,
        irreps_receiver_input,
        irreps_receiver_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        num_neighbors,
    ) -> None:
        super().__init__()

        # Parse irreps
        self.irreps_sender_input = o3.Irreps(irreps_sender_input)
        self.irreps_sender_attr = o3.Irreps(irreps_sender_attr)
        self.irreps_receiver_input = o3.Irreps(irreps_receiver_input)
        self.irreps_receiver_attr = o3.Irreps(irreps_receiver_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.num_neighbors = num_neighbors

        # Initialize layers
        self.sc = FullyConnectedTensorProduct(
            self.irreps_receiver_input, self.irreps_receiver_attr, self.irreps_node_output
        )
        self.lin1 = FullyConnectedTensorProduct(
            self.irreps_sender_input, self.irreps_sender_attr, self.irreps_sender_input
        )
        irreps_mid, instructions = self._compute_intermediate_irreps_and_instructions()
        self.tp = TensorProduct(
            self.irreps_sender_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            fc_neurons + [self.tp.weight_numel], torch.nn.functional.silu
        )
        self.lin2 = FullyConnectedTensorProduct(
            irreps_mid, self.irreps_receiver_attr, self.irreps_node_output
        )
        self.lin3 = FullyConnectedTensorProduct(irreps_mid, self.irreps_receiver_attr, "0e")

    def _compute_intermediate_irreps_and_instructions(self):
        """
        Compute intermediate irreps and instructions for the tensor product.
        """
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_sender_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]


        return irreps_mid, instructions

    def forward(
        self,
        sender_input,
        sender_attr,
        receiver_input,
        receiver_attr,
        edge_src,
        edge_dst,
        edge_attr,
        edge_scalars,
    ) -> torch.Tensor:
        # Compute weights using fully connected network
        weight = self.fc(edge_scalars)

        # Compute self-connection
        receiver_self_connection = self.sc(receiver_input, receiver_attr)

        # Compute sender features
        sender_features = self.lin1(sender_input, sender_attr)

        # Compute edge features
        edge_features = self.tp(sender_features[edge_src], edge_attr, weight)

        # Scatter edge features to receiver nodes
        receiver_features = scatter(
            edge_features, edge_dst, dim_size=receiver_input.shape[0]
        ).div(self.num_neighbors**0.5)

        # Compute convolution output and angles
        receiver_conv_out = self.lin2(receiver_features, receiver_attr)
        receiver_angle = 0.1 * self.lin3(receiver_features, receiver_attr)

        # Combine outputs using trigonometric functions
        cos, sin = receiver_angle.cos(), receiver_angle.sin()
        sin = (1 - self.sc.output_mask) + sin * self.sc.output_mask
        return cos * receiver_self_connection + sin * receiver_conv_out
