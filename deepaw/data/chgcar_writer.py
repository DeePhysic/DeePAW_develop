from typing import List
import torch
import numpy as np
import ase
import ase.neighborlist
from ase.db.row import AtomsRow
import time
from tqdm import tqdm
from ase.db import connect
from scipy.spatial import KDTree, cKDTree
from scipy.spatial.distance import euclidean

# GPU 图构建后端（可选依赖）
try:
    from torch_cluster import radius as tc_radius
    HAS_TORCH_CLUSTER = True
except ImportError:
    HAS_TORCH_CLUSTER = False

# DeePAW imports
# DeePAW imports
# Inline implementation of pad_and_stack (was in layer.py)
def pad_and_stack(tensors: List[torch.Tensor]):
    """Pad list of tensors if tensors are arrays and stack if they are scalars"""
    if tensors[0].shape:
        return torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=0
        )
    return torch.stack(tensors)



class DensityData(torch.utils.data.Dataset):
    def __init__(self, mysql_url, **kwargs):
        super().__init__(**kwargs)
        db = connect(mysql_url)
        self.data = [i for i in range(len(db) + 1)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    



class GraphConstructor(object):
    def __init__(self, cutoff, num_probes=None, disable_pbc=True, sorted_edges=False,
                 use_gpu_graph=None, device=None):
        super().__init__()
        self.cutoff = cutoff
        self.disable_pbc = disable_pbc
        self.sorted_edges = sorted_edges
        self.default_type = torch.get_default_dtype()
        self.num_probes = num_probes

        # GPU 图构建后端
        if use_gpu_graph is True and not HAS_TORCH_CLUSTER:
            raise ImportError(
                "use_gpu_graph=True 但 torch_cluster 未安装。"
                "请安装: pip install torch-cluster"
            )
        if use_gpu_graph is None:
            # 自动检测：有 torch_cluster + CUDA 设备就用 GPU
            self._use_gpu_graph = (
                HAS_TORCH_CLUSTER
                and device is not None
                and "cuda" in str(device)
            )
        else:
            self._use_gpu_graph = use_gpu_graph
        self._device = device

    def __call__(self,
        density,
        atoms,
        grid_pos,
        threshold_distance,
        ):

        if self.disable_pbc:
            atoms = atoms.copy()
            atoms.set_pbc(False)

        probe_pos, probe_target  = self.sample_probes(grid_pos, density, atoms, threshold_distance)

        

        graph_dict = self.atoms_and_probes_to_graph(atoms, probe_pos)
        
        # pylint: disable=E1102
        graph_dict.update(
            probe_target=torch.tensor(probe_target, dtype=self.default_type),
            num_nodes=torch.tensor(graph_dict["nodes"].shape[0]),
            num_atom_edges=torch.tensor(graph_dict["atom_edges"].shape[0]),
            num_probes=torch.tensor(probe_target.shape[0]),
            num_probe_edges=torch.tensor(graph_dict["probe_edges"].shape[0]),
            probe_xyz=torch.tensor(probe_pos, dtype=self.default_type),
            atom_xyz=torch.tensor(atoms.get_positions(), dtype=self.default_type),
            cell=torch.tensor(np.array(atoms.get_cell()), dtype=self.default_type),
        )

        return graph_dict

    def sample_probes(self, grid_pos, density, atoms, threshold_distance=1.0):
        probe_pos = grid_pos.reshape(-1,3)
        probe_target = density.flatten()
        # positions = atoms.positions
        # tree = cKDTree(positions)
        # valid_indices = []
        # invalid_indices = []
        # for i, target in enumerate(probe_pos):
        #     # 查找在给定半径内的原子索引
        #     indices = tree.query_ball_point(target, threshold_distance)
        #     if len(indices) > 0:
        #         valid_indices.append(i)
        #     else:
        #         invalid_indices.append(i)
        # probe_pos_valid = probe_pos[valid_indices, :]
        # probe_target_valid = probe_target[valid_indices]


        return probe_pos, probe_target
    # def sample_probes(self, grid_pos, density):
    #     if self.num_probes is not None:
    #         # # probe_choice_max = np.prod(grid_pos.shape[0:3])
    #         # probe_choice = np.random.randint(grid_pos.shape[0], size=self.num_probes)
    #         # # probe_choice = np.unravel_index(probe_choice, grid_pos.shape[0:3])
    #         # probe_pos = grid_pos[probe_choice]
    #         # probe_target = density[probe_choice]
    #         probe_choice_max = np.prod(grid_pos.shape[0:3])
    #         probe_choice = np.random.randint(probe_choice_max, size=self.num_probes)
    #         probe_choice = np.unravel_index(probe_choice, grid_pos.shape[0:3])
    #         probe_pos = grid_pos[probe_choice]
    #         probe_target = density[probe_choice]            
    #     else:
    #         probe_pos = grid_pos.reshape(-1,3)
    #         if len(density.shape) == 4: # spin density TODO: have actual arg for spin
    #             probe_target = density.reshape(-1, 2)
    #         else:
    #             probe_target = density.flatten()
    #     return probe_pos, probe_target

    
    def precompute_atom_data(self, atoms):
        """预计算原子相关的不变数据，供 batch 循环复用"""
        if self.disable_pbc:
            atoms = atoms.copy()
            atoms.set_pbc(False)

        atom_edges, atom_edges_disp, neighborlist, inv_cell_T = self.atoms_to_graph(atoms)

        # 预计算超胞（probes_to_graph 中每次重复的部分）
        atom_positions = atoms.positions
        atom_idx = np.arange(len(atoms))
        pbc = atoms.get_pbc()
        cell_heights = _cell_heights(atoms.get_cell())
        n_rep = np.ceil(self.cutoff / (cell_heights + 1e-12))
        _rep = lambda dim: np.arange(-n_rep[dim], n_rep[dim] + 1) if pbc[dim] else [0]
        repeat_offsets = np.array([(x, y, z) for x in _rep(0) for y in _rep(1) for z in _rep(2)])
        total_repeats = repeat_offsets.shape[0]
        repeat_offsets = np.dot(repeat_offsets, atoms.get_cell())
        supercell_atom_pos = np.repeat(atom_positions[..., None, :], total_repeats, axis=-2)
        supercell_atom_pos += repeat_offsets
        supercell_atom_idx = np.repeat(atom_idx[:, None], total_repeats, axis=-1)
        supercell_atom_positions = supercell_atom_pos.reshape(-1, 3)
        supercell_atom_idx_flat = supercell_atom_idx.reshape(-1)

        # 预创建不变的 tensor
        nodes_tensor = torch.tensor(atoms.get_atomic_numbers())
        atom_edges_tensor = torch.tensor(np.concatenate(atom_edges, axis=0))
        atom_edges_disp_tensor = torch.tensor(
            np.concatenate(atom_edges_disp, axis=0), dtype=self.default_type)
        atom_xyz_tensor = torch.tensor(atoms.get_positions(), dtype=self.default_type)
        cell_tensor = torch.tensor(np.array(atoms.get_cell()), dtype=self.default_type)

        cache = {
            'atoms': atoms,
            'atom_positions': atom_positions,
            'inv_cell_T': inv_cell_T,
            'supercell_atom_positions': supercell_atom_positions,
            'supercell_atom_idx': supercell_atom_idx_flat,
            'nodes': nodes_tensor,
            'atom_edges': atom_edges_tensor,
            'atom_edges_displacement': atom_edges_disp_tensor,
            'atom_xyz': atom_xyz_tensor,
            'cell': cell_tensor,
            'num_nodes': torch.tensor(nodes_tensor.shape[0]),
            'num_atom_edges': torch.tensor(atom_edges_tensor.shape[0]),
        }

        if self._use_gpu_graph:
            # GPU 模式：缓存 GPU tensor，不建 KDTree
            dev = self._device
            cache['supercell_atom_positions_gpu'] = torch.tensor(
                supercell_atom_positions, dtype=self.default_type, device=dev)
            cache['supercell_atom_idx_gpu'] = torch.tensor(
                supercell_atom_idx_flat, dtype=torch.long, device=dev)
            cache['atom_positions_gpu'] = torch.tensor(
                atom_positions, dtype=self.default_type, device=dev)
            cache['inv_cell_T_gpu'] = torch.tensor(
                inv_cell_T, dtype=self.default_type, device=dev)
        else:
            # CPU 模式：建 KDTree
            cache['atom_kdtree'] = KDTree(supercell_atom_positions)

        return cache

    def build_graph_with_cache(self, density, grid_pos, cache, inference=False):
        """使用预计算缓存构建图，只计算 probe 相关部分"""
        probe_pos = grid_pos.reshape(-1, 3)
        num_probes = probe_pos.shape[0]

        if self._use_gpu_graph:
            probe_edges, probe_edges_displacement = self._build_graph_gpu(probe_pos, cache)
        else:
            probe_edges, probe_edges_displacement = self._build_graph_kdtree(probe_pos, cache)

        graph_dict = {
            "nodes": cache['nodes'],
            "atom_edges": cache['atom_edges'],
            "atom_edges_displacement": cache['atom_edges_displacement'],
            "probe_edges": torch.tensor(probe_edges),
            "probe_edges_displacement": torch.tensor(
                probe_edges_displacement, dtype=self.default_type),
            "num_nodes": cache['num_nodes'],
            "num_atom_edges": cache['num_atom_edges'],
            "num_probes": torch.tensor(num_probes),
            "num_probe_edges": torch.tensor(probe_edges.shape[0]),
            "probe_xyz": torch.tensor(probe_pos, dtype=self.default_type),
            "atom_xyz": cache['atom_xyz'],
            "cell": cache['cell'],
        }
        if not inference:
            probe_target = density.flatten()
            graph_dict["probe_target"] = torch.tensor(probe_target, dtype=self.default_type)
        return graph_dict

    def _build_graph_kdtree(self, probe_pos, cache):
        """CPU KDTree 后端：scipy KDTree 邻居搜索"""
        probe_kdtree = KDTree(probe_pos)
        query = probe_kdtree.query_ball_tree(cache['atom_kdtree'], r=self.cutoff)

        edges_per_probe = np.array([len(q) for q in query])
        dest_node_idx = np.repeat(np.arange(len(edges_per_probe)), edges_per_probe)
        supercell_neigh_idx = np.concatenate(query).astype(int)
        src_node_idx = cache['supercell_atom_idx'][supercell_neigh_idx]
        probe_edges = np.stack((src_node_idx, dest_node_idx), axis=1)

        atom_positions = cache['atom_positions']
        supercell_atom_positions = cache['supercell_atom_positions']
        inv_cell_T = cache['inv_cell_T']

        src_pos = atom_positions[src_node_idx]
        dest_pos = probe_pos[dest_node_idx]
        neigh_vecs = supercell_atom_positions[supercell_neigh_idx] - dest_pos
        neigh_origin = neigh_vecs + dest_pos - src_pos
        probe_edges_displacement = np.round(inv_cell_T.dot(neigh_origin.T).T)

        return probe_edges, probe_edges_displacement

    def _build_graph_gpu(self, probe_pos, cache):
        """GPU torch_cluster 后端：CUDA 加速邻居搜索"""
        dev = self._device
        probe_pos_gpu = torch.tensor(probe_pos, dtype=self.default_type, device=dev)
        supercell_pos_gpu = cache['supercell_atom_positions_gpu']

        # torch_cluster.radius: 找 y 中每个点在 x 中的邻居
        # 返回 edge_index: [2, num_edges], row=y_idx(probe), col=x_idx(supercell_atom)
        # 注意: torch_cluster 用 strict < ，scipy 用 <=，加 epsilon 保持一致
        edge_index = tc_radius(
            x=supercell_pos_gpu,
            y=probe_pos_gpu,
            r=self.cutoff + 1e-6,
        )
        probe_idx = edge_index[0]          # probe 索引
        supercell_neigh_idx = edge_index[1]  # 超胞原子索引

        # 映射回原始原子索引
        src_node_idx = cache['supercell_atom_idx_gpu'][supercell_neigh_idx]

        # GPU 上计算 displacement
        atom_pos_gpu = cache['atom_positions_gpu']
        inv_cell_T_gpu = cache['inv_cell_T_gpu']

        src_pos = atom_pos_gpu[src_node_idx]
        dest_pos = probe_pos_gpu[probe_idx]
        neigh_vecs = supercell_pos_gpu[supercell_neigh_idx] - dest_pos
        neigh_origin = neigh_vecs + dest_pos - src_pos
        probe_edges_displacement_gpu = torch.round(
            (inv_cell_T_gpu @ neigh_origin.T).T
        )

        # 转回 numpy
        probe_edges = torch.stack(
            (src_node_idx, probe_idx), dim=1
        ).cpu().numpy()
        probe_edges_displacement = probe_edges_displacement_gpu.cpu().numpy()

        return probe_edges, probe_edges_displacement

    def atoms_and_probes_to_graph(self, atoms, probe_pos):
        atom_edges, atom_edges_displacement, neighborlist, inv_cell_T = self.atoms_to_graph(atoms)
        
        probe_edges, probe_edges_displacement = self.probes_to_graph(atoms, probe_pos, 
            neighborlist=neighborlist, inv_cell_T=inv_cell_T)        

        if self.sorted_edges:
            # Sort probe edges for reproducibility
            concat_pe = _sort_by_rows(np.concatenate((probe_edges, probe_edges_displacement), axis=1))
            probe_edges = concat_pe[:,:2].astype(int)
            probe_edges_displacement = concat_pe[:,2:]

        graph_dict = {
            "nodes": torch.tensor(atoms.get_atomic_numbers()),
            "atom_edges": torch.tensor(np.concatenate(atom_edges, axis=0)),
            "atom_edges_displacement": torch.tensor(
                np.concatenate(atom_edges_displacement, axis=0), dtype=self.default_type
            ),
            "probe_edges": torch.tensor(probe_edges),
            "probe_edges_displacement": torch.tensor(
                probe_edges_displacement, dtype=self.default_type
            ),
        }
        return graph_dict

    def atoms_to_graph(self, atoms):
        atom_edges = []
        atom_edges_displacement = []

        inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

        # Compute neighborlist
        if (
            True # force ASE
            or np.any(atoms.get_cell().lengths() <= 0.0001)
            or (
                np.any(atoms.get_pbc())
                and np.any(_cell_heights(atoms.get_cell()) < self.cutoff)
            )
        ):
            neighborlist = AseNeigborListWrapper(self.cutoff, atoms)
        else:
            # neighborlist = AseNeigborListWrapper(cutoff, atoms)
            neighborlist = asap3.FullNeighborList(self.cutoff, atoms)

        atom_positions = atoms.get_positions()

        for i in range(len(atoms)):
            neigh_idx, neigh_vec, _ = neighborlist.get_neighbors(i, self.cutoff)

            self_index = np.ones_like(neigh_idx) * i
            edges = np.stack((neigh_idx, self_index), axis=1)

            neigh_pos = atom_positions[neigh_idx]
            this_pos = atom_positions[i]
            neigh_origin = neigh_vec + this_pos - neigh_pos
            neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

            atom_edges.append(edges)
            atom_edges_displacement.append(neigh_origin_scaled)

        return atom_edges, atom_edges_displacement, neighborlist, inv_cell_T

    def probes_to_graph(self, atoms, probe_pos, neighborlist=None, inv_cell_T=None):
        # FIXME: can turn this into atoms_and_probes_to_graph. The atoms NNs can be extracted
        # from the KD tree. This will circumvent ASAP/Ase completely
        atom_positions = atoms.positions
        atom_idx = np.arange(len(atoms))

        if inv_cell_T is None:
            inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

        # get number of repeats in each dimension
        pbc = atoms.get_pbc()
        cell_heights = _cell_heights(atoms.get_cell())
        n_rep = np.ceil(self.cutoff / (cell_heights + 1e-12))
        _rep = lambda dim: np.arange(-n_rep[dim], n_rep[dim] + 1) if pbc[dim] else [0]
        repeat_offsets = np.array([(x, y, z) for x in _rep(0) for y in _rep(1) for z in _rep(2)])

        # total repeats in all dimensions
        total_repeats = repeat_offsets.shape[0]
        # project repeat cell offsets into cartesian space
        repeat_offsets = np.dot(repeat_offsets, atoms.get_cell())
        # tile grid positions, subtract offsets
        # (subtracting grid positions is like adding atom positions)
        supercell_atom_pos = np.repeat(atom_positions[..., None, :], total_repeats, axis=-2)
        supercell_atom_pos += repeat_offsets

        # store the original index of each atom
        supercell_atom_idx = np.repeat(atom_idx[:, None], total_repeats, axis=-1)

        # flatten
        supercell_atom_positions = supercell_atom_pos.reshape(np.prod(supercell_atom_pos.shape[:2]), 3)
        supercell_atom_idx = supercell_atom_idx.reshape(np.prod(supercell_atom_pos.shape[:2]))

        if self._use_gpu_graph:
            # GPU 后端
            dev = self._device
            probe_pos_gpu = torch.tensor(probe_pos, dtype=self.default_type, device=dev)
            supercell_pos_gpu = torch.tensor(
                supercell_atom_positions, dtype=self.default_type, device=dev)
            supercell_idx_gpu = torch.tensor(
                supercell_atom_idx, dtype=torch.long, device=dev)
            atom_pos_gpu = torch.tensor(
                atom_positions, dtype=self.default_type, device=dev)
            inv_cell_T_gpu = torch.tensor(
                inv_cell_T, dtype=self.default_type, device=dev)

            edge_index = tc_radius(
                x=supercell_pos_gpu,
                y=probe_pos_gpu,
                r=self.cutoff + 1e-6,
            )
            probe_idx = edge_index[0]
            supercell_neigh_idx = edge_index[1]

            src_node_idx = supercell_idx_gpu[supercell_neigh_idx]
            src_pos = atom_pos_gpu[src_node_idx]
            dest_pos = probe_pos_gpu[probe_idx]
            neigh_vecs = supercell_pos_gpu[supercell_neigh_idx] - dest_pos
            neigh_origin = neigh_vecs + dest_pos - src_pos
            probe_edges_displacement_gpu = torch.round(
                (inv_cell_T_gpu @ neigh_origin.T).T
            )

            probe_edges = torch.stack(
                (src_node_idx, probe_idx), dim=1
            ).cpu().numpy()
            probe_edges_displacement = probe_edges_displacement_gpu.cpu().numpy()
        else:
            # CPU KDTree 后端
            atom_kdtree = KDTree(supercell_atom_positions)
            probe_kdtree = KDTree(probe_pos)

            # query points between kd tree
            query = probe_kdtree.query_ball_tree(atom_kdtree, r=self.cutoff)

            # set up vector of destination nodes (probes)
            edges_per_probe = np.array([len(q) for q in query])
            dest_node_idx = np.repeat(np.arange(len(edges_per_probe)), edges_per_probe)

            # get original atom idx from supercell idx
            supercell_neigh_idx = np.concatenate(query).astype(int)
            src_node_idx = supercell_atom_idx[supercell_neigh_idx]
            # create edges from src/dest nodes
            probe_edges = np.stack((src_node_idx, dest_node_idx), axis=1)

            # get non-supercell atom positions
            src_pos = atom_positions[src_node_idx]
            dest_pos = probe_pos[dest_node_idx]

            # edge vector between supercell atoms and probe
            neigh_vecs = supercell_atom_positions[supercell_neigh_idx] - dest_pos
            # compute displacement (number of unitcells in each dim)
            neigh_origin = neigh_vecs + dest_pos - src_pos
            probe_edges_displacement = np.round(inv_cell_T.dot(neigh_origin.T).T)

        return probe_edges, probe_edges_displacement

# class KdTreeGraphConstructor(GraphConstructor):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def probes_to_graph(self, atoms, probe_pos, neighborlist=None, inv_cell_T=None):
#         # FIXME: can turn this into atoms_and_probes_to_graph. The atoms NNs can be extracted
#         # from the KD tree. This will circumvent ASAP/Ase completely
#         atom_positions = atoms.positions
#         atom_idx = np.arange(len(atoms))

#         if inv_cell_T is None:
#             inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

#         # get number of repeats in each dimension
#         pbc = atoms.get_pbc()
#         cell_heights = _cell_heights(atoms.get_cell())
#         n_rep = np.ceil(self.cutoff / (cell_heights + 1e-12))
#         _rep = lambda dim: np.arange(-n_rep[dim], n_rep[dim] + 1) if pbc[dim] else [0]
#         repeat_offsets = np.array([(x, y, z) for x in _rep(0) for y in _rep(1) for z in _rep(2)])

#         # total repeats in all dimensions
#         total_repeats = repeat_offsets.shape[0]
#         # project repeat cell offsets into cartesian space
#         repeat_offsets = np.dot(repeat_offsets, atoms.get_cell())
#         # tile grid positions, subtract offsets 
#         # (subtracting grid positions is like adding atom positions)
#         supercell_atom_pos = np.repeat(atom_positions[..., None, :], total_repeats, axis=-2)
#         supercell_atom_pos += repeat_offsets
        
#         # store the original index of each atom
#         supercell_atom_idx = np.repeat(atom_idx[:, None], total_repeats, axis=-1)

#         # flatten
#         supercell_atom_positions = supercell_atom_pos.reshape(np.prod(supercell_atom_pos.shape[:2]), 3)
#         supercell_atom_idx = supercell_atom_idx.reshape(np.prod(supercell_atom_pos.shape[:2]))

#         # create KDTrees for atoms and probes
#         atom_kdtree = KDTree(supercell_atom_positions)
#         probe_kdtree = KDTree(probe_pos)

#         # query points between kd tree
#         query = probe_kdtree.query_ball_tree(atom_kdtree, r=self.cutoff)

#         # set up vector of destination nodes (probes)
#         edges_per_probe = [len(q) for q in query]
#         dest_node_idx = np.concatenate([[i]*n for i,n in enumerate(edges_per_probe)]).astype(int)

#         # get original atom idx from supercell idx
#         supercell_neigh_idx = np.concatenate(query).astype(int)
#         src_node_idx = supercell_atom_idx[supercell_neigh_idx]
#         # create edges from src/dest nodes
#         probe_edges = np.stack((src_node_idx, dest_node_idx), axis=1)

#         # get non-supercell atom positions
#         src_pos = atom_positions[src_node_idx]
#         dest_pos = probe_pos[dest_node_idx]

#         # FIXME: on the next two lines, what is the purpose of dest_pos? 
#         # edge vector between supercell atoms and probe
#         neigh_vecs = supercell_atom_positions[supercell_neigh_idx] - dest_pos
#         # compute displacement (number of unitcells in each dim)
#         neigh_origin = neigh_vecs + dest_pos - src_pos
#         probe_edges_displacement = np.round(inv_cell_T.dot(neigh_origin.T).T)

#         return probe_edges, probe_edges_displacement

class AseNeigborListWrapper:
    """
    Wrapper around ASE neighborlist to have the same interface as asap3 neighborlist

    """

    def __init__(self, cutoff, atoms):
        self.neighborlist = ase.neighborlist.NewPrimitiveNeighborList(
            cutoff, skin=0.0, self_interaction=False, bothways=True
        )
        self.neighborlist.build(
            atoms.get_pbc(), atoms.get_cell(), atoms.get_positions()
        )
        self.cutoff = cutoff
        self.atoms_positions = atoms.get_positions()
        self.atoms_cell = atoms.get_cell()

    def get_neighbors(self, i, cutoff):
        assert (
            cutoff == self.cutoff
        ), "Cutoff must be the same as used to initialise the neighborlist"

        indices, offsets = self.neighborlist.get_neighbors(i)

        rel_positions = (
            self.atoms_positions[indices]
            + offsets @ self.atoms_cell
            - self.atoms_positions[i][None]
        )

        dist2 = np.sum(np.square(rel_positions), axis=1)
        return indices, rel_positions, dist2

def _cell_heights(cell_object):
    volume = cell_object.volume
    crossproducts = np.cross(cell_object[[1, 2, 0]], cell_object[[2, 0, 1]])
    crosslengths = np.sqrt(np.sum(np.square(crossproducts), axis=1))
    heights = volume / crosslengths
    return heights

def _sort_by_rows(arr):
    assert len(arr.shape) == 2, "Only 2D arrays"
    return np.array(sorted([tuple(x) for x in arr.tolist()]))





class MyCollator(object):
    def __init__(self, mysql_url, cutoff=4.0, num_probes=None, if_disable_pbc=False, data_batch_size=3000, inference=False):
        self.mysql_url = mysql_url
        self.graph_constructor = GraphConstructor(cutoff=cutoff, num_probes=num_probes, disable_pbc=if_disable_pbc)
        self.cutoff = cutoff
        self.data_batch_size = data_batch_size
        self.inference = inference
        
    def __call__(self, examples):
        # ind_examples = sorted(range(len(examples)), key=lambda k: examples[k])


        rows = query_database(self.mysql_url, examples)

        list_of_dicts = []
        list_of_dicts_invalid = []
        for row in rows:
            atoms = row.toatoms()
            atoms.set_pbc(True)
            nx, ny, nz = row.data['nx'], row.data['ny'], row.data['nz']
            probe_pos = get_grid_centers(atoms, nx, ny, nz)
            # density = row.data['chg']
            try:
                density = row.data['chg']
                density = row.data['chg'].reshape(nx, ny, nz)
            except:
                density = np.zeros((nx, ny, nz))
            
            num_positions = nx * ny * nz
            batch_size = self.data_batch_size

            # 预计算原子不变数据（只算一次，所有 batch 复用）
            atom_cache = self.graph_constructor.precompute_atom_data(atoms)
            total_num_probes_tensor = torch.tensor([nx, ny, nz]).long()

            for start_idx in tqdm(
                            range(0, num_positions, batch_size),
                            desc="Processing Batches",
                            total=(num_positions + batch_size - 1) // batch_size,
                            leave=False
                        ):


                end_idx = min(start_idx + batch_size, num_positions)

                unravel_indices = np.unravel_index(np.arange(start_idx, end_idx), (nx, ny, nz))
                probe_pos_batch = probe_pos[unravel_indices]
                density_batch = density[unravel_indices]

                graph_dict = self.graph_constructor.build_graph_with_cache(
                    density=density_batch, grid_pos=probe_pos_batch, cache=atom_cache,
                    inference=self.inference)

                if not self.inference:
                    graph_dict.update(total_num_probes=total_num_probes_tensor)

                # 单元素直接 unsqueeze，避免 pad_and_stack 开销
                list_of_dicts.append({k: v.unsqueeze(0) for k, v in graph_dict.items()})
            

        return list_of_dicts




def res_input_constructor(atoms, grid_pos, graph_dict, threshold_distance):
    positions = atoms.positions
    tree = cKDTree(positions)
    chemical_symbols = atoms.get_atomic_numbers()
    atom_inputs = []
    num_atom_inputs = []
    distances_list = []
    need_fix = []
    for i, target in enumerate(grid_pos):
        # 查找在给定半径内的原子索引
        indices = tree.query_ball_point(target, threshold_distance)
        distances = torch.tensor([euclidean(target, positions[index]) for index in indices])
        atom_chem = torch.tensor([chemical_symbols[item] for item in indices])
        # atom_indx = torch.tensor(indices)
        if len(indices) > 0:
            need_fix.append(torch.tensor(1).unsqueeze(0))
        else:
            need_fix.append(torch.tensor(0).unsqueeze(0))


        distances_list.append(distances)
        num_atoms = len(indices)
        atom_inputs.append(atom_chem)
        # atom_inputs.append(atom_indx)
        num_atom_inputs.append(num_atoms)
    
    graph_dict.update(
        neighbour_atom=torch.cat(atom_inputs, dim=0).long(),
        num_neighbour_atoms=torch.tensor(num_atom_inputs).long(),
        distances=torch.cat(distances_list, dim=0).float(),
        need_fix=torch.cat(need_fix, dim=0).float()
        )
    return graph_dict


def collate_list_of_dicts(list_of_dicts, pin_memory=False):
    # Convert from "list of dicts" to "dict of lists"
    dict_of_lists = {k: [d[k] for d in list_of_dicts] for k in list_of_dicts[0]}

    # Convert each list of tensors to single tensor with pad and stack
    if pin_memory:
        pin = lambda x: x.pin_memory()
    else:
        pin = lambda x: x

    collated = {}
    for k,v  in dict_of_lists.items():
        if not k in ["filename",
                     "load_time"]:
            collated[k] = pin(pad_and_stack(v))
        else:
            collated[k] = torch.cat(v, dim=0)

    # collated = {k: pin(pad_and_stack(dict_of_lists[k])) for k in dict_of_lists}
    return collated



def get_grid_centers(atoms, nx, ny, nz):
    """
    Read atoms and cell information from a VASP file and generate Cartesian coordinates of the grid center points.

    Parameters:
    filename (str): File name, VASP file containing cell information (such as POSCAR).
    nx, ny, nz (int): Number of grid divisions.

    Returns:
    np.ndarray: Cartesian coordinate array of grid center points, shape (nx*ny*nz, 3).
    """

    ngridpts = np.array([nx, ny, nz])  # grid matrix
    grid_pos = np.meshgrid(
        np.arange(ngridpts[0]) / nx,
        np.arange(ngridpts[1]) / ny,
        np.arange(ngridpts[2]) / nz,
        indexing="ij",
    )
    grid_pos = np.stack(grid_pos, 3)
    grid_pos = np.dot(grid_pos, atoms.get_cell())
    grid_pos = grid_pos

    
    return np.array(grid_pos)



def query_database(mysql_url, idx_list):
    if "db" in str(mysql_url):
        db = connect(mysql_url)
        rows = []
        for idx in idx_list:
            #print(idx)
            try:
                it = db.get(id=idx)
                rows.append(it)
            except:
                print(f"err match id {idx}")
                pass
        return rows
    
    
    else:
        count = 0
        while True:
            try:
                db = connect(mysql_url)
                con = db._connect().cursor()

                
                input = {}
                cmps = [('id', ' IN ', '1,2')]

                keys = []  # No additional keys are needed for a simple ID query
                sort = None  # Assuming no sorting is required
                order = None  # Default order, can be 'DESC' for descending
                sort_table = None  # Not needed for an ID query
                columns = 'all'  # If you want all columns, otherwise specify the columns you need

                values = np.array([None for i in range(27)])
                values[25] = '{}'
                columnindex = list(range(27))

                what = ', '.join('systems.' + name
                                    for name in
                                    np.array(db.columnnames)[np.array(columnindex)])


                sql, args = db.create_select_statement(keys, cmps, sort, order,
                                                            sort_table, what)

                args = [tuple(idx_list)]


                con.execute(sql, args)

                deblob = db.deblob
                decode = db.decode

                rows = []

                for shortvalues in con.fetchall():
                    values[columnindex] = shortvalues
                    dct = {'id': values[0],
                                'unique_id': values[1],
                                'ctime': values[2],
                                'mtime': values[3],
                                'user': values[4],
                                'numbers': deblob(values[5], np.int32),
                                'positions': deblob(values[6], shape=(-1, 3)),
                                'cell': deblob(values[7], shape=(3, 3))}

                    # if values[8] is not None:
                    #     dct['pbc'] = (values[8] & np.array([1, 2, 4])).astype(bool)
                    # if values[9] is not None:
                    #     dct['initial_magmoms'] = deblob(values[9])
                    # if values[10] is not None:
                    #     dct['initial_charges'] = deblob(values[10])
                    # if values[11] is not None:
                    #     dct['masses'] = deblob(values[11])
                    # if values[12] is not None:
                    #     dct['tags'] = deblob(values[12], np.int32)
                    # if values[13] is not None:
                    #     dct['momenta'] = deblob(values[13], shape=(-1, 3))
                    # if values[14] is not None:
                    #     dct['constraints'] = values[14]
                    # if values[15] is not None:
                    #     dct['calculator'] = values[15]
                    # if values[16] is not None:
                    #     dct['calculator_parameters'] = decode(values[16])
                    # if values[17] is not None:
                    #     dct['energy'] = values[17]
                    # if values[18] is not None:
                    #     dct['free_energy'] = values[18]
                    # if values[19] is not None:
                    #     dct['forces'] = deblob(values[19], shape=(-1, 3))
                    # if values[20] is not None:
                    #     dct['stress'] = deblob(values[20])
                    # if values[21] is not None:
                    #     dct['dipole'] = deblob(values[21])
                    # if values[22] is not None:
                    #     dct['magmoms'] = deblob(values[22])
                    # if values[23] is not None:
                    #     dct['magmom'] = values[23]
                    # if values[24] is not None:
                    #     dct['charges'] = deblob(values[24])
                    # if values[25] != '{}':
                    #     dct['key_value_pairs'] = decode(values[25])
                    if len(values) >= 27 and values[26] != 'null':
                        dct['data'] = decode(values[26], lazy=True)
                    
                    # external_tab = db._get_external_table_names()
                    # tables = {}
                    # for tab in external_tab:
                    #     row = self._read_external_table(tab, dct["id"])
                    #     tables[tab] = row

                    # dct.update(tables)
                    rows.append(AtomsRow(dct))
                return rows
            except Exception as e:
                time.sleep(1)  # 等待一秒再重试
                count += 1
                print(f"trying {count} times")