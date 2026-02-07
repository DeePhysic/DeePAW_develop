import torch
import numpy as np
import ase
import ase.neighborlist
from ase.db.row import AtomsRow
import time
import traceback
from tqdm import tqdm
from ase.db import connect
from scipy.spatial import KDTree, cKDTree
from scipy.spatial.distance import euclidean

# DeePAW imports
from deepaw.data.layer import pad_and_stack

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
    def __init__(self, cutoff, num_probes=None, disable_pbc=False, sorted_edges=False):
        super().__init__()
        self.cutoff = cutoff
        self.disable_pbc = disable_pbc
        self.sorted_edges = sorted_edges
        self.default_type = torch.get_default_dtype()
        self.num_probes = num_probes

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
        atom_kdtree = KDTree(supercell_atom_positions)

        nodes_tensor = torch.tensor(atoms.get_atomic_numbers())
        atom_edges_tensor = torch.tensor(np.concatenate(atom_edges, axis=0))
        atom_edges_disp_tensor = torch.tensor(
            np.concatenate(atom_edges_disp, axis=0), dtype=self.default_type)
        atom_xyz_tensor = torch.tensor(atoms.get_positions(), dtype=self.default_type)
        cell_tensor = torch.tensor(np.array(atoms.get_cell()), dtype=self.default_type)

        return {
            'atoms': atoms,
            'atom_positions': atom_positions,
            'inv_cell_T': inv_cell_T,
            'supercell_atom_positions': supercell_atom_positions,
            'supercell_atom_idx': supercell_atom_idx_flat,
            'atom_kdtree': atom_kdtree,
            'nodes': nodes_tensor,
            'atom_edges': atom_edges_tensor,
            'atom_edges_displacement': atom_edges_disp_tensor,
            'atom_xyz': atom_xyz_tensor,
            'cell': cell_tensor,
            'num_nodes': torch.tensor(nodes_tensor.shape[0]),
            'num_atom_edges': torch.tensor(atom_edges_tensor.shape[0]),
        }

    def build_graph_with_cache(self, density, grid_pos, cache, inference=False):
        """使用预计算缓存构建图，只计算 probe 相关部分"""
        probe_pos = grid_pos.reshape(-1, 3)
        num_probes = probe_pos.shape[0]

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

        # create KDTrees for atoms and probes
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

        # FIXME: on the next two lines, what is the purpose of dest_pos? 
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





# class MyCollator(object):
class MyCollator(object):
    def __init__(self, mysql_url, cutoff=4.0, num_probes=None, data_batch_size=3000, inference=False):
        self.mysql_url = mysql_url
        self.graph_constructor = GraphConstructor(cutoff=cutoff, num_probes=num_probes)
        self.data_batch_size = data_batch_size
        self.inference = inference
        print(f"初始化SimpleMyCollator: data_batch_size={data_batch_size}")
        
    def __call__(self, examples):
        print(f"SimpleMyCollator处理 {len(examples)} 个示例")
        
        rows = query_database(self.mysql_url, examples)
        print(f"从数据库获取了 {len(rows)} 条记录")
        
        list_of_dicts = []
        
        for row_idx, row in enumerate(rows):
            print(f"处理行 {row_idx+1}/{len(rows)}")
            atoms = row.toatoms()
            atoms.set_pbc(True)
            nx, ny, nz = row.data['nx'], row.data['ny'], row.data['nz']
            print(f"网格大小: {nx} x {ny} x {nz}")
            
            probe_pos = get_grid_centers(atoms, nx, ny, nz)
            
            try:
                density = row.data['chg']
                density = density.reshape(nx, ny, nz)
                print(f"成功加载电荷密度，形状: {density.shape}")
            except Exception as e:
                print(f"加载电荷密度失败: {str(e)}，使用零密度替代")
                density = np.zeros((nx, ny, nz))
            
            # 直接处理整个密度，但使用批处理
            print(f"使用批量大小 {self.data_batch_size} 处理整个网格")
            results = self.process_density(density, atoms, probe_pos, nx, ny, nz)
            if results:
                list_of_dicts.extend(results)
            else:
                print("处理未生成结果")
        
        print(f"SimpleMyCollator共生成 {len(list_of_dicts)} 个结果")
        return list_of_dicts
    
    def process_density(self, density, atoms, probe_pos, nx, ny, nz):
        """处理密度网格，使用批处理"""
        list_of_dicts = []
        num_positions = nx * ny * nz
        batch_size = self.data_batch_size

        # 预计算原子不变数据（只算一次，所有 batch 复用）
        atom_cache = self.graph_constructor.precompute_atom_data(atoms)
        total_num_probes_tensor = torch.tensor([nx, ny, nz]).long()

        for start_idx in tqdm(
                range(0, num_positions, batch_size),
                desc="Processing Positions",
                total=(num_positions + batch_size - 1) // batch_size,
                leave=False
            ):

            end_idx = min(start_idx + batch_size, num_positions)

            unravel_indices = np.unravel_index(np.arange(start_idx, end_idx), (nx, ny, nz))
            probe_pos_batch = probe_pos[unravel_indices]
            density_batch = density[unravel_indices]

            try:
                graph_dict = self.graph_constructor.build_graph_with_cache(
                    density=density_batch,
                    grid_pos=probe_pos_batch,
                    cache=atom_cache,
                    inference=self.inference
                )

                if not self.inference:
                    graph_dict.update(total_num_probes=total_num_probes_tensor)

                list_of_dicts.append({k: v.unsqueeze(0) for k, v in graph_dict.items()})
            except Exception as e:
                print(f"处理位置 {start_idx}-{end_idx} 时出错: {str(e)}")
                traceback.print_exc()
                continue

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