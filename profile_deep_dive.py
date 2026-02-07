#!/usr/bin/env python
"""
DeePAW 深度瓶颈分析
====================
在 profile_bottleneck.py 基础上，进一步拆解:
1. res_input_constructor 循环内部各操作的耗时
2. Tensor 转换中哪些操作最慢
3. Edge 列表构建中 Python list comprehension 的开销
4. 模拟大网格场景的预估
"""

import os, sys, time
import numpy as np
import torch
from scipy.spatial import KDTree, cKDTree
from scipy.spatial.distance import euclidean

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from ase.db import connect
from deepaw.data.chgcar_writer import (
    GraphConstructor, AseNeigborListWrapper,
    get_grid_centers, _cell_heights,
)

DB_PATH = os.path.join(script_dir, 'examples', 'hfo2_chgd.db')
CUTOFF = 4.0
BATCH_SIZE = 3000

db = connect(DB_PATH)
row = db.get(1)
atoms = row.toatoms()
atoms.set_pbc(True)
nx, ny, nz = row.data['nx'], row.data['ny'], row.data['nz']

probe_pos = get_grid_centers(atoms, nx, ny, nz)
density = np.zeros((nx, ny, nz))  # 数据库无chg数据，用零代替（不影响计时）

# 取第一个 batch
start_idx, end_idx = 0, BATCH_SIZE
unravel_indices = np.unravel_index(np.arange(start_idx, end_idx), (nx, ny, nz))
probe_pos_batch = probe_pos[unravel_indices]
probe_pos_flat = probe_pos_batch.reshape(-1, 3)
density_batch = density[unravel_indices]
n_probes = len(probe_pos_flat)

# 预先构建好 probe edges (模拟 probes_to_graph 的输出)
atom_positions = atoms.get_positions()
atom_idx = np.arange(len(atoms))
inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)
pbc = atoms.get_pbc()
cell_heights = _cell_heights(atoms.get_cell())
n_rep = np.ceil(CUTOFF / (cell_heights + 1e-12))
_rep = lambda dim: np.arange(-n_rep[dim], n_rep[dim] + 1) if pbc[dim] else [0]
repeat_offsets = np.array([(x, y, z) for x in _rep(0) for y in _rep(1) for z in _rep(2)])
total_repeats = repeat_offsets.shape[0]
repeat_offsets_cart = np.dot(repeat_offsets, atoms.get_cell())
supercell_atom_pos = np.repeat(atom_positions[..., None, :], total_repeats, axis=-2)
supercell_atom_pos += repeat_offsets_cart
supercell_atom_idx = np.repeat(atom_idx[:, None], total_repeats, axis=-1)
supercell_atom_positions = supercell_atom_pos.reshape(-1, 3)
supercell_atom_idx_flat = supercell_atom_idx.reshape(-1)

atom_kdtree = KDTree(supercell_atom_positions)
probe_kdtree = KDTree(probe_pos_flat)
query = probe_kdtree.query_ball_tree(atom_kdtree, r=CUTOFF)

print("=" * 70)
print("深度瓶颈分析")
print("=" * 70)
print(f"Batch: {n_probes} probes, {len(atoms)} atoms, {total_repeats} repeats")
print(f"总 batch 数: {(nx*ny*nz + BATCH_SIZE - 1) // BATCH_SIZE}")
print()

# ============================================================
# 分析 1: res_input_constructor 循环内部拆解
# ============================================================
print("-" * 70)
print("分析 1: res_input_constructor 内部拆解")
print("-" * 70)

positions = atoms.positions
chemical_symbols = atoms.get_atomic_numbers()

# 1a: cKDTree 构建
t0 = time.perf_counter()
for _ in range(100):
    tree = cKDTree(positions)
t1 = time.perf_counter()
print(f"  cKDTree(atoms) 构建 (x100): {(t1-t0)*10:.2f}ms/次")

# 1b: 循环内各操作分别计时
t_query_ball = 0
t_euclidean = 0
t_tensor_create = 0
t_append = 0
t_total_loop = 0

tree = cKDTree(positions)
t_loop_start = time.perf_counter()

n_with_neighbors = 0
for i in range(n_probes):
    target = probe_pos_flat[i]

    t0 = time.perf_counter()
    indices = tree.query_ball_point(target, 0.1)
    t1 = time.perf_counter()
    t_query_ball += (t1 - t0)

    t0 = time.perf_counter()
    dists = [euclidean(target, positions[idx]) for idx in indices]
    t1 = time.perf_counter()
    t_euclidean += (t1 - t0)

    t0 = time.perf_counter()
    _ = torch.tensor(dists)
    _ = torch.tensor([chemical_symbols[item] for item in indices])
    if len(indices) > 0:
        _ = torch.tensor(1).unsqueeze(0)
        n_with_neighbors += 1
    else:
        _ = torch.tensor(0).unsqueeze(0)
    t1 = time.perf_counter()
    t_tensor_create += (t1 - t0)

t_total_loop = time.perf_counter() - t_loop_start

print(f"\n  循环总耗时:                    {t_total_loop*1000:.2f}ms")
print(f"    cKDTree.query_ball_point:   {t_query_ball*1000:.2f}ms  "
      f"({t_query_ball/t_total_loop*100:.1f}%)")
print(f"    euclidean() 距离计算:        {t_euclidean*1000:.2f}ms  "
      f"({t_euclidean/t_total_loop*100:.1f}%)")
print(f"    torch.tensor() 创建:        {t_tensor_create*1000:.2f}ms  "
      f"({t_tensor_create/t_total_loop*100:.1f}%)")
overhead = t_total_loop - t_query_ball - t_euclidean - t_tensor_create
print(f"    Python 循环开销 (overhead):  {overhead*1000:.2f}ms  "
      f"({overhead/t_total_loop*100:.1f}%)")
print(f"\n  有邻居的探针数: {n_with_neighbors}/{n_probes} "
      f"({n_with_neighbors/n_probes*100:.1f}%)")
print(f"  (threshold_distance=0.1 非常小，大部分探针没有邻居)")

# ============================================================
# 分析 2: Tensor 转换拆解
# ============================================================
print("\n" + "-" * 70)
print("分析 2: Tensor 转换拆解")
print("-" * 70)

# 模拟 graph_dict 中各个 tensor 的创建
edges_per_probe = [len(q) for q in query]
dest_node_idx = np.concatenate([[i]*n for i, n in enumerate(edges_per_probe)]).astype(int)
supercell_neigh_idx = np.concatenate(query).astype(int)
src_node_idx = supercell_atom_idx_flat[supercell_neigh_idx]
probe_edges = np.stack((src_node_idx, dest_node_idx), axis=1)
src_pos = atom_positions[src_node_idx]
dest_pos = probe_pos_flat[dest_node_idx]
neigh_vecs = supercell_atom_positions[supercell_neigh_idx] - dest_pos
neigh_origin = neigh_vecs + dest_pos - src_pos
probe_edges_displacement = np.round(inv_cell_T.dot(neigh_origin.T).T)

print(f"  probe_edges shape: {probe_edges.shape}")
print(f"  probe_edges_displacement shape: {probe_edges_displacement.shape}")

tensors_to_test = {
    "probe_edges (int)": (probe_edges, {}),
    "probe_edges_displacement (float32)": (probe_edges_displacement, {"dtype": torch.float32}),
    "density_batch (float32)": (density_batch.flatten(), {"dtype": torch.float32}),
    "probe_pos_flat (float32)": (probe_pos_flat, {"dtype": torch.float32}),
}

for name, (arr, kwargs) in tensors_to_test.items():
    t0 = time.perf_counter()
    for _ in range(100):
        _ = torch.tensor(arr, **kwargs)
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / 100 * 1000
    print(f"  torch.tensor({name}): {avg_ms:.3f}ms  "
          f"(shape={arr.shape}, {arr.nbytes/1024:.1f}KB)")

# 测试 torch.tensor vs torch.from_numpy
print("\n  对比 torch.tensor vs torch.from_numpy (probe_edges_displacement):")
arr = probe_edges_displacement.astype(np.float32)
t0 = time.perf_counter()
for _ in range(100):
    _ = torch.tensor(arr)
t1 = time.perf_counter()
print(f"    torch.tensor():     {(t1-t0)/100*1000:.3f}ms")

t0 = time.perf_counter()
for _ in range(100):
    _ = torch.from_numpy(arr.copy())
t1 = time.perf_counter()
print(f"    torch.from_numpy(): {(t1-t0)/100*1000:.3f}ms")

# ============================================================
# 分析 3: Edge 列表构建中的 Python 开销
# ============================================================
print("\n" + "-" * 70)
print("分析 3: Edge 列表构建 Python 开销拆解")
print("-" * 70)

t0 = time.perf_counter()
edges_per_probe = [len(q) for q in query]
t1 = time.perf_counter()
print(f"  [len(q) for q in query]:  {(t1-t0)*1000:.3f}ms")

t0 = time.perf_counter()
dest_node_idx = np.concatenate([[i]*n for i, n in enumerate(edges_per_probe)]).astype(int)
t1 = time.perf_counter()
print(f"  dest_node_idx 构建:       {(t1-t0)*1000:.3f}ms  <<<")

# 对比: 用 np.repeat 替代 list comprehension
t0 = time.perf_counter()
dest_node_idx_v2 = np.repeat(np.arange(len(edges_per_probe)), edges_per_probe)
t1 = time.perf_counter()
print(f"  dest_node_idx (np.repeat): {(t1-t0)*1000:.3f}ms  (优化版)")

t0 = time.perf_counter()
supercell_neigh_idx = np.concatenate(query).astype(int)
t1 = time.perf_counter()
print(f"  np.concatenate(query):    {(t1-t0)*1000:.3f}ms")

# ============================================================
# 分析 4: 重复计算量化
# ============================================================
print("\n" + "-" * 70)
print("分析 4: 重复计算量化")
print("-" * 70)

num_batches = (nx*ny*nz + BATCH_SIZE - 1) // BATCH_SIZE

# atoms_to_graph 在 MyCollator 中每 batch 调用一次
t0 = time.perf_counter()
atoms_copy = atoms.copy()
atoms_copy.set_pbc(False)
nl = AseNeigborListWrapper(CUTOFF, atoms_copy)
for i in range(len(atoms_copy)):
    nl.get_neighbors(i, CUTOFF)
t1 = time.perf_counter()
t_atoms_graph = t1 - t0
print(f"  atoms_to_graph 单次耗时: {t_atoms_graph*1000:.2f}ms")
print(f"  当前: 每 batch 调用 1 次 x {num_batches} batches = {num_batches} 次")
print(f"  浪费: {t_atoms_graph * (num_batches - 1) * 1000:.2f}ms")

# 超胞 + atom_kdtree 也是重复的
t0 = time.perf_counter()
_ = KDTree(supercell_atom_positions)
t1 = time.perf_counter()
t_atom_kd = t1 - t0
print(f"\n  atom KDTree 单次构建: {t_atom_kd*1000:.2f}ms")
print(f"  当前: 每 batch 重建 x {num_batches} = 浪费 {t_atom_kd*(num_batches-1)*1000:.2f}ms")

# res_input 中 cKDTree 也是重复的
t0 = time.perf_counter()
_ = cKDTree(positions)
t1 = time.perf_counter()
t_ckd = t1 - t0
print(f"\n  cKDTree(atoms) 单次构建: {t_ckd*1000:.4f}ms")
print(f"  当前: 每 batch 重建 x {num_batches} = 浪费 {t_ckd*(num_batches-1)*1000:.2f}ms")

# ============================================================
# 分析 5: 大网格场景预估
# ============================================================
print("\n" + "-" * 70)
print("分析 5: 大网格场景预估")
print("-" * 70)

scenarios = [
    ("当前 HfO2", nx, ny, nz, len(atoms)),
    ("中等体系", 100, 100, 100, 30),
    ("大体系", 144, 144, 200, 50),
    ("超大体系", 144, 144, 448, 100),
]

avg_batch_time_ms = np.mean([
    # 从 profile_bottleneck.py 的结果
    # 这里用当前测量的数据
    t_total_loop * 1000  # res_input 循环
]) + 41  # tensor 转换 (从之前的结果)
# 更准确: 用实际测量的总 batch 时间
# 从之前的 profiling: ~112ms per batch for 80x80x80

for name, gx, gy, gz, n_atoms in scenarios:
    n_grid = gx * gy * gz
    n_batches = (n_grid + BATCH_SIZE - 1) // BATCH_SIZE
    # 粗略线性外推 (实际可能更慢因为更多原子)
    est_s = n_batches * 112.5 / 1000  # 用 112.5ms/batch
    print(f"  {name:15s}: {gx}x{gy}x{gz} = {n_grid:>12,} points, "
          f"{n_batches:>5} batches, 预估 ~{est_s:.0f}s ({est_s/60:.1f}min)")

print("\n  注意: 原子数增多时, KDTree query 和 edge 构建会更慢,")
print("  实际耗时可能比线性外推更长。")

print("\n" + "=" * 70)
print("深度分析完成!")
print("=" * 70)
