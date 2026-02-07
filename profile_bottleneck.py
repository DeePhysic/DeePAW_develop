#!/usr/bin/env python
"""
DeePAW CPU Bottleneck Profiler
逐步计时预处理管线的每个阶段，定位 CPU 瓶颈
"""

import os
import sys
import time
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from ase.db import connect
from scipy.spatial import KDTree, cKDTree
from scipy.spatial.distance import euclidean
import torch

from deepaw.data.chgcar_writer import (
    GraphConstructor, AseNeigborListWrapper,
    get_grid_centers, _cell_heights, query_database,
)

# ============================================================
# 配置
# ============================================================
DB_PATH = os.path.join(script_dir, 'examples', 'hfo2_chgd.db')
CUTOFF = 4.0
BATCH_SIZE = 3000
NUM_BATCHES_TO_PROFILE = 5  # 只测前5个batch就够了

if not os.path.exists(DB_PATH):
    print(f"数据库不存在: {DB_PATH}")
    sys.exit(1)

# ============================================================
# 加载数据
# ============================================================
print("=" * 70)
print("DeePAW CPU Bottleneck Profiler")
print("=" * 70)

db = connect(DB_PATH)
row = db.get(1)
atoms = row.toatoms()
atoms.set_pbc(True)

nx, ny, nz = row.data['nx'], row.data['ny'], row.data['nz']
num_positions = nx * ny * nz
num_batches_total = (num_positions + BATCH_SIZE - 1) // BATCH_SIZE

print(f"化学式: {row.formula}")
print(f"原子数: {len(atoms)}")
print(f"网格尺寸: {nx} x {ny} x {nz} = {num_positions:,} 个探针点")
print(f"Batch size: {BATCH_SIZE}")
print(f"总 batch 数: {num_batches_total}")
print(f"Profiling 前 {NUM_BATCHES_TO_PROFILE} 个 batch")
print()

# ============================================================
# Stage 1: 网格生成
# ============================================================
print("-" * 70)
print("Stage 1: 网格生成 (get_grid_centers)")
print("-" * 70)

t0 = time.perf_counter()
probe_pos = get_grid_centers(atoms, nx, ny, nz)
t1 = time.perf_counter()
print(f"  网格形状: {probe_pos.shape}")
print(f"  耗时: {t1 - t0:.4f} 秒")
print(f"  预估全量: {t1 - t0:.4f} 秒 (只执行一次)")
print()

# ============================================================
# Stage 2: 密度加载和 reshape
# ============================================================
print("-" * 70)
print("Stage 2: 密度加载和 reshape")
print("-" * 70)

t0 = time.perf_counter()
try:
    density = row.data['chg'].reshape(nx, ny, nz)
except:
    density = np.zeros((nx, ny, nz))
t1 = time.perf_counter()
print(f"  密度形状: {density.shape}")
print(f"  耗时: {t1 - t0:.4f} 秒")
print()

# ============================================================
# Stage 3: Atom-Atom 邻居表 (只执行一次)
# ============================================================
print("-" * 70)
print("Stage 3: Atom-Atom 邻居表构建 (ASE NeighborList)")
print("-" * 70)

inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

t0 = time.perf_counter()
neighborlist = AseNeigborListWrapper(CUTOFF, atoms)
t1 = time.perf_counter()
print(f"  ASE NeighborList 构建耗时: {t1 - t0:.4f} 秒")

t0 = time.perf_counter()
atom_positions = atoms.get_positions()
atom_edges = []
for i in range(len(atoms)):
    neigh_idx, neigh_vec, _ = neighborlist.get_neighbors(i, CUTOFF)
    self_index = np.ones_like(neigh_idx) * i
    edges = np.stack((neigh_idx, self_index), axis=1)
    neigh_pos = atom_positions[neigh_idx]
    this_pos = atom_positions[i]
    neigh_origin = neigh_vec + this_pos - neigh_pos
    neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)
    atom_edges.append(edges)
t1 = time.perf_counter()
total_atom_edges = sum(len(e) for e in atom_edges)
print(f"  Atom edge 构建耗时: {t1 - t0:.4f} 秒")
print(f"  总 atom edges: {total_atom_edges}")
print(f"  预估全量: {t1 - t0:.4f} 秒 (只执行一次)")
print()

# ============================================================
# Stage 4-7: 逐 Batch Profiling
# ============================================================
print("-" * 70)
print(f"Stage 4-7: 逐 Batch 预处理 (前 {NUM_BATCHES_TO_PROFILE} 个 batch)")
print("-" * 70)

# 收集每个阶段的耗时
times_supercell = []
times_kdtree_build = []
times_kdtree_query = []
times_edge_construct = []
times_displacement = []
times_res_input = []
times_res_kdtree = []
times_res_loop = []
times_tensor_convert = []
times_total_batch = []

pbc = atoms.get_pbc()
cell_heights = _cell_heights(atoms.get_cell())
n_rep = np.ceil(CUTOFF / (cell_heights + 1e-12))
_rep = lambda dim: np.arange(-n_rep[dim], n_rep[dim] + 1) if pbc[dim] else [0]

for batch_idx in range(NUM_BATCHES_TO_PROFILE):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, num_positions)

    t_batch_start = time.perf_counter()

    # --- 索引解包和切片 ---
    unravel_indices = np.unravel_index(
        np.arange(start_idx, end_idx), (nx, ny, nz)
    )
    probe_pos_batch = probe_pos[unravel_indices]
    density_batch = density[unravel_indices]

    probe_pos_flat = probe_pos_batch.reshape(-1, 3)
    n_probes = probe_pos_flat.shape[0]

    # --- 4a: 超胞构建 ---
    t0 = time.perf_counter()
    atom_idx = np.arange(len(atoms))
    repeat_offsets = np.array([
        (x, y, z) for x in _rep(0) for y in _rep(1) for z in _rep(2)
    ])
    total_repeats = repeat_offsets.shape[0]
    repeat_offsets_cart = np.dot(repeat_offsets, atoms.get_cell())

    supercell_atom_pos = np.repeat(
        atom_positions[..., None, :], total_repeats, axis=-2
    )
    supercell_atom_pos += repeat_offsets_cart
    supercell_atom_idx = np.repeat(atom_idx[:, None], total_repeats, axis=-1)

    supercell_atom_positions = supercell_atom_pos.reshape(
        np.prod(supercell_atom_pos.shape[:2]), 3
    )
    supercell_atom_idx_flat = supercell_atom_idx.reshape(
        np.prod(supercell_atom_pos.shape[:2])
    )
    t1 = time.perf_counter()
    times_supercell.append(t1 - t0)

    # --- 4b: KDTree 构建 ---
    t0 = time.perf_counter()
    atom_kdtree = KDTree(supercell_atom_positions)
    probe_kdtree = KDTree(probe_pos_flat)
    t1 = time.perf_counter()
    times_kdtree_build.append(t1 - t0)

    # --- 4c: KDTree query_ball_tree (核心嫌疑) ---
    t0 = time.perf_counter()
    query = probe_kdtree.query_ball_tree(atom_kdtree, r=CUTOFF)
    t1 = time.perf_counter()
    times_kdtree_query.append(t1 - t0)

    # --- 4d: Edge 构建 ---
    t0 = time.perf_counter()
    edges_per_probe = [len(q) for q in query]
    dest_node_idx = np.concatenate(
        [[i] * n for i, n in enumerate(edges_per_probe)]
    ).astype(int)
    supercell_neigh_idx = np.concatenate(query).astype(int)
    src_node_idx = supercell_atom_idx_flat[supercell_neigh_idx]
    probe_edges = np.stack((src_node_idx, dest_node_idx), axis=1)
    t1 = time.perf_counter()
    times_edge_construct.append(t1 - t0)
    total_probe_edges = len(probe_edges)

    # --- 4e: Displacement 计算 ---
    t0 = time.perf_counter()
    src_pos = atom_positions[src_node_idx]
    dest_pos = probe_pos_flat[dest_node_idx]
    neigh_vecs = supercell_atom_positions[supercell_neigh_idx] - dest_pos
    neigh_origin = neigh_vecs + dest_pos - src_pos
    probe_edges_displacement = np.round(inv_cell_T.dot(neigh_origin.T).T)
    t1 = time.perf_counter()
    times_displacement.append(t1 - t0)

    # --- 5: res_input_constructor (第二个嫌疑) ---
    t0 = time.perf_counter()
    positions = atoms.positions
    tree = cKDTree(positions)
    t_kdtree2 = time.perf_counter()
    times_res_kdtree.append(t_kdtree2 - t0)

    chemical_symbols = atoms.get_atomic_numbers()
    atom_inputs = []
    num_atom_inputs = []
    distances_list = []
    need_fix = []

    t_loop_start = time.perf_counter()
    for i, target in enumerate(probe_pos_flat):
        indices = tree.query_ball_point(target, 0.1)
        distances = torch.tensor(
            [euclidean(target, positions[index]) for index in indices]
        )
        atom_chem = torch.tensor(
            [chemical_symbols[item] for item in indices]
        )
        if len(indices) > 0:
            need_fix.append(torch.tensor(1).unsqueeze(0))
        else:
            need_fix.append(torch.tensor(0).unsqueeze(0))
        distances_list.append(distances)
        num_atoms = len(indices)
        atom_inputs.append(atom_chem)
        num_atom_inputs.append(num_atoms)
    t_loop_end = time.perf_counter()
    times_res_loop.append(t_loop_end - t_loop_start)
    times_res_input.append(t_loop_end - t0)

    # --- 6: Tensor 转换 ---
    t0 = time.perf_counter()
    _ = torch.tensor(probe_edges)
    _ = torch.tensor(probe_edges_displacement, dtype=torch.float32)
    _ = torch.tensor(density_batch.flatten(), dtype=torch.float32)
    _ = torch.tensor(probe_pos_flat, dtype=torch.float32)
    t1 = time.perf_counter()
    times_tensor_convert.append(t1 - t0)

    t_batch_end = time.perf_counter()
    times_total_batch.append(t_batch_end - t_batch_start)

    if batch_idx == 0:
        print(f"\n  Batch 0 详细信息:")
        print(f"    探针数: {n_probes}")
        print(f"    超胞原子数: {len(supercell_atom_positions)}"
              f" ({len(atoms)} atoms x {total_repeats} repeats)")
        print(f"    Probe edges: {total_probe_edges}")
        print(f"    平均每探针邻居数: {total_probe_edges / n_probes:.1f}")

# ============================================================
# 汇总结果
# ============================================================
print("\n" + "=" * 70)
print("PROFILING 结果汇总")
print("=" * 70)

def report(name, times, num_batches_total):
    avg = np.mean(times)
    std = np.std(times)
    estimated_total = avg * num_batches_total
    print(f"  {name:<40s}  avg={avg*1000:8.2f}ms  "
          f"std={std*1000:6.2f}ms  "
          f"预估全量={estimated_total:8.2f}s")
    return avg, estimated_total

print(f"\n每 Batch 耗时 (基于 {NUM_BATCHES_TO_PROFILE} 个 batch 的平均值):")
print(f"  总 batch 数: {num_batches_total}")
print()

results = {}
results['supercell'] = report(
    "4a. 超胞构建", times_supercell, num_batches_total)
results['kdtree_build'] = report(
    "4b. KDTree 构建 (两棵树)", times_kdtree_build, num_batches_total)
results['kdtree_query'] = report(
    "4c. KDTree query_ball_tree <<<", times_kdtree_query, num_batches_total)
results['edge_construct'] = report(
    "4d. Edge 列表构建", times_edge_construct, num_batches_total)
results['displacement'] = report(
    "4e. Displacement 矩阵运算", times_displacement, num_batches_total)
results['res_kdtree'] = report(
    "5a. res_input cKDTree 构建", times_res_kdtree, num_batches_total)
results['res_loop'] = report(
    "5b. res_input Python 循环 <<<", times_res_loop, num_batches_total)
results['res_total'] = report(
    "5.  res_input_constructor 总计", times_res_input, num_batches_total)
results['tensor'] = report(
    "6.  Tensor 转换", times_tensor_convert, num_batches_total)
results['total'] = report(
    ">>> 单 Batch 总计", times_total_batch, num_batches_total)

# ============================================================
# 占比分析
# ============================================================
print("\n" + "=" * 70)
print("耗时占比分析")
print("=" * 70)

avg_total = np.mean(times_total_batch)
items = [
    ("超胞构建", np.mean(times_supercell)),
    ("KDTree 构建", np.mean(times_kdtree_build)),
    ("KDTree query_ball_tree", np.mean(times_kdtree_query)),
    ("Edge 列表构建", np.mean(times_edge_construct)),
    ("Displacement 矩阵运算", np.mean(times_displacement)),
    ("res_input_constructor", np.mean(times_res_input)),
    ("Tensor 转换", np.mean(times_tensor_convert)),
]

for name, avg in sorted(items, key=lambda x: -x[1]):
    pct = avg / avg_total * 100
    bar = "#" * int(pct / 2)
    print(f"  {name:<30s}  {pct:5.1f}%  {bar}")

# ============================================================
# 预估总时间
# ============================================================
print("\n" + "=" * 70)
print("预估全量预处理时间")
print("=" * 70)

total_preprocess = avg_total * num_batches_total
print(f"  网格点总数: {num_positions:,}")
print(f"  总 batch 数: {num_batches_total}")
print(f"  单 batch 平均耗时: {avg_total * 1000:.2f} ms")
print(f"  预估总预处理时间: {total_preprocess:.1f} 秒")
print(f"  (GPU 推理通常只需几秒)")
print()

# 最大瓶颈
top = sorted(items, key=lambda x: -x[1])
print("TOP 瓶颈:")
for i, (name, avg) in enumerate(top[:3]):
    est = avg * num_batches_total
    print(f"  #{i+1}  {name}: "
          f"每batch {avg*1000:.2f}ms, 全量预估 {est:.1f}s "
          f"({avg/avg_total*100:.1f}%)")

print("\n" + "=" * 70)
print("Profiling 完成!")
print("=" * 70)
