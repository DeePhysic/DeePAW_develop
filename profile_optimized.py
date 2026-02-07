#!/usr/bin/env python
"""优化前后对比: 直接测试 MyCollator 的完整流程耗时"""
import os, sys, time
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from ase.db import connect
from deepaw.data.chgcar_writer import (
    GraphConstructor, get_grid_centers, MyCollator
)

DB_PATH = os.path.join(script_dir, 'examples', 'hfo2_chgd.db')
CUTOFF = 4.0
BATCH_SIZE = 3000

db = connect(DB_PATH)
row = db.get(1)
atoms = row.toatoms()
atoms.set_pbc(True)
nx, ny, nz = row.data['nx'], row.data['ny'], row.data['nz']
num_positions = nx * ny * nz
num_batches = (num_positions + BATCH_SIZE - 1) // BATCH_SIZE

print("=" * 60)
print("优化后 Pipeline 性能测试")
print("=" * 60)
print(f"化学式: {row.formula}, 原子数: {len(atoms)}")
print(f"网格: {nx}x{ny}x{nz} = {num_positions:,} probes")
print(f"Batch size: {BATCH_SIZE}, 总 batch 数: {num_batches}")

# 测试优化后的缓存路径
gc = GraphConstructor(cutoff=CUTOFF, num_probes=None, disable_pbc=True)
probe_pos = get_grid_centers(atoms, nx, ny, nz)
density = np.zeros((nx, ny, nz))

print("\n--- 优化后: precompute + build_graph_with_cache ---")
t_start = time.perf_counter()

t0 = time.perf_counter()
cache = gc.precompute_atom_data(atoms)
t_precompute = time.perf_counter() - t0
print(f"  precompute_atom_data: {t_precompute*1000:.2f}ms (一次性)")

batch_times = []
for batch_idx in range(num_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, num_positions)

    unravel_indices = np.unravel_index(
        np.arange(start_idx, end_idx), (nx, ny, nz))
    probe_pos_batch = probe_pos[unravel_indices]
    density_batch = density[unravel_indices]

    t0 = time.perf_counter()
    gd = gc.build_graph_with_cache(
        density=density_batch, grid_pos=probe_pos_batch, cache=cache)
    gd['total_num_probes'] = torch.tensor([nx, ny, nz]).long()
    batched = {k: v.unsqueeze(0) for k, v in gd.items()}
    t1 = time.perf_counter()
    batch_times.append(t1 - t0)

t_total = time.perf_counter() - t_start

avg_batch = np.mean(batch_times) * 1000
print(f"  平均每 batch: {avg_batch:.2f}ms")
print(f"  总耗时: {t_total:.2f}s")

# 对比: 原始路径 (不含 res_input_constructor)
print("\n--- 对比: 原始 __call__ 路径 (已去掉 res_input) ---")
t_start2 = time.perf_counter()

batch_times_old = []
for batch_idx in range(num_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, num_positions)

    unravel_indices = np.unravel_index(
        np.arange(start_idx, end_idx), (nx, ny, nz))
    probe_pos_batch = probe_pos[unravel_indices]
    density_batch = density[unravel_indices]

    t0 = time.perf_counter()
    gd = gc(density=density_batch, atoms=atoms,
            grid_pos=probe_pos_batch, threshold_distance=0.1)
    gd['total_num_probes'] = torch.tensor([nx, ny, nz]).long()
    batched = {k: v.unsqueeze(0) for k, v in gd.items()}
    t1 = time.perf_counter()
    batch_times_old.append(t1 - t0)

t_total_old = time.perf_counter() - t_start2

avg_batch_old = np.mean(batch_times_old) * 1000
print(f"  平均每 batch: {avg_batch_old:.2f}ms")
print(f"  总耗时: {t_total_old:.2f}s")

# 汇总
print("\n" + "=" * 60)
print("对比结果")
print("=" * 60)
print(f"  原始路径 (无 res_input): {t_total_old:.2f}s")
print(f"  优化后 (cache):          {t_total:.2f}s")
speedup = t_total_old / t_total if t_total > 0 else float('inf')
print(f"  加速比: {speedup:.2f}x")
print(f"\n  原始 profiling 基线 (含 res_input): ~19.2s")
print(f"  优化后总耗时: {t_total:.2f}s")
overall = 19.2 / t_total if t_total > 0 else float('inf')
print(f"  相对原始基线加速: {overall:.2f}x")
