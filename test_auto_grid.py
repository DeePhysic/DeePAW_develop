"""
测试自动网格计算功能

验证当只提供 atoms 而不指定 grid_shape 时，系统能自动计算合适的网格尺寸。
"""
import torch
from ase.io import read
from deepaw.inference import InferenceEngine

# 读取 HfO2 结构
atoms = read("examples/hfo2_chgd.db@1")

print("=" * 60)
print("测试 1: 自动计算网格尺寸（默认 ENCUT=500 eV）")
print("=" * 60)

engine = InferenceEngine(use_gpu_graph=True, use_compile=False)

# 不指定 grid_shape，应该自动计算
result = engine.predict(atoms=atoms, grid_shape=None)

print(f"\n自动计算的网格尺寸: {result['grid_shape']}")
print(f"电荷密度形状: {result['density_3d'].shape}")
print(f"总电荷: {result['density_3d'].sum():.2f}")

print("\n" + "=" * 60)
print("测试 2: 手动指定网格尺寸")
print("=" * 60)

# 手动指定 grid_shape
result2 = engine.predict(atoms=atoms, grid_shape=(80, 80, 80))

print(f"\n手动指定的网格尺寸: {result2['grid_shape']}")
print(f"电荷密度形状: {result2['density_3d'].shape}")
print(f"总电荷: {result2['density_3d'].sum():.2f}")

print("\n" + "=" * 60)
print("测试 3: 不同 ENCUT 值的网格尺寸")
print("=" * 60)

from deepaw.utils import calculate_grid_size

for encut in [300, 400, 500, 600]:
    nx, ny, nz = calculate_grid_size(atoms, encut=encut)
    print(f"ENCUT={encut} eV: {nx}x{ny}x{nz} = {nx*ny*nz:,} 点")

print("\n✓ 自动网格计算功能测试完成！")
