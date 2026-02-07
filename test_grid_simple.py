"""
快速测试自动网格计算功能
"""
from ase.io import read
from deepaw.utils import calculate_grid_size

# 读取 HfO2 结构
atoms = read("examples/hfo2_chgd.db@1")

print("=" * 60)
print("自动网格计算测试")
print("=" * 60)
print(f"\n晶体结构:")
print(f"  化学式: {atoms.get_chemical_formula()}")
print(f"  原子数: {len(atoms)}")
print(f"  晶胞参数: a={atoms.cell.cellpar()[0]:.3f}, b={atoms.cell.cellpar()[1]:.3f}, c={atoms.cell.cellpar()[2]:.3f} Å")

print(f"\n不同 ENCUT 值的网格尺寸:")
print(f"{'ENCUT (eV)':<12} {'网格尺寸':<15} {'总点数':<12}")
print("-" * 60)

for encut in [300, 400, 500, 600, 700, 800]:
    nx, ny, nz = calculate_grid_size(atoms, encut=encut)
    total_points = nx * ny * nz
    print(f"{encut:<12} {nx}x{ny}x{nz:<10} {total_points:<12,}")

print("\n✓ 自动网格计算功能正常！")
