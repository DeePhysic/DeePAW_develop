"""
从 CIF/POSCAR 文件直接预测电荷密度

演示如何使用 InferenceEngine 从单个结构文件预测电荷密度，
无需手动指定网格尺寸（自动根据 ENCUT 计算）。
"""
import torch
from ase.io import read
from deepaw.inference import InferenceEngine

# 初始化推理引擎（启用 GPU 图构建加速）
print("初始化 DeePAW 推理引擎...")
engine = InferenceEngine(
    use_gpu_graph=True,      # 使用 GPU 加速图构建
    use_compile=True,        # 使用 torch.compile 加速
    use_dual_model=True,     # 使用双模型（更高精度）
)

# 从数据库读取结构（也可以从 CIF/POSCAR 读取）
print("\n读取晶体结构...")
atoms = read("examples/hfo2_chgd.db@1")
print(f"  化学式: {atoms.get_chemical_formula()}")
print(f"  原子数: {len(atoms)}")
print(f"  晶胞参数: {atoms.cell.cellpar()[:3]} Å")

# 方式 1: 自动计算网格尺寸（推荐）
print("\n" + "=" * 60)
print("方式 1: 自动计算网格尺寸（ENCUT=500 eV）")
print("=" * 60)

result = engine.predict(atoms=atoms)  # grid_shape=None 时自动计算

print(f"\n预测完成！")
print(f"  网格尺寸: {result['grid_shape']}")
print(f"  总网格点: {result['grid_shape'][0] * result['grid_shape'][1] * result['grid_shape'][2]:,}")
print(f"  总电荷: {result['density_3d'].sum():.2f}")

# 方式 2: 指定不同的 ENCUT 值
print("\n" + "=" * 60)
print("方式 2: 使用更高的 ENCUT（600 eV）")
print("=" * 60)

# 注意：predict() 方法目前不直接接受 encut 参数
# 需要先手动计算网格尺寸
from deepaw.utils import calculate_grid_size

nx, ny, nz = calculate_grid_size(atoms, encut=600.0)
print(f"ENCUT=600 eV 对应网格: {nx}x{ny}x{nz}")

result2 = engine.predict(atoms=atoms, grid_shape=(nx, ny, nz))
print(f"\n预测完成！")
print(f"  总电荷: {result2['density_3d'].sum():.2f}")

# 方式 3: 输出 CHGCAR 文件
print("\n" + "=" * 60)
print("方式 3: 直接输出 CHGCAR 文件")
print("=" * 60)

engine.predict_and_write_chgcar(
    atoms=atoms,
    output_path="CHGCAR_auto"  # 会自动添加 .vasp 后缀
)

print(f"\n✓ CHGCAR 文件已保存到: CHGCAR_auto.vasp")

# 对比不同 ENCUT 的网格尺寸
print("\n" + "=" * 60)
print("不同 ENCUT 值的网格尺寸对比")
print("=" * 60)
print(f"{'ENCUT (eV)':<12} {'网格尺寸':<15} {'总点数':<12} {'相对大小'}")
print("-" * 60)

base_points = None
for encut in [300, 400, 500, 600, 700, 800]:
    nx, ny, nz = calculate_grid_size(atoms, encut=encut)
    total_points = nx * ny * nz
    if base_points is None:
        base_points = total_points
    ratio = total_points / base_points
    print(f"{encut:<12} {nx}x{ny}x{nz:<10} {total_points:<12,} {ratio:.2f}x")

print("\n" + "=" * 60)
print("使用建议")
print("=" * 60)
print("• 快速测试: ENCUT=300-400 eV")
print("• 标准精度: ENCUT=500 eV (默认)")
print("• 高精度: ENCUT=600-700 eV")
print("• 发表级: ENCUT=800+ eV")
print("\n注意: 网格点数越多，推理时间越长，显存占用越大")
