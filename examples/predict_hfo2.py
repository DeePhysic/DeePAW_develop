#!/usr/bin/env python
"""
HfO2 Charge Density Prediction - Test Script
使用 F_nonlocal + F_local 双模型预测 HfO2 的电荷密度并写入 CHGCAR 文件

重要：F_local 不能使用 eval() 模式（KAN 网络有 bug）
"""

import os
import sys

# Add DeePAW to path
script_dir = os.path.dirname(os.path.abspath(__file__))
deepaw_root = os.path.abspath(os.path.join(script_dir, '..'))
if deepaw_root not in sys.path:
    sys.path.insert(0, deepaw_root)

import torch
import numpy as np
from ase.db import connect
from torch.utils.data import DataLoader
from tqdm import tqdm
from ase.calculators.vasp import VaspChargeDensity

# DeePAW imports
from deepaw import F_nonlocal, F_local
from deepaw.data.chgcar_writer import DensityData, MyCollator

print("="*80)
print("HfO2 电荷密度预测测试")
print("="*80)

# 配置
DB_PATH = os.path.join(deepaw_root, 'examples', 'hfo2_chgd.db')
CHECKPOINT_DIR = os.path.join(deepaw_root, 'checkpoints')
OUTPUT_DIR = os.path.join(deepaw_root, 'outputs', 'hfo2_predictions')
CUTOFF = 4.0
NUM_PROBES = None  # 使用所有探针点

# 设备配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n使用设备: {device}")

# 检查数据库
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"数据库文件不存在: {DB_PATH}")
print(f"数据库路径: {DB_PATH}")

# 加载数据库
db = connect(DB_PATH)
dataset = DensityData(DB_PATH)
total_count = len(dataset)
print(f"数据库中的结构数: {total_count}")

# 显示第一个结构信息
row = db.get(1)
print(f"第一个结构: {row.formula}")
print(f"网格尺寸: {row.data.nx} x {row.data.ny} x {row.data.nz}")

# 初始化模型
print("\n" + "="*80)
print("初始化模型")
print("="*80)

f_nonlocal = F_nonlocal(
    num_interactions=3,
    num_neighbors=20,
    mul=500,
    lmax=4,
    cutoff=CUTOFF,
    num_basis=10,
)

f_local = F_local(
    input_dim=992,
    hidden_dim=32,
)

# 加载预训练权重
checkpoint_nonlocal = os.path.join(CHECKPOINT_DIR, 'f_nonlocal.pth')
checkpoint_local = os.path.join(CHECKPOINT_DIR, 'f_local.pth')

if not os.path.exists(checkpoint_nonlocal):
    raise FileNotFoundError(f"F_nonlocal 权重文件不存在: {checkpoint_nonlocal}")
if not os.path.exists(checkpoint_local):
    raise FileNotFoundError(f"F_local 权重文件不存在: {checkpoint_local}")

print(f"加载 F_nonlocal 权重: {checkpoint_nonlocal}")
f_nonlocal.load_state_dict(torch.load(checkpoint_nonlocal, map_location=device))

print(f"加载 F_local 权重: {checkpoint_local}")
f_local.load_state_dict(torch.load(checkpoint_local, map_location=device))

# 移动到设备
f_nonlocal = f_nonlocal.to(device)
f_local = f_local.to(device)

# 重要：只对 F_nonlocal 使用 eval()，F_local 不能用 eval()（KAN 网络 bug）
print("\n⚠️  重要：F_local 不使用 eval() 模式（KAN 网络 bug）")
f_nonlocal.eval()  # F_nonlocal 可以使用 eval()
# f_local.eval()   # ❌ 不要对 F_local 使用 eval()

print("✓ 模型加载成功！")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


def predict_structure(db_id):
    """预测单个结构的电荷密度

    Args:
        db_id: 数据库ID（从1开始）
    """
    print(f"\n预测结构 DB ID: {db_id}")

    # 创建数据加载器 - 直接使用数据库ID
    dataloader = DataLoader(
        [db_id],  # 直接传入数据库ID
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=True if device == 'cuda' else False,
        collate_fn=MyCollator(DB_PATH, cutoff=CUTOFF, num_probes=NUM_PROBES, inference=True)
    )

    all_predictions = []

    # 预测（不使用 eval() 模式）
    with torch.no_grad():
        for big_batch in dataloader:
            for batch in tqdm(big_batch, desc="预测中", total=len(big_batch)):
                # 移动到设备（跳过推理无关键，使用 non_blocking 加速传输）
                _skip = {'probe_target', 'total_num_probes'}
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k not in _skip}

                # 1. F_nonlocal 预测基础电荷密度
                output_nonlocal, node_rep = f_nonlocal(batch)
                output_nonlocal = output_nonlocal.view(-1)

                # 2. F_local 预测局部修正
                correction, _ = f_local(None, node_rep)
                correction = correction.view(-1)

                # 3. 最终预测 = 基础 + 修正
                output_final = output_nonlocal + correction

                all_predictions.append(output_final.detach().cpu())
            break

    # 合并所有预测
    predictions = torch.cat(all_predictions, dim=0).numpy()
    return predictions


def write_chgcar(prediction_array, atoms, nx, ny, nz, output_path):
    """写入 VASP CHGCAR 文件"""
    # 重塑为 3D 网格
    density = prediction_array.reshape(nx, ny, nz)

    # 创建 VaspChargeDensity 对象
    vcd = VaspChargeDensity(filename=None)
    vcd.atoms.append(atoms)
    vcd.chg.append(density)

    # 写入文件
    vcd.write(f'{output_path}.vasp', format='chgcar')
    print(f"✓ CHGCAR 已保存: {output_path}.vasp")


# 主预测循环
print("\n" + "="*80)
print("开始预测")
print("="*80)

# 预测前几个结构作为测试（你可以修改范围）
NUM_STRUCTURES_TO_PREDICT = 1  # 预测1个结构进行测试
START_ID = 1  # 数据库ID从1开始

for i in range(NUM_STRUCTURES_TO_PREDICT):
    db_id = START_ID + i  # 数据库ID（从1开始）

    print(f"\n{'='*80}")
    print(f"处理结构 {i + 1}/{NUM_STRUCTURES_TO_PREDICT} (DB ID: {db_id})")
    print(f"{'='*80}")

    # 获取结构信息
    row = db.get(db_id)
    atoms = row.toatoms()
    formula = row.formula
    nx, ny, nz = row.data.nx, row.data.ny, row.data.nz

    print(f"化学式: {formula}")
    print(f"原子数: {len(atoms)}")
    print(f"网格尺寸: {nx} x {ny} x {nz}")

    # 预测电荷密度（使用数据库ID）
    prediction = predict_structure(db_id)
    print(f"预测数组形状: {prediction.shape}")
    print(f"预测值范围: [{prediction.min():.6f}, {prediction.max():.6f}]")

    # 构建输出文件名
    try:
        # 尝试获取 mpid
        mpid = row.data.get("mpid", f"hfo2_{db_id}")
    except:
        mpid = f"hfo2_{db_id}"

    output_path = os.path.join(OUTPUT_DIR, mpid)

    # 写入 CHGCAR 文件
    write_chgcar(prediction, atoms, nx, ny, nz, output_path)

    # 可选：如果数据库中有真实值，计算误差
    if hasattr(row.data, 'chg'):
        true_chg = row.data.chg
        mae = np.mean(np.abs(prediction - true_chg))
        print(f"平均绝对误差 (MAE): {mae:.6f}")

print("\n" + "="*80)
print("预测完成！")
print("="*80)
print(f"\n输出目录: {OUTPUT_DIR}")
print(f"共预测 {NUM_STRUCTURES_TO_PREDICT} 个结构")
print(f"\nCHGCAR 文件已保存为 .vasp 格式")

# 列出生成的文件
print("\n生成的文件:")
for file in sorted(os.listdir(OUTPUT_DIR)):
    if file.endswith('.vasp'):
        filepath = os.path.join(OUTPUT_DIR, file)
        filesize = os.path.getsize(filepath) / 1024  # KB
        print(f"  - {file} ({filesize:.1f} KB)")

print("\n" + "="*80)
print("测试完成！")
print("="*80)
