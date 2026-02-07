#!/usr/bin/env python
"""
InferenceEngine 使用示例

演示模型常驻显存的推理方式：初始化一次，反复调用 predict()。
"""

import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
deepaw_root = os.path.abspath(os.path.join(script_dir, ".."))
if deepaw_root not in sys.path:
    sys.path.insert(0, deepaw_root)

from deepaw import InferenceEngine

# ============================================================
# 1. 初始化引擎（模型加载到 GPU，只需执行一次）
# ============================================================
print("=" * 60)
print("初始化 InferenceEngine（模型加载到显存）")
print("=" * 60)

t0 = time.time()
engine = InferenceEngine(
    checkpoint_dir=os.path.join(deepaw_root, "checkpoints"),
    # device="cuda",          # 默认自动选择
    # data_batch_size=3000,   # 24GB GPU 建议 3000
    # use_dual_model=True,    # 使用 F_nonlocal + F_local
)
print(f"引擎初始化耗时: {time.time() - t0:.2f}s")
print("模型已常驻显存，后续预测无需重新加载。\n")

# ============================================================
# 2. 方式一：从数据库预测
# ============================================================
DB_PATH = os.path.join(deepaw_root, "examples", "hfo2_chgd.db")

if os.path.exists(DB_PATH):
    print("=" * 60)
    print("方式一：从 ASE 数据库预测")
    print("=" * 60)

    t0 = time.time()
    result = engine.predict(db_path=DB_PATH, db_id=1)
    elapsed = time.time() - t0

    nx, ny, nz = result["grid_shape"]
    density = result["density_3d"]
    print(f"网格尺寸: {nx} x {ny} x {nz}")
    print(f"预测值范围: [{density.min():.6f}, {density.max():.6f}]")
    print(f"推理耗时: {elapsed:.2f}s\n")

    # 再次预测同一结构（模型已在显存，无加载开销）
    print("再次预测同一结构（验证模型常驻）...")
    t0 = time.time()
    result2 = engine.predict(db_path=DB_PATH, db_id=1)
    elapsed2 = time.time() - t0
    print(f"第二次推理耗时: {elapsed2:.2f}s\n")

# ============================================================
# 3. 方式二：从 ASE Atoms 对象预测
# ============================================================
print("=" * 60)
print("方式二：从 ASE Atoms 对象预测")
print("=" * 60)

if os.path.exists(DB_PATH):
    from ase.db import connect

    db = connect(DB_PATH)
    row = db.get(1)
    atoms = row.toatoms()
    nx, ny, nz = row.data["nx"], row.data["ny"], row.data["nz"]

    t0 = time.time()
    result = engine.predict(atoms=atoms, grid_shape=(nx, ny, nz))
    elapsed = time.time() - t0

    density = result["density_3d"]
    print(f"预测值范围: [{density.min():.6f}, {density.max():.6f}]")
    print(f"推理耗时: {elapsed:.2f}s\n")

# ============================================================
# 4. 方式三：预测并直接写 CHGCAR 文件
# ============================================================
print("=" * 60)
print("方式三：预测并写入 CHGCAR 文件")
print("=" * 60)

if os.path.exists(DB_PATH):
    output_dir = os.path.join(deepaw_root, "outputs", "engine_demo")
    output_path = os.path.join(output_dir, "hfo2_engine")

    t0 = time.time()
    result = engine.predict_and_write_chgcar(
        output_path=output_path, db_path=DB_PATH, db_id=1
    )
    elapsed = time.time() - t0

    print(f"CHGCAR 已保存: {output_path}.vasp")
    filesize = os.path.getsize(f"{output_path}.vasp") / 1024
    print(f"文件大小: {filesize:.1f} KB")
    print(f"耗时: {elapsed:.2f}s\n")

print("=" * 60)
print("演示完成")
print("=" * 60)
