# DeePAW (develop)

Deep Learning for PAW Charge Density Prediction — 开发分支。

## 环境搭建

```bash
conda create -n DeePAW python=3.10
conda activate DeePAW
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install e3nn ase pymatgen scipy tqdm accelerate pykan
pip install -e .
```

验证：
```bash
python -c "from deepaw import F_nonlocal, F_local, InferenceEngine; print('OK')"
```

## 项目结构

```
deepaw/
├── models/
│   ├── f_nonlocal.py       # E3-equivariant GNN (~1.9M params)
│   ├── f_local.py          # KAN correction (~36K params)
│   └── irreps_tools.py     # e3nn 工具函数
├── data/
│   └── chgcar_writer.py    # 图构建 + CHGCAR I/O
├── inference.py             # InferenceEngine (统一推理入口)
├── server/                  # 推理服务器 (Unix socket + HTTP)
│   ├── server.py            # DeePAWServer
│   ├── client.py            # DeePAWClient
│   ├── cli.py               # deepaw-server / deepaw-predict 命令
│   ├── protocol.py          # 序列化协议
│   ├── http_handler.py      # HTTP 处理器
│   └── unix_handler.py      # Unix socket 处理器
├── scripts/                 # 预测脚本
├── config.py                # 全局配置
└── __init__.py              # 导出 F_nonlocal, F_local, InferenceEngine
checkpoints/                 # 预训练权重
examples/                    # 示例脚本
docs/                        # 文档
```

## 快速预测

```python
from deepaw import InferenceEngine

engine = InferenceEngine(use_dual_model=True)
result = engine.predict(db_path="examples/hfo2_chgd.db", db_id=1)
# result["density_3d"].shape → (80, 80, 80)
```

## 推理服务器

模型常驻 GPU，避免每次 ~7s 的加载开销：

```bash
# 启动（一次性）
deepaw-server start

# 预测（秒级响应）
deepaw-predict --db examples/hfo2_chgd.db --id 1 -o CHGCAR

# HTTP API
curl http://localhost:8265/status
curl -X POST http://localhost:8265/predict \
  -H "Content-Type: application/json" \
  -d '{"db_path": "/abs/path/to/db.db", "db_id": 1}'
```

详见 [docs/server/SERVER_GUIDE.md](docs/server/SERVER_GUIDE.md)。

## 推理加速记录

| 优化 | 提升 | 说明 |
|------|------|------|
| CPU 预处理优化 | 4.93x | KDTree 缓存、超胞预计算、去除冗余代码 |
| torch.compile | 1.15x | probe_model 编译加速（需 `--compile`） |
| 推理服务器 | 消除 7s 启动 | 模型常驻显存 |

详见 [docs/inference_acceleration/](docs/inference_acceleration/)。

## 开发命令

```bash
# 运行示例
python examples/predict_hfo2.py

# 服务器管理
deepaw-server start [--compile] [--daemon] [--port 8265]
deepaw-server status
deepaw-server stop

# 客户端预测
deepaw-predict --db PATH --id N [-o OUTPUT] [--format {chgcar,npy}]
deepaw-predict --poscar PATH --grid NX NY NZ [-o OUTPUT]
```

## Git 提交历史

```
d9e378b 添加推理服务器使用手册
7c5a650 添加推理服务器：Unix socket + HTTP API
f75f7ce torch.compile 加速 probe_model (1.15x)
ba17f9c GPU profiling 基准测试
e26ca93 初始版本：两轮推理加速 + InferenceEngine
```

## 文档

- [docs/server/SERVER_GUIDE.md](docs/server/SERVER_GUIDE.md) — 推理服务器使用指南
- [docs/inference_acceleration/](docs/inference_acceleration/) — 推理加速工作记录
- [docs/QUICKSTART.md](docs/QUICKSTART.md) — 快速上手
- [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) — 项目架构
