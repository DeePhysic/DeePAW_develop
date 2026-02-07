# DeePAW 推理服务器使用指南

DeePAW 提供常驻 GPU 的推理服务，模型只需加载一次（~7s），后续预测请求直接进入推理阶段，无需重复初始化。

支持两种访问方式：
- **Unix socket** — 本机调用，零网络开销
- **HTTP API** — 支持远程调用、跨机器访问、curl 测试

---

## 📋 目录

- [快速开始](#-快速开始)
- [服务器命令详解](#-服务器命令详解)
- [客户端命令详解](#-客户端命令详解)
- [HTTP API 参考](#-http-api-参考)
- [Python API](#-python-api)
- [后台运行](#-后台运行)
- [架构说明](#-架构说明)
- [常见问题](#-常见问题)

---

## 🚀 快速开始

```bash
# 确保已安装 DeePAW
pip install -e .

# 终端 A：启动服务（模型加载一次，常驻显存）
deepaw-server start

# 终端 B：发送预测请求（秒级响应，无启动开销）
deepaw-predict --db examples/hfo2_chgd.db --id 1 --output CHGCAR
```

启动后输出示例：

```
正在加载模型到 GPU...
模型加载完成 (6.1s)
Unix socket: /home/user/.deepaw/server.sock
HTTP API:    http://0.0.0.0:8265
服务已就绪，等待请求...
```

---

## 🔧 服务器命令详解

### `deepaw-server start`

启动推理服务，加载模型到 GPU 并监听请求。

```bash
deepaw-server start [选项]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--port PORT` | 8265 | HTTP 监听端口 |
| `--host HOST` | 0.0.0.0 | HTTP 监听地址 |
| `--no-http` | - | 禁用 HTTP 服务，仅保留 Unix socket |
| `--compile` | - | 启用 torch.compile 加速（首次请求会有编译开销） |
| `--daemon` | - | 后台运行（日志写入 `~/.deepaw/server.log`） |
| `--checkpoint-dir DIR` | ./checkpoints | 模型权重目录 |
| `--batch-size N` | 3000 | 每批 probe 点数量 |
| `--socket PATH` | ~/.deepaw/server.sock | Unix socket 路径 |

**示例：**

```bash
# 默认启动（Unix socket + HTTP 8265）
deepaw-server start

# 指定端口，后台运行
deepaw-server start --port 9000 --daemon

# 仅本地 Unix socket，不开 HTTP
deepaw-server start --no-http

# 启用 torch.compile 加速（推理更快，但首次请求需要编译）
deepaw-server start --compile
```

### `deepaw-server stop`

停止正在运行的服务。

```bash
deepaw-server stop
```

### `deepaw-server status`

查看服务运行状态。

```bash
deepaw-server status
```

输出示例：
```
状态:     运行中
PID:      12345
设备:     cuda
compile:  False
dual:     True
batch:    3000
```

---

## 📡 客户端命令详解

### `deepaw-predict`

向运行中的服务发送预测请求。

```bash
deepaw-predict [输入选项] [输出选项]
```

**输入方式（二选一）：**

| 选项 | 说明 |
|------|------|
| `--db PATH --id N` | 从 ASE 数据库读取结构 |
| `--poscar PATH --grid NX NY NZ` | 从 POSCAR 文件读取结构 |

**输出选项：**

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--output PATH` / `-o PATH` | 无（打印摘要） | 输出文件路径 |
| `--format {chgcar,npy}` | chgcar | 输出格式 |
| `--socket PATH` | ~/.deepaw/server.sock | Unix socket 路径 |

**示例：**

```bash
# 从数据库预测，输出 CHGCAR
deepaw-predict --db data/structures.db --id 1 -o CHGCAR

# 从 POSCAR 预测，指定网格
deepaw-predict --poscar POSCAR --grid 80 80 80 -o CHGCAR

# 输出为 numpy 格式
deepaw-predict --db data/structures.db --id 1 -o density.npy --format npy

# 仅查看摘要（不保存文件）
deepaw-predict --db examples/hfo2_chgd.db --id 1
```

输出示例：
```
网格: 80x80x80
密度范围: [0.025600, 8.122800]
服务端推理: 35.52s
总耗时: 35.53s
```

---

## 🌐 HTTP API 参考

服务器默认在 `http://0.0.0.0:8265` 提供 HTTP API。

### GET /status

查询服务状态。

```bash
curl http://localhost:8265/status
```

响应：
```json
{
  "status": "running",
  "pid": 12345,
  "device": "cuda",
  "cuda_available": true,
  "use_compile": false,
  "use_dual_model": true,
  "data_batch_size": 3000
}
```

### GET /health

健康检查（同 `/status`）。

```bash
curl http://localhost:8265/health
```

### POST /predict

发送预测请求。

**请求体（JSON）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `db_path` | string | ASE 数据库的**绝对路径** |
| `db_id` | int | 数据库中的结构 ID |
| `atoms` | object | Atoms 字典（与 db_path 二选一） |
| `grid_shape` | [int, int, int] | 网格尺寸（使用 atoms 时必填） |

**方式一：从数据库预测**

```bash
curl -X POST http://localhost:8265/predict \
  -H "Content-Type: application/json" \
  -d '{
    "db_path": "/absolute/path/to/database.db",
    "db_id": 1
  }'
```

**方式二：直接传入原子结构**

```bash
curl -X POST http://localhost:8265/predict \
  -H "Content-Type: application/json" \
  -d '{
    "atoms": {
      "numbers": [72, 72, 8, 8, 8, 8],
      "positions": [[0,0,0], [2.5,2.5,2.5], ...],
      "cell": [[5,0,0], [0,5,0], [0,0,5]],
      "pbc": [true, true, true]
    },
    "grid_shape": [80, 80, 80]
  }'
```

**响应：**

```json
{
  "density_b64": "<base64 编码的 float32 密度数组>",
  "grid_shape": [80, 80, 80],
  "atoms": {
    "numbers": [72, 72, 8, 8, 8, 8],
    "positions": [...],
    "cell": [...],
    "pbc": [true, true, true]
  },
  "elapsed": 35.52
}
```

> **注意**：`density_b64` 是 base64 编码的 float32 字节流。解码方式见下方 Python API 示例。

---

## 🐍 Python API

### DeePAWClient（推荐）

通过 Unix socket 连接服务器，适合 Python 脚本和 Jupyter notebook。

```python
from deepaw.server import DeePAWClient

client = DeePAWClient()

# 检查服务是否运行
if not client.is_running():
    print("请先启动服务: deepaw-server start")
    exit()

# 从数据库预测
result = client.predict(db_path="examples/hfo2_chgd.db", db_id=1)
print(f"密度形状: {result['density_3d'].shape}")  # (80, 80, 80)
print(f"原子: {result['atoms'].get_chemical_formula()}")  # Hf4O8
print(f"推理耗时: {result['elapsed']:.2f}s")

# 直接传入 ASE Atoms 对象
from ase.io import read
atoms = read("POSCAR")
result = client.predict(atoms=atoms, grid_shape=(80, 80, 80))

# 预测并直接写 CHGCAR 文件
result = client.predict_chgcar(
    "output/CHGCAR",
    db_path="data/structures.db",
    db_id=1,
)
```

### 返回值说明

`client.predict()` 返回字典：

| 字段 | 类型 | 说明 |
|------|------|------|
| `density_3d` | np.ndarray (nx,ny,nz) | 三维电荷密度 |
| `density_flat` | np.ndarray (nx*ny*nz,) | 展平的电荷密度 |
| `atoms` | ase.Atoms | 原子结构 |
| `grid_shape` | tuple (nx,ny,nz) | 网格尺寸 |
| `elapsed` | float | 服务端推理耗时（秒） |

### HTTP 响应解码（Python）

如果通过 HTTP API 获取结果，需要手动解码密度数组：

```python
import base64, json, numpy as np
import urllib.request

# 发送请求
req_data = json.dumps({
    "db_path": "/absolute/path/to/database.db",
    "db_id": 1,
}).encode("utf-8")

req = urllib.request.Request(
    "http://localhost:8265/predict",
    data=req_data,
    headers={"Content-Type": "application/json"},
)
resp = urllib.request.urlopen(req)
data = json.loads(resp.read())

# 解码密度
density_bytes = base64.b64decode(data["density_b64"])
density = np.frombuffer(density_bytes, dtype=np.float32)
density_3d = density.reshape(data["grid_shape"])
```

---

## 🔄 后台运行

### Daemon 模式

使用 `--daemon` 参数将服务放到后台运行：

```bash
deepaw-server start --daemon
# 输出: 服务已在后台启动 (PID 12345)
```

日志输出到 `~/.deepaw/server.log`：

```bash
tail -f ~/.deepaw/server.log
```

管理后台服务：

```bash
# 查看状态
deepaw-server status

# 停止服务
deepaw-server stop
```

### 文件位置

| 文件 | 路径 | 说明 |
|------|------|------|
| Unix socket | `~/.deepaw/server.sock` | 本地通信端点 |
| PID 文件 | `~/.deepaw/server.pid` | 进程 ID，用于 stop/status |
| 日志文件 | `~/.deepaw/server.log` | daemon 模式的输出日志 |

---

## 🏗️ 架构说明

```
┌─────────────────────────────────────────────────┐
│                  DeePAWServer                    │
│                                                  │
│  ┌──────────────┐    ┌───────────────────────┐  │
│  │ Unix Socket   │    │ HTTP Server            │  │
│  │ (本地快速)    │    │ (远程/API 调用)        │  │
│  │ :server.sock  │    │ :8265                  │  │
│  └──────┬───────┘    └──────────┬────────────┘  │
│         │                       │                │
│         └───────────┬───────────┘                │
│                     ▼                            │
│           ┌─────────────────┐                    │
│           │ handle_request  │ ← threading.Lock   │
│           └────────┬────────┘                    │
│                    ▼                             │
│           ┌─────────────────┐                    │
│           │ InferenceEngine │ ← 模型常驻 GPU     │
│           │  F_nonlocal     │                    │
│           │  F_local        │                    │
│           └─────────────────┘                    │
└─────────────────────────────────────────────────┘
```

- **单进程双线程**：Unix socket 和 HTTP 各一个监听线程，共享同一个 InferenceEngine
- **线程安全**：`threading.Lock` 保护 predict 调用，确保 GPU 推理串行执行
- **零额外依赖**：HTTP 使用 Python 标准库 `http.server`，无需安装 Flask/FastAPI

---

## ❓ 常见问题

### 服务启动失败："Address already in use"

端口被占用，换一个端口或关闭占用进程：

```bash
# 换端口
deepaw-server start --port 9000

# 或查找并关闭占用进程
lsof -i :8265
```

### 服务启动失败：socket 文件已存在

上次服务未正常关闭，手动清理：

```bash
rm ~/.deepaw/server.sock ~/.deepaw/server.pid
deepaw-server start
```

### 预测报错 "CUDA out of memory"

GPU 显存不足。可以减小 batch size：

```bash
deepaw-server start --batch-size 1000
```

### torch.compile 相关

`--compile` 选项可以加速推理约 15%，但：
- 首次预测请求会触发编译，需要额外等待
- 需要 PyTorch 2.0+
- 编译后的 kernel 会缓存，后续启动更快

```bash
# 推荐：需要频繁预测时使用
deepaw-server start --compile
```

### 如何从远程机器调用？

确保服务器启动时开启了 HTTP（默认开启），然后从远程机器：

```bash
curl http://<server-ip>:8265/status
curl -X POST http://<server-ip>:8265/predict \
  -H "Content-Type: application/json" \
  -d '{"db_path": "/path/on/server/database.db", "db_id": 1}'
```

> **注意**：`db_path` 必须是**服务器上的绝对路径**。

---

**Last Updated**: 2025-02-08

