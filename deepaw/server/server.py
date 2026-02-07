"""
DeePAWServer: 模型常驻显存的推理服务。

单进程双线程架构：Unix socket + HTTP 共享同一个 InferenceEngine。
"""

import os
import signal
import sys
import threading
import time
import traceback

from .protocol import atoms_to_dict, dict_to_atoms, encode_density


class DeePAWServer:
    """模型常驻显存的推理服务。

    Args:
        socket_path: Unix socket 路径
        http_host: HTTP 监听地址
        http_port: HTTP 监听端口
        enable_http: 是否启用 HTTP 服务
        checkpoint_dir: 模型权重目录
        use_compile: 是否使用 torch.compile
        data_batch_size: 每批 probe 点数量
        use_dual_model: 是否使用 F_local 修正
    """

    def __init__(
        self,
        socket_path=None,
        http_host="0.0.0.0",
        http_port=8265,
        enable_http=True,
        checkpoint_dir=None,
        use_compile=False,
        data_batch_size=3000,
        use_dual_model=True,
    ):
        from deepaw.config import SERVER_DEFAULTS

        defaults = SERVER_DEFAULTS
        self.socket_path = os.path.expanduser(
            socket_path or defaults["socket_path"]
        )
        self.pid_file = os.path.expanduser(defaults["pid_file"])
        self.http_host = http_host
        self.http_port = http_port
        self.enable_http = enable_http
        self._checkpoint_dir = checkpoint_dir
        self._use_compile = use_compile
        self._data_batch_size = data_batch_size
        self._use_dual_model = use_dual_model

        self.engine = None
        self._lock = threading.Lock()
        self._unix_server = None
        self._http_server = None
        self._running = False

    def start(self):
        """启动服务（阻塞，直到收到停止信号）。"""
        # 加载模型
        t0 = time.time()
        print("正在加载模型到 GPU...")
        self._load_engine()
        print(f"模型加载完成 ({time.time() - t0:.1f}s)")

        # 准备 socket 目录
        os.makedirs(os.path.dirname(self.socket_path), exist_ok=True)
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # 写 PID 文件
        os.makedirs(os.path.dirname(self.pid_file), exist_ok=True)
        with open(self.pid_file, "w") as f:
            f.write(str(os.getpid()))

        self._running = True

        # 启动 Unix socket 线程
        from .unix_handler import UnixSocketServer

        self._unix_server = UnixSocketServer(self.socket_path, self)
        unix_thread = threading.Thread(
            target=self._unix_server.serve_forever, daemon=True
        )
        unix_thread.start()
        print(f"Unix socket: {self.socket_path}")

        # 启动 HTTP 线程
        if self.enable_http:
            from .http_handler import DeePAWHTTPServer

            self._http_server = DeePAWHTTPServer(
                self.http_host, self.http_port, self
            )
            http_thread = threading.Thread(
                target=self._http_server.serve_forever, daemon=True
            )
            http_thread.start()
            print(f"HTTP API:    http://{self.http_host}:{self.http_port}")

        print("服务已就绪，等待请求...")

        # 注册信号处理（仅在主线程中有效）
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)

        # 主线程等待
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        """停止服务。"""
        self._running = False
        if self._unix_server:
            self._unix_server.shutdown()
        if self._http_server:
            self._http_server.shutdown()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        if os.path.exists(self.pid_file):
            os.unlink(self.pid_file)
        print("服务已停止")

    def handle_request(self, request: dict) -> dict:
        """处理请求（Unix socket 和 HTTP 共用）。"""
        action = request.get("action", "predict")

        if action == "status":
            return self._handle_status()
        elif action == "predict":
            return self._handle_predict(request)
        else:
            return {"error": f"未知操作: {action}"}

    def _handle_status(self) -> dict:
        import torch

        return {
            "status": "running",
            "pid": os.getpid(),
            "device": str(self.engine.device) if self.engine else "N/A",
            "cuda_available": torch.cuda.is_available(),
            "use_compile": self._use_compile,
            "use_dual_model": self._use_dual_model,
            "data_batch_size": self._data_batch_size,
        }

    def _handle_predict(self, request: dict) -> dict:
        with self._lock:
            return self._do_predict(request)

    def _do_predict(self, request: dict) -> dict:
        try:
            # 解析输入
            db_path = request.get("db_path")
            db_id = request.get("db_id")
            atoms_dict = request.get("atoms")
            grid_shape = request.get("grid_shape")

            atoms = dict_to_atoms(atoms_dict) if atoms_dict else None
            grid_shape = tuple(grid_shape) if grid_shape else None

            t0 = time.time()
            result = self.engine.predict(
                db_path=db_path,
                db_id=db_id,
                atoms=atoms,
                grid_shape=grid_shape,
            )
            elapsed = time.time() - t0

            return {
                "density_b64": encode_density(result["density_flat"]),
                "grid_shape": list(result["grid_shape"]),
                "atoms": atoms_to_dict(result["atoms"]),
                "elapsed": elapsed,
            }
        except Exception as e:
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def _load_engine(self):
        from deepaw.inference import InferenceEngine

        self.engine = InferenceEngine(
            checkpoint_dir=self._checkpoint_dir,
            use_compile=self._use_compile,
            data_batch_size=self._data_batch_size,
            use_dual_model=self._use_dual_model,
        )

    def _signal_handler(self, signum, frame):
        self._running = False
