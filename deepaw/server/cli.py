"""
CLI 入口：deepaw-server 和 deepaw-predict 命令。

用法:
    deepaw-server start [--port 8265] [--no-http] [--compile] [--daemon]
    deepaw-server stop
    deepaw-server status

    deepaw-predict --db path.db --id 1 [--output CHGCAR]
    deepaw-predict --poscar POSCAR --grid 80 80 80 [--output CHGCAR]
"""

import argparse
import os
import signal
import sys
import time


def server_main():
    """deepaw-server 命令入口。"""
    parser = argparse.ArgumentParser(
        prog="deepaw-server",
        description="DeePAW 推理服务器：模型常驻显存，随时接受预测请求",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # start
    p_start = sub.add_parser("start", help="启动服务")
    p_start.add_argument("--port", type=int, default=8265, help="HTTP 端口")
    p_start.add_argument("--host", default="0.0.0.0", help="HTTP 监听地址")
    p_start.add_argument(
        "--no-http", action="store_true", help="禁用 HTTP 服务"
    )
    p_start.add_argument(
        "--compile", action="store_true", help="启用 torch.compile 加速"
    )
    p_start.add_argument(
        "--daemon", action="store_true", help="后台运行"
    )
    p_start.add_argument(
        "--checkpoint-dir", default=None, help="模型权重目录"
    )
    p_start.add_argument(
        "--batch-size", type=int, default=3000, help="每批 probe 点数量"
    )
    p_start.add_argument(
        "--socket", default=None, help="Unix socket 路径"
    )

    # stop
    sub.add_parser("stop", help="停止服务")

    # status
    sub.add_parser("status", help="查看服务状态")

    args = parser.parse_args()

    if args.command == "start":
        _cmd_start(args)
    elif args.command == "stop":
        _cmd_stop()
    elif args.command == "status":
        _cmd_status()


def _cmd_start(args):
    """启动服务器。"""
    if args.daemon:
        _daemonize()

    from deepaw.server.server import DeePAWServer

    server = DeePAWServer(
        socket_path=args.socket,
        http_host=args.host,
        http_port=args.port,
        enable_http=not args.no_http,
        checkpoint_dir=args.checkpoint_dir,
        use_compile=args.compile,
        data_batch_size=args.batch_size,
    )
    server.start()


def _cmd_stop():
    """停止服务器。"""
    from deepaw.config import SERVER_DEFAULTS

    pid_file = os.path.expanduser(SERVER_DEFAULTS["pid_file"])
    if not os.path.exists(pid_file):
        print("服务未运行（PID 文件不存在）")
        return

    with open(pid_file) as f:
        pid = int(f.read().strip())

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"已发送停止信号到 PID {pid}")
        # 等待进程退出
        for _ in range(30):
            try:
                os.kill(pid, 0)
                time.sleep(0.1)
            except ProcessLookupError:
                print("服务已停止")
                return
        print("警告: 进程未在 3 秒内退出")
    except ProcessLookupError:
        print(f"进程 {pid} 不存在，清理 PID 文件")
        os.unlink(pid_file)


def _cmd_status():
    """查看服务器状态。"""
    from deepaw.server.client import DeePAWClient

    client = DeePAWClient()
    if client.is_running():
        status = client.status()
        print(f"状态:     运行中")
        print(f"PID:      {status.get('pid', 'N/A')}")
        print(f"设备:     {status.get('device', 'N/A')}")
        print(f"compile:  {status.get('use_compile', 'N/A')}")
        print(f"dual:     {status.get('use_dual_model', 'N/A')}")
        print(f"batch:    {status.get('data_batch_size', 'N/A')}")
    else:
        print("服务未运行")


def _daemonize():
    """Fork 到后台运行。"""
    pid = os.fork()
    if pid > 0:
        print(f"服务已在后台启动 (PID {pid})")
        sys.exit(0)
    os.setsid()
    # 重定向 stdout/stderr 到日志
    log_dir = os.path.expanduser("~/.deepaw")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "server.log")
    log_fd = open(log_path, "a")
    os.dup2(log_fd.fileno(), sys.stdout.fileno())
    os.dup2(log_fd.fileno(), sys.stderr.fileno())


# ============================================================
# deepaw-predict 客户端命令
# ============================================================

def predict_main():
    """deepaw-predict 命令入口。"""
    parser = argparse.ArgumentParser(
        prog="deepaw-predict",
        description="向 DeePAW 服务发送预测请求",
    )

    # 输入方式
    input_group = parser.add_argument_group("输入（二选一）")
    input_group.add_argument("--db", help="ASE 数据库路径")
    input_group.add_argument("--id", type=int, help="数据库结构 ID")
    input_group.add_argument("--poscar", help="POSCAR 文件路径")
    input_group.add_argument(
        "--grid", type=int, nargs=3, metavar=("NX", "NY", "NZ"),
        help="网格尺寸",
    )

    # 输出
    parser.add_argument(
        "--output", "-o", help="输出文件路径（默认打印摘要）"
    )
    parser.add_argument(
        "--format", choices=["chgcar", "npy"], default="chgcar",
        help="输出格式",
    )
    parser.add_argument("--socket", default=None, help="Unix socket 路径")

    args = parser.parse_args()

    from deepaw.server.client import DeePAWClient

    client = DeePAWClient(socket_path=args.socket)

    if not client.is_running():
        print("错误: DeePAW 服务未运行，请先执行 deepaw-server start")
        sys.exit(1)

    # 构建预测参数
    kwargs = {}
    if args.db:
        kwargs["db_path"] = args.db
        kwargs["db_id"] = args.id
    elif args.poscar:
        from ase.io import read

        atoms = read(args.poscar)
        kwargs["atoms"] = atoms
        if not args.grid:
            print("错误: 使用 --poscar 时必须指定 --grid")
            sys.exit(1)
        kwargs["grid_shape"] = tuple(args.grid)
    else:
        print("错误: 必须指定 --db 或 --poscar")
        sys.exit(1)

    # 执行预测
    t0 = time.time()
    if args.output:
        if args.format == "npy":
            import numpy as np

            result = client.predict(**kwargs)
            np.save(args.output, result["density_3d"])
            print(f"已保存: {args.output}")
        else:
            result = client.predict_chgcar(args.output, **kwargs)
            out_path = args.output
            if not out_path.endswith(".vasp"):
                out_path += ".vasp"
            print(f"已保存: {out_path}")
    else:
        result = client.predict(**kwargs)

    elapsed = time.time() - t0
    nx, ny, nz = result["grid_shape"]
    d = result["density_3d"]
    print(f"网格: {nx}x{ny}x{nz}")
    print(f"密度范围: [{d.min():.6f}, {d.max():.6f}]")
    print(f"服务端推理: {result['elapsed']:.2f}s")
    print(f"总耗时: {elapsed:.2f}s")
