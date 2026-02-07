"""
DeePAWClient: 连接 DeePAW 推理服务的 Python 客户端。

通过 Unix socket 与 DeePAWServer 通信，避免每次预测都加载模型。
"""

import os
import socket

import numpy as np

from .protocol import (
    atoms_to_dict,
    decode_density,
    dict_to_atoms,
    recv_message,
    send_message,
)


class DeePAWClient:
    """DeePAW 推理服务客户端。

    Args:
        socket_path: Unix socket 路径（需与服务器一致）
    """

    def __init__(self, socket_path=None):
        from deepaw.config import SERVER_DEFAULTS

        self.socket_path = os.path.expanduser(
            socket_path or SERVER_DEFAULTS["socket_path"]
        )

    def predict(
        self, db_path=None, db_id=None, atoms=None, grid_shape=None
    ) -> dict:
        """预测电荷密度。

        Args:
            db_path: ASE 数据库路径
            db_id: 数据库中的结构 ID
            atoms: ASE Atoms 对象
            grid_shape: 网格尺寸 (nx, ny, nz)

        Returns:
            dict: {
                "density_3d": np.ndarray (nx, ny, nz),
                "density_flat": np.ndarray (nx*ny*nz,),
                "atoms": ASE Atoms 对象,
                "grid_shape": (nx, ny, nz),
                "elapsed": float,
            }
        """
        request = {"action": "predict"}

        if db_path is not None:
            request["db_path"] = os.path.abspath(db_path)
        if db_id is not None:
            request["db_id"] = db_id
        if atoms is not None:
            request["atoms"] = atoms_to_dict(atoms)
        if grid_shape is not None:
            request["grid_shape"] = list(grid_shape)

        response = self._send(request)

        if "error" in response:
            raise RuntimeError(f"服务器错误: {response['error']}")

        grid_shape = tuple(response["grid_shape"])
        density_flat = decode_density(response["density_b64"], (-1,))
        density_3d = density_flat.reshape(grid_shape)

        return {
            "density_3d": density_3d,
            "density_flat": density_flat,
            "atoms": dict_to_atoms(response["atoms"]),
            "grid_shape": grid_shape,
            "elapsed": response.get("elapsed", 0),
        }

    def predict_chgcar(self, output_path, **kwargs):
        """预测并写入 CHGCAR 文件。"""
        from ase.calculators.vasp import VaspChargeDensity

        result = self.predict(**kwargs)

        vcd = VaspChargeDensity(filename=None)
        vcd.atoms.append(result["atoms"])
        vcd.chg.append(result["density_3d"])

        if not output_path.endswith(".vasp"):
            output_path = f"{output_path}.vasp"

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        vcd.write(output_path, format="chgcar")
        return result

    def status(self) -> dict:
        """查询服务器状态。"""
        return self._send({"action": "status"})

    def is_running(self) -> bool:
        """检查服务器是否在运行。"""
        try:
            self.status()
            return True
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return False

    def _send(self, request: dict) -> dict:
        """发送请求并接收响应。"""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(self.socket_path)
            send_message(sock, request)
            return recv_message(sock)
        finally:
            sock.close()
