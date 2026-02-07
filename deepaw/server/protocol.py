"""
消息帧协议与序列化工具。

Unix socket 协议：[4 bytes 大端长度][JSON payload]
密度数组：base64 编码 float32 bytes
Atoms：{numbers, positions, cell, pbc} JSON dict
"""

import base64
import json
import struct

import numpy as np

HEADER_FORMAT = ">I"  # 4 字节大端无符号整数
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAX_MESSAGE_SIZE = 256 * 1024 * 1024  # 256 MB


def send_message(sock, data: dict):
    """发送带长度前缀的 JSON 消息。"""
    payload = json.dumps(data).encode("utf-8")
    header = struct.pack(HEADER_FORMAT, len(payload))
    sock.sendall(header + payload)


def recv_message(sock) -> dict:
    """接收并解析带长度前缀的 JSON 消息。"""
    header = _recv_exact(sock, HEADER_SIZE)
    if not header:
        raise ConnectionError("连接已关闭")
    (length,) = struct.unpack(HEADER_FORMAT, header)
    if length > MAX_MESSAGE_SIZE:
        raise ValueError(f"消息过大: {length} bytes")
    payload = _recv_exact(sock, length)
    return json.loads(payload.decode("utf-8"))


def _recv_exact(sock, n: int) -> bytes:
    """从 socket 精确读取 n 字节。"""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            if buf:
                raise ConnectionError(f"连接中断，已读 {len(buf)}/{n} 字节")
            return b""
        buf.extend(chunk)
    return bytes(buf)


# === Atoms 序列化 ===

def atoms_to_dict(atoms) -> dict:
    """ASE Atoms → JSON-safe dict。"""
    return {
        "numbers": atoms.get_atomic_numbers().tolist(),
        "positions": atoms.get_positions().tolist(),
        "cell": atoms.get_cell().tolist(),
        "pbc": atoms.get_pbc().tolist(),
    }


def dict_to_atoms(d: dict):
    """JSON dict → ASE Atoms。"""
    from ase import Atoms

    return Atoms(
        numbers=d["numbers"],
        positions=d["positions"],
        cell=d["cell"],
        pbc=d["pbc"],
    )


# === 密度数组序列化 ===

def encode_density(arr: np.ndarray) -> str:
    """numpy float32 数组 → base64 字符串。"""
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")


def decode_density(s: str, shape: tuple) -> np.ndarray:
    """base64 字符串 → numpy float32 数组。"""
    raw = base64.b64decode(s)
    return np.frombuffer(raw, dtype=np.float32).reshape(shape)
