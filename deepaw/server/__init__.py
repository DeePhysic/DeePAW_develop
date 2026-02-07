"""DeePAW Server: 模型常驻显存的推理服务"""

from .server import DeePAWServer
from .client import DeePAWClient

__all__ = ["DeePAWServer", "DeePAWClient"]
