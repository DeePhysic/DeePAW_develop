"""Unix socket 请求处理器。"""

import json
import socketserver
import traceback

from .protocol import recv_message, send_message


class UnixRequestHandler(socketserver.StreamRequestHandler):
    """处理单个 Unix socket 连接的请求。"""

    def handle(self):
        try:
            request = recv_message(self.connection)
        except (ConnectionError, json.JSONDecodeError) as e:
            self._send_error(f"请求解析失败: {e}")
            return

        try:
            result = self.server.deepaw_server.handle_request(request)
            send_message(self.connection, result)
        except Exception as e:
            self._send_error(f"处理失败: {e}\n{traceback.format_exc()}")

    def _send_error(self, msg: str):
        try:
            send_message(self.connection, {"error": msg})
        except Exception:
            pass


class UnixSocketServer(socketserver.UnixStreamServer):
    """Unix socket 服务器，持有 DeePAWServer 引用。"""

    def __init__(self, socket_path: str, deepaw_server):
        self.deepaw_server = deepaw_server
        super().__init__(socket_path, UnixRequestHandler)
