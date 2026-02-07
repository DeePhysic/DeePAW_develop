"""HTTP 请求处理器。"""

import json
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer


class HTTPRequestHandler(BaseHTTPRequestHandler):
    """处理 HTTP 请求。"""

    def do_GET(self):
        if self.path in ("/status", "/health"):
            result = self.server.deepaw_server.handle_request(
                {"action": "status"}
            )
            self._send_json(200, result)
        else:
            self._send_json(404, {"error": f"未知路径: {self.path}"})

    def do_POST(self):
        if self.path == "/predict":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                request = json.loads(body.decode("utf-8"))
            except (json.JSONDecodeError, ValueError) as e:
                self._send_json(400, {"error": f"请求解析失败: {e}"})
                return

            request.setdefault("action", "predict")
            try:
                result = self.server.deepaw_server.handle_request(request)
                status = 500 if "error" in result else 200
                self._send_json(status, result)
            except Exception as e:
                self._send_json(500, {
                    "error": f"处理失败: {e}",
                    "traceback": traceback.format_exc(),
                })
        else:
            self._send_json(404, {"error": f"未知路径: {self.path}"})

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        """静默日志，避免刷屏。"""
        pass


class DeePAWHTTPServer(HTTPServer):
    """HTTP 服务器，持有 DeePAWServer 引用。"""

    def __init__(self, host: str, port: int, deepaw_server):
        self.deepaw_server = deepaw_server
        super().__init__((host, port), HTTPRequestHandler)
