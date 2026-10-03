"""Loopback bridge for the laptop viewer; launched only through its SSH tunnel.

Extends the existing read-only server without changing serve.py. The only write is
the viewer lease. Camera capture, configuration and detector ownership stay put.
"""
import json
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from ..serve import Handler, box_data, stop_when_stdin_closes
from ..preview import PreviewReader


class LiveHandler(Handler):
    def do_POST(self):
        if self.path != "/viewer": return super().do_POST()
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 16384: raise ValueError("Invalid demand")
            demand = json.loads(self.rfile.read(length))
            names = demand.get("cameras", [])
            if not isinstance(names, list) or len(names)>32 or not all(isinstance(n,str) for n in names):
                raise ValueError("Invalid cameras")
            allowed = self.data.cameras().get("active", [])
            names = [n for n in names if n in allowed]
            hero = demand.get("hero")
            PreviewReader(self.data.preview_dir).touch(hero if hero in names else None,
                visible=demand.get("visible") is True, cameras=names)
            self._json({"ok": True})
        except (OSError, ValueError, TypeError, AttributeError):
            self._json({"error": "Invalid viewer demand"}, HTTPStatus.BAD_REQUEST)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8765), type("LiveHandler",(LiveHandler,),{"data":box_data()}))
    stop_when_stdin_closes(server)
    try: server.serve_forever()
    finally: server.server_close()


if __name__ == "__main__": main()
