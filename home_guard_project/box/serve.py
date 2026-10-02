"""What the box shows, served to the laptop app: a small read-only HTTP server on 127.0.0.1.

The window on the box reads its files directly (logs/preview, ai_status.json,
telegram_chat.jsonl). A window on the installer's laptop cannot, so it runs
this server over the ssh link it already has and forwards one port::

    ssh -i <key> -L 8765:127.0.0.1:8765 <user>@<box> \\
        "cd /d C:\\home_guard && .venv\\Scripts\\python.exe -m home_guard_project.box.serve"

The server binds to the box's own loopback only, so nothing on the box's
network can reach it; it stops when its stdin closes (the ssh session ends),
so it never outlives the window. It changes nothing on the box except the
preview's ``viewer.alive`` marker, which tells the running program that a
window is watching and which camera is large.

    GET  /status                     one JSON with the box's state, its settings, the camera
                                     lists (active and disabled), the AI's status and the
                                     names the preview publishes
    GET  /ai_status.json             the AI status file as it is
    GET  /chat?limit=200             the Telegram conversation (chat_feed.read_feed)
    GET  /chat_images/<name>.jpg     a picture from the conversation
    GET  /preview/<camera>.jpg       the latest preview picture of a camera (404 when none)
    POST /viewer  {"hero": "<camera>"}  keep the preview alive; the hero camera gets more frames
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qs, urlsplit

log = logging.getLogger("box.serve")

DEFAULT_PORT = 8765
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


class BoxData:
    """Everything the server hands out, read fresh on each request. Pure file reads; never raises to the handler."""

    def __init__(self, log_dir: str, cameras_path: Optional[str] = None, status: Optional[Callable[[], Dict]] = None,
                 settings: Optional[Callable[[], Dict]] = None) -> None:
        self.log_dir = log_dir
        self.preview_dir = os.path.join(log_dir, "preview")
        self.cameras_path = cameras_path
        self._status = status
        self._settings = settings

    def status(self) -> Dict[str, Any]:
        from .ai_status import read_status  # noqa: PLC0415
        from .preview import PreviewReader  # noqa: PLC0415

        out: Dict[str, Any] = {"box": {}, "settings": {}, "cameras": {"active": [], "disabled": []}}
        for key, fn in (("box", self._status), ("settings", self._settings)):
            if fn is not None:
                try:
                    out[key] = fn()
                except Exception as exc:  # noqa: BLE001
                    out[key] = {"error": str(exc)}
        out["cameras"] = self.cameras()
        out["preview_names"] = PreviewReader(self.preview_dir).names()
        out["ai"] = read_status(os.path.join(self.log_dir, "ai_status.json"))
        return out

    def cameras(self) -> Dict[str, Any]:
        """Camera names only, never their addresses or logins."""
        if not self.cameras_path or not os.path.isfile(self.cameras_path):
            return {"active": [], "disabled": []}
        try:
            import yaml  # noqa: PLC0415

            with open(self.cameras_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except Exception:  # noqa: BLE001
            return {"active": [], "disabled": []}
        return {"active": sorted((raw.get("cameras") or {}).keys()),
                "disabled": sorted((raw.get("disabled") or {}).keys())}

    def chat(self, limit: int) -> Any:
        from .chat_feed import read_feed  # noqa: PLC0415

        return read_feed(os.path.join(self.log_dir, "telegram_chat.jsonl"), limit=limit)

    def chat_image(self, name: str) -> Optional[str]:
        path = os.path.join(self.log_dir, "chat_images", name)
        return path if _NAME.match(name) and name.endswith(".jpg") and os.path.isfile(path) else None

    def preview(self, camera: str) -> Optional[str]:
        from .preview import camera_key  # noqa: PLC0415

        path = os.path.join(self.preview_dir, camera_key(camera) + ".jpg")
        return path if _NAME.match(camera) and os.path.isfile(path) else None

    def viewer(self, hero: Optional[str]) -> None:
        from .preview import PreviewReader  # noqa: PLC0415

        PreviewReader(self.preview_dir).touch(hero if hero and _NAME.match(hero) else None)


class Handler(BaseHTTPRequestHandler):
    data: BoxData

    def log_message(self, fmt: str, *args: Any) -> None:   # one quiet line per request, through logging
        log.debug("%s " + fmt, self.address_string(), *args)

    # -- helpers -----------------------------------------------------------------
    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Optional[str], content_type: str) -> None:
        if path is None:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            with open(path, "rb") as f:
                body = f.read()
            mtime = os.stat(path).st_mtime
        except OSError:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Modified", repr(mtime))
        self.end_headers()
        self.wfile.write(body)

    # -- routes ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - http.server's name
        url = urlsplit(self.path)
        parts = [p for p in url.path.split("/") if p]
        try:
            if parts == ["status"]:
                self._json(self.data.status())
            elif parts == ["ai_status.json"]:
                self._file(os.path.join(self.data.log_dir, "ai_status.json"), "application/json")
            elif parts == ["chat"]:
                limit = int((parse_qs(url.query).get("limit") or ["200"])[0])
                self._json(self.data.chat(max(1, min(limit, 1000))))
            elif len(parts) == 2 and parts[0] == "chat_images":
                self._file(self.data.chat_image(parts[1]), "image/jpeg")
            elif len(parts) == 2 and parts[0] == "preview" and parts[1].endswith(".jpg"):
                self._file(self.data.preview(parts[1][:-4]), "image/jpeg")
            else:
                self._json({"error": "no such path"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001 - one bad request must not take the server down
            log.warning("GET %s failed: %s", self.path, exc)
            self._json({"error": "failed"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        parts = [p for p in urlsplit(self.path).path.split("/") if p]
        if parts != ["viewer"]:
            self._json({"error": "no such path"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}") if length else {}
            self.data.viewer(body.get("hero") if isinstance(body, dict) else None)
            self._json({"ok": True})
        except Exception as exc:  # noqa: BLE001
            log.warning("POST /viewer failed: %s", exc)
            self._json({"error": "failed"}, HTTPStatus.BAD_REQUEST)


def make_server(data: BoxData, bind: str = "127.0.0.1", port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    handler = type("BoxHandler", (Handler,), {"data": data})
    return ThreadingHTTPServer((bind, port), handler)


def stop_when_stdin_closes(server: ThreadingHTTPServer, stdin: Any = None) -> threading.Thread:
    """End the server when the ssh session that started it goes away (its stdin reaches EOF)."""
    stream = stdin if stdin is not None else sys.stdin

    def watch() -> None:
        try:
            while stream.readline():
                pass
        except Exception:  # noqa: BLE001
            pass
        server.shutdown()

    thread = threading.Thread(target=watch, name="serve-stdin", daemon=True)
    thread.start()
    return thread


def box_data() -> BoxData:
    """The box's own folders and status, for the real server."""
    from .boxconfig import LOG_DIR, load_box_config, load_box_settings  # noqa: PLC0415
    from .find_cameras import CAMERAS_PATH  # noqa: PLC0415

    def status() -> Dict[str, Any]:
        from .__main__ import _status  # noqa: PLC0415

        return _status(load_box_config())

    def settings() -> Dict[str, Any]:
        from .boxconfig import OPTIONS  # noqa: PLC0415

        raw = load_box_settings()
        return {key: raw.get(key) for key in OPTIONS if key != "telegram_chat_ids"} | {"site": raw.get("site")}

    return BoxData(LOG_DIR, CAMERAS_PATH, status, settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve what the box shows to the laptop app (loopback only).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    server = make_server(box_data(), args.bind, args.port)
    stop_when_stdin_closes(server)
    print(f"serving on http://{args.bind}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
