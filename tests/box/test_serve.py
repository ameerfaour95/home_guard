from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from home_guard_project.box.chat_feed import ChatFeed
from home_guard_project.box.preview import camera_key
from home_guard_project.box.serve import BoxData, make_server, stop_when_stdin_closes


class ServeTest(unittest.TestCase):
    """The laptop app's view of the box: what the server hands out, and what it refuses."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.logs = os.path.join(tmp.name, "logs")
        os.makedirs(os.path.join(self.logs, "preview"))
        with open(os.path.join(self.logs, "ai_status.json"), "w", encoding="utf-8") as f:
            json.dump({"updated": 1.0, "cameras": {}, "decisions": [{"summary": "hi"}]}, f)
        with open(os.path.join(self.logs, "preview", "cameras.json"), "w", encoding="utf-8") as f:
            json.dump(["front", "yard"], f)
        with open(os.path.join(self.logs, "preview", camera_key("front") + ".jpg"), "wb") as f:
            f.write(b"\xff\xd8front")
        ChatFeed(os.path.join(self.logs, "telegram_chat.jsonl")).add("box", "alert", "front: a person",
                                                                     alert_id="front_1_alert", image=b"\xff\xd8pic", now=5.0)
        cameras = os.path.join(tmp.name, "cameras.yaml")
        with open(cameras, "w", encoding="utf-8") as f:
            f.write("cameras:\n  front: rtsp://admin:secret@10.0.0.9/a\ndisabled:\n  yard: rtsp://admin:secret@10.0.0.9/b\n")
        data = BoxData(self.logs, cameras, status=lambda: {"stopped": False, "site": "test"},
                       settings=lambda: {"inference_conf": 0.4})
        self.server = make_server(data, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)

    def _get(self, path: str):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()

    def test_status_has_the_box_the_settings_the_cameras_and_the_ai(self) -> None:
        status, ctype, body = self._get("/status")
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(data["box"], {"stopped": False, "site": "test"})
        self.assertEqual(data["settings"], {"inference_conf": 0.4})
        self.assertEqual(data["cameras"], {"active": ["front"], "disabled": ["yard"]})
        self.assertEqual(data["preview_names"], ["front", "yard"])
        self.assertEqual(data["ai"]["decisions"], [{"summary": "hi"}])
        self.assertNotIn("secret", body.decode())                         # never a camera login

    def test_pictures_and_the_conversation(self) -> None:
        status, ctype, body = self._get("/preview/front.jpg")
        self.assertEqual((status, ctype, body), (200, "image/jpeg", b"\xff\xd8front"))
        with self.assertRaises(urllib.error.HTTPError) as missing:
            self._get("/preview/yard.jpg")
        self.assertEqual(missing.exception.code, 404)
        status, _, body = self._get("/chat?limit=10")
        (entry,) = json.loads(body)
        self.assertEqual(entry["text"], "front: a person")
        status, ctype, body = self._get(f"/chat_images/{entry['image']}")
        self.assertEqual((status, ctype, body), (200, "image/jpeg", b"\xff\xd8pic"))
        status, _, body = self._get("/ai_status.json")
        self.assertEqual(json.loads(body)["decisions"], [{"summary": "hi"}])

    def test_only_known_paths_and_plain_names(self) -> None:
        for path in ("/", "/logs/runner.log", "/chat_images/../ai_status.json", "/preview/..%2Fcameras.json",
                     "/chat_images/x.txt"):
            with self.assertRaises(urllib.error.HTTPError, msg=path) as err:
                self._get(path)
            self.assertEqual(err.exception.code, 404, path)

    def test_the_viewer_keeps_the_preview_alive_with_a_hero(self) -> None:
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/viewer", data=json.dumps({"hero": "front"}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertEqual(json.loads(r.read()), {"ok": True})
        with open(os.path.join(self.logs, "preview", "viewer.alive"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["hero"], "front")

    def test_the_server_stops_when_its_stdin_closes(self) -> None:
        server = make_server(BoxData(self.logs), "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        watcher = stop_when_stdin_closes(server, stdin=io.StringIO(""))     # EOF at once
        watcher.join(timeout=5)
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
