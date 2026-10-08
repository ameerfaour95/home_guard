"""The Telegram conversation as the box's window shows it.

Alerts the box sent, what the owner tapped or wrote, and what the assistant
answered, in the order they happened: one JSON object per line in
``logs/telegram_chat.jsonl``, oldest first.

    {"ts": <epoch>, "who": "box" | "owner" | "assistant", "name": "Ameer",
     "kind": "alert" | "video" | "message" | "button" | "answer",
     "text": "front_door: a person at the door", "camera": "front_door",
     "alert_id": "front_door_1790972263_alert", "image": "front_door_1790972263_alert.jpg",
     "delivered": true, "error": ""}

``image`` names a JPEG in ``logs/chat_images/`` (the picture sent with an
alert). ``delivered`` false with ``error`` set means Telegram refused the
message. The file holds no token and no chat id. Writing never raises: the
window's feed must not be able to stop an alert.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

FEED_NAME = "telegram_chat.jsonl"   # in the logs folder (paths.logs_dir())
KEEP = 300            # messages kept; the file is trimmed back to this when it reaches twice as many
_SAFE = re.compile(r"[^A-Za-z0-9_-]")


class ChatFeed:
    """Appends to the conversation file. Safe to call from several threads."""

    def __init__(self, path: str, keep: int = KEEP) -> None:
        self.path = path
        self.keep = keep
        self.image_dir = os.path.join(os.path.dirname(path) or ".", "chat_images")
        self._lock = threading.Lock()

    def add(self, who: str, kind: str, text: str, name: str = "", camera: str = "", alert_id: str = "",
            image: Optional[bytes] = None, delivered: bool = True, error: str = "",
            now: Optional[float] = None) -> None:
        entry: Dict[str, Any] = {
            "ts": time.time() if now is None else now, "who": who, "name": name, "kind": kind, "text": text,
            "camera": camera, "alert_id": alert_id, "image": "", "delivered": bool(delivered), "error": error,
        }
        try:
            with self._lock:
                if image and alert_id:
                    entry["image"] = self._save_image(alert_id, image)
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self._trim()
        except OSError:
            pass

    def _save_image(self, alert_id: str, image: bytes) -> str:
        name = f"{_SAFE.sub('_', alert_id)}.jpg"
        os.makedirs(self.image_dir, exist_ok=True)
        with open(os.path.join(self.image_dir, name), "wb") as f:
            f.write(image)
        return name

    def _trim(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < self.keep * 2:
            return
        kept = lines[-self.keep:]
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(kept)
        os.replace(tmp, self.path)
        wanted = {entry.get("image") for entry in _parse(kept)}
        if os.path.isdir(self.image_dir):
            for name in os.listdir(self.image_dir):
                if name not in wanted:
                    os.remove(os.path.join(self.image_dir, name))


def _parse(lines: List[str]) -> List[Dict[str, Any]]:
    entries = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue                       # a line cut off by a power loss
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def read_feed(path: str, limit: int = 100) -> List[Dict[str, Any]]:
    """The newest *limit* messages, oldest first. Empty when there is no file yet."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    return _parse(lines)[-limit:]
