"""What the detector and the AI just saw, written to one small file for the box's window to show.

Inference mode runs as a background task and the window is another program, so
they share ``logs/ai_status.json``:

    {"updated": <epoch>,
     "cameras":   {"<camera>": {"checked_ts": <epoch>, "ts": <epoch of the last detection>,
                                "objects": [{"label": "person", "conf": 0.71,
                                             "box": [x1, y1, x2, y2]}]}},     # box: 0..1 of the picture
     "thinking": {"camera": "...", "labels": ["person"], "ts": <epoch>} or null,  # the AI is looking now
     "settings": {"conf": 0.4, "alert_start_hour": 0, "alert_end_hour": 0, "cooldown_sec": 120.0},  # in force now
     "decisions": [{"ts": <epoch>, "camera": "...", "labels": ["person"],
                    "summary": "A person is walking in the driveway.", "label": "normal",   # or suspicious / escalation
                    "command": "[send_message]", "sent": true, "false_positive": false,
                    "muted": false, "error": ""}],                            # newest last
     "mode": "guard" | "assistant" | null,
     "status_line": "🛡️ Guarding until 06:00" or null}

A camera's entry may also carry ``frame_ts``: when the camera last delivered a
new picture (not a repeat of an old frame). Live/offline follows it.

The file holds no picture, address or login. It is rewritten at most about
once a second, through a temp file, so a reader never sees half of it.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

KEEP_DECISIONS = 50

# What the window is told about. The detector knows 80 kinds of things and names a "sink" or a
# "potted plant" as readily as a person; only the kinds the house cares about are shown.
SHOWN_LABELS = frozenset({"person", "bicycle", "car", "motorcycle", "bus", "truck", "bird", "cat", "dog"})


def objects_from_result(result: Any, shown: frozenset = SHOWN_LABELS) -> List[Dict[str, Any]]:
    """The objects of interest in one ultralytics result: label, confidence, and box as fractions of the picture."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    names = getattr(result, "names", {})
    objects = []
    for box in boxes:
        label = names.get(int(box.cls[0]), str(int(box.cls[0])))
        if label not in shown:
            continue
        x1, y1, x2, y2 = (round(float(v), 4) for v in box.xyxyn[0])
        objects.append({"label": label, "conf": round(float(box.conf[0]), 2), "box": [x1, y1, x2, y2]})
    return objects


class AiStatus:
    """Collects the latest detections and decisions, and writes them out. Safe to call from several threads."""

    def __init__(self, path: str, min_interval: float = 1.0) -> None:
        self.path = path
        self.min_interval = min_interval
        self._cameras: Dict[str, Dict[str, Any]] = {}
        self._decisions: List[Dict[str, Any]] = []
        self._thinking: Optional[Dict[str, Any]] = None
        self._settings: Dict[str, Any] = {}
        self._mode: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._written = 0.0

    def detection(self, camera: str, objects: List[Dict[str, Any]], now: Optional[float] = None) -> None:
        """Record what the detector found in *camera*'s latest picture (an empty list: nothing)."""
        now = time.time() if now is None else now
        with self._lock:
            entry = self._cameras.setdefault(camera, {"checked_ts": now, "ts": None, "objects": []})
            entry["checked_ts"] = now
            if objects:
                entry["ts"], entry["objects"] = now, objects
            self._write(now, force=False)

    def settings(self, values: Dict[str, Any], now: Optional[float] = None) -> None:
        """The values in force right now (the detector's threshold, the alert hours, the cooldown)."""
        now = time.time() if now is None else now
        with self._lock:
            self._settings = dict(values)
            self._write(now, force=True)

    def thinking(self, camera: str, labels: List[str], now: Optional[float] = None) -> None:
        """The detector fired on *camera* and the AI is now looking at the clip; cleared by the decision."""
        now = time.time() if now is None else now
        with self._lock:
            self._thinking = {"camera": camera, "labels": list(labels), "ts": now}
            self._write(now, force=True)

    def decision(self, camera: str, labels: List[str], summary: str, command: str, sent: bool,
                 false_positive: bool = False, muted: bool = False, error: str = "",
                 now: Optional[float] = None, label: str = "") -> None:
        """Record what the AI said about one trigger, and what happened to the alert.

        *label* is the AI's own label for the scene: normal, suspicious or escalation.
        """
        now = time.time() if now is None else now
        with self._lock:
            self._thinking = None
            self._decisions.append({
                "ts": now, "camera": camera, "labels": list(labels), "summary": summary, "command": command,
                "label": label, "sent": bool(sent), "false_positive": bool(false_positive), "muted": bool(muted),
                "error": error,
            })
            self._decisions = self._decisions[-KEEP_DECISIONS:]
            self._write(now, force=True)

    def mode(self, mode: str, line: str, now: Optional[float] = None) -> None:
        """Guard or Assistant, and the status line the app and Telegram show."""
        now = time.time() if now is None else now
        with self._lock:
            self._mode = {"mode": mode, "line": line}
            self._write(now, force=True)

    def frame_seen(self, camera: str, ts: float) -> None:
        """The camera delivered a new picture at *ts* (not a repeat of an old frame)."""
        with self._lock:
            entry = self._cameras.setdefault(camera, {"checked_ts": None, "ts": None, "objects": []})
            entry["frame_ts"] = ts

    def offline(self, now: float, after: float = 60.0, cameras: Sequence[str] = (), since: float = 0.0) -> List[str]:
        """Cameras with no new picture for *after* seconds; a listed camera that never sent one is offline too.

        *since* is when watching began: a camera is not counted offline before it had *after* seconds to deliver.
        """
        with self._lock:
            names = set(self._cameras) | set(cameras)
            return sorted(n for n in names
                          if now - max(float((self._cameras.get(n) or {}).get("frame_ts") or 0), float(since))
                          > after)

    def _write(self, now: float, force: bool) -> None:
        if not force and now - self._written < self.min_interval:
            return
        self._written = now
        data = {"updated": now, "cameras": self._cameras, "thinking": self._thinking,
                "settings": self._settings, "decisions": self._decisions,
                "mode": self._mode.get("mode"), "status_line": self._mode.get("line")}
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = f"{self.path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
        except OSError:
            pass    # the window holds the file open for a moment: the next write goes through


def read_status(path: str) -> Dict[str, Any]:
    """The status file as a dict; empty when it is missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}
