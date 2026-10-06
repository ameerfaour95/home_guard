"""The eval set's conventions, as ``home_guard_project/box/eval_prompt.py`` defines them.

The Admin Center checkout has no box/ package, so the few rules the studio needs are restated here: how a
tagger's text becomes our truth word (``[alert]`` -> alert, "No special activity" -> empty, else normal), which
rows are dropped (``[delete]`` or no text), where a clip's 5 frames live in an eval folder, and the clip's local
time from the epoch in its name. Keep them in step with eval_prompt.py.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

FRAME_COUNT = 5
FRAMES_DIR = "frames"
JPEG_QUALITY = 90
MAX_SIDE = 1280

_TAG_RE = re.compile(r"\[[^\]]*\]")
_ALERT_RE = re.compile(r"\[(alert|alet)\]", re.IGNORECASE)
_EPOCH_RE = re.compile(r"_(\d{10})(?:_|$)")


def parse_truth(description: str) -> Tuple[str, str]:
    """Our label (``alert`` | ``empty`` | ``normal``) and the text without its ``[...]`` tag."""
    text = re.sub(r"\s+", " ", _TAG_RE.sub("", description or "")).strip()
    if _ALERT_RE.search(description or ""):
        return "alert", text
    if text.rstrip(".").strip().lower() == "no special activity":
        return "empty", text
    return "normal", text


def is_dropped(description: str) -> bool:
    return not (description or "").strip() or "[delete]" in description.lower()


def frame_paths(clip_id: str) -> List[str]:
    return [f"{FRAMES_DIR}/{clip_id}_{i}.jpg" for i in range(FRAME_COUNT)]


def clip_epoch(clip_id: str) -> Optional[int]:
    m = _EPOCH_RE.search(clip_id or "")
    return int(m.group(1)) if m else None


def clip_local_time(clip_id: str) -> Optional[str]:
    epoch = clip_epoch(clip_id)
    return datetime.fromtimestamp(epoch).strftime("%H:%M:%S") if epoch is not None else None


def read_jsonl(path: str) -> Tuple[List[Dict[str, Any]], int]:
    """``(rows, bad lines)``; a missing file is no rows."""
    rows, bad = [], 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except FileNotFoundError:
        pass
    return rows, bad


def sample_indices(n: int, k: int = FRAME_COUNT) -> List[int]:
    if n <= 0:
        return []
    if n <= k:
        return list(range(n)) + [n - 1] * (k - n)
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def sample_frames(path: str, k: int = FRAME_COUNT) -> List[Any]:
    """``k`` evenly spaced frames of the clip, long side at most MAX_SIDE; ``[]`` if it will not decode."""
    import cv2  # noqa: PLC0415

    frames = []
    cap = cv2.VideoCapture(path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    picked = [frames[i] for i in sample_indices(len(frames), k)]
    out = []
    for frame in picked:
        h, w = frame.shape[:2]
        scale = min(1.0, MAX_SIDE / max(h, w))
        out.append(cv2.resize(frame, (round(w * scale), round(h * scale))) if scale < 1 else frame)
    return out


def exists(path: str) -> bool:
    return bool(path) and os.path.isfile(path)
