"""The alerts a box in inference mode has saved, for looking things up later.

An alert is a clip plus its ``.meta.json`` in the normal dataset layout, with
the alert itself (summary, command) in the meta. They sit in the production
folders on the box for PRODUCTION_RETENTION_DAYS, so the owner can ask "what
happened last night?" or "send me the video from 3pm" and get an answer from
the box's own disk. With a few thousand alerts at most, a scan of the meta
files is enough: no database and no index to keep in step.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .feedback import Query
from .outbox import META_SUFFIX

log = logging.getLogger("box.archive")

FEEDBACK_SUFFIX = ".feedback.json"


@dataclass(frozen=True)
class AlertRecord:
    alert_id: str               # the clip's file stem
    camera: str
    ts: float                   # when the clip ended (epoch seconds)
    summary: str
    command: str
    clip_path: Optional[str]    # the mp4, if it is still on the box
    verdicts: Tuple[str, ...]   # what the owner answered, oldest first


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _verdicts(roots: Sequence[str]) -> Dict[str, List[Tuple[str, str]]]:
    """Map alert id -> ``[(time_utc, verdict)]`` from every feedback file under *roots*."""
    found: Dict[str, List[Tuple[str, str]]] = {}
    for root in roots:
        for dirpath, _, filenames in os.walk(os.path.join(root, "feedback")):
            for name in filenames:
                data = _load_json(os.path.join(dirpath, name)) if name.endswith(FEEDBACK_SUFFIX) else None
                alert_id = ((data or {}).get("alert") or {}).get("alert_id")
                if alert_id and data.get("verdict") not in (None, "none"):
                    found.setdefault(alert_id, []).append((str(data.get("time_utc")), data["verdict"]))
    return found


def load_records(roots: Sequence[str]) -> List[AlertRecord]:
    """Every saved alert under *roots* (dataset folders), oldest first. Damaged metas are skipped."""
    verdicts = _verdicts(roots)
    records: List[AlertRecord] = []
    for root in roots:
        for dirpath, _, filenames in os.walk(os.path.join(root, "meta")):
            for name in filenames:
                if not name.endswith(META_SUFFIX):
                    continue
                meta_path = os.path.join(dirpath, name)
                meta = _load_json(meta_path)
                if meta is None:
                    continue
                alert_id = name[: -len(META_SUFFIX)]
                alert = meta.get("alert") if isinstance(meta.get("alert"), dict) else {}
                clip = os.path.join(root, str(meta.get("clip_path") or "").replace("\\", os.sep))
                records.append(AlertRecord(
                    alert_id=alert_id,
                    camera=str(meta.get("camera_name") or ""),
                    ts=float(meta.get("clip_end_ts") or os.path.getmtime(meta_path)),
                    summary=str(alert.get("summary") or ""),
                    command=str(alert.get("alert_command") or ""),
                    clip_path=clip if meta.get("clip_path") and os.path.isfile(clip) else None,
                    verdicts=tuple(v for _, v in sorted(verdicts.get(alert_id, []))),
                ))
    return sorted(records, key=lambda r: r.ts)


def _words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in ("the", "and", "was")]


def search(records: Sequence[AlertRecord], query: Query, limit: int = 5) -> List[AlertRecord]:
    """The alerts that match *query*: in time order, or newest first for a "latest" query.

    The words in ``query.what`` narrow the result when some summary contains
    them. When none does, everything in the range is returned: the owner's
    words and the summary's words often differ, and the caller reads the
    summaries anyway.
    """
    hits = [
        r for r in records
        if query.start_ts <= r.ts <= query.end_ts and (query.camera is None or r.camera == query.camera)
    ]
    words = _words(query.what)
    if words:
        matching = [r for r in hits if any(w in f"{r.summary} {r.camera}".lower() for w in words)]
        hits = matching or hits
    if query.latest:
        hits = hits[::-1]
    return hits[:limit]


def expire_old_files(root_dir: str, max_age_days: float, now: Optional[float] = None) -> int:
    """Delete files under *root_dir* older than *max_age_days*, and the folders that empties. Returns files deleted."""
    now = time.time() if now is None else now
    cutoff = now - max_age_days * 86400
    deleted = 0
    for dirpath, _, filenames in os.walk(root_dir, topdown=False):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                if os.path.getmtime(path) <= cutoff:
                    os.remove(path)
                    deleted += 1
            except OSError as exc:
                log.warning("Could not delete %s: %s", path, exc)
        if dirpath != root_dir and not os.listdir(dirpath):
            try:
                os.rmdir(dirpath)
            except OSError:
                pass
    return deleted
