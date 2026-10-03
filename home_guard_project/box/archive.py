"""The alerts a box in inference mode has saved, for looking things up later.

An alert is a clip plus its ``.meta.json`` in the normal dataset layout, with
the alert itself (summary, command) in the meta. They sit in the production
folders on the box for PRODUCTION_RETENTION_DAYS, so the owner can ask "what
happened last night?" or "send me the video from 3pm" and get an answer from
the box's own disk. With a few thousand alerts at most, a scan of the meta
files is enough: no database and no index to keep in step.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .embeddings import cosine
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
    kind: str = "alert"                      # "alert" (guard hours) or "quiet" (outside them)
    label: str = ""                          # the AI's normal / suspicious / escalation, when it looked
    people: Optional[int] = None             # how many people the AI counted
    mode: str = ""                           # "guard" or "assistant" when the event happened
    detector_labels: Tuple[str, ...] = ()    # what the detector fired on
    described: bool = False                  # True when some AI description exists
    clip_start_ts: Optional[float] = None
    trigger_ts: Optional[float] = None       # when the detector fired (inside the clip)


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("saved record must be an object")
        return data
    except (OSError, ValueError, RecursionError) as exc:
        log.warning("Could not read saved record %s: %s", path, exc)
        return None


def _verdicts(roots: Sequence[str]) -> Dict[str, List[Tuple[str, str]]]:
    """Map alert id -> ``[(time_utc, verdict)]`` from every feedback file under *roots*."""
    found: Dict[str, List[Tuple[str, str]]] = {}
    for root in roots:
        for dirpath, _, filenames in os.walk(os.path.join(root, "feedback")):
            for name in filenames:
                data = _load_json(os.path.join(dirpath, name)) if name.endswith(FEEDBACK_SUFFIX) else None
                if data is None:
                    continue
                alert = data.get("alert")
                if not isinstance(alert, dict) or not isinstance(alert.get("alert_id"), str):
                    log.warning("Skipping malformed feedback %s", name)
                    continue
                alert_id = alert["alert_id"]
                if not isinstance(data.get("verdict"), (str, type(None))):
                    log.warning("Skipping malformed feedback %s", name)
                    continue
                if alert_id and data.get("verdict") not in (None, "none"):
                    found.setdefault(alert_id, []).append((str(data.get("time_utc")), data["verdict"]))
    return found


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("timestamp must be finite")
    # Check platform-supported dates before records reach a renderer.
    dt.datetime.fromtimestamp(number)
    return number


def _optional_ts(value: Any) -> Optional[float]:
    """A usable timestamp, or None when the field is missing or bad (one bad field never hides an alert)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return _finite(value)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _text(value: Any, default: str) -> str:
    return value if isinstance(value, str) and value else default


def _people(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return int(value)


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
                try:
                    # Falsy or missing falls back to the file time; a present but unusable value is a bad record.
                    ts = _finite(meta.get("clip_end_ts") or os.path.getmtime(meta_path))
                    clip = os.path.join(root, str(meta.get("clip_path") or "").replace("\\", os.sep))
                    summary = str(alert.get("summary") or "")
                    yolo = meta.get("yolo") if isinstance(meta.get("yolo"), dict) else {}
                    people = alert.get("people")
                    if people is None:
                        people = (alert.get("vlm") or {}).get("people") if isinstance(alert.get("vlm"), dict) else None
                    detector_labels = yolo.get("trigger_classes") or alert.get("labels") or []
                    if not isinstance(detector_labels, (list, tuple)):
                        detector_labels = []
                    records.append(AlertRecord(
                        alert_id=alert_id,
                        camera=_text(meta.get("camera_name"), ""),
                        ts=ts,
                        summary=summary,
                        command=_text(alert.get("alert_command"), ""),
                        clip_path=clip if meta.get("clip_path") and os.path.isfile(clip) else None,
                        verdicts=tuple(v for _, v in sorted(verdicts.get(alert_id, []))),
                        kind=_text(meta.get("kind"), "alert"),
                        label=_text(alert.get("label"), ""),
                        people=_people(people),
                        mode=_text(meta.get("mode"), ""),
                        detector_labels=tuple(str(x) for x in detector_labels),
                        described=bool(summary),
                        clip_start_ts=_optional_ts(meta.get("clip_start_ts")),
                        trigger_ts=_optional_ts(meta.get("trigger_ts")),
                    ))
                except (OSError, TypeError, ValueError, OverflowError) as exc:
                    log.warning("Skipping damaged meta %s: %s", meta_path, exc)
    return sorted(records, key=lambda r: r.ts)


def record_doc(record: AlertRecord, score: Optional[float] = None) -> Dict[str, Any]:
    """One alert as a JSON document for the agent: what it was, when, and whether the video is still here."""
    doc: Dict[str, Any] = {
        "id": record.alert_id,
        "time": dt.datetime.fromtimestamp(record.ts).strftime("%a %d %b %H:%M"),
        "camera": record.camera,
        "summary": record.summary or "no description",
        "owner_said": list(record.verdicts),
        "has_video": record.clip_path is not None,
    }
    if score is not None:
        doc["match"] = round(float(score), 3)
    return doc


def window(records: Sequence[AlertRecord], query: Query) -> List[AlertRecord]:
    """Every saved alert inside the query's time range (and camera), oldest first.

    Unlike :func:`search` this neither ranks nor caps - it is for summarizing a
    whole period ("what happened today?"), where the caller wants the full set.
    """
    return [
        r for r in records
        if query.start_ts <= r.ts <= query.end_ts and (query.camera is None or r.camera == query.camera)
    ]


def _words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in ("the", "and", "was")]


def _semantic_rank(hits: Sequence[AlertRecord], what: str, embedder: Any) -> Optional[List[AlertRecord]]:
    """*hits* ordered by how close each summary is in meaning to *what*.

    Returns None if the embedder cannot be reached at all, so the caller falls
    back to the keyword search. Records whose summary could not be embedded keep
    their time order at the end.
    """
    if not hits:
        return []
    query_vec = embedder.embed_one(what)
    if query_vec is None:
        return None
    vectors = embedder.embed([r.summary or r.camera for r in hits])
    if vectors is None:
        return None
    scored = sorted(
        ((cosine(query_vec, vec), i, r) for i, (r, vec) in enumerate(zip(hits, vectors)) if vec is not None),
        key=lambda t: (t[0], -t[1]), reverse=True,
    )
    ranked = [r for _, _, r in scored]
    ranked += [r for r, vec in zip(hits, vectors) if vec is None]
    return ranked


def search(records: Sequence[AlertRecord], query: Query, limit: int = 5, embedder: Any = None) -> List[AlertRecord]:
    """The alerts that match *query*: in time order, newest first for a "latest" query, or - when
    *query.what* is set and an *embedder* is given - the ones closest in meaning first.

    Time and camera are always the first filter (most questions are about a time
    or a camera). Semantic ranking, when available, orders what is left; without
    an embedder, or if it cannot be reached, ``query.what`` narrows by keyword
    and, when nothing matches, everything in the range is returned (the owner's
    words and the summary's often differ, and the caller reads the summaries).
    """
    hits = [
        r for r in records
        if query.start_ts <= r.ts <= query.end_ts and (query.camera is None or r.camera == query.camera)
    ]
    if query.latest:
        return hits[::-1][:limit]
    what = (query.what or "").strip()
    if what:
        ranked = _semantic_rank(hits, what, embedder) if embedder is not None else None
        if ranked is not None:
            return ranked[:limit]
        words = _words(what)
        if words:
            matching = [r for r in hits if any(w in f"{r.summary} {r.camera}".lower() for w in words)]
            return (matching or hits)[:limit]
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
