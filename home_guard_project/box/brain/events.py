# home_guard_project/box/brain/events.py
"""Saved events for the assistant: alerts from the guard hours and quiet events from the rest of the day.

A quiet event has no AI description until the owner asks about it; its
description is then cached in ``<desc_dir>/<alert_id>.json`` (one folder for
the live and the archived clips, because the uploader moves clips between
them). Searching never invents a match: a meaning search ranks only events
that have a description, a keyword search returns only real matches, and an
undescribed event is offered only when the detector saw what the owner asked
about - always marked "not confirmed".
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import math
from functools import wraps
import os
import re
from uuid import uuid4
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from ..archive import AlertRecord, _finite, _words, load_records
from ..embeddings import cosine
from ..inference import PERSON_CLASSES, VEHICLE_CLASSES

log = logging.getLogger("box.brain.events")

DESC_DIR_NAME = ".desc"
MIN_SIMILARITY = 0.3  # a meaning search keeps only events at least this close in meaning
SEVERITY = {"escalation": 0, "suspicious": 1, "normal": 2, "": 3}

_PERSON_WORDS = ("person", "people", "someone", "somebody", "anyone", "anybody", "man", "woman", "men", "women",
                 "kid", "child", "visitor", "courier", "delivery", "stranger", "intruder", "guy")
_VEHICLE_WORDS = ("car", "cars", "vehicle", "vehicles", "truck", "van", "bus", "motorcycle", "bike", "taxi")
_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _safe(default):
    """Contain bad input at the loop boundary, with one diagnostic per failed call."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                log.warning("%s failed: %s", function.__name__, exc)
                return default()
        return wrapped
    return decorate


@_safe(lambda: "unknown")
def local(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%a %d %b %H:%M")


def _desc_path(desc_dir: str, alert_id: str) -> str:
    return os.path.join(desc_dir, f"{_SAFE.sub('_', alert_id)}.json")


@_safe(dict)
def read_desc(desc_dir: str, alert_id: str) -> Dict[str, Dict[str, Any]]:
    """Every cached description of *alert_id*, by key (``"<mode>:<prompt version>:<question>"``)."""
    if not desc_dir:
        return {}
    try:
        with open(_desc_path(desc_dir, alert_id), encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("description cache must be an object")
        return data
    except FileNotFoundError:
        return {}


@_safe(lambda: None)
def write_desc(desc_dir: str, alert_id: str, key: str, value: Dict[str, Any]) -> None:
    """Cache one description. Never raises."""
    try:
        if not isinstance(key, str) or not isinstance(value, dict):
            raise TypeError("description key must be text and value must be an object")
        data = read_desc(desc_dir, alert_id)
        data[key] = value
        serialized = json.dumps(data, ensure_ascii=False, allow_nan=False)
        os.makedirs(desc_dir, exist_ok=True)
        path = _desc_path(desc_dir, alert_id)
        tmp = f"{path}.{os.getpid()}.{uuid4().hex[:6]}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(serialized)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
    except OSError as exc:
        log.warning("Description of %s not cached: %s", alert_id, exc)


@_safe(lambda: None)
def latest_desc(entries: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid = []
    damaged = False
    for value in entries.values():
        try:
            if not isinstance(value, dict) or not isinstance(value.get("text"), str):
                raise ValueError("invalid description")
            timestamp = float(value.get("ts") or 0)
            if not math.isfinite(timestamp):
                raise ValueError("nonfinite description timestamp")
            if value["text"]:
                valid.append((timestamp, value))
        except (TypeError, ValueError, OverflowError):
            damaged = True
    if damaged:
        log.warning("Skipping malformed cached descriptions")
    return max(valid, key=lambda item: item[0])[1] if valid else None


@_safe(list)
def load_events(roots: Sequence[str], desc_dir: str = "") -> List[AlertRecord]:
    """Every saved event, oldest first, with cached descriptions filled in for undescribed ones."""
    out = []
    for record in load_records(roots):
        if not record.summary and desc_dir:
            cached = latest_desc(read_desc(desc_dir, record.alert_id))
            if cached:
                record = dataclasses.replace(record, summary=str(cached["text"]), described=True,
                                             label=record.label or str(cached.get("label") or ""))
        out.append(record)
    return out


@_safe(list)
def filter_events(records: Iterable[AlertRecord], start_ts: float, end_ts: float,
                  cameras: Optional[Set[str]] = None, kinds: Optional[Set[str]] = None,
                  labels: Optional[Set[str]] = None) -> List[AlertRecord]:
    start_ts, end_ts = _finite(start_ts), _finite(end_ts)
    return [
        r for r in records
        if start_ts <= r.ts <= end_ts
        and (not cameras or r.camera in cameras)
        and (not kinds or r.kind in kinds)
        and (not labels or r.label in labels)
    ]


@_safe(set)
def class_words(what: str) -> Set[str]:
    """Detector classes the owner's words are about: ``{"person"}``, the vehicle classes, both, or nothing."""
    if what is not None and not isinstance(what, str):
        raise TypeError("question must be text")
    words = set(re.findall(r"[a-z]+", (what or "").lower()))
    out: Set[str] = set()
    if words & set(_PERSON_WORDS):
        out |= PERSON_CLASSES
    if words & set(_VEHICLE_WORDS):
        out |= VEHICLE_CLASSES
    return out


def _meaning_rank(described: Sequence[AlertRecord], what: str, embedder: Any) -> Optional[List[AlertRecord]]:
    """Described events close enough in meaning to *what*, closest first; None if the embedder is unavailable."""
    if not described:
        return []
    query_vec = embedder.embed_one(what)
    if query_vec is None:
        return None
    vectors = embedder.embed([r.summary for r in described])
    if vectors is None:
        return None
    scored = [(cosine(query_vec, vec), i, r) for i, (r, vec) in enumerate(zip(described, vectors)) if vec is not None]
    scored = [t for t in scored if t[0] >= MIN_SIMILARITY]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [r for _, _, r in scored]


@_safe(list)
def rank_events(records: Sequence[AlertRecord], what: str, embedder: Any = None) -> List[AlertRecord]:
    """Events matching *what*: described ones by meaning (or keyword), then undescribed detector hits."""
    if not (what or "").strip():
        return list(records)
    described = [r for r in records if r.described]
    undescribed = [r for r in records if not r.described]
    try:
        ranked = _meaning_rank(described, what, embedder) if embedder is not None else None
    except Exception as exc:
        log.warning("Meaning search failed; using keywords: %s", exc)
        ranked = None
    if ranked is None:
        words = _words(what)
        ranked = [r for r in described if any(w in r.summary.casefold() for w in words)]
    wanted = class_words(what)
    hits = [r for r in undescribed if wanted and set(r.detector_labels) & wanted]
    return list(ranked) + hits


@_safe(list)
def order_for_mode(records: Sequence[AlertRecord], mode: str) -> List[AlertRecord]:
    """Guard: the most serious first, then newest. Assistant: time order."""
    if mode == "guard":
        return sorted(records, key=lambda r: (SEVERITY.get(r.label, 3), -_finite(r.ts)))
    return sorted(records, key=lambda r: _finite(r.ts))


def _detector_phrase(record: AlertRecord) -> str:
    labels = set(record.detector_labels)
    what = "a person" if labels & PERSON_CLASSES else ("a vehicle" if labels & VEHICLE_CLASSES else "movement")
    return f"detector saw {what}, not confirmed"


@_safe(dict)
def event_doc(record: AlertRecord, handle: str) -> Dict[str, Any]:
    return {
        "handle": handle,
        "time": local(record.ts),
        "camera": record.camera,
        "kind": record.kind,
        "label": record.label or None,
        "people": record.people,
        "summary": record.summary if record.described else _detector_phrase(record),
        "described": record.described,
        "has_video": record.clip_path is not None,
        "owner_said": list(record.verdicts),
    }


@_safe(dict)
def coverage(snapshot: Any, all_records: Sequence[AlertRecord], start_ts: float, end_ts: float) -> Dict[str, Any]:
    """What a search could and could not see, so "nothing found" is never read as "nobody came"."""
    quiet = [r.ts for r in all_records if r.kind == "quiet"]
    if not getattr(snapshot, "quiet_log", False):
        since = "off"
    elif getattr(snapshot, "quiet_since", None):
        since = local(snapshot.quiet_since)
    else:
        since = "unknown"
    return {
        "from": local(start_ts),
        "to": local(end_ts),
        "cameras_off": [c.name for c in snapshot.cameras if not c.enabled],
        "cameras_offline": list(snapshot.offline_names),
        "quiet_log_since": since,
        "oldest_quiet_event_kept": local(min(quiet)) if quiet else None,
        "oldest_kept": local(snapshot.now - snapshot.retention_days * 86400),
        "note": ("Only moments when the detector saw a person or a moving vehicle are saved. Outside the alert "
                 "hours nothing is saved while the quiet log is off, nor before quiet_log_since; cameras that "
                 "were off saw nothing. 'unknown' means the start of the quiet log is not known."),
    }
