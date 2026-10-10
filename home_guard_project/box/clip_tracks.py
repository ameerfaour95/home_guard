"""The live tracker's tracks of one alert clip, saved next to it as ``responses/<camera>/<day>/<stem>.tracks.json``.

Why: labeling starts from the box's own YOLO boxes with stable ids (P1, P2, CAR1) instead of drawing every box by
hand, and the Admin Center can show what the tracker followed in the clip. The file sits in ``responses/``, which the
outbox moves and the uploader sends with the clip (``outbox._CLIP_DIRS``).

The format is the one of the offline replay (``analysis/replay_tracks.py``), so the same readers (split statistics,
renders) take both; ``source`` says which made it::

    {"clip": "clips/<camera>/<day>/<stem>.mp4", "camera": ..., "start_local": "HH:MM", "source": "live",
     "params": {"person_conf": 0.5, "alert_person_conf": 0.8, "tracker": "tb1"},
     "clip_start_ts": ..., "clip_end_ts": ..., "frames": <frames written>,
     "looks": [{"frame", "ts", "detections", "tracks": [ids]}],
     "tracks": [{"id", "kind", "cls", "first_seen", "last_seen", "hits", "confirmed", "shown", "max_conf",
                 "prev_id", "returns", "entity": "P1" (when the event book mapped it), "boxes": [{"frame", "ts",
                 "box": [x1, y1, x2, y2] normalised}]}],
     "stats": {...}}

``frame`` is the clip frame nearest the look's time: the clip's frames are spread evenly from ``clip_start_ts`` to
``clip_end_ts`` (``inference._prepare_alert``). The live tracker is not fed every frame (about 1-3 looks a second),
so a track has a box only on the frames nearest its looks; ``looks.detections`` counts the tracks the look saw
(person and vehicle boxes, before the tracker's MAX_ACTIVE cap).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

log = logging.getLogger("box.clip_tracks")

SUFFIX = ".tracks.json"
SOURCE = "live"


def frame_index(ts: float, start: float, end: float, frames: int) -> int:
    """The clip frame nearest *ts*, the frames spread evenly over ``[start, end]``; 0 for a one-frame clip."""
    if frames <= 1 or end <= start:
        return 0
    i = int(math.floor((float(ts) - start) / (end - start) * (frames - 1) + 0.5))   # halves go up, not to even
    return min(frames - 1, max(0, i))


def entity_ids(session: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """Tracker key (``entities.track_key``: ``"<id>@<first_seen>"``) -> entity id (P1, CAR1) of an event session."""
    out: Dict[str, str] = {}
    for e in (session or {}).get("entities") or ():
        if isinstance(e, Mapping) and e.get("id"):
            for key in e.get("tracks") or ():
                out[str(key)] = str(e["id"])
    return out


def _stats(tracks: Sequence[Dict[str, Any]], looks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    people = [t for t in tracks if t["kind"] == "person"]
    shown = {t["id"] for t in tracks if t["kind"] != "person" and t.get("shown", True)}
    person_ids = {t["id"] for t in people}
    return {"person_tracks": len(people),
            "vehicle_tracks": len(shown),
            "parked_vehicles": sum(1 for t in tracks if t["kind"] == "vehicle" and not t.get("shown", True)),
            "most_people_at_once": max((sum(1 for i in lk["tracks"] if i in person_ids) for lk in looks), default=0),
            "returns_linked": sum(1 for t in people if t.get("prev_id") is not None),
            "entities": sorted({t["entity"] for t in tracks if t.get("entity")})}


def build(meta: Mapping[str, Any], tracks: Iterable[Mapping[str, Any]],
          looks: Iterable[Tuple[float, Sequence[int]]] = (), params: Optional[Mapping[str, Any]] = None,
          entities: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The ``.tracks.json`` document for the clip *meta* describes, from ``tracker.tracks_with_boxes`` (*tracks*),
    ``tracker.looks_between`` (*looks*) and the event's tracker-key -> entity map (*entities*)."""
    from .entities import track_key  # noqa: PLC0415

    start = float(meta["clip_start_ts"])
    end = float(meta.get("clip_end_ts") or start)
    n = int(meta.get("frames_written") or 0)
    entities = entities or {}
    out_tracks: List[Dict[str, Any]] = []
    for t in tracks:
        boxes = [{"frame": frame_index(b["ts"], start, end, n), "ts": round(float(b["ts"]), 3),
                  "box": [round(float(v), 4) for v in b["box"]]}
                 for b in t.get("boxes") or () if start <= float(b["ts"]) <= end]
        if not boxes:
            continue
        row = {"id": int(t["id"]), "kind": t["kind"], "cls": int(t["cls"]),
               "first_seen": round(float(t["first_seen"]), 3), "last_seen": round(float(t["last_seen"]), 3),
               "hits": int(t.get("hits") or len(boxes)), "confirmed": bool(t.get("confirmed", True)),
               "shown": bool(t.get("shown", True)), "max_conf": t.get("max_conf"), "prev_id": t.get("prev_id"),
               "returns": int(t.get("returns") or 0)}
        entity = entities.get(track_key(t))
        if entity:
            row["entity"] = entity
        row["boxes"] = boxes
        out_tracks.append(row)
    kept = {t["id"] for t in out_tracks}
    out_looks = [{"frame": frame_index(ts, start, end, n), "ts": round(float(ts), 3), "detections": len(ids),
                  "tracks": [int(i) for i in ids if int(i) in kept]}
                 for ts, ids in looks if start <= float(ts) <= end]
    clip = str(meta.get("clip_path") or "").replace("\\", "/")
    return {"clip": clip, "camera": meta.get("camera_name", ""),
            "start_local": dt.datetime.fromtimestamp(start).strftime("%H:%M"), "source": SOURCE,
            "params": dict(params or {}), "clip_start_ts": start, "clip_end_ts": end, "frames": n,
            "looks": out_looks, "tracks": out_tracks, "stats": _stats(out_tracks, out_looks)}


def path_for(root_dir: str, meta_path: str) -> str:
    """``<root>/responses/<camera>/<day>/<stem>.tracks.json`` for the meta ``<root>/meta/<camera>/<day>/<stem>.meta.json``."""
    from .outbox import META_SUFFIX  # noqa: PLC0415

    camera_day = os.path.relpath(os.path.dirname(meta_path), os.path.join(root_dir, "meta"))
    stem = os.path.basename(meta_path)[: -len(META_SUFFIX)]
    return os.path.join(root_dir, "responses", camera_day, stem + SUFFIX)


def write(root_dir: str, meta_path: str, tracks: Iterable[Mapping[str, Any]],
          looks: Iterable[Tuple[float, Sequence[int]]] = (), params: Optional[Mapping[str, Any]] = None,
          entities: Optional[Mapping[str, str]] = None) -> str:
    """Write the clip's ``.tracks.json`` (through a temp file) and return its path. Raises on a bad meta or disk."""
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    doc = build(meta, tracks, looks, params, entities)
    path = path_for(root_dir, meta_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path
