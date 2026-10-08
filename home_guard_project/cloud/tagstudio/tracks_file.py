"""The box tracker's own tracks of a clip (``<stem>.tracks.json``), as editable tracks.

The file is written next to an alert clip (or its meta) in the format of ``analysis/replay_tracks.py`` (home_guard,
branch w23-replay): ``{"frames": n, "looks": [{"frame", "ts", ...}], "tracks": [{"id", "kind", "cls" (COCO id),
"first_seen", "last_seen", "hits", "confirmed", "prev_id", "returns", "boxes": [{"frame", "ts", "box": xyxy}]}]}``,
boxes normalised, ``ts`` epoch seconds.

Read here the way the box's entity layer reads it (box/entities.py): a track the tracker linked as a return
(``prev_id``) continues the one it returned from, so one person stays one track ("P1 stays P1"); tracks that never
got confirmed are flicker the entity layer ignores and are left out. Each kept track's boxes become keyframes where
linear interpolation would be off by more than TOLERANCE, and the box is hidden (an ``enabled=False`` keyframe)
from the first look it was missing for more than MAX_GAP looks, and after its last look. Tracks are ``t-1``,
``t-2``, ... in order of first appearance, ``source="yolo"``: machine boxes nobody has checked yet.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from ...fleet_contract import tracks as ft
from ...fleet_contract.classes import COCO_NAMES

SUFFIX = ".tracks.json"
TOLERANCE = 0.02
MAX_GAP = 3


def _number(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _boxes(track: Dict[str, Any]) -> List[tuple]:
    out = []
    for b in track.get("boxes") or []:
        if not isinstance(b, dict) or type(b.get("frame")) is not int or b["frame"] < 0:
            continue
        box = b.get("box")
        if not isinstance(box, list) or len(box) != 4 or any(_number(v) is None for v in box):
            continue
        x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in box)
        if x2 > x1 and y2 > y1:
            out.append((b["frame"], _number(b.get("ts")), [x1, y1, x2, y2]))
    return out


def _chains(tracks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Tracks joined along ``prev_id`` (a return continues the track it returned from), oldest first."""
    by_id = {t["id"]: t for t in tracks}
    root: Dict[Any, Any] = {}

    def find(tid):
        seen = []
        while tid in by_id and by_id[tid].get("prev_id") in by_id and tid not in seen:
            seen.append(tid)
            tid = by_id[tid]["prev_id"]
        return tid

    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for t in tracks:
        root[t["id"]] = find(t["id"])
        groups.setdefault(root[t["id"]], []).append(t)
    return list(groups.values())


def read_tracks(doc: Dict[str, Any], fps: Optional[float] = None, confirmed_only: bool = True) -> List[ft.Track]:
    """The tracker tracks of a tracks.json document as editable tracks (see the module docstring); [] if none.

    A keyframe's time is frame / `fps` (the clip's own frame rate, as for the weak labels), else its ``ts`` from the
    clip's first look."""
    if not isinstance(doc, dict):
        return []
    raw = [t for t in doc.get("tracks") or [] if isinstance(t, dict) and "id" in t
           and COCO_NAMES.get(t.get("cls")) and (t.get("confirmed") or not confirmed_only)]
    looks = sorted({lk["frame"] for lk in doc.get("looks") or [] if isinstance(lk, dict) and type(lk.get("frame")) is int})
    starts = [_number(lk.get("ts")) for lk in doc.get("looks") or [] if isinstance(lk, dict)]
    t0 = min((s for s in starts if s is not None), default=None)
    frames_total = doc.get("frames") if type(doc.get("frames")) is int else None
    rate = fps if fps and fps > 0 else None

    def time_of(frame: int, ts: Optional[float]) -> float:
        if rate:
            return round(frame / rate, 4)
        if ts is not None and t0 is not None:
            return round(ts - t0, 4)
        return round(frame / 7.0, 4)

    built = []
    for chain in _chains(raw):
        label = COCO_NAMES[chain[0]["cls"]]
        points: Dict[int, tuple] = {}
        for t in chain:
            for frame, ts, box in _boxes(t):
                points.setdefault(frame, (frame, time_of(frame, ts), box))
        pts = [points[f] for f in sorted(points)]
        if not pts:
            continue
        built.append((pts[0][1], label, _keyframes(pts, looks, frames_total, time_of)))
    built.sort(key=lambda b: b[0])
    return [ft.Track(track_id=f"t-{n}", label=label, keyframes=kfs, source="yolo")
            for n, (_, label, kfs) in enumerate(built, start=1)]


def _keyframes(pts: List[tuple], looks: List[int], frames_total: Optional[int], time_of) -> List[ft.Keyframe]:
    """Visible runs of `pts` (split where more than MAX_GAP looks went by without a box), each thinned to the
    keyframes interpolation needs, each followed by a hidden keyframe at the first look it was missing."""
    order = {f: i for i, f in enumerate(looks)}
    runs, run = [], [pts[0]]
    for prev, cur in zip(pts, pts[1:]):
        missed = (order[cur[0]] - order[prev[0]] - 1) if prev[0] in order and cur[0] in order else cur[0] - prev[0] - 1
        if missed > MAX_GAP:
            runs.append(run)
            run = []
        run.append(cur)
    runs.append(run)
    out: List[ft.Keyframe] = []
    for run in runs:
        out += [ft.Keyframe(frame=run[i][0], t_sec=run[i][1], xyxy=list(run[i][2]), enabled=True)
                for i in ft._sparse(run, TOLERANCE)]
        last = run[-1][0]
        after = next((f for f in looks if f > last), last + 1)
        if frames_total is None or after < frames_total:
            out.append(ft.Keyframe(frame=after, t_sec=time_of(after, None), xyxy=list(run[-1][2]), enabled=False))
    # a hidden keyframe never lands on the next run's first frame
    seen: Dict[int, ft.Keyframe] = {}
    for k in out:
        if k.frame not in seen or k.enabled:
            seen[k.frame] = k
    return sorted(seen.values(), key=lambda k: k.t_sec)


def load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def candidates(paths: Iterable[str]) -> List[str]:
    """Where a clip's tracks.json may be: next to its meta (``<stem>.meta.json``) or its video (``<stem>.mp4``)."""
    out = []
    for path in paths:
        if not path:
            continue
        base = path[:-len(".meta.json")] if path.endswith(".meta.json") else os.path.splitext(path)[0]
        out.append(base + SUFFIX)
    return out


def local_tracks(paths: Iterable[str], fps: Optional[float]) -> List[ft.Track]:
    """The tracks of the first readable tracks.json next to any of `paths` (meta or video); [] if none."""
    for candidate in candidates(paths):
        if os.path.isfile(candidate):
            doc = load(candidate)
            if doc is not None:
                return read_tracks(doc, fps)
    return []
