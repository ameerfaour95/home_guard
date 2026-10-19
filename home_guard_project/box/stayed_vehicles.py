""""It stayed": a car the plate reader confirmed keeps its name on its parking spot, day and night (owner, 2026-10-19).

The owner asked how the box would know "my Tesla" when the plate cannot be read: at night, far away, at an angle, from
behind. Once a plate read (car_plates.py) matches a car he named (known_vehicles.py) on a camera, that identity stays
with the vehicle's spot:

    {"id", "camera", "box" (normalised x1, y1, x2, y2: the same view in the sub and the main stream), "vehicle_id",
     "label" (his words), "plate" (digits, kept on the box only), "confirmed_at", "last_seen", "ended_at", "end"}

- **Kept alive** by every detector look at that camera (:meth:`StayedVehicles.look`, from the guard loop, IR looks
  included): a vehicle box overlapping the spot by ``KEEP_IOU`` or more is the car still there.
- **Named**: a vehicle the tracker followed in a later alert, at that camera, whose box sits on the spot (median box,
  IoU >= ``STAY_IOU``) is called by his words in the alert text and on the event ("since HH:MM" in the meta). Naming
  only: the alert's level never changes.
- **Ended** when the spot stays empty (no overlapping vehicle for ``empty_sec`` AND at least ``EMPTY_LOOKS`` looks:
  the camera was looking), when a vehicle that started on the spot left it (``LEFT_IOU``; that departing car is still
  named once), when a plate read on the spot is another car's, or when the box was off too long to know
  (``RESTART_GAP_SEC`` since it was last seen).
- One JSON file beside the plate reads (``<events>/plates/stayed.json``), atomic replace + fsync, written on a change
  and at most every ``SAVE_EVERY_SEC`` for ``last_seen``. The full plate is in it, so it never leaves the box; what
  the meta and the owner see is his label and the time.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import statistics
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import plates

log = logging.getLogger("box.stayed_vehicles")

FILE_NAME = os.path.join("plates", "stayed.json")
STAY_IOU = 0.6              # a vehicle on the spot (the name is given)
KEEP_IOU = 0.4              # the car still there in a detector look (boxes jitter; a person may cover part of it)
LEFT_IOU = 0.2              # a vehicle that started on the spot and ends this far off it has left
EMPTY_SEC = 15 * 60.0       # box.yaml stayed_empty_min: the spot empty this long ...
EMPTY_LOOKS = 30            # ... over at least this many looks (a camera that was not looking says nothing)
RESTART_GAP_SEC = 30 * 60.0  # at start: an identity not seen for this long is not trusted any more
SAVE_EVERY_SEC = 120.0
ENDED_KEPT = 40
VEHICLE_CLASSES = (2, 3, 5, 7)   # COCO car, motorcycle, bus, truck (scene_map.Detection class ids)

Box = Tuple[float, float, float, float]


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def median_box(boxes: Sequence[Sequence[float]]) -> Optional[Box]:
    boxes = [b for b in boxes if b is not None and len(b) == 4]
    if not boxes:
        return None
    return tuple(float(statistics.median(float(b[i]) for b in boxes)) for i in range(4))  # type: ignore[return-value]


def track_boxes(track: Mapping[str, Any]) -> List[Box]:
    """A tracker track's normalised boxes in time order (tracker.tracks_with_boxes)."""
    out = []
    for b in sorted(track.get("boxes") or (), key=lambda x: float(x.get("ts") or 0)):
        box = b.get("box")
        if box is not None and len(box) == 4:
            out.append(tuple(float(v) for v in box))
    return out


def clock(ts: float) -> str:
    return dt.datetime.fromtimestamp(float(ts)).strftime("%H:%M")


class StayedVehicles:
    def __init__(self, path: str, empty_sec: float = EMPTY_SEC, now: Callable[[], float] = time.time) -> None:
        self.path = path
        self.empty_sec = float(empty_sec)
        self._now = now
        self._lock = threading.Lock()
        self._rows: List[Dict[str, Any]] = []
        self._saved = 0.0
        self._load()

    # ---------- files ----------
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                rows = json.load(f)
            self._rows = [r for r in rows if isinstance(r, dict) and r.get("camera") and len(r.get("box") or ()) == 4] \
                if isinstance(rows, list) else []
        except FileNotFoundError:
            self._rows = []
        except (OSError, ValueError) as exc:
            log.warning("stayed vehicles unreadable (%s); starting empty", exc)
            self._rows = []
        now = self._now()
        stale = [r for r in self._active() if now - float(r.get("last_seen") or 0) > RESTART_GAP_SEC]
        for r in stale:
            self._end(r, "not seen since the box was off", now)
        if stale:
            self._save()

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            ended = [r for r in self._rows if r.get("ended_at")][-ENDED_KEPT:]
            rows = [r for r in self._rows if not r.get("ended_at")] + ended
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=1)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            self._rows = rows
            self._saved = self._now()
        except OSError as exc:
            log.warning("stayed vehicles not saved: %s", exc)

    # ---------- reading ----------
    def _active(self, camera: str = "") -> List[Dict[str, Any]]:
        return [r for r in self._rows if not r.get("ended_at") and (not camera or r.get("camera") == camera)]

    def active(self, camera: str = "") -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._active(camera)]

    def at(self, camera: str, box: Sequence[float], min_iou: float = STAY_IOU) -> Optional[Dict[str, Any]]:
        """The identity whose spot *box* sits on, or None."""
        with self._lock:
            best = max(((iou(r["box"], box), r) for r in self._active(camera)), key=lambda x: x[0], default=None)
            return dict(best[1]) if best and best[0] >= min_iou else None

    # ---------- changes ----------
    def _end(self, r: Dict[str, Any], why: str, ts: float) -> None:
        r["ended_at"], r["end"] = float(ts), why
        log.info("stayed: %s at %s ended (%s; since %s)", r.get("label"), r.get("camera"), why,
                 clock(r.get("confirmed_at") or ts))

    def plate_read(self, camera: str, box: Sequence[float], plate: str, vehicle: Optional[Mapping[str, Any]],
                   ts: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """The plate reader read *plate* on *box* at *camera*: a car he named (*vehicle*, a known_vehicles record)
        starts or refreshes its identity there; any other plate on an identity's spot ends it. Returns the identity
        the box now holds for that spot, or None."""
        ts = self._now() if ts is None else float(ts)
        digits = plates.digits_only(plate)
        if not plates.valid_plate(digits) or box is None or len(box) != 4:
            return None
        box = tuple(float(v) for v in box)
        changed = False
        with self._lock:
            mine: Optional[Dict[str, Any]] = None
            for r in self._active(camera):
                if iou(r["box"], box) < STAY_IOU:
                    continue
                if mine is None and plates.levenshtein(str(r.get("plate") or ""), digits) <= 1:
                    mine = r                  # the same car (a read one digit off is still it)
                else:
                    self._end(r, "another plate on the spot", ts)
                    changed = True
            if mine is not None and vehicle is not None and mine.get("vehicle_id") != vehicle.get("id"):
                self._end(mine, "another plate on the spot", ts)
                mine, changed = None, True
            if mine is not None:
                mine.update(last_seen=ts, empty_since=0.0, empty_looks=0, box=list(box))
                if vehicle is not None:
                    mine["label"] = str(vehicle.get("label") or mine.get("label") or "")
                changed = True
            elif vehicle is not None:
                mine = {"id": "S" + uuid.uuid4().hex[:8], "camera": camera, "box": list(box),
                        "vehicle_id": str(vehicle.get("id") or ""), "label": str(vehicle.get("label") or ""),
                        "plate": digits, "confirmed_at": ts, "last_seen": ts, "empty_since": 0.0,
                        "empty_looks": 0, "ended_at": 0.0, "end": ""}
                self._rows.append(mine)
                log.info("stayed: %s confirmed at %s by its plate %s", mine["label"], camera, plates.mask(digits))
                changed = True
            if changed:
                self._save()
            return dict(mine) if mine else None

    def look(self, camera: str, ts: float, detections: Sequence[Sequence[float]]) -> None:
        """One detector look at *camera* (scene_map.Detection: class, conf, x1, y1, x2, y2 normalised). Cheap when
        the camera holds no identity."""
        with self._lock:
            rows = self._active(camera)
            if not rows:
                return
            boxes = [tuple(float(v) for v in d[2:6]) for d in detections or ()
                     if len(d) >= 6 and int(d[0]) in VEHICLE_CLASSES]
            changed = False
            for r in rows:
                if any(iou(r["box"], b) >= KEEP_IOU for b in boxes):
                    r.update(last_seen=float(ts), empty_since=0.0, empty_looks=0)
                    continue
                if not r.get("empty_since"):
                    r["empty_since"] = float(ts)
                r["empty_looks"] = int(r.get("empty_looks") or 0) + 1
                if ts - float(r["empty_since"]) >= self.empty_sec and r["empty_looks"] >= EMPTY_LOOKS:
                    self._end(r, "the spot is empty", ts)
                    changed = True
            if changed or ts - self._saved >= SAVE_EVERY_SEC:
                self._save()

    def names_for(self, camera: str, tracks: Sequence[Mapping[str, Any]], ts: Optional[float] = None
                  ) -> Dict[str, Dict[str, Any]]:
        """Per tracker key, the vehicles of an alert that sit on (or just left) a confirmed spot at *camera*:
        ``{"label", "vehicle_id", "since" (HH:MM), "left"}``. A vehicle that started on the spot and left it is
        named once and ends the identity."""
        from .entities import track_key  # noqa: PLC0415

        ts = self._now() if ts is None else float(ts)
        out: Dict[str, Dict[str, Any]] = {}
        changed = False
        with self._lock:
            rows = self._active(camera)
            if not rows:
                return out
            for t in tracks or ():
                if t.get("kind") != "vehicle":
                    continue
                boxes = track_boxes(t)
                mid = median_box(boxes)
                if mid is None:
                    continue
                scored = max(((iou(r["box"], mid), r) for r in rows), key=lambda x: x[0])
                left = None
                if scored[0] < STAY_IOU:
                    starts = max(((iou(r["box"], boxes[0]), r) for r in rows), key=lambda x: x[0])
                    if starts[0] >= STAY_IOU and iou(starts[1]["box"], boxes[-1]) < LEFT_IOU:
                        left = starts[1]
                    else:
                        continue
                r = left or scored[1]
                out[track_key(dict(t))] = {"label": str(r.get("label") or ""), "vehicle_id": str(r.get("vehicle_id")
                                                                                                  or ""),
                                           "since": clock(r.get("confirmed_at") or ts), "left": left is not None}
                if left is not None and not r.get("ended_at"):
                    self._end(r, "it left", ts)
                    changed = True
            if changed:
                self._save()
        return out


def store_in(events_dir: str, empty_sec: float = EMPTY_SEC) -> StayedVehicles:
    return StayedVehicles(os.path.join(events_dir, FILE_NAME), empty_sec=empty_sec)


def empty_sec_of(box_settings: Optional[Mapping[str, Any]]) -> float:
    try:
        minutes = float((box_settings or {}).get("stayed_empty_min") or EMPTY_SEC / 60.0)
    except (TypeError, ValueError):
        minutes = EMPTY_SEC / 60.0
    return max(1.0, minutes) * 60.0
