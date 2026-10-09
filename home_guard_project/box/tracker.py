"""The person tracker: how long someone has been in view, where they walked, and whether they came back.

Spec ``docs/superpowers/specs/2026-10-06-weeks-2-3-design.md`` section 1. The Eye sees five frames of about ten
seconds; it cannot know that someone has stood at the door for three minutes, came back for the third time, or
walked street > parking > window. Code measures this from every detector look; the model never guesses it.

One ``CameraTracker`` per camera, fed by the detection loop with every look (``update``, about 1-3 a second, once a
second during the cooldown). It is not ``model.track``: one YOLO model serves every camera, so its own tracker state
would mix them up.

- **Matching** is greedy: by box overlap first (IoU >= IOU_MATCH), then by foot-point distance, with a gate that
  grows with the gap between looks (GATE_PER_SEC picture widths a second, at most MAX_GATE, at least MIN_GATE so
  back-to-back looks still match). People and vehicles never share a track. A person box still left over then
  **joins** a person track seen within JOIN_SEC, if their boxes overlap at all (IoU >= JOIN_IOU) or its foot point
  is within JOIN_GATE: the replay of the Oct 7-8 clips (``analysis/replay_tracks.py``) found one walk split in two
  after a gap of 0.16-0.78 s, because a shrinking or growing box moves the foot point more than the tight gate.
- A track is **confirmed** after CONFIRM_HITS looks and **lost** after LOST_SEC unseen; confirmed lost tracks stay
  HISTORY_SEC as history. Memory is capped (MAX_ACTIVE tracks, MAX_HISTORY lost ones, MAX_POINTS foot points each,
  thinned evenly).
- A new person track that starts within RETURN_SEC of a lost one, in the same scene-map area (or within
  RETURN_DISTANCE of its last foot point when the map cannot place them), **came back**: it links to it.
- **Vehicles that never moved** (less than ``scene_map.PARKED_DISTANCE`` from where they were first seen - a
  parked car, a trailer read as a vehicle) are followed inside but are not tracks to anyone outside: no facts, no
  snapshot (so no entity), no vehicle event. The moment one drives off it is handed over whole, from its start.
- **"People" that are a fixture** (2026-10-09 17:44 ch1: the wall lamp was P1, P2, P4) the same way: a person track
  whose foot point never moved STATIC_PERSON_DISTANCE in STATIC_PERSON_MIN_SEC or more (its returns to the same spot
  count as one stay), and that the detector was never sure of (max below STATIC_PERSON_MAX_CONF, the alert's own
  person score, and mean below STATIC_PERSON_MEAN_CONF). A real person standing still for long is seen surely at
  least once, so stays a person.
- Each track keeps the best detector score it had (``max_conf``). The loop feeds people from a lower score than
  the alert needs (box.yaml ``tracker_person_conf``), so a reader that must know a person was seen at the alert's
  own certainty checks ``max_conf`` (``inference.investigate_lingering``).
- **Areas and lines** use the foot point with inertia: a person enters an area (or is across a line) only after
  ENTRY_LOOKS looks in a row there, a vehicle after one; a stay outlives EXIT_GRACE_SEC outside the area. Time
  standing still (the foot point within STATIONARY_DISTANCE of where it stopped) is measured too.
- Dwell, paths and returns are facts, never a verdict: nothing here labels or alerts.

Vehicle tracks that crossed a line, entered another area or ended are handed over with
``drain_vehicle_events()`` (pure data, for the owner-car feature: plates, "car left / came back").

Readers (the alert workers) ask ``facts(t0, t1)`` and ``tracks_between(t0, t1)``; the loop writes. Each track also
keeps its box per look (thinned with the foot points), so an alert clip gets its tracks with boxes
(``tracks_with_boxes``, written as the clip's ``.tracks.json`` by ``clip_tracks.py``). One lock per
camera keeps them apart; every call is pure Python and well under a millisecond for a few tracks.

``TrackerFacts.line()`` is the one line the Eye may get (box.yaml ``eye_tracker_facts: on``). It never holds counts,
classes, confidences or coordinates: the Eye's own count from the frames is what cancels a false YOLO trigger
(``inference.vlm_confirms``), so the box must not hand it one.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from . import scene_map as sm

log = logging.getLogger("box.tracker")

# Bumped whenever the facts line or its rule changes; the prompt version gets "+<this>" when the line is shown.
TRACKER_FACTS_VERSION = "tf1"
# Bumped whenever what ``tracks_with_boxes`` hands out (the clip's ``.tracks.json``) changes meaning.
TRACKS_VERSION = "tb1"
TRACKER_FACTS_RULE = ("Measured by code over the whole visit. Use them for how long and where; count people from "
                      "the frames, not from here.")

IOU_MATCH = 0.3
GATE_PER_SEC = 0.12          # picture widths a foot point may move per second between looks
MAX_GATE = 0.25
MIN_GATE = 0.05              # back-to-back looks (a few hundredths of a second) must still match
JOIN_SEC = 1.0               # a left-over person box joins a person track seen this recently ...
JOIN_IOU = 0.1               # ... whose box it overlaps at all ...
JOIN_GATE = 0.2              # ... or whose last foot point is this close (picture widths)
CONFIRM_HITS = 2
LOST_SEC = {"person": 4.0, "vehicle": 6.0}
HISTORY_SEC = 600.0          # lost tracks kept this long, for returns and for an alert's pre-roll
RETURN_SEC = 600.0
RETURN_DISTANCE = 0.15       # unmapped: this close (picture widths) to where they were last seen is the same place
MAX_POINTS = 300
MAX_ACTIVE = 40
MAX_HISTORY = 200
MAX_LOOKS = 2400             # (ts, track ids) per look, for how many were seen together
CLOCK_BACK_SEC = 1.0         # a look this much older than the last one: the clock jumped, start over
MAP_REFRESH_SEC = 30.0       # the registry re-reads a camera's scene map at most this often
LINE_LIMIT = sm.FACTS_LIMIT
LINE_PEOPLE = sm.MAX_TRACKS_PER_KIND
LONG_VISIT_SEC = 15.0        # longer than the alert clip shows: worth telling the Eye
PATH_SHOWN = 4               # the last areas of a long path
MAX_CROSSINGS_SHOWN = 3
EDGE_MARGIN = 0.08           # a foot point this close to the picture's border entered or left there
# Area inertia (as Frigate's zones): a person is in an area after this many looks in a row there (a vehicle after
# one), and the same holds for being on the far side of a line; a stay survives EXIT_GRACE_SEC outside it, so one
# missed or wobbly look does not split it.
ENTRY_LOOKS = {"person": 3, "vehicle": 1}
EXIT_GRACE_SEC = 2.0
STATIONARY_DISTANCE = 0.02   # a foot point that stays this close to where it stopped is standing, not walking
STATIONARY_MIN_SEC = 2.0     # shorter pauses are part of walking
STATIONARY_SHOWN_SEC = 5.0
# A fixture read as a person (a wall lamp): never moved, never sure, for a long time. The lamp at ch1 (17:44): moved
# ~0.001, max 0.71-0.78 on the box, mean 0.59-0.72 (yolo11s re-run on the clip), in view 611 s with 3 returns.
STATIC_PERSON_DISTANCE = 0.015
STATIC_PERSON_MIN_SEC = 60.0
STATIC_PERSON_MAX_CONF = 0.8     # box.yaml conf_person: a person the alert itself would trust
STATIC_PERSON_MEAN_CONF = 0.75

Box = Tuple[float, float, float, float]
Point3 = Tuple[float, float, float]          # (ts, x, y) foot point


def distance_gate(gap: float) -> float:
    """How far (picture widths) a foot point may move between two looks *gap* seconds apart."""
    return min(MAX_GATE, max(MIN_GATE, GATE_PER_SEC * float(gap)))


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _kind(cls_id: int) -> str:
    return "person" if cls_id in sm.PERSON_IDS else "vehicle" if cls_id in sm.VEHICLE_IDS else ""


def _edge(x: float, y: float) -> str:
    """The picture border a foot point is at (``left``/``right``/``top``/``bottom``), "" in the middle."""
    if x <= EDGE_MARGIN:
        return "left"
    if x >= 1.0 - EDGE_MARGIN:
        return "right"
    if y <= EDGE_MARGIN:
        return "top"
    if y >= 1.0 - EDGE_MARGIN:
        return "bottom"
    return ""


def _usable(scene: Any) -> Any:
    """A map only when it says more than today's drawn zone (inside it everything is one implicit area)."""
    return scene if scene is not None and getattr(scene, "informative", False) else None


@dataclass
class _Track:
    id: int
    kind: str
    cls: int
    box: Box
    first_seen: float
    last_seen: float
    points: List[Point3] = field(default_factory=list)
    hits: int = 1
    prev_id: Optional[int] = None       # the lost track this one is a return of
    returns: int = 0                    # earlier visits in the chain within RETURN_SEC
    followed: bool = False              # a later track already counts as this one coming back
    max_conf: float = 0.0               # the best detector score of any of its looks
    last_conf: float = 0.0              # the detector score of its latest look (reid.py picks its best looks)
    moved: float = 0.0                  # farthest its foot point got from the first one (picture widths)
    boxes: List[Tuple[float, Box]] = field(default_factory=list)   # (ts, box) per look, thinned like the points
    conf_sum: float = 0.0               # the detector scores of all its looks, for mean_conf
    static_since: Optional[float] = None   # a return to the same spot by a fixture: when the stay began

    @property
    def confirmed(self) -> bool:
        return self.hits >= CONFIRM_HITS

    @property
    def mean_conf(self) -> float:
        return self.conf_sum / self.hits if self.hits else 0.0

    @property
    def still(self) -> bool:
        """Never moved and never sure: what a fixture read as a person looks like, however long so far."""
        return (self.kind == "person" and self.moved < STATIC_PERSON_DISTANCE
                and self.max_conf < STATIC_PERSON_MAX_CONF and self.mean_conf < STATIC_PERSON_MEAN_CONF)

    @property
    def fixture(self) -> bool:
        """A "person" that is a fixture (a wall lamp): :attr:`still` for STATIC_PERSON_MIN_SEC or more."""
        start = self.first_seen if self.static_since is None else min(self.static_since, self.first_seen)
        return self.still and self.last_seen - start >= STATIC_PERSON_MIN_SEC

    @property
    def shown(self) -> bool:
        """A person unless a fixture; a vehicle only once it has moved (a parked one is no track to anyone outside)."""
        if self.kind == "vehicle":
            return self.moved >= sm.PARKED_DISTANCE
        return not self.fixture

    @property
    def first_foot(self) -> Tuple[float, float]:
        return (self.points[0][1], self.points[0][2])        # thinning always keeps the first point

    @property
    def last_foot(self) -> Tuple[float, float]:
        return (self.points[-1][1], self.points[-1][2])

    def add(self, ts: float, box: Box, conf: float = 0.0) -> None:
        self.box, self.last_seen, self.hits = box, ts, self.hits + 1
        self.max_conf = max(self.max_conf, float(conf))
        self.last_conf = float(conf)
        self.conf_sum += float(conf)
        foot = sm.foot_point(box)
        self.moved = max(self.moved, math.hypot(foot[0] - self.points[0][1], foot[1] - self.points[0][2]))
        self.points.append((ts,) + foot)
        if len(self.points) > MAX_POINTS:
            # Every other point, keeping the first and the latest: the shape and the times survive.
            self.points = self.points[:-1:2] + [self.points[-1]]
        self.boxes.append((ts, box))
        if len(self.boxes) > MAX_POINTS:
            self.boxes = self.boxes[:-1:2] + [self.boxes[-1]]


# ----------------------------------------------------------------------------
# Facts
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class TrackFacts:
    """What code measured about one track over its whole visit (up to the end of the window)."""
    id: int
    kind: str
    first_seen: float
    last_seen: float
    time_in_view_s: int
    seconds_per_area: Tuple[Tuple[str, int], ...] = ()     # owner's area names, in order of first visit
    path: Tuple[str, ...] = ()                             # owner's area names in order, repeats collapsed
    zone_path: Tuple[str, ...] = ()                        # the same as taxonomy zone words (case memory)
    last_area_s: int = 0                                   # seconds in the area where the path ends
    stationary_s: int = 0                                  # seconds standing still (STATIONARY_DISTANCE)
    crossings: Tuple[Tuple[str, str], ...] = ()            # (line name, in | out)
    returns: int = 0
    prev_id: Optional[int] = None
    entry_edge: str = ""
    exit_edge: str = ""
    max_conf: float = 0.0                                  # the best detector score of any of its looks

    @property
    def useful(self) -> bool:
        """Something the Eye cannot see in its five frames."""
        return bool(self.time_in_view_s >= LONG_VISIT_SEC or len(self.path) > 1 or self.crossings or self.returns)

    def parts(self, label: str) -> List[str]:
        """The pieces of this track's text, most important first."""
        head = f"{label} in view {self.time_in_view_s}s"
        if self.path and self.last_area_s:
            head += f", {self.last_area_s}s in '{self.path[-1]}'"
        parts = [head]
        if self.stationary_s >= STATIONARY_SHOWN_SEC:
            parts.append(f"stationary {self.stationary_s}s")
        if len(self.path) > 1:
            shown = self.path[-PATH_SHOWN:]
            parts.append("path " + ("... > " if len(self.path) > PATH_SHOWN else "")
                         + " > ".join(f"'{name}'" for name in shown))
        parts += [f"crossed '{name}' {'inward' if way == sm.IN else 'outward'}"
                  for name, way in self.crossings[:MAX_CROSSINGS_SHOWN]]
        if self.returns:
            parts.append(f"came back {self.returns} time{'s' if self.returns != 1 else ''} in "
                         f"{int(RETURN_SEC // 60)} min")
        return parts

    def record(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "first_seen": round(self.first_seen, 3),
                "last_seen": round(self.last_seen, 3), "time_in_view_s": self.time_in_view_s,
                "stationary_s": self.stationary_s, "seconds_per_area": dict(self.seconds_per_area), "path": list(self.path),
                "zones": list(self.zone_path), "crossings": [list(c) for c in self.crossings],
                "returns": self.returns, "prev_id": self.prev_id, "entry_edge": self.entry_edge,
                "exit_edge": self.exit_edge, "max_conf": round(self.max_conf, 3)}


@dataclass(frozen=True)
class TrackerFacts:
    """The camera's tracks over an alert's window ``[t0, t1]``: people and moving vehicles, each over its whole
    visit. ``line()`` for the Eye, ``case_memory_dict()`` for the investigator, ``record()`` for meta and training."""
    t0: float = 0.0
    t1: float = 0.0
    people: Tuple[TrackFacts, ...] = ()
    vehicles: Tuple[TrackFacts, ...] = ()
    people_together: int = 0          # the most people seen at one look in the window
    vehicles_together: int = 0        # the most moving vehicles seen at one look in the window
    mapped: bool = False

    @property
    def empty(self) -> bool:
        return not self.people and not self.vehicles

    def line(self) -> str:
        """``TRACKER FACTS (from code): person 1 in view 38s, 22s in 'entrance'; path 'street' > 'entrance'; came
        back 2 times in 10 min.`` or "" when no person has anything the frames cannot show. People only, at most
        LINE_PEOPLE of them, at most LINE_LIMIT characters."""
        told = [p for p in self.people if p.useful][:LINE_PEOPLE]
        if not told:
            return ""
        line = "TRACKER FACTS (from code): "
        first = True
        for i, person in enumerate(told, 1):
            for j, part in enumerate(person.parts(f"person {i}")):
                piece = ("" if first else "; ") + part
                if len(line) + len(piece) + 1 > LINE_LIMIT:
                    if j == 0:
                        return line.rstrip("; ") + "."
                    break
                line += piece
                first = False
        return line + "."

    def _primary(self) -> Optional[TrackFacts]:
        pool = self.people or self.vehicles
        return max(pool, key=lambda t: (t.time_in_view_s, -t.first_seen)) if pool else None

    def case_memory_dict(self) -> Dict[str, Any]:
        """What ``case_memory.signature.build_signature`` reads as *tracker*, from the person who stayed longest
        (a moving vehicle when nobody was seen); {} when the tracker saw nothing in the window."""
        main = self._primary()
        if main is None:
            return {}
        return {"time_in_view_s": float(main.time_in_view_s), "path": list(main.zone_path),
                "entry_edge": main.entry_edge, "exit_edge": main.exit_edge,
                "people": max(self.people_together, 1 if self.people else 0),
                "vehicles": max(self.vehicles_together, 1 if self.vehicles else 0)}

    def record(self) -> Dict[str, Any]:
        """``tracker`` in the clip's .meta.json and the teacher record."""
        return {"version": TRACKER_FACTS_VERSION, "window": [round(self.t0, 3), round(self.t1, 3)],
                "mapped": self.mapped, "people": [p.record() for p in self.people],
                "vehicles": [v.record() for v in self.vehicles], "people_together": self.people_together,
                "vehicles_together": self.vehicles_together, "line": self.line(),
                "case_memory": self.case_memory_dict()}


NO_FACTS = TrackerFacts()


def prompt_block(line: str) -> str:
    """The facts line and its one rule, for a prompt; "" without a line."""
    return f"{line}\n{TRACKER_FACTS_RULE}" if line else ""


def _collapse(words: Iterable[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for word in words:
        if word and (not out or out[-1] != word):
            out.append(word)
    return tuple(out)


def _area_runs(kind: str, points: Sequence[Point3], scene: Any) -> List[List[Any]]:
    """``[area, first ts, last ts]`` stays, in order, with inertia: entering takes ENTRY_LOOKS looks in a row in
    the area (the stay starts at the first of them); a stay ends when another area is entered or after
    EXIT_GRACE_SEC outside it."""
    need = ENTRY_LOOKS.get(kind, 1)
    runs: List[List[Any]] = []
    current = None
    cand, cand_count, cand_first = None, 0, 0.0
    for ts, x, y in points:
        area = scene.area_at((x, y))
        if current is not None and area == current:
            runs[-1][2] = ts
            cand, cand_count = None, 0
            continue
        if current is not None and ts - runs[-1][2] > EXIT_GRACE_SEC:
            current = None
        if area is None:
            cand, cand_count = None, 0
            continue
        if area == cand:
            cand_count += 1
        else:
            cand, cand_count, cand_first = area, 1, ts
        if cand_count >= need:
            runs.append([cand, cand_first, ts])
            current, cand, cand_count = cand, None, 0
    return runs


def _timed_crossings(kind: str, points: Sequence[Point3], scene: Any) -> List[Tuple[str, str, float]]:
    """``(line, in | out, ts)`` in time order, *ts* the first look on the new side. The track's side of each line
    changes only after ENTRY_LOOKS looks in a row on the other side, and it counts as a crossing only when the
    track went across the drawn segment itself (not around its end): a person who wobbles on the gate has not
    crossed it. Looks exactly on the line are skipped."""
    need = ENTRY_LOOKS.get(kind, 1)
    found: List[Tuple[float, str, str]] = []
    for line in scene.lines:
        committed, cand, count, crossed, cand_ts = "", "", 0, False, 0.0
        last: Optional[Tuple[float, float]] = None      # the last look that was on one side of the line
        for ts, x, y in points:
            side = sm._side(line.a, line.b, (x, y))
            if not side:
                continue
            if last is not None and line.crossing(last, (x, y)):
                crossed = True
            last = (x, y)
            if not committed:
                committed = side
                continue
            if side == committed:
                cand, count, crossed = "", 0, False
                continue
            if side != cand:
                cand, count, cand_ts = side, 0, ts
            count += 1
            if count >= need:
                if crossed:
                    found.append((cand_ts, line.name, sm.IN if side == line.inward else sm.OUT))
                committed, cand, count, crossed = side, "", 0, False
    out: List[Tuple[str, str, float]] = []
    for ts, name, way in sorted(found):
        if not out or out[-1][:2] != (name, way):
            out.append((name, way, ts))
    return out


def _crossings(kind: str, points: Sequence[Point3], scene: Any) -> List[Tuple[str, str]]:
    """``(line, in | out)`` in time order (``_timed_crossings`` without the times)."""
    return [(name, way) for name, way, _ in _timed_crossings(kind, points, scene)]


def _stationary(points: Sequence[Point3]) -> float:
    """Seconds standing still: stretches of at least STATIONARY_MIN_SEC in which the foot point stayed within
    STATIONARY_DISTANCE of where the stretch began (a slow walk leaves that circle within a second or two)."""
    total, anchor = 0.0, points[0]
    start = last = points[0][0]
    for p in points[1:]:
        if math.hypot(p[1] - anchor[1], p[2] - anchor[2]) <= STATIONARY_DISTANCE:
            last = p[0]
            continue
        if last - start >= STATIONARY_MIN_SEC:
            total += last - start
        anchor, start, last = p, p[0], p[0]
    if last - start >= STATIONARY_MIN_SEC:
        total += last - start
    return total


def _track_facts(track: _Track, points: Sequence[Point3], scene: Any) -> TrackFacts:
    runs = _area_runs(track.kind, points, scene) if scene is not None else []
    crossings = _crossings(track.kind, points, scene) if scene is not None else []
    per_area: Dict[str, float] = {}
    for area, first, last in runs:
        per_area[area.name] = per_area.get(area.name, 0.0) + (last - first)
    zone_path = _collapse(str(area.zone).strip().lower() for area, _, _ in runs)
    first_pt, last_pt = points[0], points[-1]
    return TrackFacts(
        id=track.id, kind=track.kind, first_seen=first_pt[0], last_seen=last_pt[0],
        time_in_view_s=int(round(last_pt[0] - first_pt[0])),
        seconds_per_area=tuple((name, int(round(s))) for name, s in per_area.items()),
        path=_collapse(area.name for area, _, _ in runs), zone_path=zone_path,
        last_area_s=int(round(runs[-1][2] - runs[-1][1])) if runs else 0,
        stationary_s=int(round(_stationary(points))), crossings=tuple(crossings),
        returns=track.returns, prev_id=track.prev_id,
        entry_edge=zone_path[0] if zone_path else _edge(first_pt[1], first_pt[2]),
        exit_edge=zone_path[-1] if zone_path else _edge(last_pt[1], last_pt[2]), max_conf=track.max_conf)


# ----------------------------------------------------------------------------
# The tracker
# ----------------------------------------------------------------------------
class CameraTracker:
    """One camera's tracks. *scene_map* is an optional getter (no arguments) for the camera's current map, used
    when a call does not pass one; the map may change while the box runs, so it is never kept."""

    def __init__(self, camera: str = "", scene_map: Optional[Callable[[], Any]] = None) -> None:
        self.camera = str(camera)
        self._map_getter = scene_map
        self._lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self._active: List[_Track] = []
        self._history: List[_Track] = []
        self._looks: Deque[Tuple[float, Tuple[int, ...]]] = deque(maxlen=MAX_LOOKS)
        self._ended: Deque[_Track] = deque(maxlen=MAX_HISTORY)   # confirmed vehicles lost since the last drain
        self._reported: Dict[int, Dict[str, Any]] = {}           # per vehicle: what drain already handed over
        self._next_id = 1
        self._last_ts: Optional[float] = None

    def _map(self, scene: Any = None) -> Any:
        if scene is None and self._map_getter is not None:
            try:
                scene = self._map_getter()
            except Exception as exc:  # noqa: BLE001 - no map: returns are judged by distance
                log.debug("[%s] scene map not read for the tracker: %s", self.camera, exc)
                scene = None
        return _usable(scene)

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    # -- writing (the detection loop) ---------------------------------------
    def update(self, ts: float, detections: Sequence[sm.Detection], scene_map: Any = None) -> None:
        """One detector look at *ts*: normalised ``(COCO class, conf, x1, y1, x2, y2)`` boxes (other classes are
        ignored). *scene_map* is only read when a new person is confirmed (to judge a return)."""
        ts = float(ts)
        dets: List[Tuple[str, int, Box]] = []
        confs: List[float] = []
        for d in detections or ():
            kind = _kind(int(d[0]))
            if kind:
                dets.append((kind, int(d[0]), (float(d[2]), float(d[3]), float(d[4]), float(d[5]))))
                confs.append(float(d[1]))
        confirmed_now: List[_Track] = []
        with self._lock:
            if self._last_ts is not None and ts < self._last_ts - CLOCK_BACK_SEC:
                log.info("[%s] the clock went back %.0f s; the tracker starts over", self.camera, self._last_ts - ts)
                self._reset()
            self._last_ts = ts
            self._expire(ts)
            candidates = [t for t in self._active if t.last_seen < ts]
            used_t: set = set()
            used_d: set = set()
            matches: List[Tuple[int, int]] = []
            # Best overlaps first, each track and each box once; then the nearest foot points within the gate.
            overlaps = sorted(((_iou(t.box, box), ti, di) for di, (kind, _, box) in enumerate(dets)
                               for ti, t in enumerate(candidates) if t.kind == kind), reverse=True)
            for overlap, ti, di in overlaps:
                if overlap < IOU_MATCH:
                    break
                if ti not in used_t and di not in used_d:
                    used_t.add(ti)
                    used_d.add(di)
                    matches.append((ti, di))
            feet = [sm.foot_point(box) for _, _, box in dets]
            near = sorted((math.hypot(feet[di][0] - t.points[-1][1], feet[di][1] - t.points[-1][2]), ti, di)
                          for di, (kind, _, _) in enumerate(dets) if di not in used_d
                          for ti, t in enumerate(candidates) if ti not in used_t and t.kind == kind)
            for dist, ti, di in near:
                if ti in used_t or di in used_d or dist > distance_gate(ts - candidates[ti].last_seen):
                    continue
                used_t.add(ti)
                used_d.add(di)
                matches.append((ti, di))
            # A person box still left over joins a person seen within JOIN_SEC whose box it overlaps, or whose foot
            # point is near: the same walk, not a new person (most overlap first, then the nearest).
            joins = sorted((-_iou(t.box, dets[di][2]), math.hypot(feet[di][0] - t.points[-1][1],
                                                                  feet[di][1] - t.points[-1][2]), ti, di)
                           for di, (kind, _, _) in enumerate(dets) if kind == "person" and di not in used_d
                           for ti, t in enumerate(candidates)
                           if ti not in used_t and t.kind == "person" and ts - t.last_seen <= JOIN_SEC)
            for neg_iou, dist, ti, di in joins:
                if ti in used_t or di in used_d or (-neg_iou < JOIN_IOU and dist > JOIN_GATE):
                    continue
                used_t.add(ti)
                used_d.add(di)
                matches.append((ti, di))
            seen: List[int] = []
            for ti, di in matches:
                track = candidates[ti]
                was = track.confirmed
                track.add(ts, dets[di][2], confs[di])
                seen.append(track.id)
                if not was and track.confirmed:
                    confirmed_now.append(track)
            for di, (kind, cls_id, box) in enumerate(dets):
                if di in used_d or len(self._active) >= MAX_ACTIVE:
                    continue
                track = _Track(self._next_id, kind, cls_id, box, ts, ts, [(ts,) + feet[di]], max_conf=confs[di],
                               boxes=[(ts, box)], last_conf=confs[di], conf_sum=confs[di])
                self._next_id += 1
                self._active.append(track)
                seen.append(track.id)
            self._looks.append((ts, tuple(seen)))
            people = [t for t in confirmed_now if t.kind == "person"]
        if people:
            scene = self._map(scene_map)
            with self._lock:
                for track in people:
                    self._link(track, scene)

    def _expire(self, ts: float) -> None:
        still = []
        for t in self._active:
            if ts - t.last_seen > LOST_SEC[t.kind]:
                if t.confirmed:
                    self._history.append(t)
                    if t.kind == "vehicle" and t.shown:
                        self._ended.append(t)
            else:
                still.append(t)
        self._active = still
        self._history = [t for t in self._history if ts - t.last_seen <= HISTORY_SEC][-MAX_HISTORY:]
        while self._looks and ts - self._looks[0][0] > HISTORY_SEC:
            self._looks.popleft()

    def _by_id(self, track_id: Optional[int]) -> Optional[_Track]:
        if track_id is None:
            return None
        return next((t for t in self._history + self._active if t.id == track_id), None)

    @staticmethod
    def _same_place(p: Tuple[float, float], q: Tuple[float, float], scene: Any) -> bool:
        if scene is not None:
            a, b = scene.area_at(p), scene.area_at(q)
            if a is not None and b is not None:
                return a == b
        return math.hypot(p[0] - q[0], p[1] - q[1]) <= RETURN_DISTANCE

    def _link(self, track: _Track, scene: Any) -> None:
        """A newly confirmed person who was here a moment ago in the same place came back: link the visits."""
        start = (track.points[0][1], track.points[0][2])
        for old in sorted(self._history, key=lambda t: t.last_seen, reverse=True):
            gap = track.first_seen - old.last_seen
            if old.kind != "person" or old.followed or gap < 0 or gap > RETURN_SEC:
                continue
            if self._same_place((old.points[-1][1], old.points[-1][2]), start, scene):
                old.followed = True
                track.prev_id = old.id
                # A fixture lost and found again at the very same spot is one long stay.
                if old.still and math.hypot(start[0] - old.points[0][1],
                                            start[1] - old.points[0][2]) < STATIC_PERSON_DISTANCE:
                    track.static_since = old.first_seen if old.static_since is None else old.static_since
                count, cur = 0, old
                while cur is not None and track.first_seen - cur.last_seen <= RETURN_SEC:
                    count += 1
                    cur = self._by_id(cur.prev_id)
                track.returns = count
                return

    # -- vehicle events (the owner-car feature) -------------------------------
    def _vehicle_event(self, t: _Track, scene: Any, state: Dict[str, Any], ended: bool) -> Optional[Dict[str, Any]]:
        timed = _timed_crossings(t.kind, t.points, scene) if scene is not None else []
        path = _collapse(a.name for a, _, _ in _area_runs(t.kind, t.points, scene)) if scene is not None else ()
        new = timed[state["crossings"]:]
        if not ended and not new and path == state["path"]:
            return None
        state["crossings"], state["path"] = len(timed), path
        whole = sm.Track(t.kind, t.points)
        return {"track_id": t.id, "camera": self.camera, "kind": "vehicle", "cls": t.cls,
                "first_ts": t.first_seen, "last_ts": t.last_seen, "first_foot": list(t.first_foot),
                "last_foot": list(t.last_foot), "last_box": list(t.box),
                "crossings": [(name, way, round(ts, 3)) for name, way, ts in new], "area_path": list(path),
                "moved": round(whole.moved, 4), "ended": ended,
                "reason": "ended" if ended else "crossed" if new else "area"}

    def drain_vehicle_events(self, scene_map: Any = None) -> List[Dict[str, Any]]:
        """Confirmed vehicle tracks that crossed a scene-map line, entered another area or ended (were lost) since
        the last call, as small dicts: ``track_id, camera, kind, cls, first_ts, last_ts, first_foot, last_foot,
        last_box`` (normalised x1, y1, x2, y2), ``crossings`` (only the new ``(line, in | out, ts)``),
        ``area_path`` (the owner's area names so far), ``moved`` (picture widths from the first foot point; parked
        is below ``scene_map.PARKED_DISTANCE``), ``ended`` and ``reason`` (crossed | area | ended).

        Pure data, no I/O; cheap enough for every look: a vehicle whose foot point has not moved since it was last
        checked (a parked car) is not looked at again."""
        scene = self._map(scene_map)
        events: List[Dict[str, Any]] = []
        with self._lock:
            for t in self._active:
                if t.kind != "vehicle" or not t.confirmed or not t.shown:
                    continue
                state = self._reported.setdefault(t.id, {"crossings": 0, "path": (), "at": None})
                at = state["at"]
                if at is not None and math.hypot(t.last_foot[0] - at[0], t.last_foot[1] - at[1]) <= STATIONARY_DISTANCE:
                    continue
                state["at"] = t.last_foot
                event = self._vehicle_event(t, scene, state, ended=False)
                if event:
                    events.append(event)
            while self._ended:
                t = self._ended.popleft()
                state = self._reported.pop(t.id, None) or {"crossings": 0, "path": (), "at": None}
                events.append(self._vehicle_event(t, scene, state, ended=True))
            live = {t.id for t in self._active}
            for track_id in [i for i in self._reported if i not in live]:
                del self._reported[track_id]
        return events

    # -- reading (the alert workers) ----------------------------------------
    def _overlapping(self, t0: float, t1: float) -> List[_Track]:
        return sorted((t for t in self._history + self._active
                       if t.confirmed and t.shown and t.first_seen <= t1 and t.last_seen >= t0),
                      key=lambda t: t.first_seen)

    def tracks_between(self, t0: float, t1: float) -> List[sm.Track]:
        """The confirmed tracks seen in ``[t0, t1]``, with their foot points in that window, as
        ``scene_map.Track`` (what ``scene_map.scene_facts`` reads). Parked vehicles are left out."""
        with self._lock:
            out = []
            for t in self._overlapping(t0, t1):
                points = [p for p in t.points if t0 <= p[0] <= t1]
                if not points:
                    continue
                track = sm.Track(t.kind, list(points))
                if t.kind == "person" or track.moved >= sm.PARKED_DISTANCE:
                    out.append(track)
            return out

    def facts(self, t0: float, t1: float, scene_map: Any = None) -> TrackerFacts:
        """Per person (and moving vehicle) seen in ``[t0, t1]``: time in view, seconds per area, the area path,
        line crossings and returns, each over the whole visit up to *t1*. *scene_map* defaults to the getter's."""
        scene = self._map(scene_map)
        with self._lock:
            people: List[TrackFacts] = []
            vehicles: List[TrackFacts] = []
            for t in self._overlapping(t0, t1):
                points = [p for p in t.points if p[0] <= t1]
                if not points:
                    continue
                if t.kind == "vehicle":
                    window = sm.Track(t.kind, [p for p in points if p[0] >= t0])
                    if window.moved < sm.PARKED_DISTANCE:
                        continue
                (people if t.kind == "person" else vehicles).append(_track_facts(t, points, scene))
            ids_p = {p.id for p in people}
            ids_v = {v.id for v in vehicles}
            together_p = together_v = 0
            for ts, ids in self._looks:
                if t0 <= ts <= t1:
                    together_p = max(together_p, sum(1 for i in ids if i in ids_p))
                    together_v = max(together_v, sum(1 for i in ids if i in ids_v))
        return TrackerFacts(t0=float(t0), t1=float(t1), people=tuple(people), vehicles=tuple(vehicles),
                            people_together=together_p, vehicles_together=together_v, mapped=scene is not None)

    def snapshot(self, t0: float, t1: float, scene_map: Any = None) -> List[Dict[str, Any]]:
        """The confirmed tracks seen in ``[t0, t1]`` as small dicts with their ids, for the event's entities
        (entities.py): ``id, kind, first_seen, last_seen, prev_id, first_foot, last_foot, moved`` (picture widths
        from the first foot point over the whole visit; a vehicle below ``scene_map.PARKED_DISTANCE`` is parked),
        ``active`` (still in view, not lost), ``path`` (the owner's area names in order, () without a map),
        ``entry_edge`` and ``exit_edge``. Pure data, oldest first."""
        scene = self._map(scene_map)
        with self._lock:
            live = {t.id for t in self._active}
            out: List[Dict[str, Any]] = []
            for t in self._overlapping(t0, t1):
                points = [p for p in t.points if p[0] <= t1]
                if not points:
                    continue
                runs = _area_runs(t.kind, points, scene) if scene is not None else []
                first, last = points[0], points[-1]
                out.append({"id": t.id, "kind": t.kind, "first_seen": first[0], "last_seen": last[0],
                            "prev_id": t.prev_id, "first_foot": (first[1], first[2]), "last_foot": (last[1], last[2]),
                            "moved": sm.Track(t.kind, list(points)).moved, "active": t.id in live,
                            "path": list(_collapse(area.name for area, _, _ in runs)),
                            "entry_edge": _edge(first[1], first[2]), "exit_edge": _edge(last[1], last[2]),
                            "max_conf": round(t.max_conf, 3), "fixture": t.fixture})
            return out

    def tracks_with_boxes(self, t0: float, t1: float) -> List[Dict[str, Any]]:
        """Every confirmed track seen in ``[t0, t1]`` with its box at each of its looks in that window, for the clip's
        ``.tracks.json`` (labeling starts from these boxes and ids): ``id, kind, cls, first_seen, last_seen`` (over
        the whole visit), ``hits, confirmed, shown`` (False: a vehicle that never moved, which nobody outside is
        told about), ``max_conf, prev_id, returns`` and ``boxes`` [{``ts``, ``box``: normalised x1, y1, x2, y2}],
        oldest first. A long visit's boxes are thinned as its foot points are (MAX_POINTS). Pure data."""
        with self._lock:
            out: List[Dict[str, Any]] = []
            for t in sorted((t for t in self._history + self._active
                             if t.confirmed and t.first_seen <= t1 and t.last_seen >= t0), key=lambda t: t.first_seen):
                boxes = [{"ts": ts, "box": list(box)} for ts, box in t.boxes if t0 <= ts <= t1]
                if not boxes:
                    continue
                out.append({"id": t.id, "kind": t.kind, "cls": t.cls, "first_seen": t.first_seen,
                            "last_seen": t.last_seen, "hits": t.hits, "confirmed": True, "shown": t.shown,
                            "max_conf": round(t.max_conf, 3), "prev_id": t.prev_id, "returns": t.returns,
                            "boxes": boxes})
            return out

    def looks_between(self, t0: float, t1: float) -> List[Tuple[float, Tuple[int, ...]]]:
        """``(ts, track ids seen)`` of every look in ``[t0, t1]``, oldest first (the ids include tracks not yet
        confirmed). Kept HISTORY_SEC, at most MAX_LOOKS."""
        with self._lock:
            return [(ts, ids) for ts, ids in self._looks if t0 <= ts <= t1]

    def person_looks(self, ts: float) -> List[Dict[str, Any]]:
        """The people boxed in the look at *ts* (the last ``update``), for the appearance memory (reid.py): ``id,
        first_seen, box`` (normalised x1, y1, x2, y2), ``conf`` (this look's detector score) and ``confirmed``."""
        with self._lock:
            return [{"id": t.id, "first_seen": t.first_seen, "box": t.box, "conf": t.last_conf,
                     "confirmed": t.confirmed}
                    for t in self._active if t.kind == "person" and t.last_seen == float(ts)]


class TrackerRegistry:
    """One tracker per camera, and each camera's scene map re-read at most every *refresh_sec* (the owner may
    redraw it while the box runs). *scene_map_loader(camera)* defaults to ``scene_map.load_scene_map``."""

    def __init__(self, scene_map_loader: Optional[Callable[[str], Any]] = None,
                 refresh_sec: float = MAP_REFRESH_SEC, clock: Callable[[], float] = time.monotonic) -> None:
        self._loader = scene_map_loader or sm.load_scene_map
        self._refresh = float(refresh_sec)
        self._clock = clock
        self._lock = threading.Lock()
        self._trackers: Dict[str, CameraTracker] = {}
        self._maps: Dict[str, Tuple[float, Any]] = {}

    def get(self, camera: str) -> CameraTracker:
        with self._lock:
            tracker = self._trackers.get(camera)
            if tracker is None:
                tracker = self._trackers[camera] = CameraTracker(camera, lambda: self.scene_map(camera))
            return tracker

    def scene_map(self, camera: str) -> Any:
        """The camera's informative map, or None (no map, only today's drawn zone, or it could not be read)."""
        now = self._clock()
        with self._lock:
            cached = self._maps.get(camera)
        if cached is not None and now - cached[0] < self._refresh:
            return cached[1]
        try:
            scene = _usable(self._loader(camera))
        except Exception as exc:  # noqa: BLE001 - without a map the tracker still measures time and returns
            log.debug("[%s] scene map not read for the tracker: %s", camera, exc)
            scene = None
        with self._lock:
            self._maps[camera] = (now, scene)
        return scene

    def update(self, camera: str, ts: float, detections: Sequence[sm.Detection]) -> None:
        self.get(camera).update(ts, detections, scene_map=self.scene_map(camera))

    def facts(self, camera: str, t0: float, t1: float) -> TrackerFacts:
        return self.get(camera).facts(t0, t1, self.scene_map(camera))

    def tracks_between(self, camera: str, t0: float, t1: float) -> List[sm.Track]:
        return self.get(camera).tracks_between(t0, t1)

    def snapshot(self, camera: str, t0: float, t1: float) -> List[Dict[str, Any]]:
        """``CameraTracker.snapshot`` for *camera*, with its cached scene map."""
        return self.get(camera).snapshot(t0, t1, self.scene_map(camera))

    def drain_vehicle_events(self, camera: str) -> List[Dict[str, Any]]:
        """``CameraTracker.drain_vehicle_events`` for *camera*, with its cached scene map."""
        return self.get(camera).drain_vehicle_events(self.scene_map(camera))

    def tracks_with_boxes(self, camera: str, t0: float, t1: float) -> List[Dict[str, Any]]:
        return self.get(camera).tracks_with_boxes(t0, t1)

    def looks_between(self, camera: str, t0: float, t1: float) -> List[Tuple[float, Tuple[int, ...]]]:
        return self.get(camera).looks_between(t0, t1)

    def person_looks(self, camera: str, ts: float) -> List[Dict[str, Any]]:
        return self.get(camera).person_looks(ts)
