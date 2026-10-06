"""The scene map: what each part of a camera's picture is, and whose ground it is.

Plan item "scene map v0" (doc "ראיון ההתקנה ומפת הסצנה"). Code decides WHERE, the model decides WHAT: the box
checks a person's feet against the owner's map (a point in a polygon, a track across a line) and hands the Eye
one short ``ZONE FACTS`` line; the Eye still says what the person does.

Per camera, in ``scene_maps.yaml`` next to ``zones.yaml`` (raw storage in ``data_collection/zones.py``):

- **areas**: a polygon (normalised 0-1, 3-32 corners), the owner's own name for it ("the dog house"), one of
  ``taxonomy.ZONES`` and a kind:
  ``mine`` (watched and alerted on as today), ``watch_no_alert`` (seen and tracked; ordinary life there is
  expected, it alerts only on suspicious or serious behaviour or when someone crosses onto the owner's ground)
  or ``black`` (blacked out at frame read, for privacy, exactly like the outside of today's zone: the AI, the
  recordings, the clips and pictures sent to the owner and the live views never contain it).
  A ``watch_no_alert`` area also says whose ground it is: ``neighbour`` or ``public`` (the default).
- **lines**: a boundary (the railing between us, the gate) from point ``a`` to point ``b``, with the inward side
  (towards the owner's ground) ``left`` or ``right`` of travel from a to b as seen on the picture.

Today's drawn watch zone migrates by itself, without writing anything: its polygon is an implicit ``mine`` area
and everything outside it stays ``black``. A camera without a zone and without a map is ``unmapped`` (watched
whole, as today). A map that only holds the migrated zone is not ``informative``: the Eye's prompt does not change.
Where areas overlap, the innermost (smallest) one wins.

UI contract (the app's map editor, the existing watch-zone dialog grown up; Codex builds it, this module is the
engine): read a camera's map with ``load_scene_map(camera).to_dict()``; edit areas (polygon + name + zone + kind
+ owner) and lines (two points + which side is the owner's: ``Line.toward(name, a, b, inside_point)``, or
``inward_from_areas``); store it with ``save_scene_map(SceneMap.from_dict(...))``, which validates everything
and raises ``ValueError`` with a plain message. A change to black areas needs the running mode restarted
(``find_cameras._restart_running_mode``), like a zone change; the other kinds are read per alert. Colours used
by the interview's confirmation picture: mine blue, neighbour orange, public grey, black black.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import taxonomy as tx

log = logging.getLogger("box.scene_map")

Point = Tuple[float, float]

MINE, WATCH, BLACK = "mine", "watch_no_alert", "black"
KINDS = (MINE, WATCH, BLACK)
NEIGHBOUR, PUBLIC = "neighbour", "public"
OWNERS = (NEIGHBOUR, PUBLIC)               # whose ground a watch_no_alert area is
GROUND_WORDS = {MINE: "mine", NEIGHBOUR: "neighbour's", PUBLIC: "public"}
SIDES = ("left", "right")
IN, OUT = "in", "out"

NAME_LIMIT = 40
WATCHED_NAME = "watched area"              # the implicit mine area that today's drawn zone becomes
MAX_TRACKS_PER_KIND = 3
FACTS_LIMIT = 400
MATCH_DISTANCE = 0.25                      # a foot point this close (picture widths) to a track's last one continues it
PARKED_DISTANCE = 0.03                     # a vehicle that moved less than this is parked, and left out
PERSON_IDS = frozenset({0})
VEHICLE_IDS = frozenset({2, 3, 5, 7})      # COCO car, motorcycle, bus, truck

# A zone word -> the camera role it suggests, for a camera whose role is not set in box.yaml.
_ROLE_OF_ZONE = {"yard": "private", "roof": "private", "window": "private", "fence": "private",
                 "entrance": "entrance", "gate": "entrance", "parking": "parking", "car": "parking",
                 "street": "street"}


def plain_name(text: Any) -> str:
    """Owner words fit for a one-line prompt: no quotes, separators or line breaks (an apostrophe becomes a
    typographic one, so "neighbour's" still reads), at most NAME_LIMIT long."""
    text = str(text or "").replace("'", "’")
    return " ".join(re.sub(r"[\"`;\r\n]", " ", text).split())[:NAME_LIMIT].strip()


def _points(points: Any) -> Tuple[Point, ...]:
    from ..data_collection.zones import validate_points  # noqa: PLC0415

    return tuple(validate_points(points))


def _point(p: Any) -> Point:
    if not isinstance(p, (list, tuple)) or len(p) != 2 or any(isinstance(v, bool) for v in p):
        raise ValueError("a point must be two numbers, x and y")
    x, y = float(p[0]), float(p[1])
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError("a point must lie inside the picture (0 to 1)")
    return (round(x, 4), round(y, 4))


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def inside(p: Point, polygon: Sequence[Point]) -> bool:
    """Ray casting: is *p* inside *polygon*?"""
    x, y = p
    hit = False
    n = len(polygon)
    for i in range(n):
        (x1, y1), (x2, y2) = polygon[i], polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


def polygon_area(polygon: Sequence[Point]) -> float:
    n = len(polygon)
    return abs(sum(polygon[i][0] * polygon[(i + 1) % n][1] - polygon[(i + 1) % n][0] * polygon[i][1]
                   for i in range(n))) / 2.0


def _cross(a: Point, b: Point, p: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _side(a: Point, b: Point, p: Point) -> str:
    """``left`` or ``right`` of travel from a to b as seen on the picture (y grows downwards), "" on the line."""
    c = _cross(a, b, p)
    return "" if c == 0 else ("left" if c < 0 else "right")


def foot_point(box: Sequence[float]) -> Point:
    """Where a detected person stands: the middle of the bottom of the box ``(x1, y1, x2, y2)``."""
    x1, _y1, x2, y2 = (float(v) for v in box[:4])
    return (round((x1 + x2) / 2.0, 4), round(y2, 4))


# ----------------------------------------------------------------------------
# The map
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Area:
    name: str
    kind: str
    zone: str
    points: Tuple[Point, ...]
    owner: str = ""            # watch_no_alert only: neighbour | public ("" reads as public)
    implicit: bool = False     # today's drawn watch zone, never stored in the scene map

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"an area's kind must be one of {', '.join(KINDS)}")
        if self.zone not in tx.ZONES:
            raise ValueError(f"an area's zone must be one of {', '.join(tx.ZONES)}")
        if self.owner and (self.kind != WATCH or self.owner not in OWNERS):
            raise ValueError(f"only a {WATCH} area has an owner, one of {', '.join(OWNERS)}")
        object.__setattr__(self, "points", _points(self.points))
        object.__setattr__(self, "name", plain_name(self.name) or self.zone)

    @property
    def ground(self) -> str:
        """Whose ground: ``mine``, ``neighbour``, ``public``, or "" for a black area."""
        if self.kind == MINE:
            return MINE
        if self.kind == WATCH:
            return self.owner or PUBLIC
        return ""

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "kind": self.kind, "zone": self.zone,
                               "points": [[x, y] for x, y in self.points]}
        if self.owner:
            out["owner"] = self.owner
        return out


@dataclass(frozen=True)
class Line:
    name: str
    a: Point
    b: Point
    inward: str                # the owner's side: left | right of travel from a to b, as seen on the picture

    def __post_init__(self) -> None:
        a, b = _point(self.a), _point(self.b)
        if a == b:
            raise ValueError("a line needs two different points")
        if self.inward not in SIDES:
            raise ValueError("a line's inward side must be left or right")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "name", plain_name(self.name) or "line")

    @classmethod
    def toward(cls, name: str, a: Point, b: Point, inside_point: Point) -> "Line":
        """The line from a to b whose inward side is the side *inside_point* is on (the owner taps their side)."""
        side = _side(_point(a), _point(b), _point(inside_point))
        if not side:
            raise ValueError("the point showing the owner's side lies on the line")
        return cls(name, a, b, side)

    def crossing(self, p: Point, q: Point) -> Optional[str]:
        """``in`` or ``out`` if the step from p to q crosses the line (touching it is not crossing), else None."""
        sp, sq = _side(self.a, self.b, p), _side(self.a, self.b, q)
        if not sp or not sq or sp == sq:
            return None
        # The step must cross the segment itself, not the line through it.
        if (_cross(p, q, self.a) < 0) == (_cross(p, q, self.b) < 0) and _cross(p, q, self.a) != 0:
            return None
        return IN if sq == self.inward else OUT

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "a": list(self.a), "b": list(self.b), "inward": self.inward}


def inward_from_areas(name: str, a: Point, b: Point, areas: Iterable[Area]) -> Line:
    """The line from a to b with its inward side where the owner's own (mine) ground is, next to its middle."""
    a, b = _point(a), _point(b)
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    length = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
    nx, ny = -(b[1] - a[1]) / length * 0.02, (b[0] - a[0]) / length * 0.02
    mine = [ar for ar in areas if ar.kind == MINE]
    for probe in ((mid[0] + nx, mid[1] + ny), (mid[0] - nx, mid[1] - ny)):
        if any(inside(probe, ar.points) for ar in mine):
            return Line(name, a, b, _side(a, b, probe) or "right")
    raise ValueError("no area of the owner's next to the line: say which side is theirs")


@dataclass(frozen=True)
class SceneMap:
    camera: str
    areas: Tuple[Area, ...] = ()
    lines: Tuple[Line, ...] = ()
    watched: Optional[Tuple[Point, ...]] = None     # today's drawn zone (zones.yaml): outside it is black

    @property
    def outside(self) -> str:
        """What the picture outside every area is: ``black`` with a drawn zone, else ``unmapped``."""
        return BLACK if self.watched else "unmapped"

    @property
    def migrated(self) -> bool:
        """Only today's drawn zone: nothing the owner told the interview yet."""
        return bool(self.watched) and not self.areas and not self.lines

    @property
    def informative(self) -> bool:
        """Something the code can tell the Eye beyond today's zone."""
        return bool(self.areas or self.lines)

    def all_areas(self) -> Tuple[Area, ...]:
        """The stored areas, then today's drawn zone as the implicit mine area."""
        if not self.watched:
            return self.areas
        return self.areas + (Area(WATCHED_NAME, MINE, "other", self.watched, implicit=True),)

    def area_at(self, p: Point) -> Optional[Area]:
        """The innermost area under foot point *p*; None outside every area or outside today's zone (black)."""
        if self.watched and not inside(p, self.watched):
            return None
        hits = [a for a in self.all_areas() if inside(p, a.points)]
        return min(hits, key=lambda a: polygon_area(a.points)) if hits else None

    def camera_role(self) -> str:
        """The role the owner's own areas suggest (the largest one), or "" when the map says nothing."""
        mine = [a for a in self.areas if a.kind == MINE and a.zone in _ROLE_OF_ZONE]
        if not mine:
            return ""
        return _ROLE_OF_ZONE[max(mine, key=lambda a: polygon_area(a.points)).zone]

    def zones(self) -> Tuple[str, ...]:
        """The taxonomy zones the camera sees (stored areas that are not black), in map order."""
        out: List[str] = []
        for a in self.areas:
            if a.kind != BLACK and a.zone not in out:
                out.append(a.zone)
        return tuple(out)

    def to_dict(self) -> Dict[str, Any]:
        return {"camera": self.camera, "outside": self.outside,
                "watched": [[x, y] for x, y in self.watched] if self.watched else None,
                "areas": [a.to_dict() for a in self.areas], "lines": [ln.to_dict() for ln in self.lines]}

    @classmethod
    def from_dict(cls, camera: str, data: Mapping[str, Any], watched: Optional[Sequence[Point]] = None) -> "SceneMap":
        """A map from its stored (or the app's) form; ValueError in plain words on anything invalid."""
        areas_raw = data.get("areas") or []
        lines_raw = data.get("lines") or []
        if not isinstance(areas_raw, list) or not isinstance(lines_raw, list):
            raise ValueError("areas and lines must be lists")
        areas = []
        for a in areas_raw:
            if not isinstance(a, Mapping):
                raise ValueError("every area must be a mapping")
            areas.append(Area(a.get("name", ""), str(a.get("kind") or ""), str(a.get("zone") or "other"),
                              a.get("points"), owner=str(a.get("owner") or "")))
        lines = []
        for ln in lines_raw:
            if not isinstance(ln, Mapping):
                raise ValueError("every line must be a mapping")
            lines.append(Line(ln.get("name", ""), ln.get("a"), ln.get("b"), str(ln.get("inward") or "")))
        return cls(str(camera), tuple(areas), tuple(lines), tuple(watched) if watched else None)


# ----------------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------------
def _paths(zones_path: Optional[str]) -> Tuple[str, str]:
    from ..data_collection.zones import ZONES_PATH, scene_maps_path_for  # noqa: PLC0415

    zp = zones_path or ZONES_PATH
    return zp, scene_maps_path_for(zp)


def load_scene_map(camera: str, zones_path: Optional[str] = None) -> SceneMap:
    """The camera's map: its stored areas and lines, with today's drawn zone. Never raises: a damaged entry is
    no stored map (logged), the drawn zone still counts."""
    from ..data_collection.zones import load_zones, read_scene_maps  # noqa: PLC0415

    zp, sp = _paths(zones_path)
    watched = load_zones(zp).get(str(camera))
    entry = read_scene_maps(sp).get(str(camera))
    if isinstance(entry, Mapping):
        try:
            return SceneMap.from_dict(camera, entry, watched)
        except (TypeError, ValueError) as exc:
            log.warning("Scene map of %s ignored: %s", camera, exc)
    elif entry is not None:
        log.warning("Scene map of %s ignored: not a mapping", camera)
    return SceneMap(str(camera), watched=tuple(watched) if watched else None)


def save_scene_map(scene: SceneMap, zones_path: Optional[str] = None) -> SceneMap:
    """Store the camera's areas and lines (today's drawn zone stays in zones.yaml), keeping other cameras'.
    Returns the map as it will load."""
    from ..data_collection.zones import read_scene_maps, write_scene_maps  # noqa: PLC0415

    _zp, sp = _paths(zones_path)
    entries = read_scene_maps(sp)
    entries[scene.camera] = {"areas": [a.to_dict() for a in scene.areas if not a.implicit],
                             "lines": [ln.to_dict() for ln in scene.lines]}
    write_scene_maps(entries, sp)
    log.info("Scene map of %s saved: %d area(s), %d line(s)", scene.camera, len(scene.areas), len(scene.lines))
    return load_scene_map(scene.camera, zones_path)


def clear_scene_map(camera: str, zones_path: Optional[str] = None) -> bool:
    """Forget the camera's map (today's drawn zone stays). True if there was one."""
    from ..data_collection.zones import read_scene_maps, write_scene_maps  # noqa: PLC0415

    _zp, sp = _paths(zones_path)
    entries = read_scene_maps(sp)
    if str(camera) not in entries:
        return False
    del entries[str(camera)]
    write_scene_maps(entries, sp)
    return True


# ----------------------------------------------------------------------------
# Tracks: foot points over the clip
# ----------------------------------------------------------------------------
Detection = Tuple[int, float, float, float, float, float]      # (COCO class, conf, x1, y1, x2, y2), normalised


@dataclass
class Track:
    kind: str                                                   # person | vehicle
    points: List[Tuple[float, float, float]] = field(default_factory=list)   # (ts, x, y) foot points

    @property
    def moved(self) -> float:
        xs = [(x, y) for _, x, y in self.points]
        return max((math.hypot(x - xs[0][0], y - xs[0][1]) for x, y in xs), default=0.0)


def _kind(cls_id: int) -> str:
    return "person" if cls_id in PERSON_IDS else "vehicle" if cls_id in VEHICLE_IDS else ""


def detections_from_result(result: Any, width: int, height: int) -> List[Detection]:
    """A YOLO result's person and vehicle boxes, normalised to the picture (read box by box, the way
    ``vlm_crop.scale_boxes`` reads them). [] for anything unreadable."""
    def number(v: Any) -> float:
        return float(v.item() if hasattr(v, "item") else v)

    try:
        out = []
        for b in (result.boxes if result is not None and result.boxes is not None else []):
            cid = int(number(b.cls))
            if not _kind(cid):
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            conf = number(b.conf) if getattr(b, "conf", None) is not None else 0.0
            out.append((cid, round(conf, 3), round(x1 / width, 4), round(y1 / height, 4),
                        round(x2 / width, 4), round(y2 / height, 4)))
        return out
    except Exception:  # noqa: BLE001 - no detections is the safe answer: no zone facts
        return []


def tracks_from_detections(looks: Iterable[Tuple[float, Sequence[Detection]]]) -> List[Track]:
    """Foot-point tracks from the looks over a clip (``(ts, detections)`` in time order).

    Each look continues the nearest track of the same kind (globally nearest pairs first, within
    MATCH_DISTANCE); the rest start new tracks. Vehicles that did not move (parked) are left out.
    """
    tracks: List[Track] = []
    for ts, dets in looks:
        feet = [(_kind(int(d[0])), foot_point(d[2:6])) for d in dets if _kind(int(d[0]))]
        pairs = sorted((math.hypot(fx - t.points[-1][1], fy - t.points[-1][2]), ti, di)
                       for di, (kind, (fx, fy)) in enumerate(feet)
                       for ti, t in enumerate(tracks) if t.kind == kind and t.points[-1][0] < ts)
        used_t, used_d = set(), set()
        for dist, ti, di in pairs:
            if dist > MATCH_DISTANCE or ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tracks[ti].points.append((float(ts),) + feet[di][1])
        for di, (kind, foot) in enumerate(feet):
            if di not in used_d:
                tracks.append(Track(kind, [(float(ts),) + foot]))
    return [t for t in tracks if t.kind == "person" or t.moved >= PARKED_DISTANCE]


# ----------------------------------------------------------------------------
# Zone facts: the one line the Eye gets, and what the priors table needs
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SceneFacts:
    line: str = ""             # "ZONE FACTS (from code): ..." or "" when the map has nothing to say
    ground: str = ""           # mine | neighbour | public | "" (unknown, as before the map)
    crossed_in: bool = False   # someone crossed a boundary line towards the owner's ground
    zone: str = ""             # the taxonomy zone where it happens, by the map

    def record(self) -> Dict[str, Any]:
        """``scene`` in the training record and .meta.json (the same line goes into training prompts)."""
        return {"zone_facts": self.line, "ground": self.ground, "crossed_in": self.crossed_in, "zone": self.zone}


NO_FACTS = SceneFacts()
_GROUND_ORDER = (MINE, NEIGHBOUR, PUBLIC)


def _track_text(scene: SceneMap, track: Track, label: str) -> Tuple[str, str, Optional[Area], bool]:
    """``(text, ground, last area, crossed in)`` for one track."""
    visits: List[Tuple[Area, float, float]] = []                # (area, first ts, last ts), consecutive runs
    grounds = set()
    for ts, x, y in track.points:
        area = scene.area_at((x, y))
        if area is None:
            continue
        grounds.add(area.ground)
        if visits and visits[-1][0] == area:
            visits[-1] = (area, visits[-1][1], ts)
        else:
            visits.append((area, ts, ts))
    crossings: List[Tuple[str, str]] = []
    for (_, x1, y1), (_, x2, y2) in zip(track.points, track.points[1:]):
        for line in scene.lines:
            way = line.crossing((x1, y1), (x2, y2))
            if way and (not crossings or crossings[-1] != (line.name, way)):
                crossings.append((line.name, way))
    ground = next((g for g in _GROUND_ORDER if g in grounds), "")
    if not visits:
        return f"{label} is outside the mapped areas", ground, None, False

    def where(area: Area) -> str:
        return f"'{area.name}' ({GROUND_WORDS.get(area.ground, area.kind)})"

    last, first_ts, last_ts = visits[-1]
    seconds = int(round(last_ts - first_ts))
    if len(visits) == 1:
        parts = [f"{label} is in {where(last)}" + (f" for {seconds}s" if seconds else "")]
    else:
        parts = [f"{label} entered {where(last)} from {where(visits[-2][0])}"]
    parts += [f"crossed '{name}' {'inward' if way == IN else 'outward'}" for name, way in crossings]
    if not crossings and scene.lines and ground != MINE:
        parts.append(f"did not cross '{scene.lines[0].name}'" if len(scene.lines) == 1 else "crossed no line")
    if len(visits) > 1 and seconds:
        parts.append(f"{seconds}s in '{last.name}'")
    return ", ".join(parts), ground, last, any(way == IN for _, way in crossings)


def scene_facts(scene: SceneMap, tracks: Sequence[Track]) -> SceneFacts:
    """What the map says about these tracks. NO_FACTS when the map is not informative or nobody was tracked.

    The ground is the people's (vehicles only decide it when nobody is seen), cautiously: anyone on the owner's
    ground makes it ``mine``; anyone the map cannot place keeps it unknown (""); only then the neighbour's, else
    public. A passing car on the street never makes a person somewhere else "public"."""
    if not scene.informative or not tracks:
        return NO_FACTS
    texts: List[str] = []
    judged: List[Tuple[str, str, Optional[Area]]] = []          # (kind, ground, last area) for every track
    crossed = False
    for kind in ("person", "vehicle"):
        for i, track in enumerate([t for t in tracks if t.kind == kind], 1):
            text, ground, last, came_in = _track_text(scene, track, f"{kind} {i}")
            if i <= MAX_TRACKS_PER_KIND:
                texts.append(text)
            judged.append((kind, ground, last))
            crossed = crossed or came_in
    deciders = [j for j in judged if j[0] == "person"] or judged
    grounds = {g for _, g, _ in deciders}
    ground = next((g for g in (MINE, "", NEIGHBOUR, PUBLIC) if g in grounds), "")
    zone = next((last.zone for _, g, last in deciders if g == ground and last is not None), "")
    line = "ZONE FACTS (from code): "
    for i, text in enumerate(texts):
        piece = ("; " if i else "") + text
        if len(line) + len(piece) > FACTS_LIMIT:
            break
        line += piece
    return SceneFacts(line=line, ground=ground, crossed_in=crossed, zone=zone)
