"""The scene map editor's model: the owner's answers for the numbered regions, the areas drawn by hand and the
boundary lines, and the SceneMap dict the box confirms (``scene_interview confirm --map-b64``). No Qt here.

Choices are the editor's four big buttons: ``mine`` (scene_map ``mine``), ``neighbour`` and ``public``
(``watch_no_alert`` with that owner) and ``hide`` (``black``). ``skip`` / ``unknown`` answers are said out loud but
leave the region out of the map.
"""
import base64
import json
import math
import re
import zlib
from dataclasses import dataclass, field

from .. import scene_map as sm

CHOICES = ("mine", "neighbour", "public", "hide")
SKIP, UNKNOWN = "skip", "unknown"
# A place that IS the boundary between us and them (the railing, the wall): ours, with a line along it whose
# inward side the engine works out from the areas around it, or the owner says (scene_interview.boundary_line).
BOUNDARY = "boundary"
# choice -> (kind, owner) in scene_map's words
KINDS = {"mine": (sm.MINE, ""), "neighbour": (sm.WATCH, sm.NEIGHBOUR), "public": (sm.WATCH, sm.PUBLIC),
         "hide": (sm.BLACK, "")}
# The engine's ownership colours (scene_interview.GROUND_COLOURS, BGR there), as RGB.
COLOURS = {"mine": "#0078ff", "neighbour": "#ff8c00", "public": "#a0a0a0", "hide": "#000000"}
# The engine's number colours (scene_interview._PALETTE, BGR there).
_PALETTE_BGR = ((66, 135, 245), (245, 66, 230), (66, 245, 135), (245, 200, 66), (66, 230, 245), (180, 66, 245),
                (245, 105, 66), (120, 245, 66), (66, 66, 245), (245, 66, 120), (66, 180, 180), (200, 200, 66))
MAX_CORNERS = 32               # data_collection.zones.validate_points: 3 to 32 corners
DECIMALS = 4


def number_colour(number):
    """The region's number colour, as '#rrggbb'."""
    b, g, r = _PALETTE_BGR[(int(number) - 1) % len(_PALETTE_BGR)]
    return f"#{r:02x}{g:02x}{b:02x}"


def choice_of(area):
    """The editor's choice for a stored area dict (``kind`` / ``owner``)."""
    kind = area.get("kind")
    if kind == sm.MINE:
        return "mine"
    if kind == sm.BLACK:
        return "hide"
    if kind == sm.WATCH:
        return "neighbour" if area.get("owner") == sm.NEIGHBOUR else "public"
    return None


def zone_for(name):
    """The taxonomy zone the owner's name suggests (gate, driveway, lawn...), else ``other``."""
    try:
        from ..scene_interview import _zone_of  # noqa: PLC0415 - the interview's own zone words
        return _zone_of(str(name or ""))
    except Exception:  # noqa: BLE001 - a zone is a hint, never worth failing a save over
        return "other"


def clean_points(points):
    return [[round(min(1., max(0., float(x))), DECIMALS), round(min(1., max(0., float(y))), DECIMALS)]
            for x, y in points]


def _distance_to_segment(p, a, b):
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0., min(1., ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _douglas_peucker(points, tolerance):
    if len(points) < 3:
        return list(points)
    far, index = 0., 0
    for i in range(1, len(points) - 1):
        d = _distance_to_segment(points[i], points[0], points[-1])
        if d > far:
            far, index = d, i
    if far <= tolerance:
        return [points[0], points[-1]]
    return _douglas_peucker(points[:index + 1], tolerance)[:-1] + _douglas_peucker(points[index:], tolerance)


def simplify(points, max_points=MAX_CORNERS, tolerance=0.):
    """A closed polygon with corners closer than *tolerance* (picture fractions) to the outline dropped, and at
    most *max_points* corners (the tolerance grows until it fits); never fewer than 3 corners."""
    points = [tuple(p) for p in points]
    if len(points) <= 3 or (tolerance <= 0 and len(points) <= max_points):
        return [list(p) for p in points]
    # Split the ring at its two farthest-apart corners so both halves simplify as open lines.
    start = 0
    far = max(range(len(points)), key=lambda i: math.hypot(points[i][0] - points[0][0], points[i][1] - points[0][1]))
    ring = points[start:] + points[:start]
    step = max(tolerance, .001)
    while True:
        out = _douglas_peucker(ring[:far + 1], step)[:-1] + _douglas_peucker(ring[far:] + [ring[0]], step)[:-1]
        if len(out) <= max_points or step > 1:
            break
        step *= 1.5
    if len(out) < 3:
        out = [ring[0], ring[far // 2] if far > 1 else ring[1], ring[far]]
    return [list(p) for p in out]


def shrink_map(scene, tolerance):
    """*scene* with every area's outline simplified by *tolerance* (to fit a command line)."""
    out = dict(scene)
    out["areas"] = [dict(a, points=clean_points(simplify(a["points"], MAX_CORNERS, tolerance)))
                    for a in scene.get("areas") or ()]
    return out


def polygon_area(points):
    n = len(points)
    if n < 3:
        return 0.
    return abs(sum(points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1]
                   for i in range(n))) / 2.


def inside(point, polygon):
    return len(polygon) >= 3 and sm.inside(tuple(point), [tuple(p) for p in polygon])


def _edge_distance(point, polygon):
    px, py = point
    best = math.inf
    n = len(polygon)
    for i in range(n):
        (x1, y1), (x2, y2) = polygon[i], polygon[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        t = 0. if dx == dy == 0 else max(0., min(1., ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        best = min(best, math.hypot(px - (x1 + t * dx), py - (y1 + t * dy)))
    return best


def label_point(points, covered=(), aspect=16 / 9, steps=24):
    """Where a region's number reads best: the sampled point deepest inside *points* (in picture proportions),
    preferring the part no polygon of *covered* (smaller regions drawn on top) hides."""
    if len(points) < 3:
        return tuple(points[0]) if points else (.5, .5)
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    scaled = [(x * aspect, y) for x, y in points]
    best, best_free = None, None
    for i in range(steps):
        for j in range(steps):
            p = (min(xs) + (max(xs) - min(xs)) * (i + .5) / steps, min(ys) + (max(ys) - min(ys)) * (j + .5) / steps)
            if not inside(p, points):
                continue
            depth = _edge_distance((p[0] * aspect, p[1]), scaled)
            if best is None or depth > best[0]:
                best = (depth, p)
            if not any(inside(p, c) for c in covered) and (best_free is None or depth > best_free[0]):
                best_free = (depth, p)
    chosen = best_free or best
    if chosen is None:
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    return chosen[1]


@dataclass(frozen=True)
class Region:
    number: int
    points: tuple
    area: float = 0.


@dataclass
class HandArea:
    """An area drawn by hand (or one the camera's map already has)."""
    points: list = field(default_factory=list)
    choice: str = None
    name: str = ""
    zone: str = ""             # kept from a stored area; empty: from the name

    @property
    def closed(self):
        return len(self.points) >= 3


@dataclass
class Boundary:
    """A boundary line from a to b; ``inward`` (left | right of travel, as seen) is the owner's side."""
    a: tuple
    b: tuple
    inward: str = "right"
    name: str = ""

    def flipped(self):
        return Boundary(self.a, self.b, "left" if self.inward == "right" else "right", self.name)


def boundary_toward(a, b, inside_point, name=""):
    """The line from a to b whose inward side is where the owner tapped (``scene_map.Line.toward``)."""
    line = sm.Line.toward(name or "line", tuple(a), tuple(b), tuple(inside_point))
    return Boundary(line.a, line.b, line.inward, name)


def boundary_from_areas(a, b, mine_polygons, name=""):
    """The line from a to b, inward towards the owner's own (mine) areas next to it
    (``scene_map.inward_from_areas``); ValueError when no mine area touches its middle."""
    areas = [sm.Area("mine", sm.MINE, "other", tuple(tuple(p) for p in polygon)) for polygon in mine_polygons]
    line = sm.inward_from_areas(name or "line", tuple(a), tuple(b), areas)
    return Boundary(line.a, line.b, line.inward, name)


def inward_vector(boundary):
    """A unit vector (x, y; y down) from the line towards the owner's side, in picture fractions."""
    (ax, ay), (bx, by) = boundary.a, boundary.b
    dx, dy = bx - ax, by - ay
    norm = math.hypot(dx, dy) or 1.
    # scene_map._side: "left" is a negative cross product, i.e. (dy, -dx) on the picture (y grows downwards).
    return (dy / norm, -dx / norm) if boundary.inward == "left" else (-dy / norm, dx / norm)


def _area_dict(choice, name, points, zone=""):
    kind, owner = KINDS["mine" if choice == BOUNDARY else choice]
    zone = zone or ("fence" if choice == BOUNDARY and zone_for(name) == "other" else "")
    out = {"name": sm.plain_name(name) or (zone if zone and zone != "other" else ""), "kind": kind,
           "zone": zone or zone_for(name), "points": clean_points(simplify(points))}
    if owner:
        out["owner"] = owner
    return out


def build_map(camera, regions=(), answers=None, names=None, hand_areas=(), lines=(), rest="", rest_owner=""):
    """The SceneMap dict for ``confirm --map-b64``: an area for each region answered with one of the four choices
    or as a boundary (ours; its line comes in *lines*), skipped, unknown and unanswered regions left out; each
    hand-drawn area with a choice (outlines simplified to MAX_CORNERS), and each line.
    *rest* / *rest_owner* carry the camera's current "rest of the picture" over (confirm would otherwise unmap it)."""
    answers, names = dict(answers or {}), dict(names or {})
    areas = []
    for region in regions:
        choice = answers.get(region.number)
        if choice in CHOICES or choice == BOUNDARY:
            area = _area_dict(choice, names.get(region.number, ""), region.points)
            area["name"] = area["name"] or f"area {region.number}"
            areas.append(area)
    for i, hand in enumerate(hand_areas, 1):
        if hand.choice in CHOICES and hand.closed:
            area = _area_dict(hand.choice, hand.name, hand.points, hand.zone)
            area["name"] = area["name"] or f"drawn area {i}"
            areas.append(area)
    out = {"camera": str(camera), "areas": areas,
           "lines": [{"name": sm.plain_name(ln.name) or "line", "a": clean_points([ln.a])[0],
                      "b": clean_points([ln.b])[0], "inward": ln.inward} for ln in lines]}
    if rest == sm.WATCH:
        out["rest"] = rest
        if rest_owner in sm.OWNERS:
            out["rest_owner"] = rest_owner
    return out


def counts(scene):
    """How many areas of each choice, and lines, a map dict holds: ``{"mine": 3, ..., "lines": 1}``."""
    out = {choice: 0 for choice in CHOICES}
    for area in scene.get("areas") or ():
        choice = choice_of(area)
        if choice:
            out[choice] += 1
    out["lines"] = len(scene.get("lines") or ())
    return out


def encode_map(scene):
    """The map for the command line: base64 of its compact JSON, zlib-compressed (``decode_map_b64`` reads it)."""
    raw = json.dumps(scene, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(zlib.compress(raw, 9)).decode("ascii")


def blacks(scene):
    return sorted(tuple(map(tuple, a.get("points") or ())) for a in scene.get("areas") or () if a.get("kind") == sm.BLACK)


def restart_expected(current, scene):
    """True when saving *scene* changes the frame mask of *current* (the camera's map now): the cameras restart."""
    current = current or {}
    return bool(current.get("watched")) or blacks(current) != blacks(scene)


def rest_after(current, scene):
    """What the rest of the picture will be after saving: ``unmapped`` (watched whole), ``neighbour`` or
    ``public`` (seen, ordinary life there does not alert)."""
    current = current or {}
    if scene.get("rest") == sm.WATCH:
        return scene.get("rest_owner") or sm.PUBLIC
    if current.get("watched"):
        return sm.NEIGHBOUR
    return "unmapped"


MASK = 200                     # cells per side of the picture when two outlines are compared
SAME_PLACE = .85               # outlines overlapping at least this much (IoU) are one place


def area_mask(points, size=MASK):
    """The outline filled on a *size* x *size* grid of the picture: ``{row: bits}``, a cell set when its centre is
    inside (even-odd, as the box draws polygons). Plain Python, so the app needs nothing new."""
    points = [(float(x), float(y)) for x, y in points or ()]
    if len(points) < 3:
        return {}
    rows = {}
    n = len(points)
    ys = [y for _x, y in points]
    for row in range(max(0, int(min(ys) * size) - 1), min(size, int(max(ys) * size) + 2)):
        y = (row + .5) / size
        xs = sorted(x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                    for (x1, y1), (x2, y2) in ((points[i], points[(i + 1) % n]) for i in range(n))
                    if (y1 <= y) != (y2 <= y))
        bits = 0
        for a, b in zip(xs[0::2], xs[1::2]):
            first, last = max(0, math.ceil(a * size - .5)), min(size - 1, math.floor(b * size - .5))
            if last >= first:
                bits |= (1 << (last + 1)) - (1 << first)
        if bits:
            rows[row] = bits
    return rows


def overlap(a, b):
    """How much two filled outlines (``area_mask``) are one place: intersection over union, 0 to 1."""
    inter = sum((bits & b.get(row, 0)).bit_count() for row, bits in a.items())
    union = sum(v.bit_count() for v in a.values()) + sum(v.bit_count() for v in b.values()) - inter
    return inter / union if union else 0.


def unique_areas(areas, same=SAME_PLACE):
    """*areas* with repeats of one place collapsed: outlines overlapping by *same* (IoU) or more are one place, and
    the LAST of them (the latest save) is kept, at its own position, with its kind and name. Saving a reopened map
    used to add the answers on top of the saved areas, from a new picture, so the repeats are near, not exact
    (ch2, 2026-10-10: 17 areas, 9 places). Returns (kept areas, how many were merged)."""
    masks = [area_mask(a.get("points")) for a in areas]
    kept = []
    for i in range(len(areas) - 1, -1, -1):
        if not any(overlap(masks[i], masks[j]) >= same for j in kept):
            kept.append(i)
    kept.reverse()
    return [areas[i] for i in kept], len(areas) - len(kept)


def merged_duplicates(current):
    """How many of the camera's stored areas repeat another one's place (``from_current`` merges them)."""
    return unique_areas([a for a in (current or {}).get("areas") or () if choice_of(a)])[1]


def has_saved_map(current):
    """The camera has a map of its own (stored areas or lines), not only today's drawn zone."""
    current = current or {}
    return bool(current.get("areas") or current.get("lines"))


_STAND_IN = re.compile(r"(drawn )?area \d+")


def owner_name(hand):
    """The name the owner gave the area, or "" for the box's own stand-ins (``area 3``, ``drawn area 1``, the zone
    word of an unnamed one, the watched area): those are not words to show in a name field."""
    name = str(hand.name or "").strip()
    if not name or name == sm.WATCHED_NAME or _STAND_IN.fullmatch(name) or (hand.zone and hand.zone != "other"
                                                                            and name == hand.zone):
        return ""
    return name


def from_current(current):
    """The camera's map now as editable areas and lines: its stored areas (repeats of one place merged, see
    ``unique_areas``), and today's watch zone as an area of ours (``confirm`` keeps it as one)."""
    current = current or {}
    stored, _merged = unique_areas([a for a in current.get("areas") or () if choice_of(a)])
    hands = [HandArea([list(p) for p in a.get("points") or ()], choice_of(a), str(a.get("name") or ""),
                      str(a.get("zone") or "other")) for a in stored]
    watched = current.get("watched")
    if watched and not any(h.points == [list(p) for p in watched] for h in hands):
        hands.append(HandArea([list(p) for p in watched], "mine", sm.WATCHED_NAME, "other"))
    lines = [Boundary(tuple(ln["a"]), tuple(ln["b"]), ln.get("inward", "right"), str(ln.get("name") or ""))
             for ln in current.get("lines") or () if ln.get("a") and ln.get("b")]
    return hands, lines
