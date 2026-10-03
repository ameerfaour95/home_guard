"""Watch zones: the part of each camera's picture the box looks at.

One optional polygon per camera, in normalised 0-1 coordinates, kept in
``zones.yaml`` next to ``cameras.yaml``. :class:`ZoneMask` blacks out every
pixel outside the polygon; it is applied once, where a frame enters the
system, so the detector, the VLM, the clips, the snapshots and the previews
all see the same masked picture. A camera without a zone is watched whole.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

log = logging.getLogger(__name__)

Point = Tuple[float, float]

_DIR = os.path.dirname(os.path.abspath(__file__))
ZONES_PATH = os.path.join(_DIR, "zones.yaml")

MIN_POINTS = 3
MAX_POINTS = 32
DECIMALS = 4

_HEADER = (
    "# ──────────────────────────────────────────────────────────────────────────────\n"
    "#  Watch zones — normalised polygon corners per camera; everything outside is\n"
    "#  blacked out. Cameras without an entry are watched whole.\n"
    "#  DO NOT COMMIT (deployment-specific)\n"
    "# ──────────────────────────────────────────────────────────────────────────────\n\n"
)
_PAIR_SPLIT = re.compile(r"[;\s]+")


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def validate_points(points: Any) -> List[Point]:
    """The polygon as (x, y) tuples rounded to 4 decimals, or ValueError in plain words."""
    if not isinstance(points, (list, tuple)):
        raise ValueError("the zone must be a list of corners")
    if not MIN_POINTS <= len(points) <= MAX_POINTS:
        raise ValueError(f"a zone needs {MIN_POINTS} to {MAX_POINTS} corners (got {len(points)})")
    out: List[Point] = []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError("every corner must be two numbers, x and y")
        if isinstance(p[0], bool) or isinstance(p[1], bool):
            raise ValueError("every corner must be two numbers, x and y")
        try:
            x, y = float(p[0]), float(p[1])
        except Exception:  # noqa: BLE001 - anything that is not two finite numbers is one error for the owner
            raise ValueError("every corner must be two numbers, x and y") from None
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("every corner must lie inside the picture (0 to 1)")
        out.append((round(x, DECIMALS), round(y, DECIMALS)))
    return out


def parse_points(text: str) -> List[Point]:
    """``"x,y;x,y;x,y"`` (whitespace between pairs is fine too) -> validated corners."""
    pairs = [t for t in _PAIR_SPLIT.split(str(text or "").strip()) if t]
    points: List[Any] = []
    for pair in pairs:
        parts = pair.split(",")
        if len(parts) != 2:
            raise ValueError(f"corner {pair!r} must be x,y")
        points.append(parts)
    return validate_points(points)


# ----------------------------------------------------------------------------
# The file
# ----------------------------------------------------------------------------
def _read_raw(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001 - a damaged file means "no zones", never a crash at startup
        log.warning("Could not read %s (%s); no zones", path, exc)
        return {}
    raw = data.get("zones", {}) if isinstance(data, dict) else {}
    return {str(k): v for k, v in raw.items()} if isinstance(raw, dict) else {}


def load_zones(path: str = ZONES_PATH) -> Dict[str, List[Point]]:
    """``{camera: [(x, y), ...]}`` with only the valid polygons; a damaged file is no zones."""
    zones: Dict[str, List[Point]] = {}
    for camera, pts in _read_raw(path).items():
        try:
            zones[str(camera)] = validate_points(pts)
        except ValueError as exc:
            log.warning("Zone for %s ignored: %s", camera, exc)
    return zones


def save_zones(zones: Dict[str, Sequence[Sequence[float]]], path: str = ZONES_PATH) -> None:
    """Write the whole file (temp file + replace, so a reader never sees half a file)."""
    clean: Dict[str, List[List[float]]] = {}
    for camera, pts in zones.items():
        try:
            clean[str(camera)] = [[x, y] for x, y in validate_points(pts)]
        except ValueError as exc:
            log.warning("Zone for %s dropped: %s", camera, exc)
    data = {"zones": clean}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(_HEADER)
            yaml.dump(data, f, default_flow_style=None, allow_unicode=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    log.info("Wrote %d zone(s) to %s", len(clean), path)


def save_zone(camera: str, points: Any, path: str = ZONES_PATH) -> List[Point]:
    """Set one camera's zone, keeping the others. Returns the corners as stored."""
    corners = validate_points(points)
    zones: Dict[str, Any] = dict(_read_raw(path))
    zones[str(camera)] = corners
    save_zones(zones, path)
    return corners


def clear_zone(camera: str, path: str = ZONES_PATH) -> bool:
    """Remove one camera's zone (watch everything). True if there was one."""
    zones: Dict[str, Any] = dict(_read_raw(path))
    if str(camera) not in zones:
        return False
    del zones[str(camera)]
    save_zones(zones, path)
    return True


def rename_zone(old: str, new: str, path: str = ZONES_PATH) -> bool:
    """Carry a zone along when its camera is renamed. True if there was one to move.

    Refuses (False) when ``new`` already has a zone: overwriting it would lose
    that zone. Several renames at once (a swap, a chain) go through
    :func:`remap_zones`, which applies them in one pass.
    """
    zones: Dict[str, Any] = dict(_read_raw(path))
    if str(old) not in zones or old == new:
        return False
    if str(new) in zones:
        log.warning("Zone for %s was not renamed: %s already has a zone", old, new)
        return False
    try:
        validate_points(zones[str(old)])
    except ValueError:
        log.warning("Zone for %s is damaged and was not renamed", old)
        return False
    zones[str(new)] = zones.pop(str(old))
    save_zones(zones, path)
    return True


def remapped_zones(entries: Dict[str, Any], renames: Dict[str, str]) -> Dict[str, Any]:
    """The zone entries after the camera renames ``{old: new}``, applied all at once.

    A renamed camera's zone moves to its new name; a camera that is not renamed
    keeps its own. A name that another camera takes over loses whatever zone it
    had, unless that camera brings one along: the old zone belonged to a
    different camera, and leaving it would mask the wrong picture.
    """
    renames = {str(k): str(v) for k, v in renames.items()}
    targets = set(renames.values())
    out = {k: v for k, v in entries.items() if k not in renames and k not in targets}
    for old, new in renames.items():
        if old in entries:
            out[new] = entries[old]
    return out


def remap_zones(renames: Dict[str, str], path: str = ZONES_PATH,
                between: Optional[Callable[[], None]] = None) -> None:
    """Apply camera renames ``{old: new}`` to the zone file in one pass (swaps and chains included).

    ``between`` (e.g. writing the renamed cameras.yaml) runs while the file holds
    the zones under BOTH the old and the new names, so a failure part-way never
    leaves a camera unmasked; the final file, without the old names, is written
    only after it returns.
    """
    entries = _read_raw(path)
    final = remapped_zones(entries, renames)
    changed = final != entries
    if between is not None:
        if changed:
            save_zones({**entries, **final}, path)
        between()
    if changed:
        save_zones(final, path)


# ----------------------------------------------------------------------------
# The mask
# ----------------------------------------------------------------------------
class ZoneMask:
    """Blacks out everything outside a camera's zone. ``ZoneMask(None)`` passes frames through.

    The pixel mask is built once per frame size and cached, so a frame costs
    one ``cv2.bitwise_and``. Not thread-safe: give each reader its own.
    """

    def __init__(self, polygon: Optional[Sequence[Point]]) -> None:
        self.polygon: Optional[List[Point]] = validate_points(polygon) if polygon else None
        self._shape: Optional[Tuple[int, int]] = None
        self._mask: Optional[np.ndarray] = None

    @property
    def active(self) -> bool:
        return self.polygon is not None

    def apply(self, frame: Any) -> Any:
        """The frame with everything outside the zone black (a new array), or the frame itself when there is no zone."""
        if self.polygon is None or frame is None:
            return frame
        import cv2  # noqa: PLC0415 - keep the import cost off modules that never mask

        h, w = frame.shape[:2]
        if self._shape != (h, w) or self._mask is None:
            mask = np.zeros((h, w), dtype=np.uint8)
            corners = np.array([[int(round(x * (w - 1))), int(round(y * (h - 1)))] for x, y in self.polygon],
                               dtype=np.int32)
            cv2.fillPoly(mask, [corners], 255)
            self._mask, self._shape = mask, (h, w)
        return cv2.bitwise_and(frame, frame, mask=self._mask)


def mask_for(zones: Dict[str, List[Point]], camera: str) -> ZoneMask:
    """The camera's mask from a loaded zones dict; a camera without a zone gets a pass-through."""
    return ZoneMask(zones.get(camera))
