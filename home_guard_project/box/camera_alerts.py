"""What each camera alerts the owner about: people, vehicles, animals.

The house default is ``alert_on`` in box.yaml (unset: people only). A camera can
have its own choice, kept in ``camera_alerts.yaml`` next to ``cameras.yaml`` -
for example the driveway camera on people and vehicles while the yard stays on
people. A camera without an entry uses the house default. Inference mode
re-reads the file while it runs (:class:`LiveCameraAlerts`), so a change applies
within seconds. A damaged file or entry means "house default", never a crash.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import yaml

log = logging.getLogger(__name__)

# The types, in the fixed order they are stored and shown. boxconfig.SET_OPTIONS["alert_on"] matches.
TYPES = ("person", "vehicle", "animal")
DEFAULT = ("person",)

CAMERA_ALERTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data_collection", "camera_alerts.yaml"))
POLL_SEC = 2.0

_HEADER = (
    "# ──────────────────────────────────────────────────────────────────────────────\n"
    "#  What each camera alerts about (person / vehicle / animal). Cameras without\n"
    "#  an entry use the house default, alert_on in box.yaml.\n"
    "#  DO NOT COMMIT (deployment-specific)\n"
    "# ──────────────────────────────────────────────────────────────────────────────\n\n"
)

Types = Tuple[str, ...]


def parse_types(value: Any) -> Types:
    """``"vehicle, person"`` or a list -> ``("person", "vehicle")``; ValueError in plain words otherwise."""
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple)):
        parts = [str(v) for v in value]
    else:
        raise ValueError(f"choose one or more of {', '.join(TYPES)}")
    picked = {p.strip().lower() for p in parts if p.strip()}
    if not picked:
        raise ValueError(f"choose one or more of {', '.join(TYPES)}")
    unknown = picked - set(TYPES)
    if unknown:
        raise ValueError(f"unknown alert type {', '.join(sorted(unknown))}; choose from {', '.join(TYPES)}")
    return tuple(t for t in TYPES if t in picked)


def _read_raw(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001 - a damaged file means "house default everywhere"
        log.warning("Could not read %s (%s); every camera uses the house default", path, exc)
        return {}
    raw = data.get("alerts", {}) if isinstance(data, dict) else {}
    return {str(k): v for k, v in raw.items()} if isinstance(raw, dict) else {}


def load_camera_alerts(path: str = CAMERA_ALERTS_PATH) -> Dict[str, Types]:
    """``{camera: types}`` for the cameras with a valid choice of their own."""
    out: Dict[str, Types] = {}
    for camera, value in _read_raw(path).items():
        try:
            out[camera] = parse_types(value)
        except ValueError as exc:
            log.warning("Alert choice for %s ignored (%s); it uses the house default", camera, exc)
    return out


def _save(entries: Dict[str, Any], path: str) -> None:
    """Write the whole file (temp file + replace, so a reader never sees half a file)."""
    clean: Dict[str, list] = {}
    for camera, value in entries.items():
        try:
            clean[str(camera)] = list(parse_types(value))
        except ValueError:
            log.warning("Alert choice for %s dropped: not valid", camera)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(_HEADER)
            yaml.dump({"alerts": clean}, f, default_flow_style=None, allow_unicode=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def set_camera_alerts(camera: str, types: Any, path: str = CAMERA_ALERTS_PATH) -> Types:
    """Give one camera its own choice, keeping the others. Returns the types as stored."""
    chosen = parse_types(types)
    entries = dict(_read_raw(path))
    entries[str(camera)] = list(chosen)
    _save(entries, path)
    return chosen


def clear_camera_alerts(camera: str, path: str = CAMERA_ALERTS_PATH) -> bool:
    """Put a camera back on the house default. True if it had a choice of its own."""
    entries = dict(_read_raw(path))
    if str(camera) not in entries:
        return False
    del entries[str(camera)]
    _save(entries, path)
    return True


def remap_camera_alerts(renames: Dict[str, str], path: str = CAMERA_ALERTS_PATH) -> None:
    """Apply camera renames ``{old: new}`` in one pass (swaps and chains included).

    A name another camera takes over loses its old choice unless that camera
    brings one along: the old choice belonged to a different picture.
    """
    entries = _read_raw(path)
    renames = {str(k): str(v) for k, v in renames.items()}
    targets = set(renames.values())
    final = {k: v for k, v in entries.items() if k not in renames and k not in targets}
    for old, new in renames.items():
        if old in entries:
            final[new] = entries[old]
    if final != entries:
        _save(final, path)


def effective(overrides: Dict[str, Types], camera: str, house: Sequence[str]) -> Types:
    """The types *camera* alerts on: its own choice, else the house default."""
    return overrides.get(camera) or tuple(house)


class LiveCameraAlerts:
    """Re-reads camera_alerts.yaml while inference runs, like LiveSettings does for box.yaml."""

    def __init__(self, path: str = CAMERA_ALERTS_PATH, poll_sec: float = POLL_SEC, now: float = 0.0) -> None:
        self.path = path
        self.poll_sec = poll_sec
        self._mtime = self._stat()
        self._checked = now
        self.overrides: Dict[str, Types] = load_camera_alerts(path)

    def _stat(self) -> Optional[float]:
        try:
            return os.stat(self.path).st_mtime
        except OSError:
            return None

    def check(self, now: float) -> bool:
        """Reload if the file changed since the last look. True if the choices changed."""
        if now - self._checked < self.poll_sec:
            return False
        self._checked = now
        mtime = self._stat()
        if mtime == self._mtime:
            return False
        self._mtime = mtime
        fresh = load_camera_alerts(self.path)
        if fresh == self.overrides:
            return False
        self.overrides = fresh
        log.info("Camera alert choices changed while running: %s",
                 ", ".join(f"{c}={'+'.join(t)}" for c, t in sorted(fresh.items())) or "all on the house default")
        return True

    def for_camera(self, camera: str, house: Sequence[str]) -> Types:
        return effective(self.overrides, camera, house)
