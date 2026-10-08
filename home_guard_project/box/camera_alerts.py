"""Each camera's own alert settings: what it alerts about, and how sure the detector must be.

The house defaults live in box.yaml: ``alert_on`` (unset: people only) and the
detector's certainty per type, ``conf_person`` / ``conf_vehicle`` /
``conf_animal`` (unset: ``inference_conf``). A camera can override them in
``camera_alerts.yaml`` next to ``cameras.yaml``::

    alerts:
      driveway: [person, vehicle]
    sensitivity:
      driveway: {vehicle: 0.85}      # types not listed use the house value

Inference mode re-reads the file while it runs (:class:`LiveCameraAlerts`), so a
change applies within seconds. A damaged file or entry means "house default",
never a crash.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml

from . import paths

log = logging.getLogger(__name__)

# The types, in the fixed order they are stored and shown. boxconfig.SET_OPTIONS["alert_on"] matches.
TYPES = ("person", "vehicle", "animal")
DEFAULT = ("person",)
# The detector's certainty, as fractions (boxconfig.DECIMAL_OPTIONS uses the same range).
CONF_MIN, CONF_MAX = 0.05, 0.95

CAMERA_ALERTS_PATH = paths.camera_alerts_yaml()
POLL_SEC = 2.0

_HEADER = (
    "# ──────────────────────────────────────────────────────────────────────────────\n"
    "#  Each camera's own alert settings: what it alerts about (person / vehicle /\n"
    "#  animal) and the detector's certainty per type. Cameras without an entry use\n"
    "#  the house defaults in box.yaml.\n"
    "#  DO NOT COMMIT (deployment-specific)\n"
    "# ──────────────────────────────────────────────────────────────────────────────\n\n"
)
_SECTIONS = ("alerts", "sensitivity")

Types = Tuple[str, ...]
Thresholds = Dict[str, float]


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
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


def parse_thresholds(value: Any) -> Thresholds:
    """``{"vehicle": 0.9}`` or ``"vehicle=0.9,person=0.45"`` -> per-type certainties, in type order.

    At least one type; each a number from 0.05 to 0.95. ValueError in plain words otherwise.
    """
    if isinstance(value, str):
        pairs = {}
        for part in value.split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise ValueError("write the certainty as type=number, e.g. person=0.5,vehicle=0.8")
            key, _, number = part.partition("=")
            pairs[key] = number
        value = pairs
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"give a certainty for one or more of {', '.join(TYPES)}")
    out: Thresholds = {}
    for key, number in value.items():
        kind = str(key).strip().lower()
        if kind not in TYPES:
            raise ValueError(f"unknown type {kind!r}; choose from {', '.join(TYPES)}")
        if isinstance(number, bool):
            raise ValueError(f"the certainty for {kind} must be a number from {CONF_MIN} to {CONF_MAX}")
        try:
            conf = round(float(number), 3)
        except (TypeError, ValueError):
            raise ValueError(f"the certainty for {kind} must be a number from {CONF_MIN} to {CONF_MAX}") from None
        if not math.isfinite(conf) or not CONF_MIN <= conf <= CONF_MAX:
            raise ValueError(f"the certainty for {kind} must be a number from {CONF_MIN} to {CONF_MAX}")
        out[kind] = conf
    return {t: out[t] for t in TYPES if t in out}


# ----------------------------------------------------------------------------
# The file
# ----------------------------------------------------------------------------
def _read_doc(path: str) -> Dict[str, Dict[str, Any]]:
    """Both sections as raw ``{camera: value}`` dicts; a damaged file is empty sections."""
    empty: Dict[str, Dict[str, Any]] = {s: {} for s in _SECTIONS}
    if not os.path.isfile(path):
        return empty
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001 - a damaged file means "house defaults everywhere"
        log.warning("Could not read %s (%s); every camera uses the house defaults", path, exc)
        return empty
    if not isinstance(data, dict):
        return empty
    out = {}
    for section in _SECTIONS:
        raw = data.get(section, {})
        out[section] = {str(k): v for k, v in raw.items()} if isinstance(raw, dict) else {}
    return out


def _write_doc(doc: Dict[str, Dict[str, Any]], path: str) -> None:
    """Write both sections, valid entries only (temp file + replace, so a reader never sees half a file)."""
    clean: Dict[str, Dict[str, Any]] = {"alerts": {}, "sensitivity": {}}
    for camera, value in doc.get("alerts", {}).items():
        try:
            clean["alerts"][str(camera)] = list(parse_types(value))
        except ValueError:
            log.warning("Alert choice for %s dropped: not valid", camera)
    for camera, value in doc.get("sensitivity", {}).items():
        try:
            clean["sensitivity"][str(camera)] = parse_thresholds(value)
        except ValueError:
            log.warning("Sensitivity for %s dropped: not valid", camera)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(_HEADER)
            yaml.dump(clean, f, default_flow_style=None, allow_unicode=True, sort_keys=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _load(section: str, parse, path: str, what: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for camera, value in _read_doc(path)[section].items():
        try:
            out[camera] = parse(value)
        except ValueError as exc:
            log.warning("%s for %s ignored (%s); it uses the house default", what, camera, exc)
    return out


def _set(section: str, camera: str, value: Any, path: str) -> None:
    doc = _read_doc(path)
    doc[section][str(camera)] = value
    _write_doc(doc, path)


def _clear(section: str, camera: str, path: str) -> bool:
    doc = _read_doc(path)
    if str(camera) not in doc[section]:
        return False
    del doc[section][str(camera)]
    _write_doc(doc, path)
    return True


# ----------------------------------------------------------------------------
# What a camera alerts about
# ----------------------------------------------------------------------------
def load_camera_alerts(path: str = CAMERA_ALERTS_PATH) -> Dict[str, Types]:
    """``{camera: types}`` for the cameras with a valid choice of their own."""
    return _load("alerts", parse_types, path, "Alert choice")


def set_camera_alerts(camera: str, types: Any, path: str = CAMERA_ALERTS_PATH) -> Types:
    """Give one camera its own choice, keeping everything else. Returns the types as stored."""
    chosen = parse_types(types)
    _set("alerts", camera, list(chosen), path)
    return chosen


def clear_camera_alerts(camera: str, path: str = CAMERA_ALERTS_PATH) -> bool:
    """Put a camera's alert types back on the house default. True if it had a choice of its own."""
    return _clear("alerts", camera, path)


def effective(overrides: Dict[str, Types], camera: str, house: Sequence[str]) -> Types:
    """The types *camera* alerts on: its own choice, else the house default."""
    return overrides.get(camera) or tuple(house)


# ----------------------------------------------------------------------------
# How sure the detector must be, per type
# ----------------------------------------------------------------------------
def load_camera_sensitivity(path: str = CAMERA_ALERTS_PATH) -> Dict[str, Thresholds]:
    """``{camera: {type: certainty}}`` for the cameras with valid values of their own."""
    return _load("sensitivity", parse_thresholds, path, "Sensitivity")


def set_camera_sensitivity(camera: str, thresholds: Any, path: str = CAMERA_ALERTS_PATH) -> Thresholds:
    """Give one camera its own certainty for some or all types. Returns the values as stored."""
    chosen = parse_thresholds(thresholds)
    _set("sensitivity", camera, chosen, path)
    return chosen


def clear_camera_sensitivity(camera: str, path: str = CAMERA_ALERTS_PATH) -> bool:
    """Put a camera's sensitivity back on the house values. True if it had values of its own."""
    return _clear("sensitivity", camera, path)


def thresholds_for(overrides: Dict[str, Thresholds], camera: str, house: Mapping[str, float]) -> Thresholds:
    """The certainty per type for *camera*: its own values where it has them, else the house's."""
    return {**{t: float(house[t]) for t in TYPES if t in house}, **overrides.get(camera, {})}


# ----------------------------------------------------------------------------
# Renames and live reload
# ----------------------------------------------------------------------------
def remap_camera_alerts(renames: Dict[str, str], path: str = CAMERA_ALERTS_PATH) -> None:
    """Apply camera renames ``{old: new}`` to both sections in one pass (swaps and chains included).

    A name another camera takes over loses its old settings unless that camera
    brings some along: the old ones belonged to a different picture.
    """
    doc = _read_doc(path)
    renames = {str(k): str(v) for k, v in renames.items()}
    targets = set(renames.values())
    final = {}
    for section, entries in doc.items():
        moved = {k: v for k, v in entries.items() if k not in renames and k not in targets}
        for old, new in renames.items():
            if old in entries:
                moved[new] = entries[old]
        final[section] = moved
    if final != doc:
        _write_doc(final, path)


class LiveCameraAlerts:
    """Re-reads camera_alerts.yaml while inference runs, like LiveSettings does for box.yaml."""

    def __init__(self, path: str = CAMERA_ALERTS_PATH, poll_sec: float = POLL_SEC, now: float = 0.0) -> None:
        self.path = path
        self.poll_sec = poll_sec
        self._mtime = self._stat()
        self._checked = now
        self.overrides: Dict[str, Types] = load_camera_alerts(path)
        self.sensitivity: Dict[str, Thresholds] = load_camera_sensitivity(path)

    def _stat(self) -> Optional[float]:
        try:
            return os.stat(self.path).st_mtime
        except OSError:
            return None

    def check(self, now: float) -> bool:
        """Reload if the file changed since the last look. True if any camera's settings changed."""
        if now - self._checked < self.poll_sec:
            return False
        self._checked = now
        mtime = self._stat()
        if mtime == self._mtime:
            return False
        self._mtime = mtime
        alerts, sensitivity = load_camera_alerts(self.path), load_camera_sensitivity(self.path)
        if alerts == self.overrides and sensitivity == self.sensitivity:
            return False
        self.overrides, self.sensitivity = alerts, sensitivity
        log.info("Camera alert settings changed while running: %s",
                 "; ".join(f"{c}: {'+'.join(alerts[c]) if c in alerts else 'house types'}"
                           + (f", certainty {sensitivity[c]}" if c in sensitivity else "")
                           for c in sorted(set(alerts) | set(sensitivity))) or "all on the house defaults")
        return True

    def for_camera(self, camera: str, house: Sequence[str]) -> Types:
        return effective(self.overrides, camera, house)

    def thresholds_for(self, camera: str, house: Mapping[str, float]) -> Thresholds:
        return thresholds_for(self.sensitivity, camera, house)
