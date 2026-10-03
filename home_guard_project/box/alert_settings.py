"""Read and change what the box alerts about and how sure its detector must be.

One plain function per operation, shared by the Telegram assistant (agent.py) and
the assistant v2. *camera* is a camera name, or None for the house default.
Each change returns the new effective state, or raises ValueError with a plain
message. Nothing here restarts the program: inference re-reads box.yaml and
camera_alerts.yaml within seconds.

House values live in box.yaml (``alert_on``, ``conf_person`` / ``conf_vehicle`` /
``conf_animal``); a camera's own values in camera_alerts.yaml.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import yaml

from . import boxconfig, camera_alerts
from .camera_alerts import CAMERA_ALERTS_PATH, TYPES

CAMERAS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_collection", "cameras.yaml"))

# Words that mean "go back to the house default" for a camera's alert types.
DEFAULT_WORDS = ("default", "house", "house_default")


def camera_names(cameras_path: str = CAMERAS_PATH) -> List[str]:
    """Every camera the box knows, active and disabled, in file order."""
    try:
        with open(cameras_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(raw, dict):
        return []
    names = list(dict(raw.get("cameras") or {})) + list(dict(raw.get("disabled") or {}))
    return [str(n) for n in names]


def resolve_camera(camera: Optional[str], cameras_path: str = CAMERAS_PATH) -> Optional[str]:
    """The camera's stored name ("Main Entrance" -> main_entrance); None / "house" -> None (the house).

    Raises ValueError naming the cameras when it matches none of them.
    """
    text = str(camera or "").strip()
    if not text or text.casefold() in ("house", "all", "all cameras", "every camera"):
        return None
    names = camera_names(cameras_path)
    wanted = "_".join(text.casefold().replace("-", " ").split())
    for name in names:
        if name.casefold() == wanted:
            return name
    raise ValueError(f"unknown camera {text!r}; the cameras are: {', '.join(names) or 'none'}")


def _house(box_path: str) -> Dict[str, Any]:
    """The house's alert types and certainty per type, from box.yaml."""
    from .inference import AlertSettings  # noqa: PLC0415 - inference imports this module's neighbours

    try:
        box = boxconfig.load_box_settings(box_path)
    except (boxconfig.BoxConfigError, OSError):
        box = {}
    settings = AlertSettings.from_box_settings(box)
    return {"alert_on": list(settings.alert_on), "sensitivity": settings.thresholds()}


def get_alert_settings(camera: Optional[str] = None, *, cameras_path: str = CAMERAS_PATH,
                       alerts_path: str = CAMERA_ALERTS_PATH, box_path: str = boxconfig.BOX_YAML) -> Dict[str, Any]:
    """What alerts and how sensitive, for one camera or for the whole house.

    One camera: ``{"camera", "alert_on", "own_alert_on", "sensitivity", "own_sensitivity"}``
    (``own_*`` is None when it follows the house). The house: ``{"house": {"alert_on",
    "sensitivity"}, "cameras": [<one camera, as above>, ...]}``.
    """
    house = _house(box_path)
    own_types = camera_alerts.load_camera_alerts(alerts_path)
    own_sens = camera_alerts.load_camera_sensitivity(alerts_path)

    def row(name: str) -> Dict[str, Any]:
        return {
            "camera": name,
            "alert_on": list(camera_alerts.effective(own_types, name, house["alert_on"])),
            "own_alert_on": list(own_types[name]) if name in own_types else None,
            "sensitivity": camera_alerts.thresholds_for(own_sens, name, house["sensitivity"]),
            "own_sensitivity": dict(own_sens[name]) if name in own_sens else None,
        }

    name = resolve_camera(camera, cameras_path)
    if name is not None:
        return row(name)
    return {"house": house, "cameras": [row(n) for n in camera_names(cameras_path)]}


def set_alert_types(camera: Optional[str], types: Any, *, cameras_path: str = CAMERAS_PATH,
                    alerts_path: str = CAMERA_ALERTS_PATH, box_path: str = boxconfig.BOX_YAML) -> Dict[str, Any]:
    """Set what a camera (or the house, camera None) alerts about. Returns the new effective state.

    *types*: person / vehicle / animal, as a list or "person,vehicle" - the complete new
    list. Or changes to the current list: "+animal" adds, "-vehicle" removes (e.g.
    ["+vehicle"]: also vehicles). For a camera, "default" (or ["default"]) puts it back
    on the house default.
    """
    name = resolve_camera(camera, cameras_path)
    words = [str(t).strip().casefold() for t in (types if isinstance(types, (list, tuple)) else str(types).split(","))]
    words = [w for w in words if w]
    if words and all(w[:1] in "+-" for w in words):
        types = _apply_changes(name, words, cameras_path, alerts_path, box_path)
    if name is None:
        if any(w in DEFAULT_WORDS for w in words):
            raise ValueError("the house default needs real types: person, vehicle or animal")
        chosen = camera_alerts.parse_types(types)
        boxconfig.set_option("alert_on", ",".join(chosen), box_path)
    elif words and all(w in DEFAULT_WORDS for w in words):
        camera_alerts.clear_camera_alerts(name, alerts_path)
    else:
        camera_alerts.set_camera_alerts(name, types, alerts_path)
    return get_alert_settings(name, cameras_path=cameras_path, alerts_path=alerts_path, box_path=box_path)


def _apply_changes(name: Optional[str], words: List[str], cameras_path: str, alerts_path: str,
                   box_path: str) -> List[str]:
    """The current types of the camera (or house) with "+type" added and "-type" removed."""
    current = get_alert_settings(name, cameras_path=cameras_path, alerts_path=alerts_path, box_path=box_path)
    types = list(current["house"]["alert_on"] if name is None else current["alert_on"])
    for word in words:
        kind = camera_alerts.parse_types(word[1:])[0]
        if word[0] == "+" and kind not in types:
            types.append(kind)
        elif word[0] == "-" and kind in types:
            types.remove(kind)
    if not types:
        raise ValueError("at least one type must stay on; to stop all alerts, pause them instead")
    return types


def set_sensitivity(camera: Optional[str], values: Any, *, cameras_path: str = CAMERAS_PATH,
                    alerts_path: str = CAMERA_ALERTS_PATH, box_path: str = boxconfig.BOX_YAML) -> Dict[str, Any]:
    """Set how sure the detector must be, per type, for a camera (or the house, camera None).

    *values*: ``{"person": 0.5}`` or "person=0.5,vehicle=0.8" (0.05-0.95; percentages
    like 50 are read as 0.5). Types not given keep their value. For a camera,
    "default" puts it back on the house values. Returns the new effective state.
    """
    name = resolve_camera(camera, cameras_path)
    if isinstance(values, str) and values.strip().casefold() in DEFAULT_WORDS:
        if name is None:
            raise ValueError("the house needs real values, e.g. person=0.5")
        camera_alerts.clear_camera_sensitivity(name, alerts_path)
        return get_alert_settings(name, cameras_path=cameras_path, alerts_path=alerts_path, box_path=box_path)
    chosen = camera_alerts.parse_thresholds(_fractions(values))
    if name is None:
        for kind, conf in chosen.items():
            boxconfig.set_option(f"conf_{kind}", str(conf), box_path)
    else:
        merged = {**(camera_alerts.load_camera_sensitivity(alerts_path).get(name) or {}), **chosen}
        camera_alerts.set_camera_sensitivity(name, merged, alerts_path)
    return get_alert_settings(name, cameras_path=cameras_path, alerts_path=alerts_path, box_path=box_path)


def _fractions(values: Any) -> Any:
    """Read 50 or "50%" as 0.5, so an owner's "people at 50%" works; anything else passes through."""
    if isinstance(values, str):
        pairs = {}
        for part in values.split(","):
            if "=" in part:
                key, _, number = part.partition("=")
                pairs[key.strip()] = number.strip()
        values = pairs or values
    if not isinstance(values, dict):
        return values
    out = {}
    for key, number in values.items():
        text = str(number).strip().rstrip("%")
        try:
            value = float(text)
        except ValueError:
            out[key] = number
            continue
        out[key] = value / 100 if value > 1 else value
    return out


def describe(state: Dict[str, Any]) -> str:
    """One plain English line for a camera's state, for the owner (the assistant translates it)."""
    kinds = {"person": "people", "vehicle": "vehicles", "animal": "animals"}
    types = " and ".join(kinds[t] for t in state["alert_on"])
    sens = ", ".join(f"{kinds[t]} {round(state['sensitivity'][t] * 100)}%" for t in TYPES if t in state["sensitivity"])
    source = "its own choice" if state.get("own_alert_on") else "the house default"
    return f"{state['camera']} alerts on {types} ({source}); the detector must be sure: {sens}."
