# home_guard_project/box/brain/registry.py
"""What the house looks like right now, rebuilt for every turn.

The assistant used to know only the camera names taken at start-up, so it told
the owner a camera "does not exist" right after turning it off. The snapshot
reads the live sources each time: cameras.yaml (on and off), the owner's names
for each camera, the guard loop's status file (when each camera last delivered
a picture), the alert pauses, and what each camera looks at. It never contains
a camera address or password.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import yaml

from .. import paths
from ..ai_status import read_status
from .aliases import ALIASES_PATH, load_aliases, normalize
from .mode import GUARD, hhmm, mode_ends_at, mode_started_at, resolve_mode

_log = logging.getLogger(__name__)
CAMERAS_PATH = paths.cameras_yaml()
STATUS_PATH = os.path.join(paths.logs_dir(), "ai_status.json")


@dataclass(frozen=True)
class CameraState:
    name: str
    enabled: bool
    aliases: Tuple[str, ...] = ()
    live: Optional[bool] = None          # None: unknown (the guard loop is not reporting)
    last_seen: Optional[float] = None    # when the detector last looked at a picture from it
    muted_until: Optional[float] = None
    sees: str = ""
    zone: bool = False                   # watches only the area the owner drew; the rest is blacked out


@dataclass(frozen=True)
class HouseSnapshot:
    now: float
    mode: str
    mode_ends: Optional[float]
    mode_started: Optional[float]
    start_hour: int
    end_hour: int
    cameras: Tuple[CameraState, ...]
    retention_days: float = 14.0
    state_known: bool = True
    quiet_log: bool = False                  # is the quiet log on (box.yaml quiet_log)
    quiet_since: Optional[float] = None      # when the current quiet logging began (written by the guard loop)

    def camera(self, name: str) -> Optional[CameraState]:
        return next((c for c in self.cameras if c.name == name), None)

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.cameras]

    @property
    def enabled_names(self) -> List[str]:
        return [c.name for c in self.cameras if c.enabled]

    @property
    def live_count(self) -> int:
        return sum(1 for c in self.cameras if c.enabled and c.live)

    @property
    def offline_names(self) -> List[str]:
        return [c.name for c in self.cameras if c.enabled and c.live is False]

    @property
    def paused(self) -> List[Tuple[str, float]]:
        return [(c.name, c.muted_until) for c in self.cameras if c.enabled and c.muted_until]


@dataclass(frozen=True)
class Resolution:
    camera: Optional[str]                # the one camera meant, or None
    candidates: Tuple[str, ...] = ()     # every camera that matched (several: ask the owner)


def _result(hits: Sequence[str]) -> Resolution:
    unique = tuple(dict.fromkeys(hits))
    return Resolution(unique[0] if len(unique) == 1 else None, unique)


_HE_PREFIXES = "הבלמוש"     # Hebrew writes "the", "in", "to", "from", "and", "that" glued to the word


def _variants(query: str) -> List[str]:
    """The query, and the query with Hebrew one-letter prefixes taken off each word ("הקדמית" -> "קדמית")."""
    words = query.split()
    stripped = [w[1:] if len(w) > 2 and w[0] in _HE_PREFIXES else w for w in words]
    return list(dict.fromkeys([" ".join(words), " ".join(stripped)]))


# "camera 3", "cam #3", "מצלמה 3", "המצלמה מספר 3": the owner's way of saying channel 3.
_CAMERA_WORD = r"(?:the\s+)?(?:camera|cam|ה?מצלמה|מצלמת)\s*(?:#|no\.?|number|מספר)?\s*(\d+)"
_CAMERA_NUMBER = re.compile(rf"^{_CAMERA_WORD}$")
_CAMERA_NUMBER_IN_TEXT = re.compile(rf"(?<!\w)[ושהבלמכ]{{0,2}}({_CAMERA_WORD})(?!\w)")


def resolve_camera(snapshot: HouseSnapshot, words: str) -> Resolution:
    """Which camera *words* means: exact name or alias, then a bare channel number ("3", "camera 3"),
    then a name or alias inside the phrase, then a unique name prefix."""
    query = normalize(words)
    if not query:
        return Resolution(None)
    numbered_phrase = _CAMERA_NUMBER.match(query)
    if numbered_phrase and not any(query in [normalize(c.name)] + [normalize(a) for a in c.aliases]
                                   for c in snapshot.cameras):
        query = numbered_phrase.group(1)
    variants = _variants(query)

    def keys(cam: CameraState) -> List[str]:
        return [k for k in [normalize(cam.name)] + [normalize(a) for a in cam.aliases] if k]

    exact = [c.name for c in snapshot.cameras if any(v in keys(c) for v in variants)]
    if exact:
        return _result(exact)
    if query.isdigit():
        numbered = [c.name for c in snapshot.cameras if re.search(rf"(?<!\d){query}$", c.name)]
        if numbered:
            return _result(numbered)
    padded = [f" {v} " for v in variants]
    inside = [c.name for c in snapshot.cameras if any(f" {k} " in p for k in keys(c) for p in padded)]
    if inside:
        return _result(inside)
    prefix = [c.name for c in snapshot.cameras if normalize(c.name).startswith(query)]
    return _result(prefix) if prefix else Resolution(None)


def mentioned_cameras(snapshot: HouseSnapshot, text: str) -> List[Tuple[str, str]]:
    """``(word, camera)`` for every camera a message names - by its name, one of the owner's names for it, or
    "camera 3" - as whole words (Hebrew prefixes allowed: "בפרגולה"), in the order they appear. Where two names
    overlap ("front door" and "front") the longer one wins. Resolved in code so the model never has to guess
    what "the pergola" is (2026-10-05)."""
    if not isinstance(text, str):
        return []
    norm = normalize(text)
    found: List[Tuple[int, int, str, str]] = []
    for cam in snapshot.cameras:
        for word in (cam.name,) + tuple(cam.aliases):
            key = normalize(word)
            if not key:
                continue
            for m in re.finditer(rf"(?<!\w)[ושהבלמכ]{{0,2}}({re.escape(key)})(?!\w)", norm):
                found.append((m.start(1), m.end(1), word, cam.name))
    for m in _CAMERA_NUMBER_IN_TEXT.finditer(norm):
        res = resolve_camera(snapshot, m.group(2))
        if res.camera:
            found.append((m.start(1), m.end(1), m.group(1), res.camera))
    kept: List[Tuple[int, int, str, str]] = []
    for hit in sorted(found, key=lambda h: (h[0] - h[1], h[0])):           # longest first
        if all(hit[1] <= k[0] or hit[0] >= k[1] for k in kept):
            kept.append(hit)
    out: List[Tuple[str, str]] = []
    for _, _, word, camera in sorted(kept):
        if (word, camera) not in out:
            out.append((word, camera))
    return out


def render_block(snapshot: HouseSnapshot) -> str:
    """The compact house description put in front of every owner message."""
    lines = [f"CAMERAS ({len(snapshot.cameras)} · {snapshot.live_count} live)"]
    if not snapshot.state_known:
        lines.append("camera state unknown (the camera list could not be read)")
    for cam in snapshot.cameras:
        parts = [cam.name]
        if cam.aliases:
            parts.append("aka: " + ", ".join(cam.aliases))
        if not cam.enabled:
            parts.append("OFF")
        else:
            if cam.live is None:
                parts.append("state unknown")
            elif cam.live:
                parts.append("live")
            else:
                seen = f" (last seen {hhmm(cam.last_seen)})" if cam.last_seen else ""
                parts.append(f"offline{seen}")
            parts.append(f"alerts paused until {hhmm(cam.muted_until)}" if cam.muted_until else "alerts on")
        lines.append("  ".join(parts))
        if cam.zone:
            lines.append("  watches only the drawn area (the rest of the picture is blacked out)")
        if cam.sees:
            lines.append(f"  sees: {cam.sees}")
    if snapshot.mode == GUARD:
        mode = f"Guard until {hhmm(snapshot.mode_ends)}" if snapshot.mode_ends else "Guard all day"
    else:
        if snapshot.quiet_log:
            since = f"quiet log on since {hhmm(snapshot.quiet_since)}" if snapshot.quiet_since else "quiet log on"
        else:
            since = "quiet log OFF: nothing is recorded outside the alert hours"
        nxt = f"; guarding from {hhmm(snapshot.mode_ends)}" if snapshot.mode_ends else ""
        mode = f"Assistant ({since}{nxt})"
    lines.append(f"MODE: {mode} · clips kept {int(snapshot.retention_days)} days")
    return "\n".join(lines)


def hours_from_box_yaml(path: Optional[str] = None) -> Callable[[], Tuple[int, int]]:
    """A callable returning the current ``(alert_start_hour, alert_end_hour)`` from box.yaml."""
    def hours() -> Tuple[int, int]:
        from ..boxconfig import BOX_YAML, load_box_settings  # noqa: PLC0415

        try:
            settings = load_box_settings(path or BOX_YAML)
            start, end = int(settings.get("alert_start_hour", 0)), int(settings.get("alert_end_hour", 0))
        except Exception:  # noqa: BLE001 - unreadable or malformed: guard all day, the safe side
            return (0, 0)
        return (start, end) if 0 <= start <= 23 and 0 <= end <= 23 else (0, 0)
    return hours


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _num(value: Any) -> Optional[float]:
    """A number read from a hand-editable file, or None when it is not one."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dict(value: Any) -> Dict[Any, Any]:
    return value if isinstance(value, dict) else {}


class HouseRegistry:
    """Builds a :class:`HouseSnapshot` from the box's live files. Never raises."""

    def __init__(self, mute: Any, hours: Callable[[], Tuple[int, int]], cameras_path: str = CAMERAS_PATH,
                 aliases_path: str = ALIASES_PATH, status_path: str = STATUS_PATH, sees_path: str = "",
                 now: Callable[[], float] = time.time, retention_days: float = 14.0,
                 offline_after: float = 60.0, zones_path: Optional[str] = None,
                 quiet_log: Callable[[], bool] = lambda: False, quiet_since_path: str = "") -> None:
        self.mute = mute
        self.zones_path = zones_path
        self.quiet_log = quiet_log
        self.quiet_since_path = quiet_since_path
        self.hours = hours
        self.cameras_path, self.aliases_path = cameras_path, aliases_path
        self.status_path, self.sees_path = status_path, sees_path
        self.now = now
        self.retention_days = retention_days
        self.offline_after = offline_after

    def snapshot(self) -> HouseSnapshot:
        now = self.now()
        try:
            return self._build(now)
        except Exception:  # noqa: BLE001 - last line of defence: guard all day, no cameras
            _log.warning("house snapshot failed; falling back to an empty picture", exc_info=True)
            return HouseSnapshot(
                now=now, mode=resolve_mode(now, 0, 0), mode_ends=mode_ends_at(now, 0, 0),
                mode_started=mode_started_at(now, 0, 0), start_hour=0, end_hour=0, cameras=(),
                retention_days=self.retention_days, state_known=False,
            )

    def _build(self, now: float) -> HouseSnapshot:
        known = True
        try:
            with open(self.cameras_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            if not isinstance(raw, dict):
                raise ValueError("not a mapping")
        except (OSError, yaml.YAMLError, ValueError):
            raw, known = {}, False
        active = list(_dict(raw.get("cameras")))
        disabled = [n for n in _dict(raw.get("disabled")) if n not in active]
        aliases = load_aliases(self.aliases_path)
        status_cams = _dict(_dict(read_status(self.status_path)).get("cameras"))
        sees = _dict(_read_json(self.sees_path).get("cameras")) if self.sees_path else {}
        try:
            from ...data_collection.zones import ZONES_PATH, load_zones  # noqa: PLC0415

            zones = load_zones(self.zones_path or ZONES_PATH)
        except Exception:  # noqa: BLE001
            zones = {}
        start, end = self.hours()
        cams = []
        for name in active + disabled:
            entry = _dict(status_cams.get(name))
            # When the camera last delivered a new picture; the detector's last look for an older status file.
            checked = _num(entry.get("frame_ts") or entry.get("checked_ts"))
            if name not in active:
                live: Optional[bool] = False
            elif not status_cams:
                live = None
            else:
                live = bool(checked) and now - checked <= self.offline_after
            try:
                muted = self.mute.muted_until(now, name)
            except Exception:  # noqa: BLE001
                muted = None
            cams.append(CameraState(
                name=name, enabled=name in active, aliases=tuple(aliases.get(name, [])), live=live,
                last_seen=checked if checked else None, muted_until=muted,
                sees=str(_dict(sees.get(name)).get("text") or ""), zone=bool(zones.get(name)),
            ))
        try:
            logging_on = bool(self.quiet_log())
        except Exception:  # noqa: BLE001
            logging_on = False
        since = _read_json(self.quiet_since_path).get("since") if (logging_on and self.quiet_since_path) else None
        return HouseSnapshot(
            now=now, mode=resolve_mode(now, start, end), mode_ends=mode_ends_at(now, start, end),
            mode_started=mode_started_at(now, start, end), start_hour=start, end_hour=end,
            cameras=tuple(cams), retention_days=self.retention_days, state_known=known,
            quiet_log=logging_on, quiet_since=_num(since) or None,
        )
