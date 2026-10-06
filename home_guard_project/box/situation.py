"""The situation of one look: the hour, the light, the house state, the camera's role and why we are looking.

Pure code, no model (spec ``docs/superpowers/specs/2026-10-05-situation-aware-eye-and-investigator-design.md``
Part 1). Everything here is computed, never guessed by the Eye. ``Situation.header()`` is the structured line the
Eye prompt carries, identical at training and inference:

    SITUATION: time 02:14, late_night, dark; house: home_asleep; camera: back_yard (private); intent: alert_triage; expecting: none

``Situation.to_taxonomy_context()`` hands the same facts to ``taxonomy.contextual_label``.

- ``phase``: ``late_night`` from 00:00 to first light, ``dawn`` from first light to sunrise, ``day`` to sunset,
  ``evening`` to midnight. ``dark`` separately (before first light or after last light). A month-level sunrise
  and sunset table for Israel (local wall clock, daylight saving included) is close enough for this.
- ``house_state``: from ``house_state.current`` (a vacation reads as ``away``).
- ``camera_role``: ``camera_roles: {camera: role}`` in box.yaml, else what the camera's scene map suggests
  (``scene_map.SceneMap.camera_role``), else guessed from the camera's name.
- ``zones``: ``camera_zones: {camera: [names]}`` in box.yaml (optional), else the scene map's zones.
- ``zone_facts``, ``ground``, ``crossed_in``, ``scene_zone``: what the scene map says about this look
  (``scene_map.scene_facts``). The ZONE FACTS line goes into the Eye's prompt next to the header, never inside
  it; ``record()["scene"]`` keeps it for training. The map's zone and ground win over the Eye's guess.
- ``expecting``: the owner's live expecting notes for this camera; ``fact_covers``: a live "lower" house note.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from . import taxonomy as tx

# Month -> (sunrise, sunset), Tel Aviv, local wall clock on the 15th (summer time April to October).
SUN_TABLE: Dict[int, Tuple[str, str]] = {
    1: ("06:40", "17:03"), 2: ("06:24", "17:28"), 3: ("05:53", "17:48"), 4: ("06:16", "19:06"),
    5: ("05:47", "19:28"), 6: ("05:33", "19:47"), 7: ("05:44", "19:48"), 8: ("06:04", "19:24"),
    9: ("06:22", "18:46"), 10: ("06:36", "18:00"), 11: ("06:03", "16:42"), 12: ("06:28", "16:37"),
}
TWILIGHT_MIN = 30          # first light this long before sunrise, last light this long after sunset

# Checked in this order on the lower-cased camera name; nothing matches -> entrance (the cautious middle).
_ROLE_WORDS = (
    ("entrance", ("door", "entrance", "entry", "gate", "porch")),
    ("parking", ("parking", "driveway", "drive", "garage", "carport")),
    ("street", ("street", "road", "sidewalk", "front")),
    ("private", ("back", "yard", "garden", "side", "pergola", "roof", "pool", "passage", "balcony", "patio",
                 "terrace")),
)
DEFAULT_ROLE = "entrance"


def _minutes(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[3:])


def phase_of(ts: float) -> Tuple[str, bool]:
    """``(phase, dark)`` at local time *ts*."""
    moment = dt.datetime.fromtimestamp(ts)
    rise, sset = (_minutes(t) for t in SUN_TABLE[moment.month])
    now = moment.hour * 60 + moment.minute
    first, last = rise - TWILIGHT_MIN, sset + TWILIGHT_MIN
    dark = now < first or now >= last
    if now < first:
        return "late_night", dark
    if now < rise:
        return "dawn", dark
    if now < sset:
        return "day", dark
    return "evening", dark


def guess_role(camera: str) -> str:
    name = str(camera or "").lower()
    for role, words in _ROLE_WORDS:
        if any(w in name for w in words):
            return role
    return DEFAULT_ROLE


def camera_role(camera: str, settings: Optional[Mapping[str, Any]] = None, scene_map: Any = None) -> str:
    """The role set for *camera* in box.yaml (``camera_roles``), else the one its scene map suggests, else one
    guessed from its name."""
    roles = (settings or {}).get("camera_roles") if isinstance(settings, Mapping) else None
    chosen = str((roles or {}).get(camera) or "").strip().lower() if isinstance(roles, Mapping) else ""
    if chosen in tx.CAMERA_ROLES:
        return chosen
    from_map = scene_map.camera_role() if scene_map is not None else ""
    return from_map if from_map in tx.CAMERA_ROLES else guess_role(camera)


def _zones(camera: str, settings: Optional[Mapping[str, Any]], zones: Optional[Iterable[str]],
           scene_map: Any = None) -> Tuple[str, ...]:
    if zones is None and isinstance(settings, Mapping) and isinstance(settings.get("camera_zones"), Mapping):
        zones = settings["camera_zones"].get(camera)
    if zones is None and scene_map is not None:
        zones = scene_map.zones()
    if not zones or isinstance(zones, str):
        return ()
    return tuple(z for z in (str(v).strip().lower() for v in zones) if z in tx.ZONES)


def _plain(text: Any, limit: int = 60) -> str:
    """Owner text that goes into the one-line header: no quotes or separators that could break the line."""
    return " ".join(re.sub(r"[\"'`;\r\n]", "", str(text or "")).split())[:limit]


@dataclass(frozen=True)
class Situation:
    camera: str
    ts: float
    time: str                 # "02:14", local
    phase: str
    dark: bool
    house_state: str          # home_awake / home_asleep / away (taxonomy words)
    camera_role: str
    intent: str
    house_detail: str = ""    # house_state's own word when it differs (vacation)
    zones: Tuple[str, ...] = ()
    expecting: Tuple[str, ...] = ()
    fact_covers: bool = False
    zone_facts: str = ""      # "ZONE FACTS (from code): ..." for the prompt, next to the header; "" without a map
    ground: str = ""          # whose ground the look happens on (taxonomy.GROUNDS), from the scene map
    crossed_in: bool = False  # someone crossed a boundary line onto the owner's ground
    scene_zone: str = ""      # where it happens by the scene map (taxonomy.ZONES); wins over the Eye's zone

    def header(self) -> str:
        expecting = ", ".join(f"'{t}'" for t in self.expecting) or "none"
        return (f"SITUATION: time {self.time}, {self.phase}, {'dark' if self.dark else 'light'}; "
                f"house: {self.house_state}; camera: {_plain(self.camera, 40)} ({self.camera_role}); "
                f"intent: {self.intent}; expecting: {expecting}")

    def to_taxonomy_context(self, movement: str = "", zone: str = "", flags: Sequence[str] = ()) -> tx.Context:
        return tx.Context(phase=self.phase, house_state=self.house_state, dark=self.dark,
                          camera_role=self.camera_role, expecting=bool(self.expecting),
                          fact_covers=self.fact_covers, movement=movement, zone=self.scene_zone or zone,
                          flags=tuple(flags), ground=self.ground, crossed_in=self.crossed_in)

    def record(self) -> Dict[str, Any]:
        """``situation`` in the training record and .meta.json (``scene`` only when the map said something)."""
        out: Dict[str, Any] = {"phase": self.phase, "dark": self.dark, "house_state": self.house_state,
                               "intent": self.intent, "camera_role": self.camera_role}
        if self.zone_facts:
            out["scene"] = {"zone_facts": self.zone_facts, "ground": self.ground, "crossed_in": self.crossed_in,
                            "zone": self.scene_zone}
        return out


def build_situation(camera: str, ts: float, intent: str = "alert_triage",
                    settings: Optional[Mapping[str, Any]] = None, house: Any = None,
                    facts: Sequence[Dict[str, Any]] = (), zones: Optional[Iterable[str]] = None,
                    state_path: Optional[str] = None, mute_path: Optional[str] = None,
                    scene_map: Any = None, scene_facts: Any = None) -> Situation:
    """The situation for one look at *camera* at local time *ts*.

    *house* is a ``house_state.HouseNow``; when None it is read with ``house_state.current`` (never raises).
    *facts* are the live house notes offered for this look; a ``lower`` note sets ``fact_covers``.
    *scene_map* (``scene_map.SceneMap``) feeds the role and zones when it says anything beyond today's drawn zone;
    *scene_facts* (``scene_map.SceneFacts``) is what it says about this look.
    """
    if scene_map is not None and not scene_map.informative:
        scene_map = None                # only today's drawn zone: the situation is as before
    ground = str(getattr(scene_facts, "ground", "") or "")
    scene_zone = str(getattr(scene_facts, "zone", "") or "")
    if intent not in tx.INTENTS:
        raise ValueError(f"intent must be one of {', '.join(tx.INTENTS)}")
    if house is None:
        from . import house_state  # noqa: PLC0415

        house = house_state.current(ts, path=state_path, mute_path=mute_path)
    phase, dark = phase_of(ts)
    notes = house.expecting_for(camera) if hasattr(house, "expecting_for") else []
    expecting = tuple(t for t in (_plain(n.get("text")) for n in notes) if t)
    return Situation(
        camera=str(camera), ts=float(ts), time=dt.datetime.fromtimestamp(ts).strftime("%H:%M"), phase=phase,
        dark=dark, house_state=house.taxonomy_state, camera_role=camera_role(camera, settings, scene_map),
        intent=intent, house_detail=house.state if house.state != house.taxonomy_state else "",
        zones=_zones(camera, settings, zones, scene_map), expecting=expecting,
        fact_covers=any(isinstance(f, dict) and f.get("effect") == "lower" for f in facts or ()),
        zone_facts=str(getattr(scene_facts, "line", "") or ""), ground=ground if ground in tx.GROUNDS else "",
        crossed_in=getattr(scene_facts, "crossed_in", False) is True,
        scene_zone=scene_zone if scene_zone in tx.ZONES else "",
    )
