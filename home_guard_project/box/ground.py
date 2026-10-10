"""Whose ground it happened on: the scene map's say in whether an alert reaches the owner. Code, never the model.

Owner, 2026-10-08 (scene map stage 2c): the box knows from the owner's map which part of the picture is ours and
which is the neighbour's or the street. So:

- A person whose whole visit stays on the neighbour's or public ground is not a message, unless the label is
  suspicious for something they DO (a hand on a car door, climbing, looking in) or an escalation.
- A person who crosses a boundary line inward (the neighbour's side -> ours), or walks into an area of ours from
  the neighbour's or the street, is worth a message even when the Eye said normal: "נכנס מהצד של השכן אל השטח שלנו".

``ground_of(tracks, scene)`` is pure: the tracker's foot points over the alert's window
(``tracker.CameraTracker.tracks_between``) against the camera's scene map, with the tracker's own inertia (a person
is in an area, or across a line, only after a few looks in a row there). It is cautious: anyone on our ground makes
it ``mine``; anyone the map cannot place keeps it unknown; only then the neighbour's, else public. No map beyond
today's drawn zone, or nobody tracked, is ``UNKNOWN``: everything goes as before.

``events.EventBook.decide(..., ground=Ground.record())`` applies the policy; ``is_action`` says whether the Eye's
reason names something done (alert_guards.ACTION without the words for a place or the hour: "at night", "near the
entrance" and "loitering" are where and when, not what).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from . import scene_map as sm

WHOLE_SHARE = 0.8          # this much of a person's foot points on someone else's ground is "their whole visit"


@dataclass(frozen=True)
class Ground:
    on: str = ""                     # mine | neighbour | public | "" (unknown: no map, or someone it cannot place)
    crossed_inward: bool = False     # a boundary line crossed towards our ground
    line: str = ""                   # the first line crossed inward
    entered_from: str = ""           # the neighbour's / public area someone walked from into an area of ours
    from_ground: str = ""            # whose ground that area is (neighbour | public)
    people: int = 0                  # people tracks judged
    placed: tuple = ()               # how their foot points were placed: (("area", n), ("near_area", n), ...)

    @property
    def entered(self) -> bool:
        """Someone came onto our ground from the neighbour's side or the street."""
        return self.crossed_inward or bool(self.entered_from)

    @property
    def off_our_ground(self) -> bool:
        """Everyone stayed on the neighbour's or public ground."""
        return self.on in (sm.NEIGHBOUR, sm.PUBLIC) and not self.entered

    def record(self) -> Dict[str, Any]:
        return {"on": self.on, "crossed_inward": self.crossed_inward, "line": self.line,
                "entered_from": self.entered_from, "from_ground": self.from_ground, "people": self.people,
                "entered": self.entered, "off_our_ground": self.off_our_ground, "placed": dict(self.placed)}


UNKNOWN = Ground()


def ground_of(tracks: Sequence[Any], scene: Any) -> Ground:
    """Where the people of these tracks were, by the camera's scene map (see the module docstring)."""
    from .tracker import _area_runs, _timed_crossings  # noqa: PLC0415 - the tracker's own inertia

    if scene is None or not getattr(scene, "informative", False):
        return UNKNOWN
    people = [t for t in tracks or () if getattr(t, "kind", "") == "person" and getattr(t, "points", None)]
    if not people:
        return UNKNOWN
    ons = []
    crossed, line, entered_from, from_ground = False, "", "", ""
    for track in people:
        points = sorted(track.points)
        runs = _area_runs("person", points, scene)
        for name, way, _ts in _timed_crossings("person", points, scene):
            if way == sm.IN and not crossed:
                crossed, line = True, name
        for prev, cur in zip(runs, runs[1:]):
            if cur[0].ground == sm.MINE and prev[0].ground in (sm.NEIGHBOUR, sm.PUBLIC) and not entered_from:
                entered_from, from_ground = prev[0].name, prev[0].ground
        grounds = {run[0].ground for run in runs}
        if sm.MINE in grounds:
            ons.append(sm.MINE)
            continue
        placed = [scene.place_at((x, y)) for _, x, y in points]
        off = [a for a in placed if a is not None and a.ground in (sm.NEIGHBOUR, sm.PUBLIC)]
        if runs and len(off) >= WHOLE_SHARE * len(points):
            ons.append(sm.NEIGHBOUR if any(a.ground == sm.NEIGHBOUR for a in off) else sm.PUBLIC)
        else:
            ons.append("")
    on = next(g for g in (sm.MINE, "", sm.NEIGHBOUR, sm.PUBLIC) if g in ons)
    how: Dict[str, int] = {}
    for track in people:
        for _, x, y in track.points:
            way = scene.ground_at((x, y))[1]
            how[way] = how.get(way, 0) + 1
    return Ground(on=on, crossed_inward=crossed, line=line, entered_from=entered_from, from_ground=from_ground,
                  people=len(people), placed=tuple(sorted(how.items())))


def from_record(record: Optional[Dict[str, Any]]) -> Ground:
    """A Ground from its record (``Ground.record``); UNKNOWN for anything else."""
    if not isinstance(record, dict):
        return UNKNOWN
    try:
        return Ground(on=str(record.get("on") or ""), crossed_inward=record.get("crossed_inward") is True,
                      line=str(record.get("line") or ""), entered_from=str(record.get("entered_from") or ""),
                      from_ground=str(record.get("from_ground") or ""), people=int(record.get("people") or 0))
    except (TypeError, ValueError):
        return UNKNOWN


# ---------- what they do, not where or when ----------
_ACTION = re.compile("|".join(f"(?:{p})" for p in [
    r"\bhandles?\b", r"\bdoors?\b", r"\bwindows?\b", r"\bgates?\b", r"\bfence", r"\blocks?\b", r"\bclimb",
    r"\bhid(?:e|es|ing)\b(?! (?:his|her|their) faces?)", r"\bcrouch", r"\bsteal", r"\bstole",
    r"\btak(?:e|es|ing|en)\b", r"\btook\b", r"\btamper", r"\bpeek", r"\bpeer", r"\blook(?:s|ed|ing)? (?:in|into|inside)\b",
    r"\bbreak", r"\bforc", r"\bpry", r"\bopen(?:s|ed|ing)?\b", r"\btr(?:y|ies|ied|ying) to\b",
    r"\b(?:cover|block|turn|mov|spray|paint|point|push|hit)\w* (?:the |a |at the )?camera",
    r"ידית", r"דלת", r"חלון", r"שער", r"גדר", r"מנעול", r"טיפוס", r"מטפס", r"מסתתר", r"מתחבא", r"כורע", r"גונב",
    r"לוקח", r"לקח", r"מציץ", r"פוגע במצלמה", r"פורץ", r"פותח", r"מנסה",
]), re.IGNORECASE)


def is_action(text: str) -> bool:
    """The Eye's reason names something a person does with hands or feet (alert_guards.ACTION without "night",
    "entrance" and "loitering", which say where or when)."""
    return bool(_ACTION.search(str(text or "")))


# ---------- what the owner reads ----------
def entered_text(ground: Ground, lang: str = "he") -> str:
    """One line for the alert: who came onto our ground and from where; "" when nobody did."""
    if not ground.entered:
        return ""
    he = str(lang).startswith("he")
    if ground.from_ground == sm.PUBLIC:
        text = "נכנס מהרחוב אל השטח שלנו" if he else "came in from the street onto our ground"
    elif ground.from_ground == sm.NEIGHBOUR:
        text = "נכנס מהצד של השכן אל השטח שלנו" if he else "came in from the neighbour's side onto our ground"
    else:
        return (f"נכנס אל השטח שלנו דרך {ground.line}" if he else f"came onto our ground across {ground.line}")
    if ground.line:
        text += f" (דרך {ground.line})" if he else f" (across {ground.line})"
    return text
