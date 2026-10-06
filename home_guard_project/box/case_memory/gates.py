"""Veto and hard gates: plain code that decides when memory may not be used at all.

The veto runs first. When it fires, memory is not consulted for this event: no memory ever cancels a suspicious
sign, a call or an escalation. Then every case's scope is checked field by field; one failed gate means no match.
Each failure is a short English reason the owner can read ("23:10 is outside 07:10-08:10").
"""
from __future__ import annotations

from typing import List, Sequence

from .. import taxonomy as tx
from .models import Case, Scope, Signature, hhmm, minutes_of, window_minutes

# Any of these on the Eye's flags vetoes memory (report: touching a handle, a tool, carrying away, covered face,
# flashlight, crouching, running, a visible weapon).
RISK_FLAGS = frozenset({"touching_handle", "tool_in_hand", "item_carried_away", "face_covered", "flashlight",
                        "crouching", "running", "weapon_visible"})

# Categories compared by family, so a new Eye model that splits N1/N2 differently does not break old cases.
FAMILIES = {"N1": "transit", "N2": "transit", "N3": "door", "N4": "door", "N5": "work", "N6": "household",
            "N7": "vehicle", "N8": "animals", "N9": "guard", "N10": "nothing"}

NIGHT_PHASE = "late_night"


def veto(sig: Signature, label: str = "", alert_command: str = "") -> List[str]:
    """Why memory must not be consulted for this event (empty list: it may be)."""
    reasons = []
    final = (label or sig.label or "").lower()
    if final == "escalation" or alert_command == "[call_owner]":
        reasons.append("escalation or call")
    group = tx.group_of(sig.category)
    if group in ("S", "E"):
        reasons.append(f"category {sig.category}")
    risky = sorted(set(sig.flags) & RISK_FLAGS)
    if risky:
        reasons.append("flags " + ",".join(risky))
    if sig.serious_behaviour:
        reasons.append("serious behaviour")
    if sig.cameras_in_incident >= 2 and (sig.phase == NIGHT_PHASE or sig.house_state in ("away", "home_asleep")):
        reasons.append("multi-camera incident at night or while away")
    return reasons


def circular_distance(a: int, b: int) -> int:
    d = abs(a - b) % 1440
    return min(d, 1440 - d)


def in_window(minute: int, hours: Sequence[str], margin: int = 0) -> bool:
    start, length = window_minutes(hours)
    if length >= 1440:
        return True
    return (minute - (start - margin)) % 1440 <= length + 2 * margin


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def path_distance(a: Sequence[str], b: Sequence[str]) -> float:
    """Edit distance over zone names, normalised to 0..1 by the longer path."""
    if not a and not b:
        return 0.0
    return edit_distance(a, b) / max(len(a), len(b))


def family(category: str) -> str:
    return FAMILIES.get(category, "")


def gate_failures(case: Case, sig: Signature, hour_margin_min: int = 15, max_path_distance: float = 0.34) -> List[str]:
    """Every gate *sig* fails for *case* (empty list: it passes all of them)."""
    s: Scope = case.scope
    out = []
    if not case.live:
        out.append(f"case is {case.status}")
    if sig.camera != s.camera:
        out.append(f"camera {sig.camera} is not {s.camera}")
    if not in_window(sig.minute, s.hours, hour_margin_min):
        out.append(f"{hhmm(sig.minute)} is outside {s.hours[0]}-{s.hours[1]}")
    if sig.weekday not in s.weekdays:
        out.append(f"weekday {sig.weekday} is not in scope")
    if sig.house_state not in s.house_states:
        out.append(f"house is {sig.house_state}")
    if sig.phase == NIGHT_PHASE and not s.night:
        out.append("night is not in scope")
    if sig.people != s.people:
        out.append(f"{sig.people} people, the case has {s.people}")
    if sig.vehicles != s.vehicles:
        out.append(f"{sig.vehicles} vehicles, the case has {s.vehicles}")
    if sig.category == tx.OTHER:
        out.append("category other")
    elif s.categories:
        wanted = {family(c) for c in s.categories}
        if not sig.category or family(sig.category) not in wanted:
            out.append(f"category {sig.category or 'unknown'} is not {'/'.join(s.categories)}")
    if s.path:
        if not sig.path:
            out.append("no path from the tracker")
        else:
            if s.entry_edge and sig.entry_edge != s.entry_edge:
                out.append(f"entered at {sig.entry_edge or 'unknown'}, not {s.entry_edge}")
            if s.exit_edge and sig.exit_edge != s.exit_edge:
                out.append(f"left at {sig.exit_edge or 'unknown'}, not {s.exit_edge}")
            if path_distance(sig.path, s.path) > max_path_distance:
                out.append(f"path {'>'.join(sig.path)} is not {'>'.join(s.path)}")
    if s.max_dwell_s is not None:
        if sig.dwell_s is None:
            out.append("no time in view from the tracker")
        elif sig.dwell_s > s.max_dwell_s:
            out.append(f"stayed {sig.dwell_s:.0f}s, more than {s.max_dwell_s:.0f}s")
    return out


def default_max_dwell(dwells: Sequence[float]) -> float:
    """Twice the longest example's time in view, at least ten seconds more than it."""
    longest = max(dwells)
    return max(2 * longest, longest + 10)


def hours_around(minute: int, before: int = 30, after: int = 30, round_to: int = 10) -> tuple:
    """A window around *minute*, rounded outward to *round_to* minutes: 07:40 -> ("07:10", "08:10")."""
    start = ((minute - before) // round_to) * round_to
    end = -((-(minute + after)) // round_to) * round_to
    return hhmm(start), hhmm(end)


__all__ = ["RISK_FLAGS", "FAMILIES", "veto", "gate_failures", "in_window", "path_distance", "edit_distance",
           "circular_distance", "family", "default_max_dwell", "hours_around", "minutes_of"]
