"""Routine proposals: the Historian notices a repeat and asks the owner, before anyone explains it.

Nightly, over the last 28 days, per camera and path: ordinary events (label normal, no risk flag, no S/E category,
no serious behaviour, not at night) are clustered by time of day. A group seen on at least 6 of the last 14 days
within about half an hour is proposed: "I see someone going from the gate to the street almost every day around
07:40. Is this a routine?". Nothing changes until the owner says yes; then it becomes a case in shadow like any
other. A "no" is remembered (a labelled negative) and never proposed again. The thresholds are starting points.

This replaces facts.py's proposal band of ten hours (PROPOSAL_BAND_HOURS), far too wide for a rule that silences.
"""
from __future__ import annotations

import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import texts
from .gates import RISK_FLAGS, circular_distance, hours_around
from .models import DIGEST, WORKWEEK, Case, Example, Scope, Signature
from .signature import build_signature, clean_path
from .. import taxonomy as tx

HISTORY_DAYS = 28
RECENT_DAYS = 14
MIN_DAYS = 6
SPAN_MIN = 30
NIGHT_HOURS = range(0, 5)          # when the record has no phase


def record_signature(rec: Mapping[str, Any]) -> Signature:
    """A stored event record (the guard loop's decision record plus observation/tracker) as a signature."""
    if isinstance(rec.get("signature"), dict):
        return Signature.from_dict(rec["signature"])
    obs = {k: rec.get(k) for k in ("category", "flags", "people", "animals", "vehicle_moving", "appearance",
                                   "movement", "zone", "serious_behaviour")}
    obs.update(rec.get("observation") or {})
    trk = dict(rec.get("tracker") or {})
    if rec.get("path") and "path" not in trk:
        trk["path"] = rec["path"]
    sit = {"phase": rec.get("phase", ""), "house_state": rec.get("house_state", "home_awake")}
    sit.update(rec.get("situation") or {})
    label = str(rec.get("final_label") or rec.get("label") or "")
    return build_signature(str(rec["camera"]), float(rec["ts"]), obs, trk, sit, label)


def eligible(rec: Mapping[str, Any], sig: Signature) -> bool:
    if sig.label != "normal" or rec.get("alert_command") == "[call_owner]":
        return False
    if set(sig.flags) & RISK_FLAGS or sig.serious_behaviour or tx.group_of(sig.category) in ("S", "E"):
        return False
    if sig.phase == "late_night" or (not sig.phase and sig.minute // 60 in NIGHT_HOURS):
        return False
    return bool(sig.people or sig.vehicles)


def _date(ts: float) -> str:
    return datetime.fromtimestamp(ts).date().isoformat()


def _best_cluster(points: List[Tuple[int, float, Dict[str, Any]]], recent_from: float) -> Optional[List[Any]]:
    """The window of SPAN_MIN minutes (circular) holding the most distinct recent days."""
    best, best_days = None, 0
    for m0, _, _ in points:
        inside = [p for p in points if (p[0] - m0) % 1440 <= SPAN_MIN]
        days = {_date(p[1]) for p in inside if p[1] >= recent_from}
        if len(days) > best_days:
            best, best_days = inside, len(days)
    return best


def _covered(store, camera: str, path: Tuple[str, ...], minute: int) -> bool:
    from .gates import in_window  # noqa: PLC0415

    for case in store.live_cases(camera):
        if case.scope.path == path and in_window(minute, case.scope.hours, SPAN_MIN):
            return True
    return False


def _answered(answers: Mapping[str, Mapping[str, Any]], camera: str, path: Tuple[str, ...], minute: int) -> bool:
    for key in answers:
        cam, _, rest = key.partition("|")
        p, _, m = rest.rpartition("|")
        if cam == camera and p == ">".join(path) and m.isdigit() and circular_distance(int(m), minute) <= SPAN_MIN:
            return True
    return False


def propose(records: Iterable[Mapping[str, Any]], store=None, now: Optional[float] = None,
            min_days: int = MIN_DAYS) -> List[Dict[str, Any]]:
    """Routine proposals from the last 28 days of event records. Skips what a live case already covers and
    what the owner already answered."""
    now = time.time() if now is None else now
    since, recent_from = now - HISTORY_DAYS * 86400, now - RECENT_DAYS * 86400
    groups: Dict[Tuple[str, Tuple[str, ...]], List[Tuple[int, float, Dict[str, Any]]]] = {}
    for rec in records:
        try:
            sig = record_signature(rec)
        except (KeyError, TypeError, ValueError):
            continue
        if sig.ts < since or sig.ts > now or not eligible(rec, sig):
            continue
        groups.setdefault((sig.camera, sig.path), []).append((sig.minute, sig.ts, {"rec": rec, "sig": sig}))
    answers = store.routine_answers() if store is not None else {}
    out = []
    for (camera, path), points in sorted(groups.items()):
        remaining = list(points)
        while remaining:
            cluster = _best_cluster(remaining, recent_from)
            if not cluster:
                break
            days = {_date(p[1]) for p in cluster if p[1] >= recent_from}
            if len(days) < min_days:
                break
            taken = {id(p) for p in cluster}
            remaining = [p for p in remaining if id(p) not in taken]
            minutes = sorted(p[0] for p in cluster)
            base = minutes[0]
            centre = (base + sorted((m - base) % 1440 for m in minutes)[len(minutes) // 2]) % 1440
            if (store is not None and _covered(store, camera, path, centre)) or _answered(answers, camera, path, centre):
                continue
            out.append(_proposal(camera, path, centre, cluster, days))
    return out


def _proposal(camera: str, path: Tuple[str, ...], centre: int, cluster: List[Any], days: set) -> Dict[str, Any]:
    seen = {datetime.fromtimestamp(p[1]).weekday() for p in cluster}
    # Days it was seen; spread over three or more workdays (and only workdays) reads as the Sun-Thu week.
    weekdays = tuple(sorted(WORKWEEK)) if seen <= set(WORKWEEK) and len(seen) >= 3 else tuple(sorted(seen))
    people = Counter(p[2]["sig"].people for p in cluster).most_common(1)[0][0]
    vehicles = Counter(p[2]["sig"].vehicles for p in cluster).most_common(1)[0][0]
    examples = sorted(cluster, key=lambda p: p[1])[-3:]
    clock = f"{centre // 60:02d}:{centre % 60:02d}"
    where_he = texts.path_sentence(path, "he") or f"במצלמת {camera}"
    where_en = texts.path_sentence(path, "en") or f"at camera {camera}"
    who_he = "רכב" if vehicles and not people else "מישהו"
    who_en = "a vehicle" if vehicles and not people else "someone"
    return {
        "key": f"{camera}|{'>'.join(path)}|{centre}", "camera": camera, "path": list(path), "minute": centre,
        "hours": list(hours_around(centre)), "weekdays": list(weekdays), "days_seen": len(days),
        "people": people, "vehicles": vehicles,
        "examples": [{"event_id": str(p[2]["rec"].get("event_id") or p[2]["rec"].get("alert_id") or ""),
                      "signature": p[2]["sig"].to_dict()} for p in examples],
        "text_he": f"אני רואה {who_he} {where_he} ב-{len(days)} מתוך {RECENT_DAYS} הימים האחרונים, בסביבות {clock}. "
                   "זו שגרה?",
        "text_en": f"I see {who_en} {where_en} on {len(days)} of the last {RECENT_DAYS} days, around {clock}. "
                   "Is this a routine?",
    }


def proposal_case(proposal: Mapping[str, Any]) -> Case:
    """The case an accepted proposal becomes (it starts in shadow, like every case)."""
    examples = [Example(e["event_id"], Signature.from_dict(e["signature"])) for e in proposal.get("examples", [])]
    first = examples[0].signature if examples else None
    dwells = [e.signature.dwell_s for e in examples if e.signature.dwell_s is not None]
    from .gates import default_max_dwell  # noqa: PLC0415

    scope = Scope(camera=proposal["camera"], hours=tuple(proposal["hours"]), weekdays=tuple(proposal["weekdays"]),
                  house_states=("home_awake",), night=False, people=int(proposal["people"]),
                  vehicles=int(proposal["vehicles"]), path=clean_path(proposal.get("path") or ()),
                  entry_edge=first.entry_edge if first else "", exit_edge=first.exit_edge if first else "",
                  categories=tuple(sorted({e.signature.category for e in examples
                                           if e.signature.category and e.signature.category != "other"})),
                  max_dwell_s=default_max_dwell(dwells) if dwells and len(dwells) == len(examples) else None)
    return Case(id="", scope=scope, who="other", effect=DIGEST, examples=examples,
                created_at=time.time(), source={"origin": "routine", "key": proposal["key"]})


def answer(store, proposal: Mapping[str, Any], yes: bool, by: str,
           embed=None) -> Optional[Case]:
    """The owner's answer. Yes: a new case in shadow. No: remembered, never proposed again."""
    if not by:
        raise ValueError("only the owner answers a routine proposal")
    if not yes:
        store.answer_routine(proposal["key"], False, by)
        return None
    from .keeper import with_embedding  # noqa: PLC0415

    case = proposal_case(proposal)
    case.examples = [with_embedding(e, embed) for e in case.examples]
    saved = store.add(case, by)
    store.answer_routine(proposal["key"], True, by, saved.id if saved else "")
    return saved


__all__: Sequence[str] = ("propose", "answer", "proposal_case", "record_signature", "eligible")
