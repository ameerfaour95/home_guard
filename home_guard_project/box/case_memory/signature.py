"""The event signature: what the case memory compares, built in code from the Eye and the tracker.

Inputs are plain dicts so the guard loop can pass what it has today:

- *observation*: the Eye's answer. The situation-aware schema (``category``, ``zone``, ``movement``, ``flags``,
  ``people``, ``vehicle_moving``, ``animals``, ``visibility``, ``raw_label``, ``label``, ``serious_behaviour``)
  and an optional ``appearance`` (clothing / bag / vehicle words). Today's gpt-4o answer has only the counts and
  labels; missing fields stay empty and the gates treat them conservatively.
- *tracker*: ``time_in_view_s`` (or ``dwell_s``), ``path`` (zone names in order), ``entry_edge``, ``exit_edge``,
  optional ``people`` / ``vehicles`` counts.
- *situation*: ``phase`` and ``house_state`` (situation.py; never guessed by a model).

The retrieval sentence (``template``) has a fixed shape built from those fields. The model's free summary is
never embedded: a different Eye model words things differently, and retrieval must not drift with it.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .. import taxonomy as tx
from .models import Signature

# Words that would describe the body rather than what the person wears or drives: never kept (privacy).
_BODY_WORDS = re.compile(r"\b(face|faces|facial|beard|moustache|mustache|skin|eyes?|tattoo\w*|tall|height|"
                         r"build|fat|thin|slim|gait|age|aged|old|young|elderly|teen\w*|child|boy|girl|man|woman|"
                         r"male|female|ethnic\w*|race|hair\w*|bald)\b", re.IGNORECASE)
MAX_APPEARANCE = 8


def _int(value: Any, default: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(0, number)


def _float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 and number == number and number != float("inf") else None


def _words(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        return value.split(",")
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return ()


def clean_appearance(value: Any) -> Tuple[str, ...]:
    """Lower-case clothing / bag / vehicle words, de-duplicated and sorted; body descriptions are dropped."""
    out = []
    for word in _words(value):
        text = " ".join(str(word).lower().split())[:40]
        if text and not _BODY_WORDS.search(text) and text not in out:
            out.append(text)
    return tuple(sorted(out)[:MAX_APPEARANCE])


def clean_path(value: Any) -> Tuple[str, ...]:
    """Zone names in order, lower case, consecutive repeats collapsed (``gate gate street`` -> ``gate>street``)."""
    if isinstance(value, str):
        value = value.split(">")
    out = []
    for item in value or ():
        zone = str(item or "").strip().lower()
        if zone and (not out or out[-1] != zone):
            out.append(zone)
    return tuple(out)


def dwell_bucket(dwell_s: Optional[float]) -> str:
    if dwell_s is None:
        return "unknown"
    if dwell_s < 15:
        return "short"
    if dwell_s < 60:
        return "medium"
    return "long"


def template(sig: Signature) -> str:
    """The fixed-shape English sentence embedded for retrieval. Same fields, same order, every time."""
    cat = tx.get(sig.category)
    category = f"{cat.id} {cat.name}" if cat else (sig.category or "unknown")
    parts = [
        f"camera={sig.camera}",
        f"category={category}",
        f"zone={sig.zone or 'unknown'}",
        f"path={'>'.join(sig.path) or 'unknown'}",
        f"movement={sig.movement or 'unknown'}",
        f"people={sig.people}",
        f"vehicles={sig.vehicles}",
        f"animals={sig.animals}",
        f"appearance={', '.join(sig.appearance) or 'none'}",
        f"dwell={dwell_bucket(sig.dwell_s)}",
    ]
    if sig.actions:                      # only when the words name one: older templates stay as they were
        parts.append(f"actions={', '.join(sig.actions)}")
    return "; ".join(parts)


def words_of(text: str) -> Dict[str, Any]:
    """What an alert's words (why + reason + summary) name, by activity_memory's patterns: ``actions``, ``ways``
    (the ways in) and ``blocked`` (why an explained action could never cover it). Empty for no text."""
    text = str(text or "").strip()
    if not text:
        return {"actions": (), "ways": (), "blocked": ""}
    from .. import activity_memory as am  # noqa: PLC0415

    return {"actions": tuple(am.actions_in(text)), "ways": tuple(p for p in am.places_in(text) if p in am.WAY_IN),
            "blocked": am.red_blocked(text)}


def build_signature(camera: str, ts: float, observation: Optional[Mapping[str, Any]] = None,
                    tracker: Optional[Mapping[str, Any]] = None, situation: Optional[Mapping[str, Any]] = None,
                    label: str = "", cameras_in_incident: int = 1, eye_model: str = "",
                    prompt_version: str = "", text: str = "") -> Signature:
    """The signature of one event. *text* is the alert's words (why + reason + summary): the actions and ways in
    they name (activity_memory), for the cases that remember an explained action. *label* is the box's label after the priors (``decision["final_label"]``);
    when empty, the Eye's ``label`` is used.

    *observation* may be the Eye's processed answer (``eye_prompt.postprocess``) as is: the cleaned values in its
    nested ``observation`` win over the model's raw top-level copies, its nested ``situation`` is used when
    *situation* is not given, and the flags are the union of both lists, so the veto never sees fewer."""
    top = dict(observation or {})
    nested = top.get("observation") if isinstance(top.get("observation"), Mapping) else {}
    obs = {**top, **nested}
    trk = dict(tracker or {})
    sit = dict(situation or (top.get("situation") if isinstance(top.get("situation"), Mapping) else None) or {})
    flag_lists = [x.get("flags") for x in (top, nested)]
    moment = datetime.fromtimestamp(ts)
    category = obs.get("category")
    category = tx.normalize_id(category) if category not in (None, "") else ""
    path = clean_path(trk.get("path") or trk.get("zones") or ())
    zone = str(obs.get("zone") or "").strip().lower()   # one Eye zone is not a path: the path gate needs the tracker
    vehicles = trk.get("vehicles")
    if vehicles is None:
        vehicles = obs.get("vehicles")
    if vehicles is None:
        vehicles = 1 if obs.get("vehicle_moving") else 0
    people = trk.get("people") if trk.get("people") is not None else obs.get("people")
    flags = tuple(sorted({str(f).strip().lower() for fl in flag_lists if isinstance(fl, (list, tuple))
                          for f in fl if str(f).strip()}))
    final = str(label or obs.get("label") or "").strip().lower()
    sig = Signature(
        camera=str(camera), ts=float(ts), minute=moment.hour * 60 + moment.minute, weekday=moment.weekday(),
        phase=str(sit.get("phase") or ""), house_state=str(sit.get("house_state") or "home_awake"),
        category=category, movement=str(obs.get("movement") or "").strip().lower(), zone=zone, path=path,
        entry_edge=str(trk.get("entry_edge") or (path[0] if path else "")).strip().lower(),
        exit_edge=str(trk.get("exit_edge") or (path[-1] if path else "")).strip().lower(),
        dwell_s=_float(trk.get("time_in_view_s", trk.get("dwell_s"))),
        people=_int(people), vehicles=_int(vehicles), animals=_int(obs.get("animals")), flags=flags,
        appearance=clean_appearance(obs.get("appearance")), label=final,
        serious_behaviour=obs.get("serious_behaviour") is True, cameras_in_incident=max(1, _int(cameras_in_incident, 1)),
        eye_model=str(eye_model or ""), prompt_version=str(prompt_version or ""), **words_of(text),
    )
    return Signature(**{**sig.__dict__, "template": template(sig)})


def kinds(sig: Signature) -> Tuple[str, ...]:
    """people / vehicles / animals present in the event."""
    out = []
    if sig.people:
        out.append("people")
    if sig.vehicles:
        out.append("vehicles")
    if sig.animals:
        out.append("animals")
    return tuple(out)


def describe_path(path: Sequence[str]) -> str:
    return ">".join(path)
