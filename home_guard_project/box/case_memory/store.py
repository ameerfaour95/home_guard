"""The case memory's storage: an append-only journal folded on every read, behind a small adapter.

Today the journal is ``<live_dir>/.registry/cases.jsonl`` (``JsonlBackend``). The plan moves cases into the house
facts store once ``brain/facts.py`` merges; that only needs another :class:`CaseBackend` (two methods) that reads
and appends these same event dicts in the facts journal. Nothing above this module changes.

Rules the store enforces whatever the caller does:
- History is never deleted. Invalidating sets ``invalid_at`` and the reason; merging invalidates the source.
- Only an owner action (``by`` names who) confirms a case. Automatic matches only update ``last_seen_at``.
- A narrowing applies at once; a widening is a proposal until the owner approves it.
- One process-wide lock covers read, id allocation and append, so the guard loop and the Telegram inbox can share
  the file.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple

from .ladder import SHADOW_STAGE, TrustLadder
from .models import (ACTIVE, ALERT, EFFECTS, INVALID, MAX_EXAMPLES, PAUSED, SHADOW, Case, Example, Expecting, Scope,
                     window_minutes)

log = logging.getLogger("box.case_memory.store")

FILE_NAME = "cases.jsonl"
_LOCK = threading.RLock()


class CaseBackend(Protocol):
    """Where the journal lives. ``read_events`` returns every event dict in order; ``append`` adds one."""

    def read_events(self) -> List[Dict[str, Any]]: ...

    def append(self, event: Dict[str, Any]) -> bool: ...


class JsonlBackend:
    """One JSON object per line. Damaged lines are skipped with one warning per read."""

    def __init__(self, path: str) -> None:
        self.path = path

    def read_events(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        damaged = False
        try:
            with open(self.path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        if not isinstance(item, dict) or not isinstance(item.get("event"), str):
                            raise ValueError("not an event")
                        out.append(item)
                    except (ValueError, RecursionError):
                        damaged = True
        except FileNotFoundError:
            return []
        except OSError as exc:
            log.warning("Case memory could not be read: %s", exc)
            return []
        if damaged:
            log.warning("Skipped damaged lines in the case memory file")
        return out

    def append(self, event: Dict[str, Any]) -> bool:
        try:
            line = json.dumps(event, ensure_ascii=False, allow_nan=False)
            with _LOCK:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            return True
        except (OSError, TypeError, ValueError, RecursionError) as exc:
            log.warning("Case memory change not saved: %s", exc)
            return False


class MemoryBackend:
    """In-memory journal (tests, the evaluation)."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def read_events(self) -> List[Dict[str, Any]]:
        return [json.loads(json.dumps(e)) for e in self.events]

    def append(self, event: Dict[str, Any]) -> bool:
        self.events.append(json.loads(json.dumps(event, allow_nan=False)))
        return True


def default_path(live_dir: Optional[str] = None) -> str:
    """The journal: in the box's state folder, or ``<live_dir>/.registry`` for any other alert folder."""
    from .. import paths  # noqa: PLC0415

    state_dir, _ = paths.state_paths_for(live_dir or paths.production_dir())
    return os.path.join(state_dir, FILE_NAME)


# -- scope comparisons ---------------------------------------------------------------------------------------------

def window_inside(inner: Tuple[str, str], outer: Tuple[str, str]) -> bool:
    """True when the *inner* hours window lies within *outer* (both may cross midnight)."""
    o_start, o_len = window_minutes(outer)
    i_start, i_len = window_minutes(inner)
    if o_len >= 1440:
        return True
    return (i_start - o_start) % 1440 + i_len <= o_len


def narrows(old: Scope, new: Scope) -> bool:
    """True when every event *new* admits, *old* admitted too (a narrowing is always safe to apply)."""
    if new.camera != old.camera or new.people != old.people or new.vehicles != old.vehicles:
        return False
    if not window_inside(new.hours, old.hours):
        return False
    if not set(new.weekdays) <= set(old.weekdays) or not set(new.house_states) <= set(old.house_states):
        return False
    if new.night and not old.night:
        return False
    for key in ("path", "entry_edge", "exit_edge"):
        if getattr(old, key) and getattr(new, key) != getattr(old, key):
            return False
    if old.categories and not (new.categories and set(new.categories) <= set(old.categories)):
        return False
    if old.max_dwell_s is not None and (new.max_dwell_s is None or new.max_dwell_s > old.max_dwell_s):
        return False
    return True


def scope_diff(old: Scope, new: Scope) -> Dict[str, Tuple[Any, Any]]:
    """Field -> (old, new) for every field that changed (shown to the owner before a widening)."""
    a, b = old.to_dict(), new.to_dict()
    return {k: (a[k], b[k]) for k in a if a[k] != b.get(k)}


def end_of_day(ts: float) -> float:
    moment = datetime.fromtimestamp(ts)
    return (datetime(moment.year, moment.month, moment.day) + timedelta(days=1)).timestamp() - 1


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("not finite")
    return number


def _keep_examples(examples: List[Example]) -> List[Example]:
    """The founding example plus the newest ones, at most MAX_EXAMPLES."""
    if len(examples) <= MAX_EXAMPLES:
        return examples
    return [examples[0]] + examples[-(MAX_EXAMPLES - 1):]


class CaseStore:
    """Cases, widening proposals, expecting notes and answered routine proposals, folded from the journal."""

    def __init__(self, backend: CaseBackend, now: Callable[[], float] = time.time,
                 ladder: TrustLadder = TrustLadder()) -> None:
        self.backend = backend
        self._now = now
        self.ladder = ladder

    @classmethod
    def at(cls, path: str, **kwargs: Any) -> "CaseStore":
        return cls(JsonlBackend(path), **kwargs)

    # -- journal --------------------------------------------------------------------------------------------------
    def _append(self, event: str, **fields: Any) -> bool:
        return self.backend.append({"event": event, "ts": _finite(self._now()), **fields})

    def _status(self, case: Case, paused: bool) -> str:
        if case.invalid_at is not None:
            return INVALID
        if paused:
            return PAUSED
        stage = self.ladder.stage(case.streak, case.contradictions)
        return SHADOW if stage == SHADOW_STAGE else ACTIVE

    def _fold(self) -> Dict[str, Any]:
        cases: Dict[str, Case] = {}
        paused: Dict[str, bool] = {}
        widen: Dict[str, Dict[str, Any]] = {}
        expecting: Dict[str, Dict[str, Any]] = {}
        routines: Dict[str, Dict[str, Any]] = {}
        matches: List[Dict[str, Any]] = []
        damaged = False
        for item in self.backend.read_events():
            event = item.get("event")
            try:
                ts = _finite(item.get("ts"))
                if event == "case_add":
                    case = Case.from_dict(item["case"])
                    if case.id in cases:
                        raise ValueError("reused case id")
                    case.created_at = case.created_at or ts
                    cases[case.id] = case
                    paused[case.id] = False
                elif event == "expecting_add":
                    note = Expecting.from_dict(item["note"])
                    expecting[note.id] = {"note": note, "cancelled": False}
                elif event == "expecting_cancel":
                    if item.get("id") in expecting:
                        expecting[item["id"]]["cancelled"] = True
                elif event == "routine_answer":
                    routines[str(item["key"])] = {"yes": bool(item.get("yes")), "ts": ts,
                                                  "case_id": item.get("case_id", "")}
                elif event == "widen_answer":
                    proposal = widen.get(str(item.get("wid")))
                    if proposal is None or proposal["answer"]:
                        continue
                    proposal["answer"] = "yes" if item.get("yes") else "no"
                    case = cases.get(proposal["id"])
                    if item.get("yes") and case is not None and case.invalid_at is None:
                        case.scope = Scope.from_dict(proposal["scope"])
                        case.revision += 1
                else:
                    case = cases.get(str(item.get("id")))
                    if case is None:
                        continue
                    if event == "confirm":
                        case.confirmations += 1
                        case.streak += 1
                        case.last_confirmed_at = case.last_seen_at = ts
                        paused[case.id] = False
                        if item.get("example"):
                            example = Example.from_dict(item["example"])
                            if all(e.event_id != example.event_id for e in case.examples):
                                case.examples = _keep_examples(case.examples + [example])
                    elif event == "contradict":
                        case.streak = self.ladder.step_back(case.streak, case.contradictions)
                        case.contradictions += 1
                        if item.get("event_id"):
                            case.negatives.append(str(item["event_id"]))
                    elif event == "matched":
                        case.last_seen_at = ts
                        matches.append(dict(item))
                        continue                      # a match is a log line: no revision bump
                    elif event == "set_effect":
                        if item.get("effect") not in EFFECTS:
                            raise ValueError("bad effect")
                        case.effect = item["effect"]
                    elif event == "narrow":
                        case.scope = Scope.from_dict(item["scope"])
                    elif event == "widen_proposed":
                        widen[str(item["wid"])] = {"wid": str(item["wid"]), "id": case.id, "scope": item["scope"],
                                                   "by": item.get("by", ""), "ts": ts, "answer": ""}
                        continue
                    elif event == "invalidate":
                        case.invalid_at = ts
                        case.invalid_reason = str(item.get("reason") or "")
                        case.merged_into = str(item.get("merged_into") or "")
                    elif event == "restore":
                        case.invalid_at, case.invalid_reason, case.merged_into = None, "", ""
                    elif event == "review_asked":
                        case.review_asked_at = ts
                        paused[case.id] = True
                    elif event == "review_answer":
                        if item.get("keep"):
                            paused[case.id] = False
                            case.last_seen_at = ts
                        else:
                            case.invalid_at = ts
                            case.invalid_reason = "no longer relevant (owner)"
                    elif event == "merge_in":
                        case.scope = Scope.from_dict(item["scope"])
                        extra = [Example.from_dict(e) for e in item.get("examples") or []]
                        known = {e.event_id for e in case.examples}
                        case.examples = _keep_examples(case.examples + [e for e in extra if e.event_id not in known])
                        case.confirmations += int(item.get("confirmations") or 0)
                    else:
                        continue
                    case.revision += 1
            except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
                damaged = True
        if damaged:
            log.warning("Skipped damaged case memory events")
        for case in cases.values():
            case.status = self._status(case, paused.get(case.id, False))
        return {"cases": cases, "widen": widen, "expecting": expecting, "routines": routines, "matches": matches}

    # -- reads ----------------------------------------------------------------------------------------------------
    def cases(self, camera: Optional[str] = None, include_invalid: bool = False) -> List[Case]:
        with _LOCK:
            state = self._fold()
        out = [c for c in state["cases"].values() if camera is None or c.camera == camera]
        return [c for c in out if include_invalid or c.status != INVALID]

    def live_cases(self, camera: str) -> List[Case]:
        """Cases that may match on *camera* now: shadow or active (paused and invalid never match)."""
        return [c for c in self.cases(camera) if c.live]

    def get(self, case_id: str) -> Optional[Case]:
        with _LOCK:
            return self._fold()["cases"].get(str(case_id))

    def pending_widenings(self, case_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with _LOCK:
            widen = self._fold()["widen"]
        return [w for w in widen.values() if not w["answer"] and (case_id is None or w["id"] == case_id)]

    def matches(self, case_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """The logged automatic matches (shadow "would have silenced" and real softenings)."""
        with _LOCK:
            items = self._fold()["matches"]
        return [m for m in items if case_id is None or m.get("id") == case_id]

    def expecting_for(self, camera: str, ts: Optional[float] = None) -> List[Expecting]:
        now = self._now() if ts is None else ts
        with _LOCK:
            notes = self._fold()["expecting"].values()
        return [n["note"] for n in notes if not n["cancelled"] and n["note"].camera == camera
                and n["note"].created_at <= now <= n["note"].until]

    def routine_answers(self) -> Dict[str, Dict[str, Any]]:
        with _LOCK:
            return self._fold()["routines"]

    # -- writes ---------------------------------------------------------------------------------------------------
    def _next_id(self, prefix: str, taken: Iterable[str]) -> str:
        numbers = [int(x[len(prefix):]) for x in taken if x.startswith(prefix) and x[len(prefix):].isdigit()]
        return f"{prefix}{max(numbers, default=0) + 1}"

    def add(self, case: Case, by: str) -> Optional[Case]:
        """Save a new case (always starting in shadow, with no confirmations). Returns it with its id."""
        if not by:
            raise ValueError("only the owner creates a case")
        with _LOCK:
            state = self._fold()
            case_id = self._next_id("C", state["cases"])
            fresh = Case.from_dict({**case.to_dict(), "id": case_id, "status": SHADOW, "confirmations": 0,
                                    "contradictions": 0, "streak": 0, "created_by": by,
                                    "created_at": case.created_at or self._now(), "invalid_at": None,
                                    "invalid_reason": "", "merged_into": "", "revision": 1})
            if not self._append("case_add", case=fresh.to_dict(), by=by):
                return None
        return self.get(case_id)

    def _case_event(self, event: str, case_id: str, **fields: Any) -> Optional[Case]:
        with _LOCK:
            if self.get(case_id) is None:
                return None
            if not self._append(event, id=case_id, **fields):
                return None
            return self.get(case_id)

    def confirm(self, case_id: str, by: str, event_id: str = "", example: Optional[Example] = None) -> Optional[Case]:
        """The owner said "yes, that's them": the only thing that strengthens a case."""
        if not by:
            raise ValueError("only the owner confirms a case")
        return self._case_event("confirm", case_id, by=by, event_id=event_id,
                                example=example.to_dict() if example else None)

    def contradict(self, case_id: str, by: str, event_id: str = "", reason: str = "not them") -> Optional[Case]:
        """ "Not them": a negative example, and the case steps back one stage."""
        return self._case_event("contradict", case_id, by=by, event_id=event_id, reason=reason)

    def set_effect(self, case_id: str, effect: str, by: str) -> Optional[Case]:
        if effect not in EFFECTS:
            raise ValueError(f"effect must be one of {EFFECTS}")
        return self._case_event("set_effect", case_id, effect=effect, by=by)

    def keep_alerting(self, case_id: str, by: str) -> Optional[Case]:
        """Shadow prompt "no, keep alerting": remembered for context only."""
        return self.set_effect(case_id, ALERT, by)

    def log_match(self, case_id: str, event_id: str, level: str, band: str, score: float, shadow: bool,
                  verdict: str = "") -> bool:
        """An automatic match, for the audit and the digest. Never strengthens the case."""
        with _LOCK:
            return self._append("matched", id=case_id, event_id=event_id, level=level, band=band,
                                score=round(float(score), 4), shadow=bool(shadow), verdict=verdict)

    def narrow(self, case_id: str, scope: Scope, by: str) -> Optional[Case]:
        """Apply an owner correction at once. Refuses anything that is not a narrowing (use propose_widen)."""
        with _LOCK:
            case = self.get(case_id)
            if case is None:
                return None
            if not narrows(case.scope, scope):
                raise ValueError("not a narrowing; a wider scope needs the owner's approval (propose_widen)")
            return self._case_event("narrow", case_id, scope=scope.to_dict(), by=by)

    def propose_widen(self, case_id: str, scope: Scope, by: str = "") -> Optional[Dict[str, Any]]:
        """A wider scope waits for the owner's yes. Returns ``{wid, id, diff}``."""
        with _LOCK:
            state = self._fold()
            case = state["cases"].get(case_id)
            if case is None or case.invalid_at is not None:
                return None
            if narrows(case.scope, scope):
                raise ValueError("not a widening; apply it with narrow")
            wid = self._next_id("W", state["widen"])
            if not self._append("widen_proposed", id=case_id, wid=wid, scope=scope.to_dict(), by=by):
                return None
        return {"wid": wid, "id": case_id, "diff": scope_diff(case.scope, scope)}

    def answer_widen(self, wid: str, yes: bool, by: str) -> Optional[Case]:
        if not by:
            raise ValueError("only the owner approves a widening")
        with _LOCK:
            proposal = self._fold()["widen"].get(wid)
            if proposal is None or proposal["answer"]:
                return None
            self._append("widen_answer", wid=wid, yes=bool(yes), by=by)
            return self.get(proposal["id"])

    def invalidate(self, case_id: str, reason: str, by: str = "", merged_into: str = "") -> Optional[Case]:
        """ "Delete" for the owner: the case stops matching, the history stays."""
        return self._case_event("invalidate", case_id, reason=reason, by=by, merged_into=merged_into)

    def restore(self, case_id: str, by: str) -> Optional[Case]:
        return self._case_event("restore", case_id, by=by)

    def ask_review(self, case_id: str) -> Optional[Case]:
        """ "Still relevant?" was sent: the case pauses until the owner answers."""
        return self._case_event("review_asked", case_id)

    def answer_review(self, case_id: str, keep: bool, by: str) -> Optional[Case]:
        return self._case_event("review_answer", case_id, keep=bool(keep), by=by)

    def merge(self, source_id: str, into_id: str, scope: Scope, by: str) -> Optional[Case]:
        """Fold *source* into *into* with *scope* (the owner approved it): examples and confirmations move,
        the source is invalidated with "merged into", never deleted."""
        if not by:
            raise ValueError("only the owner approves a merge")
        with _LOCK:
            source, target = self.get(source_id), self.get(into_id)
            if source is None or target is None or source_id == into_id:
                return None
            self._append("merge_in", id=into_id, scope=scope.to_dict(), by=by, source=source_id,
                         examples=[e.to_dict() for e in source.examples], confirmations=source.confirmations)
            self._append("invalidate", id=source_id, reason=f"merged into {into_id}", by=by, merged_into=into_id)
            return self.get(into_id)

    def add_expecting(self, camera: str, text: str, by: str, ts: Optional[float] = None,
                      until: Optional[float] = None, source: Optional[Dict[str, Any]] = None) -> Optional[Expecting]:
        """ "Only today": an expecting note that ends at the end of that day."""
        now = self._now() if ts is None else ts
        with _LOCK:
            note_id = self._next_id("X", self._fold()["expecting"])
            note = Expecting(id=note_id, camera=camera, text=str(text)[:200], until=until or end_of_day(now),
                             created_at=now, created_by=by, source=dict(source or {}))
            if not self._append("expecting_add", note=note.to_dict(), by=by):
                return None
        return note

    def cancel_expecting(self, note_id: str, by: str) -> bool:
        with _LOCK:
            return self._append("expecting_cancel", id=note_id, by=by)

    def answer_routine(self, key: str, yes: bool, by: str, case_id: str = "") -> bool:
        """Remember the owner's answer to a routine proposal (a "no" is a labelled negative, never asked again)."""
        with _LOCK:
            return self._append("routine_answer", key=key, yes=bool(yes), by=by, case_id=case_id)
