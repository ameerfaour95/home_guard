"""House state: is the family awake, asleep, away or on vacation, and what is the owner expecting.

The situation engine (``situation.py``) reads it for every look; the Eye judges the same scene differently
when the family sleeps or nobody is home (``taxonomy.contextual_label``). Spec:
``docs/superpowers/specs/2026-10-05-situation-aware-eye-and-investigator-design.md`` Part 1.

States: ``home_awake``, ``home_asleep``, ``away`` and ``vacation`` (an ``away`` with an end date; the
taxonomy sees it as ``away``). Without any command the default schedule applies: ``home_asleep`` 00:00-06:00,
``home_awake`` otherwise. The owner can move that window (``set_schedule``).

Who may change what:

- An owner command applies directly ("going to sleep", "we left", "back home").
- Any other source (``schedule``, ``proposal``, ``system``) may make the house stricter by itself
  (awake -> asleep -> away/vacation). Anything that RELAXES (away -> home, asleep -> awake before the schedule
  ends, cancelling a vacation early, a new "expecting" note) becomes a pending proposal that waits for the
  owner's ``approve`` (or ``reject``). Pending proposals lapse after ``PROPOSAL_HOURS``.
- A state without ``until`` lasts until the next schedule boundary (06:00 or 00:00 by default), except
  ``away``, which lasts until the owner says otherwise. ``vacation`` needs ``until``; ``since`` sets its start.
  A newer state ends any older one that was running when it started.

Storage: one append-only JSON-lines change log, ``<paths.state_dir()>/house_state.jsonl`` (next to
the brain's facts.jsonl). One writer: a module-level lock covers read-compare-append, every change is a single
appended line, and the state is folded from the log on every read. Damaged lines are skipped with one warning
per read. Today's temporary mutes are a read-only view of the pause file that ``feedback.MuteState`` writes
(``<LOG_DIR>/alert_mute.json``); this module never writes it.

Public API (the assistant in brain/ calls these; every one takes optional ``path=`` and ``now=`` for tests):

    set_house_state(state, source, until=None, by="", since=None) -> entry   # status "applied" | "pending"
    add_expecting(text, camera=None, until=..., source="owner", by="") -> entry
    cancel_expecting(expect_id, by="") -> bool
    cancel_entry(entry_id, source="owner", by="") -> entry | None   # ends a state early (e.g. a future vacation)
    set_schedule(asleep_from="00:00", asleep_until="06:00", source="owner", by="")   # None, None: no schedule
    current(now=None) -> HouseNow      # never raises; falls back to the default schedule
    pending_proposals() -> [proposal]
    approve(proposal_id, by="") -> bool
    reject(proposal_id, by="") -> bool
    history(n=20) -> [event]           # newest first
    scheduled(ts) -> HouseNow          # the default schedule alone, no file (the eval uses it)

``HouseNow`` has ``state``, ``taxonomy_state``, ``source``, ``set_at``, ``expires_at``, ``by``, ``entry_id``,
``expecting`` (each with id, text, camera, source, set_at, expires_at, by), ``mutes`` (camera or None for
the whole house, source, set_at, expires_at), ``schedule``, ``pending`` and ``to_dict()``. Times are local
ISO strings (seconds).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("box.house_state")

FILE_NAME = "house_state.jsonl"
STATES = ("home_awake", "home_asleep", "away", "vacation")
SOURCES = ("owner", "schedule", "proposal", "system")
TAXONOMY_STATE = {"home_awake": "home_awake", "home_asleep": "home_asleep", "away": "away", "vacation": "away"}
STRICTNESS = {"home_awake": 0, "home_asleep": 1, "away": 2, "vacation": 2}
DEFAULT_SCHEDULE = ("00:00", "06:00")      # home_asleep from, until
PROPOSAL_HOURS = 12.0
MAX_TEXT = 120

_LOCK = threading.RLock()
_HHMM = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _finite(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("number must be finite")
    return number


def _iso(ts: Optional[float]) -> Optional[str]:
    return None if ts is None else dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _time_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.timestamp()
    if isinstance(value, str):
        return dt.datetime.fromisoformat(value).timestamp()
    return _finite(value)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("`", "").replace('"', "'").split())[:MAX_TEXT]


def _hhmm(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) == 4 and text[1] == ":":
        text = "0" + text
    if not _HHMM.match(text):
        raise ValueError('times must look like "06:00"')
    return text


def _at(day: dt.date, hhmm: str) -> float:
    hour, minute = map(int, hhmm.split(":"))
    return dt.datetime(day.year, day.month, day.day, hour, minute).timestamp()


def _in_window(ts: float, start: str, end: str) -> bool:
    moment = dt.datetime.fromtimestamp(ts)
    now_m = moment.hour * 60 + moment.minute
    s = int(start[:2]) * 60 + int(start[3:])
    e = int(end[:2]) * 60 + int(end[3:])
    return s <= now_m < e if s < e else (now_m >= s or now_m < e)


def _boundaries(ts: float, schedule: Optional[Tuple[str, str]]) -> Tuple[Optional[float], Optional[float]]:
    """The schedule's last change at or before *ts* and its next change after it; (None, None) without one."""
    if not schedule:
        return None, None
    day = dt.datetime.fromtimestamp(ts).date()
    moments = sorted(_at(day + dt.timedelta(days=d), hhmm) for d in (-1, 0, 1) for hhmm in schedule)
    before = [m for m in moments if m <= ts]
    after = [m for m in moments if m > ts]
    return (before[-1] if before else None), (after[0] if after else None)


def _scheduled_state(ts: float, schedule: Optional[Tuple[str, str]]) -> str:
    return "home_asleep" if schedule and _in_window(ts, *schedule) else "home_awake"


# ----------------------------------------------------------------------------
# What the house is now
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class HouseNow:
    state: str
    source: str
    set_at: Optional[str]
    expires_at: Optional[str]
    by: str = ""
    entry_id: str = ""
    expecting: List[Dict[str, Any]] = field(default_factory=list)
    mutes: List[Dict[str, Any]] = field(default_factory=list)
    schedule: Dict[str, Any] = field(default_factory=dict)
    pending: int = 0

    @property
    def taxonomy_state(self) -> str:
        """``home_awake`` / ``home_asleep`` / ``away`` (a vacation is away)."""
        return TAXONOMY_STATE.get(self.state, "home_awake")

    def expecting_for(self, camera: str) -> List[Dict[str, Any]]:
        """The expecting notes that cover *camera* (notes without a camera cover the whole house)."""
        return [e for e in self.expecting if not e.get("camera") or e.get("camera") == camera]

    def to_dict(self) -> Dict[str, Any]:
        return {"state": {"value": self.state, "taxonomy": self.taxonomy_state, "source": self.source,
                          "set_at": self.set_at, "expires_at": self.expires_at, "by": self.by,
                          "id": self.entry_id},
                "expecting": [dict(e) for e in self.expecting], "mutes": [dict(m) for m in self.mutes],
                "schedule": dict(self.schedule), "pending": self.pending}


def _schedule_dict(schedule: Optional[Tuple[str, str]], source: str, set_at: Optional[float]) -> Dict[str, Any]:
    return {"asleep_from": schedule[0] if schedule else None, "asleep_until": schedule[1] if schedule else None,
            "source": source, "set_at": _iso(set_at), "expires_at": None}


def scheduled(ts: float, schedule: Optional[Tuple[str, str]] = DEFAULT_SCHEDULE) -> HouseNow:
    """The state the schedule alone gives at *ts* (no file, no commands)."""
    last, nxt = _boundaries(ts, schedule)
    return HouseNow(state=_scheduled_state(ts, schedule), source="schedule", set_at=_iso(last),
                    expires_at=_iso(nxt), schedule=_schedule_dict(schedule, "schedule", None))


# ----------------------------------------------------------------------------
# The store
# ----------------------------------------------------------------------------
class HouseStateStore:
    """The house-state change log. All mutations serialize under the module lock."""

    def __init__(self, path: str, mute_path: Optional[str] = None, now: Callable[[], float] = time.time) -> None:
        self.path = path
        self.mute_path = mute_path
        self._now = now

    # -- the file -----------------------------------------------------------------
    def _read(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
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
                        _finite(item.get("ts"))
                        events.append(item)
                    except (ValueError, TypeError, OverflowError, RecursionError):
                        damaged = True
        except FileNotFoundError:
            return []
        except OSError as exc:
            log.warning("House state could not be read: %s", exc)
            return []
        if damaged:
            log.warning("Skipped damaged lines in the house state file")
        return events

    def _append(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Append one event as one line; a cut last line (a killed writer) is closed first so it stays alone."""
        line = json.dumps(event, ensure_ascii=False, allow_nan=False)
        with _LOCK:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            prefix = ""
            try:
                with open(self.path, "rb") as f:
                    f.seek(0, os.SEEK_END)
                    if f.tell():
                        f.seek(-1, os.SEEK_END)
                        prefix = "" if f.read(1) == b"\n" else "\n"
            except FileNotFoundError:
                pass
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(prefix + line + "\n")
                f.flush()
        return event

    def _clock(self, now: Optional[float] = None) -> float:
        return _finite(self._now() if now is None else now)

    @staticmethod
    def _next_id(events: List[Dict[str, Any]], prefix: str) -> str:
        top = 0
        for e in events:
            value = str(e.get("id") or "")
            if value.startswith(prefix) and value[1:].isdigit():
                top = max(top, int(value[1:]))
        return f"{prefix}{top + 1}"

    # -- folding ------------------------------------------------------------------
    def _fold(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        schedule: Optional[Tuple[str, str]] = DEFAULT_SCHEDULE
        schedule_src: Tuple[str, Optional[float]] = ("schedule", None)
        entries: List[Dict[str, Any]] = []
        expecting: Dict[str, Dict[str, Any]] = {}
        proposals: Dict[str, Dict[str, Any]] = {}

        def apply_entry(entry: Dict[str, Any]) -> None:
            start = entry["start"]
            for older in entries:
                if older["start"] <= start and (older["until"] is None or start < older["until"]):
                    older["until"] = start
            entries.append(entry)

        for e in events:
            try:
                kind = e["event"]
                ts = _finite(e["ts"])
                if kind == "schedule":
                    frm, until = e.get("asleep_from"), e.get("asleep_until")
                    schedule = (_hhmm(frm), _hhmm(until)) if frm and until else None
                    schedule_src = (str(e.get("source") or "owner"), ts)
                elif kind == "state":
                    if e.get("state") not in STATES:
                        raise ValueError("unknown state")
                    apply_entry({"id": str(e["id"]), "state": e["state"], "source": e.get("source", "owner"),
                                 "by": str(e.get("by") or ""), "set_at": ts,
                                 "start": _time_or_none(e.get("since")) or ts,
                                 "until": _time_or_none(e.get("until"))})
                elif kind == "cancel":
                    for entry in entries:
                        if entry["id"] == e.get("entry_id"):
                            entry["until"] = ts if entry["until"] is None else min(entry["until"], ts)
                            entry["start"] = min(entry["start"], entry["until"])
                elif kind == "expect":
                    expecting[str(e["id"])] = {"id": str(e["id"]), "text": _clean_text(e.get("text")),
                                               "camera": e.get("camera") or None,
                                               "source": e.get("source", "owner"), "by": str(e.get("by") or ""),
                                               "set_at": ts, "until": _finite(e["until"])}
                elif kind == "expect_cancel":
                    expecting.pop(str(e.get("expect_id")), None)
                elif kind == "proposal":
                    proposals[str(e["id"])] = dict(e, status="pending")
                elif kind in ("approve", "reject"):
                    p = proposals.get(str(e.get("proposal_id")))
                    if p is None or p["status"] != "pending":
                        continue
                    p["status"] = "approved" if kind == "approve" else "rejected"
                    p["answered_by"] = str(e.get("by") or "")
                    if kind == "reject":
                        continue
                    if p.get("kind") == "state":
                        apply_entry({"id": p["id"], "state": p["state"], "source": "proposal",
                                     "by": p.get("by", ""), "set_at": ts,
                                     "start": max(ts, _time_or_none(p.get("since")) or ts),
                                     "until": _time_or_none(e.get("until", p.get("until")))})
                    elif p.get("kind") == "expect":
                        expecting[p["id"]] = {"id": p["id"], "text": _clean_text(p.get("text")),
                                              "camera": p.get("camera") or None, "source": "proposal",
                                              "by": p.get("by", ""), "set_at": ts, "until": _finite(p["until"])}
                    elif p.get("kind") == "cancel":
                        for entry in entries:
                            if entry["id"] == p.get("entry_id"):
                                entry["until"] = ts if entry["until"] is None else min(entry["until"], ts)
                                entry["start"] = min(entry["start"], entry["until"])
            except (KeyError, ValueError, TypeError, OverflowError):
                log.warning("Skipped a damaged house state event: %s", str(e)[:120])
        return {"schedule": schedule, "schedule_src": schedule_src, "entries": entries,
                "expecting": expecting, "proposals": proposals}

    def _state_at(self, folded: Dict[str, Any], ts: float) -> HouseNow:
        schedule = folded["schedule"]
        live = [e for e in folded["entries"] if e["start"] <= ts and (e["until"] is None or ts < e["until"])]
        sched = _schedule_dict(schedule, *folded["schedule_src"])
        if live:
            e = live[-1]
            return HouseNow(state=e["state"], source=e["source"], set_at=_iso(e["set_at"]),
                            expires_at=_iso(e["until"]), by=e["by"], entry_id=e["id"], schedule=sched)
        base = scheduled(ts, schedule)
        return HouseNow(state=base.state, source="schedule", set_at=base.set_at, expires_at=base.expires_at,
                        schedule=sched)

    def _mutes(self, now: float) -> List[Dict[str, Any]]:
        if not self.mute_path:
            return []
        try:
            from .feedback import MuteState  # noqa: PLC0415 - read only: snapshot() never writes

            snap = MuteState(self.mute_path).snapshot()
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read today's pauses: %s", exc)
            return []
        out = []
        if snap.get("all", 0) > now:
            out.append({"camera": None, "source": "owner", "set_at": None, "expires_at": _iso(snap["all"])})
        for camera, until in sorted((snap.get("cameras") or {}).items()):
            if until > now:
                out.append({"camera": camera, "source": "owner", "set_at": None, "expires_at": _iso(until)})
        return out

    def _default_until(self, folded: Dict[str, Any], state: str, start: float) -> Optional[float]:
        if state in ("away", "vacation"):
            return None
        return _boundaries(start, folded["schedule"])[1]

    # -- reading ------------------------------------------------------------------
    def current(self, now: Optional[float] = None) -> HouseNow:
        ts = self._clock(now)
        folded = self._fold(self._read())
        base = self._state_at(folded, ts)
        expecting = [{"id": x["id"], "text": x["text"], "camera": x["camera"], "source": x["source"],
                      "by": x["by"], "set_at": _iso(x["set_at"]), "expires_at": _iso(x["until"])}
                     for x in folded["expecting"].values() if x["set_at"] <= ts < x["until"]]
        return HouseNow(state=base.state, source=base.source, set_at=base.set_at, expires_at=base.expires_at,
                        by=base.by, entry_id=base.entry_id, expecting=expecting, mutes=self._mutes(ts),
                        schedule=base.schedule, pending=len(self._pending(folded, ts)))

    def _pending(self, folded: Dict[str, Any], now: float) -> List[Dict[str, Any]]:
        out = []
        for p in folded["proposals"].values():
            if p["status"] == "pending" and now - _finite(p["ts"]) < PROPOSAL_HOURS * 3600:
                public = {k: v for k, v in p.items() if k not in ("event",)}
                for key in ("ts", "until", "since"):
                    if key in public:
                        public[key] = _iso(_time_or_none(public[key]))
                public["set_at"] = public.pop("ts")
                public["expires_at"] = _iso(_finite(p["ts"]) + PROPOSAL_HOURS * 3600)
                out.append(public)
        return out

    def pending_proposals(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        return self._pending(self._fold(self._read()), self._clock(now))

    def history(self, n: int = 20) -> List[Dict[str, Any]]:
        events = self._read()
        return list(reversed(events[-max(0, int(n)):])) if n else []

    # -- writing ------------------------------------------------------------------
    def _propose(self, events: List[Dict[str, Any]], now: float, kind: str, source: str, by: str,
                 **fields: Any) -> Dict[str, Any]:
        pid = self._next_id(events, "P")
        event = {"event": "proposal", "id": pid, "kind": kind, "source": source, "by": by, "ts": now, **fields}
        self._append(event)
        log.info("House state proposal %s (%s from %s) waits for the owner", pid, kind, source)
        return {"id": pid, "status": "pending", "kind": kind, **fields}

    def set_house_state(self, state: str, source: str, until: Any = None, by: str = "", since: Any = None,
                        now: Optional[float] = None) -> Dict[str, Any]:
        if state not in STATES:
            raise ValueError(f"state must be one of {', '.join(STATES)}")
        if source not in SOURCES:
            raise ValueError(f"source must be one of {', '.join(SOURCES)}")
        with _LOCK:
            ts = self._clock(now)
            start = _time_or_none(since) or ts
            end = _time_or_none(until)
            if state == "vacation" and end is None:
                raise ValueError("a vacation needs an end date (until)")
            if end is not None and end <= max(ts, start):
                raise ValueError("until must be later than now and than since")
            events = self._read()
            folded = self._fold(events)
            if end is None:
                end = self._default_until(folded, state, start)
            fields = {"state": state, "until": end, "since": _time_or_none(since)}
            before = self._state_at(folded, max(ts, start)).state
            if source != "owner" and STRICTNESS[state] < STRICTNESS[before]:
                return self._propose(events, ts, "state", source, by, **fields)
            entry_id = self._next_id(events, "H")
            self._append({"event": "state", "id": entry_id, "source": source, "by": by, "ts": ts, **fields})
            return {"id": entry_id, "status": "applied", "source": source, **fields}

    def cancel_entry(self, entry_id: str, source: str = "owner", by: str = "",
                     now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """End a state entry now (or drop a future one). From a non-owner source, ending anything stricter than
        awake is a relaxation and becomes a proposal."""
        if source not in SOURCES:
            raise ValueError(f"source must be one of {', '.join(SOURCES)}")
        with _LOCK:
            ts = self._clock(now)
            events = self._read()
            entry = next((e for e in self._fold(events)["entries"] if e["id"] == entry_id), None)
            if entry is None or (entry["until"] is not None and entry["until"] <= ts):
                return None
            if source != "owner" and STRICTNESS[entry["state"]] > 0:
                return self._propose(events, ts, "cancel", source, by, entry_id=entry_id)
            self._append({"event": "cancel", "entry_id": entry_id, "source": source, "by": by, "ts": ts})
            return {"id": entry_id, "status": "applied"}

    def add_expecting(self, text: str, camera: Optional[str] = None, until: Any = None, source: str = "owner",
                      by: str = "", now: Optional[float] = None) -> Dict[str, Any]:
        """An owner note like "a package today": it relaxes the priors (taxonomy ``expecting``), so a non-owner
        source only proposes it."""
        if source not in SOURCES:
            raise ValueError(f"source must be one of {', '.join(SOURCES)}")
        clean = _clean_text(text)
        if not clean:
            raise ValueError("an expecting note needs text")
        end = _time_or_none(until)
        with _LOCK:
            ts = self._clock(now)
            if end is None or end <= ts:
                raise ValueError("an expecting note needs an expiry later than now (until)")
            events = self._read()
            fields = {"text": clean, "camera": camera or None, "until": end}
            if source != "owner":
                return self._propose(events, ts, "expect", source, by, **fields)
            xid = self._next_id(events, "X")
            self._append({"event": "expect", "id": xid, "source": source, "by": by, "ts": ts, **fields})
            return {"id": xid, "status": "applied", **fields}

    def cancel_expecting(self, expect_id: str, by: str = "", now: Optional[float] = None) -> bool:
        """Dropping an expectation only makes the house stricter, so any caller may do it."""
        with _LOCK:
            if str(expect_id) not in self._fold(self._read())["expecting"]:
                return False
            self._append({"event": "expect_cancel", "expect_id": str(expect_id), "by": by, "ts": self._clock(now)})
            return True

    def set_schedule(self, asleep_from: Optional[str] = DEFAULT_SCHEDULE[0],
                     asleep_until: Optional[str] = DEFAULT_SCHEDULE[1], source: str = "owner", by: str = "",
                     now: Optional[float] = None) -> Dict[str, Any]:
        """The nightly ``home_asleep`` window; ``None, None`` turns the schedule off. Owner only."""
        if source != "owner":
            raise ValueError("only the owner changes the sleep schedule")
        frm = _hhmm(asleep_from) if asleep_from is not None else None
        until = _hhmm(asleep_until) if asleep_until is not None else None
        if (frm is None) != (until is None) or (frm is not None and frm == until):
            raise ValueError("give both times (different), or neither to turn the schedule off")
        event = {"event": "schedule", "asleep_from": frm, "asleep_until": until, "source": source, "by": by,
                 "ts": self._clock(now)}
        with _LOCK:
            self._append(event)
        return event

    def _answer(self, proposal_id: str, approve: bool, by: str, now: Optional[float]) -> bool:
        with _LOCK:
            ts = self._clock(now)
            events = self._read()
            folded = self._fold(events)
            if not any(p["id"] == proposal_id for p in self._pending(folded, ts)):
                return False
            event: Dict[str, Any] = {"event": "approve" if approve else "reject", "proposal_id": proposal_id,
                                     "by": by, "ts": ts}
            p = folded["proposals"][proposal_id]
            if approve and p.get("kind") == "state":
                start = max(ts, _time_or_none(p.get("since")) or ts)
                until = _time_or_none(p.get("until"))
                if until is None or until <= start:
                    until = self._default_until(folded, p["state"], start)
                event["until"] = until
            self._append(event)
            return True

    def approve(self, proposal_id: str, by: str = "", now: Optional[float] = None) -> bool:
        return self._answer(str(proposal_id), True, by, now)

    def reject(self, proposal_id: str, by: str = "", now: Optional[float] = None) -> bool:
        return self._answer(str(proposal_id), False, by, now)


# ----------------------------------------------------------------------------
# Module API (the box's own file)
# ----------------------------------------------------------------------------
def default_path() -> str:
    from . import paths  # noqa: PLC0415

    return os.path.join(paths.state_dir(), FILE_NAME)


def default_mute_path() -> str:
    from . import paths  # noqa: PLC0415

    return os.path.join(paths.logs_dir(), "alert_mute.json")


def _store(path: Optional[str] = None, mute_path: Optional[str] = None, now: Optional[float] = None
           ) -> HouseStateStore:
    clock = (lambda: now) if now is not None else time.time
    return HouseStateStore(path or default_path(), mute_path=mute_path or default_mute_path(), now=clock)


def set_house_state(state: str, source: str, until: Any = None, by: str = "", since: Any = None,
                    path: Optional[str] = None, now: Optional[float] = None) -> Dict[str, Any]:
    return _store(path, now=now).set_house_state(state, source, until=until, by=by, since=since)


def add_expecting(text: str, camera: Optional[str] = None, until: Any = None, source: str = "owner", by: str = "",
                  path: Optional[str] = None, now: Optional[float] = None) -> Dict[str, Any]:
    return _store(path, now=now).add_expecting(text, camera=camera, until=until, source=source, by=by)


def cancel_expecting(expect_id: str, by: str = "", path: Optional[str] = None, now: Optional[float] = None) -> bool:
    return _store(path, now=now).cancel_expecting(expect_id, by=by)


def cancel_entry(entry_id: str, source: str = "owner", by: str = "", path: Optional[str] = None,
                 now: Optional[float] = None) -> Optional[Dict[str, Any]]:
    return _store(path, now=now).cancel_entry(entry_id, source=source, by=by)


def set_schedule(asleep_from: Optional[str] = DEFAULT_SCHEDULE[0], asleep_until: Optional[str] = DEFAULT_SCHEDULE[1],
                 source: str = "owner", by: str = "", path: Optional[str] = None,
                 now: Optional[float] = None) -> Dict[str, Any]:
    return _store(path, now=now).set_schedule(asleep_from, asleep_until, source=source, by=by)


def current(now: Optional[float] = None, path: Optional[str] = None, mute_path: Optional[str] = None) -> HouseNow:
    """The house now. Never raises: on any problem, the default schedule (and a warning)."""
    ts = time.time() if now is None else now
    try:
        return _store(path, mute_path, now=ts).current(ts)
    except Exception as exc:  # noqa: BLE001 - the guard loop must never stop for the house state
        log.warning("House state unavailable, using the default schedule: %s", exc)
        return scheduled(ts)


def pending_proposals(path: Optional[str] = None, now: Optional[float] = None) -> List[Dict[str, Any]]:
    return _store(path, now=now).pending_proposals()


def approve(proposal_id: str, by: str = "", path: Optional[str] = None, now: Optional[float] = None) -> bool:
    return _store(path, now=now).approve(proposal_id, by=by)


def reject(proposal_id: str, by: str = "", path: Optional[str] = None, now: Optional[float] = None) -> bool:
    return _store(path, now=now).reject(proposal_id, by=by)


def history(n: int = 20, path: Optional[str] = None) -> List[Dict[str, Any]]:
    return _store(path).history(n)
