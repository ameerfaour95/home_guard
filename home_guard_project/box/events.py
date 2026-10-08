"""Events, not triggers: one ongoing activity per camera is one event, and a message goes out only on a change.

Why (owner, 2026-10-08): the box sent 200 alerts in 42 hours, 124 of them about the same workers on the pergola,
each "independent of the one before although it all happens now". The owner wants a session that keeps who is
there, continues the story, and resets after the people are gone for a while. Fix plan:
https://claude.ai/artifact/9Fh6q2W5XbndKSkNgVJa4M (section 3).

- **Session** (one per camera, like a Frigate review item): opened by the first alert job or by detector
  activity, kept alive by every look that sees a person or vehicle, closed after ``IDLE_SEC`` with none.
  After ``ROLL_SEC`` it is closed and a linked one opens (``parent``), so an all-day work crew is one chain.
- **Policy** (``decide``), code only, the VLM never decides whether to send:
  - "normal" is never a message in inference mode (owner decision 2026-10-08); it is kept in the event.
  - "suspicious" is sent once per session; again only when more people are seen than the session already
    reported, or the level rises.
  - "escalation" is always sent unless the same session already reported an escalation with the same people
    in the last ``ESCALATION_REPEAT_SEC``.
  - What the owner marked as known ("these are my workers", until a time) silences "suspicious" for that camera
    while the head-count stays within what was known (plus KNOWN_EXTRA_PEOPLE); never an escalation.
  - Whose ground (optional ``ground``, ``ground.Ground.record()`` plus ``action``; scene map stage 2c): everyone
    stayed on the neighbour's or public ground -> not sent, unless an escalation or a suspicious for something
    done there (``action``); someone came onto our ground from the neighbour's side or the street (``entered``)
    -> at least a suspicious, even when the Eye said normal.
- **Known** (``mark_known``) is written only from the owner's own words, through the assistant, with a receipt.
  It covers the camera until ``until`` and the session it was said in.
- **Entities** (stage 2a, entities.py): with the tracker's tracks (``decide(tracks=...)``) the session keeps who is
  who (P1, P2, CAR1). Then "more people" counts entities the owner was not told about instead of the Eye's
  head-count, and under the owner's words for this session only the entities present when they were said are
  covered: a person who arrives later is "not one of those the owner marked" (sent only when suspicious or higher).
  Without tracker data everything above holds as it was.

The book keeps open sessions in memory and appends every closed one to ``events.jsonl`` (for the assistant's
"what happened at the pergola today" and for the investigator). Thread-safe: one lock for everything.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import entities as ent

log = logging.getLogger("box.events")

IDLE_SEC = 60.0                  # no person or vehicle seen this long: the activity is over
ROLL_SEC = 1800.0                # a session this old closes and a linked one continues
ESCALATION_REPEAT_SEC = 600.0
# The owner's "these are my workers" still covers a head-count this much above what was there when it was said:
# the Eye's count of a working group jumps around (3, 5, 2...). A real new person is the entity layer's job (stage 2).
KNOWN_EXTRA_PEOPLE = 2
KEEP_OBSERVATIONS = 40           # per session, newest kept
ENTITY_LOOKBACK_SEC = 10.0       # tracks that ended this long before the session opened are not its entities
LEVELS = {"none": 0, "normal": 1, "suspicious": 2, "escalation": 3}


@dataclass
class Known:
    """Something the owner said about who is there ("עובדים בפרגולה"), with its scope."""

    text: str
    by: str
    at: float
    until: float
    people: int = 0              # how many were there when the owner said it (0 = not counted)
    camera: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    def live(self, now: float) -> bool:
        return self.at <= now < self.until


@dataclass
class Session:
    id: str
    camera: str
    opened: float
    last_active: float
    parent: str = ""
    closed: float = 0.0
    people_max: int = 0
    observations: List[Dict[str, Any]] = field(default_factory=list)
    reported_level: str = "none"         # the highest level already sent to the owner
    reported_people: int = 0             # how many people the owner was already told about
    reported_at: float = 0.0
    messages: List[Dict[str, Any]] = field(default_factory=list)   # {alert_id, chat_id, message_id, ts}
    known: List[Dict[str, Any]] = field(default_factory=list)
    # Stage 2a (entities.py): who is who in this event (P1, P2, CAR1) and which of them the owner was told about.
    entities: List[Dict[str, Any]] = field(default_factory=list)
    reported_entities: List[str] = field(default_factory=list)

    def people_when_said(self, known_id: str) -> int:
        """People present in this session when the owner's words *known_id* were said here (0 when said elsewhere)."""
        return max([_int(k.get("people")) for k in self.known if k.get("id") == known_id] or [0])

    def first_message(self) -> Optional[Dict[str, Any]]:
        return self.messages[0] if self.messages else None

    def summary_line(self) -> str:
        """A short text of the activity for the assistant: time range, most people, the latest observation."""
        start = datetime.fromtimestamp(self.opened).strftime("%H:%M")
        end = datetime.fromtimestamp(self.closed or self.last_active).strftime("%H:%M")
        last = self.observations[-1]["summary"] if self.observations else ""
        return f"{start}-{end}, up to {self.people_max} people. Latest: {last}".strip()


@dataclass
class Decision:
    notify: bool
    session_id: str
    reason: str                    # why sent or not, for the log and the clip's meta
    reply_to: Optional[Dict[str, Any]] = None   # the session's first message: updates go in its thread
    new_people: int = 0            # people beyond what the owner was already told about
    known_text: str = ""           # the owner's own words that covered this (when silenced by them)
    # Stage 2a, with the tracker's data (``tracks``): who is in view, who of them is new to the owner, and whether a
    # new one arrived although the owner marked the others as known ("not one of the workers you marked").
    entities: List[str] = field(default_factory=list)
    fresh: List[str] = field(default_factory=list)
    unmarked: bool = False
    counted_by: str = "head-count"   # head-count (the Eye's people) | entities (the tracker's)

    def record(self) -> Dict[str, Any]:
        return asdict(self)


def _int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


class EventBook:
    def __init__(self, directory: str, notify_normal: bool = False, idle_sec: float = IDLE_SEC,
                 roll_sec: float = ROLL_SEC) -> None:
        self.directory = directory
        self.notify_normal = notify_normal
        self.idle_sec = idle_sec
        self.roll_sec = roll_sec
        self._lock = threading.RLock()
        self._open: Dict[str, Session] = {}
        self._known: List[Known] = []
        self._by_alert: Dict[str, str] = {}
        self._closed: List[Session] = []          # newest last, capped; the file holds the rest
        # Called with every session as it is archived (event_memory.attach adds the long-term memory's writer).
        # A hook must not raise; one that does is logged and the others still run.
        self.archive_hooks: List[Any] = []          # callables taking the Session
        os.makedirs(directory, exist_ok=True)
        self._load_known()

    # ---------- files ----------
    @property
    def events_path(self) -> str:
        return os.path.join(self.directory, "events.jsonl")

    @property
    def known_path(self) -> str:
        return os.path.join(self.directory, "known.json")

    def _load_known(self) -> None:
        try:
            with open(self.known_path, encoding="utf-8") as f:
                rows = json.load(f)
            self._known = [Known(**r) for r in rows if isinstance(r, dict)]
        except (OSError, ValueError, TypeError):
            self._known = []

    def _save_known(self) -> None:
        tmp = self.known_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump([asdict(k) for k in self._known], f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.known_path)

    def _archive(self, s: Session) -> None:
        self._closed = (self._closed + [s])[-200:]
        try:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("event %s not archived: %s", s.id, exc)
        for hook in list(self.archive_hooks):
            try:
                hook(s)
            except Exception as exc:  # noqa: BLE001 - the long-term memory must never stop the book
                log.warning("event %s: archive hook failed: %s", s.id, exc)

    # ---------- sessions ----------
    def _close(self, camera: str, now: float) -> Optional[Session]:
        s = self._open.pop(camera, None)
        if s is not None:
            s.closed = min(now, s.last_active + self.idle_sec)
            self._archive(s)
        return s

    def _session(self, camera: str, now: float) -> Session:
        s = self._open.get(camera)
        parent = ""
        if s is not None and now - s.last_active > self.idle_sec:
            self._close(camera, now)
            s = None
        elif s is not None and now - s.opened > self.roll_sec:
            old = self._close(camera, now)
            parent = old.id if old else ""
            s = Session(id=uuid.uuid4().hex[:12], camera=camera, opened=now, last_active=now, parent=parent,
                        people_max=old.people_max if old else 0,
                        reported_level=old.reported_level if old else "none",
                        reported_people=old.reported_people if old else 0,
                        reported_at=old.reported_at if old else 0.0,
                        messages=list(old.messages[:1]) if old else [], known=list(old.known) if old else [],
                        entities=copy.deepcopy(old.entities) if old else [],
                        reported_entities=list(old.reported_entities) if old else [])
            self._open[camera] = s
            return s
        if s is None:
            s = Session(id=uuid.uuid4().hex[:12], camera=camera, opened=now, last_active=now)
            self._open[camera] = s
        return s

    def activity(self, camera: str, now: float, people: int = 0, vehicles: int = 0) -> None:
        """One detector look. A look with a person or vehicle keeps the camera's session alive."""
        with self._lock:
            s = self._open.get(camera)
            if people or vehicles:
                if s is None:
                    return          # sessions open with an alert job; quiet activity alone does not open one
                if now - s.last_active > self.idle_sec or now - s.opened > self.roll_sec:
                    s = self._session(camera, now)
                s.last_active = max(s.last_active, now)
            elif s is not None and now - s.last_active > self.idle_sec:
                self._close(camera, now)

    def tick(self, now: float) -> None:
        """Close every session idle longer than IDLE_SEC (call now and then from the main loop)."""
        with self._lock:
            for cam in [c for c, s in self._open.items() if now - s.last_active > self.idle_sec]:
                self._close(cam, now)

    # ---------- the owner's words ----------
    def known_for(self, camera: str, now: float) -> Optional[Known]:
        with self._lock:
            live = [k for k in self._known if k.live(now) and (not k.camera or k.camera == camera)]
            return live[-1] if live else None

    def mark_known(self, camera: str, text: str, by: str, until: float, now: Optional[float] = None,
                   people: Optional[int] = None) -> Dict[str, Any]:
        """The owner said who is there. Returns a receipt; raises ValueError on a bad request."""
        now = time.time() if now is None else now
        text = str(text or "").strip()
        if not text:
            raise ValueError("nothing to remember")
        if until <= now:
            raise ValueError("the end time is already past")
        with self._lock:
            s = self._open.get(camera)
            count = _int(people) if people is not None else (s.people_max if s else 0)
            k = Known(text=text[:200], by=str(by or "owner"), at=now, until=until, people=count, camera=camera)
            self._known = [x for x in self._known if x.until > now] + [k]
            self._save_known()
            if s is not None:
                s.known.append(asdict(k))
                if s.entities and now - s.last_active <= self.idle_sec:
                    ent.label_present(s.entities, now, k.text, k.id)
            log.info("known: %s on %s until %s (%d people)", k.text, camera or "all cameras",
                     datetime.fromtimestamp(until).strftime("%Y-%m-%d %H:%M"), count)
            return {"store": "events.known", "id": k.id, "camera": camera, "text": k.text, "until": until,
                    "people": count, "undo_token": f"known:{k.id}"}

    def cancel_known(self, known_id: str) -> bool:
        with self._lock:
            before = len(self._known)
            self._known = [k for k in self._known if k.id != known_id]
            if len(self._known) != before:
                self._save_known()
                return True
            return False

    def list_known(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        now = time.time() if now is None else now
        with self._lock:
            return [asdict(k) for k in self._known if k.live(now)]

    # ---------- the policy ----------
    def _base_entities(self, camera: str, ts: float) -> List[Dict[str, Any]]:
        """A copy of the entities the session at *ts* starts from: the open one's (kept across a roll-over), none
        when it is idle and a new session would open."""
        s = self._open.get(camera)
        if s is None or ts - s.last_active > self.idle_sec:
            return []
        return copy.deepcopy(s.entities)

    def _not_before(self, camera: str, ts: float) -> float:
        s = self._open.get(camera)
        opened = ts if s is None or ts - s.last_active > self.idle_sec else s.opened
        return opened - ENTITY_LOOKBACK_SEC

    def roster(self, camera: str, ts: float, tracks: Any, since: Optional[float] = None) -> Dict[str, Any]:
        """What the entities will be when *tracks* are handed to ``decide`` at *ts*, without changing the book: the
        ids in view since *since* and the Eye's roster line (box.yaml ``eye_entities: on``)."""
        with self._lock:
            rows = self._base_entities(camera, ts)
            in_view = ent.ingest(rows, tracks or (), ts, since=since, not_before=self._not_before(camera, ts))
            return {"in_view": in_view, "line": ent.roster_line(rows, in_view, ts, since)}

    def decide(self, camera: str, ts: float, label: str, people: Any = None, summary: str = "",
               alert_id: str = "", tracks: Any = None, since: Optional[float] = None, note: str = "",
               per_entity: Any = None, ground: Optional[Dict[str, Any]] = None) -> Decision:
        """Should this alert job reach the owner? Records the observation in the camera's session either way.

        With *tracks* (the tracker's ``snapshot`` around the alert, stage 2a) the session's entities are brought up to
        date first; the people seen since *since* are this alert's people, and when the tracker saw at least one of
        them "more people" and the owner's known group are judged by those entities instead of the Eye's head-count
        *people*. *note* (the owner-language summary) and *per_entity* (the Eye's ``per_entity`` answer) are what the
        entities in view did (entities.attribute). *ground* (optional, stage 2c): where it happened by the scene map
        (ground.Ground.record() plus ``action``)."""
        label = label if label in LEVELS else "normal"
        count = _int(people)
        with self._lock:
            in_view: List[str] = []
            not_before = self._not_before(camera, ts)
            s = self._session(camera, ts)
            if tracks is not None:
                in_view = ent.ingest(s.entities, tracks, ts, since=since, not_before=not_before)
            s.last_active = max(s.last_active, ts)
            s.people_max = max(s.people_max, count)
            observation = {"ts": ts, "alert_id": alert_id, "label": label,
                           "summary": str(summary or "")[:400], "people": count}
            if tracks is not None:
                observation["entities"] = list(in_view)
                if note:
                    observation["note"] = str(note)[:400]
                observation["noted"] = ent.attribute(s.entities, in_view, ts, note, label, per_entity,
                                                     people=count if people is not None else None)
            s.observations = (s.observations + [observation])[-KEEP_OBSERVATIONS:]
            if alert_id:
                self._by_alert[alert_id] = s.id
            known = self.known_for(camera, ts)
            new_people = max(0, count - s.reported_people)
            reply_to = s.first_message()
            lvl, reported = LEVELS[label], LEVELS[s.reported_level]
            persons = ent.people(s.entities, in_view) if in_view else []
            by_entities = bool(persons)
            fresh: List[str] = []
            said_here = False
            if by_entities:
                live = {k.id for k in self._known if k.live(ts) and (not k.camera or k.camera == camera)}
                fresh = ent.fresh_people(s.entities, in_view, s.reported_entities, live)
                new_people = len(fresh)
                said_here = known is not None and any(known.id in e.get("known_ids", ()) for e in s.entities)

            def make(notify: bool, reason: str, known_text: str = "", unmarked: bool = False) -> Decision:
                return Decision(notify, s.id, reason, reply_to, new_people, known_text, list(in_view), list(fresh),
                                unmarked, "entities" if by_entities else "head-count")

            def no(reason: str, known_text: str = "") -> Decision:
                return make(False, reason, known_text)

            where = ground if isinstance(ground, dict) else {}
            if where.get("off_our_ground") and label != "escalation" and not (label == "suspicious"
                                                                               and where.get("action")):
                return no(f"{label} on the {where.get('on')} ground, nothing done there: not ours")
            entered = bool(where.get("entered")) and label == "normal"
            if entered:
                label, lvl = "suspicious", LEVELS["suspicious"]     # came onto our ground: worth a message

            if label == "normal":
                if self.notify_normal and reported == 0:
                    return make(True, "normal (notify_normal on), first in this event")
                return no("normal: kept in the event, not sent")
            if label == "escalation":
                if (s.reported_level == "escalation" and new_people == 0
                        and ts - s.reported_at < ESCALATION_REPEAT_SEC):
                    return no("escalation already reported in this event, same people")
                return make(True, "escalation")
            # suspicious
            if said_here:
                # The owner marked the people of this event; only someone the tracker saw arrive since is not theirs.
                if fresh:
                    return make(True, f"suspicious, {len(fresh)} new not among those the owner marked", known.text,
                                unmarked=True)
                if all(ent.covered(e, live) for e in persons):
                    return no("suspicious, but the owner said who is here", known.text)
                return no("suspicious, the one the owner did not mark was already reported")
            head = len(persons) if by_entities else count
            covered = max(known.people, s.people_when_said(known.id)) + KNOWN_EXTRA_PEOPLE if known else 0
            if known is not None and (known.people == 0 or head <= covered):
                return no("suspicious, but the owner said who is here", known.text)
            if reported >= lvl and new_people == 0:
                return no("suspicious already reported in this event, nobody new")
            why = "suspicious, first in this event" if reported < lvl else f"suspicious, {new_people} more people"
            if entered:
                why = f"came onto our ground ({why})"
            return make(True, why)

    def record_sent(self, session_id: str, label: str, people: Any, ts: float, alert_id: str = "",
                    chat_id: Any = None, message_id: Any = None, entities: Optional[List[str]] = None) -> None:
        """The alert of *session_id* reached the owner (call only after a successful delivery). *entities* are the
        ids that were in view (``Decision.entities``): the owner now knows about them."""
        with self._lock:
            s = next((x for x in self._open.values() if x.id == session_id), None)
            if s is None:
                return
            if LEVELS.get(label, 0) > LEVELS[s.reported_level]:
                s.reported_level = label
            s.reported_people = max(s.reported_people, _int(people))
            s.reported_at = ts
            if entities:
                ent.told(s.entities, entities, s.reported_entities, ts)
                s.reported_entities = s.reported_entities + [i for i in entities if i not in s.reported_entities]
            if message_id is not None:
                s.messages.append({"alert_id": alert_id, "chat_id": str(chat_id), "message_id": int(message_id),
                                   "ts": ts})

    # ---------- reading ----------
    def session_of_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            sid = self._by_alert.get(alert_id)
            for s in list(self._open.values()) + list(reversed(self._closed)):
                if s.id == sid:
                    return asdict(s)
            return None

    def recent(self, since: float, camera: str = "") -> List[Dict[str, Any]]:
        """Sessions (open and closed) active since *since*, oldest first."""
        with self._lock:
            rows = [s for s in self._closed if (s.closed or s.last_active) >= since]
            rows += [s for s in self._open.values() if s.last_active >= since]
            if camera:
                rows = [s for s in rows if s.camera == camera]
            return [dict(asdict(s), summary_line=s.summary_line()) for s in sorted(rows, key=lambda x: x.opened)]

    def close_all(self, now: Optional[float] = None) -> None:
        with self._lock:
            for cam in list(self._open):
                self._close(cam, time.time() if now is None else now)


_BOOK: Optional[EventBook] = None
_BOOK_LOCK = threading.Lock()


def book(directory: Optional[str] = None, notify_normal: bool = False) -> EventBook:
    """The box's one event book (the guard loop and the assistant share it)."""
    global _BOOK
    with _BOOK_LOCK:
        if _BOOK is None:
            if directory is None:
                from . import paths  # noqa: PLC0415

                directory = os.path.join(paths.state_dir(), "events")
            _BOOK = EventBook(directory, notify_normal=notify_normal)
        return _BOOK
