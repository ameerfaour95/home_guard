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
  - What is usual here (optional ``baseline``, baseline.py, task 2.9): a "normal" the camera's history calls rare is
    one quiet message per event with box.yaml ``baseline_alerts: on``. It only raises; nothing else changes.
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
- **Appearance** (stage 3.2, reid.py, box.yaml ``reid: off | shadow | on``): with ``configure(reid=...)`` the
  entities also hear what the clothes say (entities.Appearance): a lost person re-linked, a geometric re-attach kept
  apart. Shadow (the default) only logs ``[cam] reid: would link P3->P1 (0.78)``.
- **One event across cameras** (stage 3.3, box.yaml ``cross_camera: off | shadow | on``, ``camera_neighbours``):
  a person entity that ENDED at one camera's picture edge and a person who STARTS at a neighbouring camera within
  ``cross_sec`` (45 s) are one house-level incident: the second camera's session gets ``incident_id`` and
  ``incident_from``. With ReID on, the clothes confirm it (a match under ``reid_veto`` is a different person, two
  candidates need a clear winner). A suspicious that continues an incident the owner was already told about does not
  open a NEW message: it replies in the first camera's thread (``Decision.incident["thread"]``); an escalation always
  goes out as before. Cameras are linked only when box.yaml names them neighbours; any other pair that fits is only
  logged as a suggestion. Shadow (the default) only logs ``[cam] cross-camera: would link``.

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
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

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
# Stage 3.3, one event across cameras.
CROSS_MODES = ("off", "shadow", "on")
CROSS_SEC = 45.0                 # a person who left one camera's picture and starts at a neighbour's within this
CROSS_OVERLAP_SEC = 3.0          # neighbouring views may show the same person at both for a moment
CROSS_LOOKBACK_SEC = 600.0       # the source camera's tracks read for its latest entities (tracker.HISTORY_SEC)


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
    # Stage 3.3: the house-level incident this session belongs to, and where it came from (``camera, session,
    # entity`` there, ``to_entity`` here, ``gap_s``, ``score`` (the clothes' cosine or None), ``at``).
    incident_id: str = ""
    incident_from: Dict[str, Any] = field(default_factory=dict)

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
    # Stage 3.3: the incident this alert continues (``id, camera, session, entity, to_entity, gap_s, score, mode``;
    # ``thread`` True when it replies in the first camera's thread, ``announce`` when this camera told nothing yet).
    incident: Dict[str, Any] = field(default_factory=dict)

    def record(self) -> Dict[str, Any]:
        return asdict(self)


def _int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def cross_mode_of(settings: Optional[Mapping[str, Any]]) -> str:
    """box.yaml ``cross_camera``: off | shadow | on (default shadow; anything else is shadow)."""
    value = (settings or {}).get("cross_camera", "shadow") if isinstance(settings, Mapping) else "shadow"
    if value is False:
        return "off"
    if value is True:
        return "on"
    value = str(value or "shadow").strip().lower()
    return value if value in CROSS_MODES else "shadow"


def cross_sec_of(settings: Optional[Mapping[str, Any]]) -> float:
    try:
        value = float((settings or {}).get("cross_sec", CROSS_SEC))
    except (TypeError, ValueError):
        return CROSS_SEC
    return value if 1.0 <= value <= 600.0 else CROSS_SEC


def neighbours_of(raw: Any) -> Dict[str, Set[str]]:
    """box.yaml ``camera_neighbours: {gate: [entrance], entrance: [door]}`` as camera -> neighbours, both ways
    (a walk from the gate to the entrance and back are the same pair). Anything unreadable is left out."""
    out: Dict[str, Set[str]] = {}
    if isinstance(raw, Mapping):
        for cam, others in raw.items():
            items = [others] if isinstance(others, str) else list(others) if isinstance(others, (list, tuple, set)) else []
            for other in items:
                a, b = str(cam).strip(), str(other).strip()
                if a and b and a != b:
                    out.setdefault(a, set()).add(b)
                    out.setdefault(b, set()).add(a)
    return out


class EventBook:
    def __init__(self, directory: str, notify_normal: bool = False, idle_sec: float = IDLE_SEC,
                 roll_sec: float = ROLL_SEC, cross_camera: str = "off", cross_sec: float = CROSS_SEC,
                 neighbours: Any = None) -> None:
        self.directory = directory
        # Stage 3.2 / 3.3 (configure()): the appearance memory (reid.Reid), the tracker's snapshot for another
        # camera's latest entities (``tracks_source(camera, t0, t1)``), and the cross-camera switch and neighbours.
        self.reid: Any = None
        self.tracks_source: Optional[Callable[[str, float, float], Any]] = None
        self.cross_mode = cross_camera if cross_camera in CROSS_MODES else "off"
        self.cross_sec = float(cross_sec)
        self.neighbours: Dict[str, Set[str]] = neighbours_of(neighbours)
        self.cross_suggested: Dict[Tuple[str, str], int] = {}    # (from, to) pairs that fit but are not neighbours
        self._cross_said: Set[Tuple[str, str]] = set()          # (session, entity) already logged
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
                        reported_entities=list(old.reported_entities) if old else [],
                        incident_id=old.incident_id if old else "",
                        incident_from=dict(old.incident_from) if old else {})
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

    def configure(self, cross_camera: Optional[str] = None, cross_sec: Optional[float] = None,
                  neighbours: Any = None, reid: Any = None, tracks_source: Any = None) -> None:
        """Stage 3.2 / 3.3 settings, each only when given (the book is started before the tracker and the ReID)."""
        with self._lock:
            if cross_camera is not None:
                self.cross_mode = cross_camera if cross_camera in CROSS_MODES else "off"
            if cross_sec is not None:
                self.cross_sec = float(cross_sec)
            if neighbours is not None:
                self.neighbours = neighbours_of(neighbours)
            if reid is not None:
                self.reid = reid
            if tracks_source is not None:
                self.tracks_source = tracks_source

    def open_track_keys(self) -> Dict[str, Set[str]]:
        """camera -> the tracker keys of its open session's entities (what reid.Reid.prune may keep)."""
        with self._lock:
            return {cam: {k for e in s.entities for k in e.get("tracks", ())} for cam, s in self._open.items()}

    def _appearance(self, camera: str) -> Any:
        if self.reid is None:
            return None
        try:
            return self.reid.appearance(camera)
        except Exception as exc:  # noqa: BLE001 - without the clothes, geometry decides
            log.debug("[%s] reid appearance not available: %s", camera, exc)
            return None

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
            in_view = ent.ingest(rows, tracks or (), ts, since=since, not_before=self._not_before(camera, ts),
                                 appearance=self._appearance(camera))
            return {"in_view": in_view, "line": ent.roster_line(rows, in_view, ts, since)}

    def decide(self, camera: str, ts: float, label: str, people: Any = None, summary: str = "",
               alert_id: str = "", tracks: Any = None, since: Optional[float] = None, note: str = "",
               per_entity: Any = None, ground: Optional[Dict[str, Any]] = None,
               baseline: Optional[Dict[str, Any]] = None) -> Decision:
        """Should this alert job reach the owner? Records the observation in the camera's session either way.

        With *tracks* (the tracker's ``snapshot`` around the alert, stage 2a) the session's entities are brought up to
        date first; the people seen since *since* are this alert's people, and when the tracker saw at least one of
        them "more people" and the owner's known group are judged by those entities instead of the Eye's head-count
        *people*. *note* (the owner-language summary) and *per_entity* (the Eye's ``per_entity`` answer) are what the
        entities in view did (entities.attribute). *ground* (optional, stage 2c): where it happened by the scene map
        (ground.Ground.record() plus ``action``). *baseline* (optional, task 2.9, baseline.py): ``{"raise": True,
        "text_en": ...}`` when what is usual at this camera says this is rare and box.yaml ``baseline_alerts`` is on;
        it only RAISES: a normal becomes one quiet message per event, a suspicious or an escalation is unchanged."""
        label = label if label in LEVELS else "normal"
        count = _int(people)
        with self._lock:
            in_view: List[str] = []
            not_before = self._not_before(camera, ts)
            s = self._session(camera, ts)
            incident: Dict[str, Any] = {}
            if tracks is not None:
                appearance = self._appearance(camera)
                in_view = ent.ingest(s.entities, tracks, ts, since=since, not_before=not_before,
                                     appearance=appearance)
                for line in appearance.lines() if appearance is not None else ():
                    log.info("[%s] %s", camera, line)
                if self.cross_mode != "off":
                    try:
                        incident = self._cross(camera, s, ts)
                    except Exception as exc:  # noqa: BLE001 - the incident only adds; the event goes on
                        log.warning("[%s] cross-camera link failed: %s", camera, exc)
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

            thread_to = self._incident_thread(s) if incident.get("mode") == "on" else None

            def make(notify: bool, reason: str, known_text: str = "", unmarked: bool = False) -> Decision:
                reply, info = reply_to, dict(incident)
                if info:
                    info["announce"] = s.reported_level == "none"
                if notify and label == "suspicious" and thread_to is not None and not s.messages:
                    # The owner already has this incident's thread at the first camera: no NEW message.
                    reply, info["thread"] = thread_to, True
                    reason = f"{reason}; continues the incident from {incident.get('camera')}: in its thread"
                return Decision(notify, s.id, reason, reply, new_people, known_text, list(in_view), list(fresh),
                                unmarked, "entities" if by_entities else "head-count", info)

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
                usual = baseline if isinstance(baseline, dict) else {}
                if usual.get("raise") and reported == 0:
                    return make(True, "normal, but rare here: a quiet message, once in this event ("
                                + str(usual.get("text_en") or "rare at this camera")[:160] + ")")
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

    # ---------- one event across cameras (stage 3.3) ----------
    def _find(self, session_id: str) -> Optional[Session]:
        for x in list(self._open.values()) + list(reversed(self._closed)):
            if x.id == session_id:
                return x
        return None

    def _incident_thread(self, s: Session) -> Optional[Dict[str, Any]]:
        """The first camera's first message when the owner was told about the incident there (suspicious or more)."""
        src = self._find(str(s.incident_from.get("session") or "")) if s.incident_id else None
        if src is None or LEVELS.get(src.reported_level, 0) < LEVELS["suspicious"]:
            return None
        return src.first_message()

    def _sources(self, camera: str, ts: float) -> List[Session]:
        """Sessions on other cameras someone could have just walked out of: open, or closed within cross_sec."""
        out = [x for cam, x in self._open.items() if cam != camera and ts - x.last_active <= self.idle_sec + self.cross_sec]
        out += [x for x in self._closed[-50:] if x.camera != camera and ts - (x.closed or x.last_active) <= self.cross_sec]
        return out

    def _latest(self, src: Session, ts: float) -> List[Dict[str, Any]]:
        """*src*'s entities brought up to *ts* from its camera's tracker, on a copy (that camera's own alert
        updates the real ones)."""
        rows = copy.deepcopy(src.entities)
        if self.tracks_source is None or self._open.get(src.camera) is not src:
            return rows
        try:
            tracks = self.tracks_source(src.camera, ts - CROSS_LOOKBACK_SEC, ts)
            ent.ingest(rows, tracks or (), ts, not_before=src.opened - ENTITY_LOOKBACK_SEC,
                       appearance=self._appearance(src.camera))
        except Exception as exc:  # noqa: BLE001 - the session's own entities then
            log.debug("[%s] latest entities not read: %s", src.camera, exc)
            return copy.deepcopy(src.entities)
        return rows

    def _clothes(self, cam_a: str, e_a: Dict[str, Any], cam_b: str, e_b: Dict[str, Any]) -> Optional[float]:
        if self.reid is None or getattr(self.reid, "mode", "off") == "off":
            return None
        try:
            value = self.reid.cross_score(cam_a, e_a.get("tracks", ()), cam_b, e_b.get("tracks", ()))
        except Exception:  # noqa: BLE001
            return None
        return None if value is None else round(float(value), 3)

    def _cross(self, camera: str, s: Session, ts: float) -> Dict[str, Any]:
        """Is a person of *s* (this camera) one who just left another camera's picture? Returns the incident
        (``mode`` on: linked and recorded; shadow: only what it would be), or {}."""
        if s.incident_id and s.incident_from:
            return dict(s.incident_from, id=s.incident_id, mode="on")
        mine = [e for e in s.entities if e.get("kind") == "person" and not e.get("from_camera")]
        if not mine:
            return {}
        acting_reid = self.reid is not None and getattr(self.reid, "mode", "off") == "on"
        link = float(getattr(getattr(self.reid, "settings", None), "link", ent.REID_LINK))
        margin = float(getattr(getattr(self.reid, "settings", None), "margin", ent.REID_MARGIN))
        veto = float(getattr(getattr(self.reid, "settings", None), "veto", ent.REID_VETO))
        options: Dict[str, List[Dict[str, Any]]] = {}
        for src in self._sources(camera, ts):
            for e_a in self._latest(src, ts):
                if e_a.get("kind") != "person" or e_a.get("state") == ent.ACTIVE or not e_a.get("exit_edge"):
                    continue
                for e_b in mine:
                    gap = float(e_b["first_seen"]) - float(e_a["last_seen"])
                    if not -CROSS_OVERLAP_SEC <= gap <= self.cross_sec:
                        continue
                    score = self._clothes(src.camera, e_a, camera, e_b)
                    if acting_reid and score is not None and score < veto:
                        continue                  # the clothes say a different person
                    options.setdefault(e_b["id"], []).append({"src": src, "e_a": e_a, "gap": gap, "score": score})
        claimed: Set[Tuple[str, str]] = set()
        for e_b in sorted(mine, key=lambda e: float(e["first_seen"])):
            found = [o for o in options.get(e_b["id"], ()) if (o["src"].id, o["e_a"]["id"]) not in claimed]
            pick = self._one(found, acting_reid, link, margin)
            if pick is None:
                continue
            claimed.add((pick["src"].id, pick["e_a"]["id"]))
            src, e_a = pick["src"], pick["e_a"]
            info = {"camera": src.camera, "session": src.id, "entity": e_a["id"], "to_entity": e_b["id"],
                    "gap_s": round(pick["gap"], 1), "score": pick["score"], "at": ts}
            neighbour = src.camera in self.neighbours.get(camera, ())
            said = (s.id, e_b["id"]) in self._cross_said
            self._cross_said.add((s.id, e_b["id"]))
            clothes = f"; clothes {pick['score']:.2f}" if pick["score"] is not None else ""
            if not neighbour:
                if not said:
                    pair = (src.camera, camera)
                    self.cross_suggested[pair] = self.cross_suggested.get(pair, 0) + 1
                    log.info("[%s] cross-camera suggestion: %s at %s fits %s here (%.0f s%s); not linked, %s is not "
                             "in camera_neighbours (seen %d times)", camera, e_a["id"], src.camera, e_b["id"],
                             pick["gap"], clothes, src.camera, self.cross_suggested[pair])
                continue
            if self.cross_mode != "on":
                if not said:
                    log.info("[%s] cross-camera: would link %s at %s -> %s here (%.0f s%s)", camera, e_a["id"],
                             src.camera, e_b["id"], pick["gap"], clothes)
                return dict(info, mode="shadow")
            incident_id = src.incident_id or uuid.uuid4().hex[:12]
            src.incident_id = incident_id
            s.incident_id, s.incident_from = incident_id, info
            e_b.update(from_camera=src.camera, from_entity=e_a["id"], from_session=src.id)
            log.info("[%s] cross-camera: %s at %s is %s here (%.0f s%s): incident %s", camera, e_a["id"], src.camera,
                     e_b["id"], pick["gap"], clothes, incident_id)
            return dict(info, id=incident_id, mode="on")
        return {}

    @staticmethod
    def _one(found: List[Dict[str, Any]], acting_reid: bool, link: float, margin: float) -> Optional[Dict[str, Any]]:
        """One clear source person, or None: a lone candidate; of several, the clothes' clear winner (ReID on)."""
        if len(found) == 1:
            return found[0]
        if len(found) < 2 or not acting_reid:
            return None
        scored = sorted((o for o in found if o["score"] is not None), key=lambda o: -o["score"])
        if len(scored) < len(found):
            return None
        if scored[0]["score"] >= link and scored[0]["score"] - scored[1]["score"] >= margin:
            return scored[0]
        return None

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
