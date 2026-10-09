"""Index a box's legacy S3 prefixes into Events, Artifacts, AI runs and Feedback.

One clip can exist twice: the production copy (`production_<site>/`, what the owner was sent) and the
training copy (`dataset_<site>/`, with the teacher inputs and answer). Both copies merge into one Event
keyed by (device, camera, stem); every object keeps its own Artifact row and every JSON body fetched is
saved as a RawRevision before merging, so differing metadata is never lost.

A pass over one device:
1. bulk-loads the device's artifacts, problems and cursors (a few statements, not one per key);
2. lists each prefix once. `Artifact.etag` is what S3 lists now; `Artifact.applied_etag` is the JSON revision
   whose content is in effect. A JSON object whose listed ETag differs from its applied one is (re)applied:
   from the stored revision when that (key, ETag) was fetched before -- no GET -- otherwise from a GET
   conditioned on the listed ETag, so a body replaced after the listing is never saved under the wrong ETag;
3. applies each body. A revision that cannot be applied (fetch failed, not JSON, unusable, timestamps out of
   range) leaves the previous applied state in place and records an IndexProblem about that ETag; a problem
   clears only when the key's current revision applies cleanly;
4. rebuilds the touched events from stored revisions only. The AI run (and the event's summary, label and
   command) comes from the best revision ever seen of either copy -- a later, poorer copy (A7: training meta
   recreated from production without the teacher) never replaces it -- and owner feedback is the union over
   every retained revision of both copies. Delivery (`dispatch`, `muted`) comes only from the production copy;
5. refreshes `expired` for events whose 14-day production retention passed with no copy of the video left (a
   training copy that outlives the production one keeps the event unexpired and exportable);
6. records the household's current names as identity aliases and fills the labelers' redacted search text
   (`Event.summary_redacted`) for every event that lacks it (for every event when a new name appeared).

Concurrent writers (two API processes, or a lock bug) never abort a pass: every row the indexer creates is
inserted with INSERT ... ON CONFLICT on its unique key and then re-read, so a row another writer created first is
taken over instead of raising a unique violation.
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from sqlalchemy import and_, exists, or_, select, tuple_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from home_guard_project.fleet_contract import event_outcome
from home_guard_project.fleet_contract._time import parse_utc
from home_guard_project.fleet_contract.keys import KeyInfo, parse_key, stem_kind
from home_guard_project.fleet_contract.legacy import ClipRecord, parse_feedback, parse_heartbeat, parse_meta

from . import redact
from .models import AiRun, Artifact, Device, Event, Feedback, IndexProblem, RawRevision, S3Cursor
from .s3 import S3, ETagMismatch

log = logging.getLogger(__name__)

ROOTS = ("production", "dataset")
COPY_NAME = {"production": "production", "dataset": "training"}
INDEXED_AREAS = {"meta", "clips", "feedback", "status", "responses", "vlm_crops", "yolo_images", "yolo_labels"}
# Second key segment of the indexed areas: a key there that parse_key rejects is an "invalid key layout".
INDEXED_DIRS = {"meta", "clips", "feedback", "_status", "responses", "vlm_crops", "yolo"}
EVENT_ROLES = ("original_video", "meta", "raw_answer", "teacher_frame", "crop_video", "yolo_image", "yolo_label",
               "tracks")
TRACKS_SUFFIX = ".tracks.json"  # the box tracker's tracks of a clip (box clip_tracks.py), in responses/
AI_RANK = {"none": 0, "fallback": 1, "failed": 2, "real": 3}
DECISION_BACKFILL = 500  # events without a parsed event-layer decision rebuilt per pass (from stored revisions)
FULL_SCAN_EVERY = timedelta(minutes=30)
PRODUCTION_RETENTION = timedelta(days=14)
INVALID_JSON = "_invalid_json"  # RawRevision body marker for an object that is not storable JSON
HEARTBEAT_SUFFIX = "/_status/heartbeat.json"
RACE_LIMIT = 3  # consecutive passes an object may change between LIST and GET before it is a problem
CHUNK = 2000  # keys per IN (...) list
INSERT_ROWS = 1000  # rows per multi-row INSERT (stays far below PostgreSQL's 65535 parameters)
_ARTIFACT_COLUMNS = ("role", "s3_key", "etag", "bytes", "mime", "available", "provenance", "detail", "camera",
                     "stem", "last_modified", "etag_mismatches")
MIN_TS = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
MAX_TS = datetime(2100, 1, 1, tzinfo=timezone.utc).timestamp()
MIME = {".mp4": "video/mp4", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".json": "application/json",
        ".txt": "text/plain"}


@dataclass
class IndexStats:
    new_events: int = 0
    updated_events: int = 0
    artifacts: int = 0
    feedback: int = 0
    problems: int = 0


@dataclass
class _Copy:
    """One usable meta revision of one copy."""
    root: str
    key: str
    etag: str
    rec: ClipRecord
    body: Any


def artifact_role(info: KeyInfo) -> Optional[str]:
    """Artifact role for an indexed key, or None when the key is not indexed."""
    area, key = info.area, info.key
    if area == "clips":
        return "original_video"
    if area == "meta":
        return "meta" if key.endswith(".meta.json") else None
    if area == "responses":
        return "tracks" if key.endswith(TRACKS_SUFFIX) else "raw_answer"
    if area == "vlm_crops":
        return {".jpg": "teacher_frame", ".jpeg": "teacher_frame", ".mp4": "crop_video"}.get(info.ext.lower())
    if area == "yolo_images":
        return "yolo_image"
    if area == "yolo_labels":
        return "yolo_label"
    if area == "feedback":
        return "feedback" if key.endswith(".feedback.json") else None
    if area == "status":
        return "status"
    return None


def owner_answer(body: dict) -> dict:
    """The owner's own answer in a box feedback file (box feedback.py ``record``), beyond what the shared contract's
    parse_feedback keeps: the Telegram tag, his words, a voice transcript and who tagged (else the sender). Strings
    only; anything else reads as ""."""
    text = lambda name: body.get(name) if isinstance(body.get(name), str) else ""  # noqa: E731
    sender = body.get("from")
    sender = sender.get("name") if isinstance(sender, dict) else sender
    return {"owner_label": text("owner_label").strip(), "owner_text": text("owner_text"),
            "transcript": text("transcript"),
            "tagged_by": text("tagged_by") or (sender if isinstance(sender, str) else ""),
            "superseded_by": text("superseded_by").strip(), "superseded_utc": body.get("superseded_utc")}


def _cut(value, n: int):
    return value[:n] if isinstance(value, str) else value


def _prefix(root: str, site: str) -> str:
    return f"{root}_{site}/"


def _like_prefix(column, prefix: str):
    """`column LIKE 'prefix%'` with LIKE wildcards in the prefix escaped (site names contain `_`)."""
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return column.like(escaped + "%", escape="\\")


def _chunks(items: Iterable, n: int = CHUNK):
    items = list(items)
    for i in range(0, len(items), n):
        yield items[i:i + n]


def _in_range(ts) -> bool:
    """A Unix timestamp the system can have produced: finite, 2020-01-01 <= ts < 2100-01-01."""
    try:
        value = float(ts)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(value) and MIN_TS <= value < MAX_TS


def _has_nul(value) -> bool:
    """PostgreSQL JSONB cannot store U+0000 anywhere in a document."""
    if isinstance(value, str):
        return "\x00" in value
    if isinstance(value, dict):
        return any(_has_nul(k) or _has_nul(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_has_nul(v) for v in value)
    return False


def _decode(text: str) -> Any:
    try:
        body = json.loads(text)
    except ValueError:
        pass
    else:
        if not _has_nul(body):
            return body
    return {INVALID_JSON: text[:100_000].replace("\x00", "")}


def _is_invalid(body: Any) -> bool:
    return isinstance(body, dict) and INVALID_JSON in body


def _in_indexed_dir(key: str) -> bool:
    parts = key.split("/")
    return len(parts) > 1 and parts[1] in INDEXED_DIRS


def _raw_muted(body: Any) -> Optional[bool]:
    alert = body.get("alert") if isinstance(body, dict) else None
    muted = alert.get("muted") if isinstance(alert, dict) else None
    return muted if isinstance(muted, bool) else None


def _expired(ev: Event, any_clip_available: bool, now: datetime) -> bool:
    """Expired = the production retention has passed and no copy of the video is left (production or training)."""
    return ev.expires_at is not None and not any_clip_available and ev.expires_at < now


class _Run:
    """State of one index pass over one device."""

    def __init__(self, session: Session, s3: S3, device: Device, full_scan: bool, now: datetime):
        self.session, self.s3, self.device = session, s3, device
        self.full_scan, self.now = full_scan, now
        self.site = device.site
        self.prefixes = [_prefix(root, self.site) for root in ROOTS]
        self.stats = IndexStats()
        self.arts: dict[str, Artifact] = {}  # only keys parse_key accepts
        self.new_arts: dict[str, Artifact] = {}  # listed keys with no row yet (transient until inserted)
        self.legacy: set[str] = set()  # persisted artifact keys parse_key now rejects (quarantined)
        self.problems: dict[str, IndexProblem] = {}
        self.body_names: set[tuple[str, str]] = set()  # names in the JSON bodies applied this pass (redact)
        self.cursors: dict[str, S3Cursor] = {}
        self.feedback: Optional[dict[str, Feedback]] = None
        self.answers: dict[str, dict] = {}   # feedback key -> owner_answer() of the revision being applied
        self.events: dict[tuple[str, str], Event] = {}
        self.events_by_id: dict[int, Event] = {}
        self.dirty: set[tuple[str, str]] = set()  # (camera, stem) of events to rebuild
        self.dirty_ids: set[int] = set()  # event ids to rebuild
        self.work: list[tuple[Artifact, KeyInfo, str, str]] = []  # JSON whose listed revision is not applied
        self.parsed: dict[tuple[str, str], tuple[ClipRecord, Optional[ClipRecord], list[str]]] = {}

    # ------------------------------------------------------------ bulk loads

    def load(self) -> None:
        session = self.session
        self.arts = {a.s3_key: a for a in session.scalars(select(Artifact).where(
            or_(*(_like_prefix(Artifact.s3_key, p) for p in self.prefixes))))}
        self.problems = {p.s3_key: p for p in session.scalars(select(IndexProblem).where(
            or_(*(_like_prefix(IndexProblem.s3_key, p) for p in self.prefixes))))}
        self.cursors = {c.prefix: c for c in session.scalars(
            select(S3Cursor).where(S3Cursor.prefix.in_(self.prefixes)))}
        self.quarantine_legacy()

    def quarantine_legacy(self) -> None:
        """Artifacts an earlier indexer created for keys parse_key now rejects (traversal, impossible day, a
        stem of another camera): made unavailable, detached from their event (which is rebuilt) and recorded
        as a problem, so no later step ever parses them."""
        for key in [k for k in self.arts if parse_key(k) is None]:
            art = self.arts.pop(key)
            self.legacy.add(key)
            if art.event_id is not None:
                self.dirty_ids.add(art.event_id)
            art.available, art.event_id = False, None
            self.problem(key, "invalid key layout (legacy)", art.etag)

    def load_feedback(self) -> dict[str, Feedback]:
        if self.feedback is None:
            self.feedback = {fb.s3_key: fb for fb in self.session.scalars(
                select(Feedback).where(Feedback.device_pk == self.device.id).order_by(Feedback.id))}
        return self.feedback

    def load_events(self, pairs: Iterable[tuple[str, str]], ids: Iterable[int], stems: Iterable[str]) -> None:
        """Load the device's events by (camera, stem), id or stem into the pass's maps (one query per chunk)."""
        pairs = sorted(p for p in pairs if p not in self.events)
        ids = sorted(i for i in ids if i not in self.events_by_id)
        stems = sorted(set(stems))
        if not (pairs or ids or stems):
            return
        for n, batch in enumerate(list(_chunks(pairs)) or [[]]):
            conds = [tuple_(Event.camera, Event.stem).in_(batch)] if batch else []
            if n == 0:
                conds += [Event.id.in_(chunk) for chunk in _chunks(ids)]
                conds += [Event.stem.in_(chunk) for chunk in _chunks(stems)]
            for ev in self.session.scalars(select(Event).where(Event.device_pk == self.device.id, or_(*conds))
                                           .order_by(Event.id)):
                self.events[(ev.camera, ev.stem)] = ev
                self.events_by_id[ev.id] = ev

    # ------------------------------------------------------------ problems

    def problem(self, key: str, reason: str, etag: Optional[str]) -> None:
        """One IndexProblem per key, about one revision; an unchanged problem is not re-counted."""
        row = self.problems.get(key)
        if row is not None and row.reason == reason and row.etag == etag:
            return
        if row is None:
            self.session.flush()  # a pending delete of this key (cleared earlier in the pass) goes first
            self.session.execute(pg_insert(IndexProblem).values(s3_key=key, reason=reason, seen_at=self.now,
                                                                etag=etag)
                                 .on_conflict_do_update(index_elements=[IndexProblem.s3_key],
                                                        set_={"reason": reason, "seen_at": self.now, "etag": etag}))
            row = self.session.get(IndexProblem, key, populate_existing=True)
            self.problems[key] = row
        else:
            row.reason, row.etag, row.seen_at = reason, etag, self.now
        self.stats.problems += 1

    def clear_problem(self, key: str) -> None:
        row = self.problems.pop(key, None)
        if row is not None:
            if row in self.session.new:
                self.session.expunge(row)
            else:
                self.session.delete(row)

    def applied(self, art: Artifact, etag: str, problems: Iterable[str] = ()) -> None:
        """The key's current revision is now in effect; its own problems replace any earlier ones."""
        art.applied_etag = etag
        problems = list(problems)
        if problems:
            self.problem(art.s3_key, "; ".join(problems), etag)
        else:
            self.clear_problem(art.s3_key)

    # ------------------------------------------------------------ listing

    def list_prefix(self, root: str) -> None:
        prefix = _prefix(root, self.site)
        seen: set[str] = set()
        for obj in self.s3.list(prefix):
            info = parse_key(obj.key)
            if info is None:
                if obj.key in self.legacy:
                    continue  # already quarantined with its own problem
                if obj.key.startswith(prefix) and _in_indexed_dir(obj.key):
                    self.problem(obj.key, "invalid key layout", obj.etag)
                continue
            if info.site != self.site or info.area not in INDEXED_AREAS:
                continue
            role = artifact_role(info)
            if role is None:
                continue
            seen.add(obj.key)
            camera, stem = info.camera, info.stem
            if role == "tracks":  # "<stem>.tracks.json": the key layout reads its stem as "<stem>.tracks"
                stem = stem[:-len(".tracks")] if stem and stem.endswith(".tracks") else stem
            art = self.arts.get(obj.key)
            if art is not None and art.role != role:  # indexed as a raw answer before the tracks role existed
                art.role, art.stem, art.event_id = role, stem, None
                art.etag = ""                         # a changed revision: re-linked to its event below
            if art is None:  # inserted (conflict-safe) once both prefixes are listed
                art = Artifact(role=role, s3_key=obj.key, etag=obj.etag, bytes=obj.size,
                               mime=MIME.get(info.ext.lower()), available=True, provenance="box",
                               detail={"copy": COPY_NAME[root]}, camera=camera, stem=stem,
                               last_modified=obj.last_modified, etag_mismatches=0)
                self.arts[obj.key] = art
                self.new_arts[obj.key] = art
                changed = revision_changed = True
            else:
                revision_changed = art.etag != obj.etag or not art.available
                changed = revision_changed or art.bytes != obj.size
                if changed:
                    art.etag, art.bytes, art.last_modified, art.available = (obj.etag, obj.size,
                                                                             obj.last_modified, True)
            if changed:
                self.stats.artifacts += 1
            if revision_changed:
                if role in EVENT_ROLES and camera and stem:
                    self.dirty.add((camera, stem))
                if art.event_id is not None:
                    self.dirty_ids.add(art.event_id)
            is_json = role in ("meta", "feedback") or (role == "status" and obj.key.endswith(HEARTBEAT_SUFFIX))
            stale = self.problems.get(obj.key)
            # A -> unusable B -> A again: A is still applied, but B's problem must go, so A is replayed (cached)
            if is_json and (art.applied_etag != obj.etag or (stale is not None and stale.etag != obj.etag)):
                self.work.append((art, info, role, obj.etag))
        self.disappearance_check(prefix, seen)

    def disappearance_check(self, prefix: str, seen: set[str]) -> None:
        cursor = self.cursors.get(prefix)
        if cursor is None:
            self.session.execute(pg_insert(S3Cursor).values(prefix=prefix)
                                 .on_conflict_do_nothing(index_elements=[S3Cursor.prefix]))
            cursor = self.session.get(S3Cursor, prefix, populate_existing=True)
            self.cursors[prefix] = cursor
        last = cursor.last_full_scan
        if not (self.full_scan or last is None or self.now - last >= FULL_SCAN_EVERY):
            return
        for key, art in self.arts.items():
            if key.startswith(prefix) and key not in seen and art.available:
                art.available = False
                if art.event_id is not None:
                    self.dirty_ids.add(art.event_id)
        cursor.last_full_scan = self.now

    def insert_new_artifacts(self) -> None:
        """Insert the listed keys that had no row (multi-row INSERT ... ON CONFLICT DO NOTHING), then re-read them:
        a row another writer created meanwhile is taken over with what this listing saw."""
        if not self.new_arts:
            return
        keys = list(self.new_arts)  # listing order (every writer lists in the same order)
        rows = [{c: getattr(self.new_arts[k], c) for c in _ARTIFACT_COLUMNS} for k in keys]
        for chunk in _chunks(rows, INSERT_ROWS):
            self.session.execute(pg_insert(Artifact).values(chunk)
                                 .on_conflict_do_nothing(index_elements=[Artifact.s3_key]))
        loaded: dict[str, Artifact] = {}
        for chunk in _chunks(keys):
            for art in self.session.scalars(select(Artifact).where(Artifact.s3_key.in_(chunk))):
                loaded[art.s3_key] = art
        for key in keys:
            new, art = self.new_arts[key], loaded[key]
            if (art.etag, art.bytes, art.available) != (new.etag, new.bytes, True):
                art.etag, art.bytes, art.last_modified, art.available = new.etag, new.bytes, new.last_modified, True
            self.arts[key] = art
        self.work = [(self.arts[a.s3_key] if a.s3_key in self.new_arts else a, info, role, etag)
                     for a, info, role, etag in self.work]
        self.new_arts.clear()

    # ------------------------------------------------------------ JSON bodies

    def obtain_bodies(self) -> dict[tuple[str, str], Any]:
        """Body of each pending (key, listed ETag): its stored revision, else a GET conditioned on that ETag."""
        bodies: dict[tuple[str, str], Any] = {}
        wanted = sorted({(art.s3_key, etag) for art, _, _, etag in self.work})
        with self.session.no_autoflush:  # new artifacts are inserted once, after their revision is applied
            for chunk in _chunks(wanted):
                for key, etag, body in self.session.execute(
                        select(RawRevision.s3_key, RawRevision.etag, RawRevision.body)
                        .where(tuple_(RawRevision.s3_key, RawRevision.etag).in_(chunk))):
                    bodies[(key, etag)] = body
        fetched: list[dict] = []
        for art, _, _, etag in self.work:
            key = art.s3_key
            if (key, etag) in bodies:
                continue
            try:
                text = self.s3.get_text(key, if_match=etag)
            except ETagMismatch:  # replaced after the listing: the next pass lists the new revision
                art.etag_mismatches = (art.etag_mismatches or 0) + 1
                if art.etag_mismatches >= RACE_LIMIT:
                    self.problem(key, f"object changed during fetch on {RACE_LIMIT}+ passes in a row", etag)
                continue
            except Exception as e:  # deleted between listing and get, throttled, ...
                self.problem(key, f"fetch failed: {type(e).__name__}", etag)
                continue
            art.etag_mismatches = 0
            body = _decode(text)
            fetched.append({"s3_key": key, "etag": etag, "fetched_at": self.now, "body": body})
            bodies[(key, etag)] = body
        for chunk in _chunks(fetched, INSERT_ROWS):  # a revision another writer stored meanwhile is the same body
            self.session.execute(pg_insert(RawRevision).values(chunk)
                                 .on_conflict_do_nothing(index_elements=[RawRevision.s3_key, RawRevision.etag]))
        return bodies

    def parse(self, key: str, etag: str, body: Any) -> tuple[ClipRecord, Optional[ClipRecord], list[str]]:
        """(record, the record if usable else None, problems) for one meta revision; cached for the pass."""
        cached = self.parsed.get((key, etag))
        if cached is not None:
            return cached
        rec = parse_meta(key, body)
        problems = list(rec.problems)
        if _sanitise_verdicts(rec.owner_feedback):
            problems.append("invalid verdict")
        bad = [name for name, ts in (("clip_start_ts", rec.start_ts), ("clip_end_ts", rec.end_ts),
                                     ("stem trigger time", rec.trigger_ts)) if ts is not None and not _in_range(ts)]
        no_start = rec.start_ts is None and rec.trigger_ts is None
        if bad:
            problems.append("timestamp out of range: " + ", ".join(bad))
        elif no_start:
            problems.append("no start time")
        result = (rec, None if bad or no_start else rec, problems)
        self.parsed[(key, etag)] = result
        return result

    def apply_bodies(self, bodies: dict[tuple[str, str], Any]) -> list[tuple[Artifact, str, Any]]:
        """Apply meta and heartbeat revisions; return the feedback revisions to apply once events are loaded."""
        feedback = []
        for art, info, role, etag in self.work:
            key = art.s3_key
            if (key, etag) not in bodies:
                continue  # not fetched this pass: the previous applied state and its problem stay
            body = bodies[(key, etag)]
            self.body_names |= redact.body_names(body)
            if _is_invalid(body):
                self.problem(key, "invalid json", etag)
            elif role == "meta":
                _, usable, problems = self.parse(key, etag, body)
                if usable is None:
                    self.problem(key, "; ".join(problems), etag)
                    continue
                self.applied(art, etag, problems)
                self.dirty.add((info.camera, info.stem))
            elif role == "feedback":
                if not isinstance(body, dict):
                    self.problem(key, "invalid feedback body", etag)
                    continue
                rec = parse_feedback(key, body)
                if rec.time_utc is not None and not _in_range(rec.time_utc.timestamp()):
                    self.problem(key, "timestamp out of range: time_utc", etag)
                    continue
                feedback.append((art, etag, rec))
                self.answers[key] = owner_answer(body)
            else:
                if not isinstance(body, dict):
                    self.problem(key, "invalid heartbeat body", etag)
                    continue
                hb = parse_heartbeat(body)
                if hb.time_utc is not None and not _in_range(hb.time_utc.timestamp()):
                    self.problem(key, "timestamp out of range: time_utc", etag)
                    continue
                self.device.last_heartbeat = body
                self.device.last_heartbeat_at = hb.time_utc
                self.applied(art, etag)
        return feedback

    def apply_feedback(self, items: list[tuple[Artifact, str, Any]]) -> None:
        rows = self.load_feedback()
        by_stem: dict[str, Event] = {}
        for ev in sorted(self.events.values(), key=lambda e: e.id or 0):
            by_stem.setdefault(ev.stem, ev)
        for art, etag, rec in items:
            fb = rows.get(art.s3_key)
            if fb is None:
                self.session.execute(pg_insert(Feedback).values(s3_key=art.s3_key, device_pk=self.device.id)
                                     .on_conflict_do_nothing(index_elements=[Feedback.s3_key]))
                fb = self.session.scalars(select(Feedback).where(Feedback.s3_key == art.s3_key)).one()
                rows[art.s3_key] = fb
            elif fb.event_id is not None:
                self.dirty_ids.add(fb.event_id)  # the event it was linked to recomputes its verdicts
            alert_stem = rec.alert_id if rec.alert_id and len(rec.alert_id) <= 255 else None
            fb.alert_stem = alert_stem
            invalid = bool(rec.verdict) and rec.verdict not in redact.VERDICTS
            fb.verdict, fb.action = redact.verdict(rec.verdict) if rec.verdict else "", _cut(rec.action, 64)
            fb.note, fb.raw_text, fb.source = rec.note, rec.raw_text, _cut(rec.source, 64)
            fb.scope_camera = _cut(rec.scope_camera, 128)
            answer = self.answers.get(art.s3_key) or owner_answer({})
            fb.owner_label, fb.owner_text = _cut(answer["owner_label"], 32), answer["owner_text"]
            fb.transcript, fb.tagged_by = answer["transcript"], _cut(answer["tagged_by"], 128)
            fb.superseded_by = _cut(answer["superseded_by"], 255)
            fb.superseded_at = parse_utc(answer["superseded_utc"]) if answer["superseded_by"] else None
            fb.received_at = rec.time_utc
            ev = by_stem.get(alert_stem) if alert_stem else None
            fb.event_id = ev.id if ev is not None else None
            art.event_id = fb.event_id  # the feedback artifact follows its row, including to no event
            if ev is not None:
                self.dirty_ids.add(ev.id)
            self.applied(art, etag, ["invalid verdict"] if invalid else ())
            self.stats.feedback += 1

    # ------------------------------------------------------------ events

    def rebuild_all(self) -> None:
        targets = set(self.dirty) | {(self.events_by_id[i].camera, self.events_by_id[i].stem)
                                     for i in self.dirty_ids if i in self.events_by_id}
        if not targets:
            return
        session = self.session
        by_pair: dict[tuple[str, str], list[Artifact]] = defaultdict(list)
        for art in self.arts.values():
            if art.role in EVENT_ROLES and art.camera and art.stem and (art.camera, art.stem) in targets:
                by_pair[(art.camera, art.stem)].append(art)
        session.flush()  # new artifacts and revisions get their ids
        # every retained revision of the targets' meta keys (never heartbeat history)
        history: dict[str, list[tuple[int, str, Any]]] = defaultdict(list)
        meta_keys = sorted(a.s3_key for arts in by_pair.values() for a in arts if a.role == "meta")
        for chunk in _chunks(meta_keys):
            for rid, key, etag, body in session.execute(
                    select(RawRevision.id, RawRevision.s3_key, RawRevision.etag, RawRevision.body)
                    .where(RawRevision.s3_key.in_(chunk)).order_by(RawRevision.id)):
                history[key].append((rid, etag, body))
        feedback_by_stem: dict[str, list[Feedback]] = defaultdict(list)
        for fb in self.load_feedback().values():
            if fb.s3_key in self.legacy:  # quarantined key: its verdict no longer counts
                fb.event_id = None
                continue
            if fb.alert_stem:
                feedback_by_stem[fb.alert_stem].append(fb)
        runs: dict[int, AiRun] = {}
        for chunk in _chunks(sorted(self.events[t].id for t in targets if t in self.events)):
            for run in session.scalars(select(AiRun).where(AiRun.event_id.in_(chunk), AiRun.purpose == "guard")
                                       .order_by(AiRun.id.desc())):
                runs[run.event_id] = run  # descending ids: the oldest run per event is kept

        plans = []
        fresh: dict[tuple[str, str], Event] = {}  # new events, transient until inserted below
        for pair in sorted(targets):
            arts = by_pair.get(pair, [])
            metas = sorted((a for a in arts if a.role == "meta"), key=lambda a: ROOTS.index(parse_key(a.s3_key).root))
            current: dict[str, _Copy] = {}
            for art in metas:
                if art.applied_etag is None:  # e.g. every artifact right after migration 0003
                    self.recover_applied(art, history.get(art.s3_key, ()))
                revs = {etag: body for _, etag, body in history.get(art.s3_key, ())}
                if art.applied_etag not in revs:
                    continue
                _, usable, _ = self.parse(art.s3_key, art.applied_etag, revs[art.applied_etag])
                if usable is not None:
                    root = parse_key(art.s3_key).root
                    current[root] = _Copy(root, art.s3_key, art.applied_etag, usable, revs[art.applied_etag])
            ev = self.events.get(pair)
            if not current and ev is None:
                continue  # orphan media: stays with event_id NULL until its meta arrives
            if ev is None:
                ev = Event(device_pk=self.device.id, site=self.site, camera=pair[0], stem=pair[1],
                           created_at=self.now)
                fresh[pair] = ev
            else:
                self.stats.updated_events += 1
            winner = self.best_ai(current, metas, history)
            if current:
                self.merge(ev, current, winner)
            elif ev.decision is None:
                ev.decision = {"v": event_outcome.VERSION}  # nothing to read: never picked up by the backfill again
            ev.updated_at = self.now
            meta_feedback = _union_owner_feedback(
                self.parse(a.s3_key, etag, body)[0]
                for a in metas for _, etag, body in history.get(a.s3_key, ()) if not _is_invalid(body))
            plans.append([ev, arts, metas, current, winner, meta_feedback, pair])
        if fresh:
            stored = self.insert_new_events(fresh, runs)
            for plan in plans:
                pair = plan[6]
                if pair in stored:
                    ev = plan[0] = stored[pair]
                    if plan[3]:
                        self.merge(ev, plan[3], plan[4])
                    ev.updated_at = self.now
        session.flush()

        for ev, arts, metas, current, winner, meta_feedback, _ in plans:
            for art in arts:
                if art.event_id != ev.id:
                    art.event_id = ev.id
            feedback = feedback_by_stem.get(ev.stem, [])
            for fb in feedback:
                fb.event_id = ev.id
                fb_art = self.arts.get(fb.s3_key)
                if fb_art is not None:
                    fb_art.event_id = ev.id
            run = self.upsert_ai_run(ev, runs.get(ev.id), winner)
            ev.owner_verdicts = _owner_verdicts(feedback, meta_feedback)
            any_clip = any(a.role == "original_video" and a.available for a in arts)
            ev.completeness = {
                "video": any_clip,
                "boxes": ("sampled" if any(c.rec.sampled_frames for c in current.values())
                          and any(a.role == "yolo_label" and a.available for a in arts) else "none"),
                "ai": run.status if run is not None else "none",
                "owner_feedback": bool(feedback) or bool(meta_feedback),
                "expired": _expired(ev, any_clip, self.now),
                "copies": sorted(COPY_NAME[parse_key(a.s3_key).root] for a in metas),
            }

    def insert_new_events(self, fresh: dict[tuple[str, str], Event], runs: dict[int, AiRun]
                          ) -> dict[tuple[str, str], Event]:
        """Insert new events (multi-row INSERT ... ON CONFLICT DO NOTHING on (device, camera, stem)) and re-read
        them. An event another writer created meanwhile is counted as updated, and its AI run is loaded so it is
        updated rather than duplicated."""
        session = self.session
        pairs = sorted(fresh)
        rows = [{"device_pk": self.device.id, "site": self.site, "camera": cam, "stem": stem,
                 "start_ts": fresh[(cam, stem)].start_ts, "kind": fresh[(cam, stem)].kind or "unknown",
                 "created_at": self.now} for cam, stem in pairs]
        inserted: set[tuple[str, str]] = set()
        for chunk in _chunks(rows, INSERT_ROWS):
            stmt = (pg_insert(Event).values(chunk)
                    .on_conflict_do_nothing(index_elements=[Event.device_pk, Event.camera, Event.stem])
                    .returning(Event.camera, Event.stem))
            inserted |= {(cam, stem) for cam, stem in session.execute(stmt)}
        stored: dict[tuple[str, str], Event] = {}
        for chunk in _chunks(pairs):
            for ev in session.scalars(select(Event).where(Event.device_pk == self.device.id,
                                                          tuple_(Event.camera, Event.stem).in_(chunk))):
                stored[(ev.camera, ev.stem)] = ev
        for pair, ev in stored.items():
            self.events[pair] = ev
            self.events_by_id[ev.id] = ev
        self.stats.new_events += len(inserted)
        taken_over = [stored[p].id for p in pairs if p not in inserted and p in stored]
        self.stats.updated_events += len(taken_over)
        for chunk in _chunks(taken_over):
            for run in session.scalars(select(AiRun).where(AiRun.event_id.in_(chunk), AiRun.purpose == "guard")
                                       .order_by(AiRun.id.desc())):
                runs[run.event_id] = run
        return stored

    def recover_applied(self, art: Artifact, revisions) -> None:
        """A meta with no applied revision (migration 0003 left every existing artifact NULL) takes its last
        usable stored revision, listed or not, available or not -- so a production copy that has since
        disappeared keeps supplying dispatch, muted and expires_at."""
        for _, etag, body in reversed(list(revisions)):
            if not _is_invalid(body) and self.parse(art.s3_key, etag, body)[1] is not None:
                art.applied_etag = etag
                return

    def best_ai(self, current: dict[str, _Copy], metas: list[Artifact],
                history: dict[str, list[tuple[int, str, Any]]]) -> Optional[_Copy]:
        """The revision whose AI answer stands: the best status ever seen in either copy.

        A current revision wins ties (production first), then the newest retained revision, so a later, poorer
        copy never replaces a real teacher answer and an equal or better one does.
        """
        best, best_rank = None, None
        for n, root in enumerate(ROOTS):
            if root in current:
                rank = (AI_RANK.get(current[root].rec.ai.status, 0), 1, -n)
                if best_rank is None or rank > best_rank:
                    best, best_rank = current[root], rank
        for art in metas:
            root = parse_key(art.s3_key).root
            for rid, etag, body in history.get(art.s3_key, ()):
                if _is_invalid(body):
                    continue
                _, usable, _ = self.parse(art.s3_key, etag, body)
                if usable is None:
                    continue
                rank = (AI_RANK.get(usable.ai.status, 0), 0, rid)
                if best_rank is None or rank > best_rank:
                    best, best_rank = _Copy(root, art.s3_key, etag, usable, body), rank
        return best

    def merge(self, ev: Event, current: dict[str, _Copy], winner: Optional[_Copy]) -> None:
        ordered = [current[root] for root in ROOTS if root in current]  # production first
        recs = [c.rec for c in ordered]
        first = lambda attr: next((getattr(r, attr) for r in recs if getattr(r, attr) is not None), None)  # noqa: E731

        ev.kind = _cut(next((r.kind for r in recs if r.kind and r.kind != "owner_feedback"), None)
                       or stem_kind(ev.stem)[2] or "unknown", 32)
        ev.trigger_ts = first("trigger_ts")
        start = first("start_ts")
        ev.start_ts = start if start is not None else float(ev.trigger_ts)
        ev.end_ts = first("end_ts")
        ev.day = parse_key(ordered[0].key).day
        # what the event says comes from the revision that supplied the AI answer
        source = winner.rec if winner is not None and winner.rec.ai.status != "none" else recs[0]
        ev.summary = source.summary or ""
        ev.summary_redacted = None  # refilled from the new summary at the end of the pass (redact.backfill)
        ev.label = redact.label(source.label)  # enum fields: anything else is stored as null
        ev.alert_command = redact.alert_command(source.alert_command)
        ev.alert_reason = source.alert_reason or ""
        # delivery is what the production copy recorded; without a production copy it is unknown
        prod = current.get("production")
        ev.dispatch = prod.rec.dispatch if prod is not None else None
        ev.muted = _raw_muted(prod.body) if prod is not None else None
        # the box's event layer: its session and what it did with the clip (the production copy, else the training one)
        decided = event_outcome.decision_of((prod or ordered[0]).body)
        ev.decision, ev.session_id = decided, event_outcome.session_id(decided) or None
        ev.would_raise = event_outcome.would_raise(decided)
        ev.detected = next((r.detected for r in recs if r.detected), [])
        ev.class_max_conf = next((r.class_max_conf for r in recs if r.class_max_conf), {})
        ev.duration_sec = first("duration_sec")
        ev.fps = first("fps")
        ev.frame_size = first("frame_size")
        ev.clip_start_local = next((redact.clip_start_local(c.body.get("clip_start_local")) for c in ordered
                                    if isinstance(c.body, dict) and isinstance(c.body.get("clip_start_local"), str)),
                                   None)  # a timestamp or nothing: free text here would bypass every projection
        ev.expires_at = (datetime.fromtimestamp(ev.start_ts, timezone.utc) + PRODUCTION_RETENTION
                         if prod is not None else None)

    def upsert_ai_run(self, ev: Event, run: Optional[AiRun], winner: Optional[_Copy]) -> Optional[AiRun]:
        """One guard run per event, from the winning revision; its links are refreshed on every rebuild."""
        if winner is None:
            return run
        if run is None:
            run = AiRun(event_id=ev.id, purpose="guard")
            self.session.add(run)
        ai = winner.rec.ai
        run.status = ai.status
        run.model = _cut(ai.model, 128)
        run.prompt_version = _cut(ai.prompt_version, 64)
        run.prompt = ai.prompt
        run.parsed = ai.parsed
        run.ai_source_key, run.ai_source_etag = winner.key, winner.etag
        prefix = _prefix(winner.root, self.site)
        raw = self.arts.get(prefix + ai.raw_rel) if ai.raw_rel else None
        run.raw_artifact_id = raw.id if raw is not None else None
        run.input_artifact_ids = [self.arts[prefix + rel].id for rel in ai.input_frames_rel
                                  if prefix + rel in self.arts]
        return run

    # ------------------------------------------------------------ driver

    def run(self) -> IndexStats:
        self.load()
        for root in ROOTS:
            self.list_prefix(root)
        self.insert_new_artifacts()
        bodies = self.obtain_bodies()
        feedback = self.apply_bodies(bodies)
        stems = {rec.alert_id for _, _, rec in feedback if rec.alert_id and len(rec.alert_id) <= 255}
        if feedback:  # the events re-applied feedback was linked to must be loaded too
            rows = self.load_feedback()
            self.dirty_ids |= {rows[a.s3_key].event_id for a, _, _ in feedback
                               if a.s3_key in rows and rows[a.s3_key].event_id is not None}
        # events indexed before the event layer was read (migration 0018): rebuilt from stored revisions, a chunk a pass
        self.dirty_ids |= set(self.session.scalars(select(Event.id).where(
            Event.device_pk == self.device.id, Event.decision.is_(None)).order_by(Event.id).limit(DECISION_BACKFILL)))
        self.load_events(self.dirty, self.dirty_ids, stems)
        self.apply_feedback(feedback)
        self.rebuild_all()
        refresh_expiry(self.session, self.device, self.now)
        # the names this pass saw (cameras, display names, heartbeat host, names in the JSON bodies) join the
        # identity history; a new name means every stored labeler search text is recomputed, else only the
        # missing ones are filled
        new_names = redact.remember(self.session, self.device, extra=self.body_names, now=self.now)
        redact.backfill(self.session, self.device, everything=bool(new_names))
        self.session.commit()
        return self.stats


def _sanitise_verdicts(entries: list[dict]) -> bool:
    """Replace verdicts outside the vocabulary by "unknown" in place; True when any entry was changed."""
    changed = False
    for entry in entries:
        if entry.get("verdict") not in (None, "") and redact.verdict(entry.get("verdict")) != entry.get("verdict"):
            entry["verdict"] = redact.UNKNOWN_VERDICT
            changed = True
    return changed


def _union_owner_feedback(records: Iterable[ClipRecord]) -> list[dict]:
    """owner_feedback entries across revisions and copies, deduplicated by (time_utc, verdict)."""
    seen, out = set(), []
    for rec in records:
        for entry in rec.owner_feedback:
            ident = (str(entry.get("time_utc")), str(entry.get("verdict")))
            if ident not in seen:
                seen.add(ident)
                out.append(entry)
    return out


def _owner_verdicts(feedback: list[Feedback], meta_feedback: list[dict]) -> list[str]:
    """Distinct owner verdicts (excluding "none"), in time order."""
    items = [(fb.received_at, fb.verdict) for fb in feedback]
    items += [(parse_utc(e.get("time_utc")), e.get("verdict")) for e in meta_feedback]
    far = datetime.max.replace(tzinfo=timezone.utc)
    items.sort(key=lambda item: item[0] or far)
    out: list[str] = []
    for _, verdict in items:
        if isinstance(verdict, str) and verdict and verdict != "none" and verdict not in out:
            out.append(verdict)
    return out


def refresh_expiry(session: Session, device: Device, now: datetime) -> int:
    """Set `completeness.expired` on the device's events as of `now`; returns how many changed.

    Expired = the production copy's 14-day retention (`expires_at`) has passed and no copy of the video
    (production or training) is available. It runs at the end of every pass, so time alone moves an event to
    expired (and a video that reappears moves it back) even when nothing in S3 changed.
    """
    any_clip = exists().where(Artifact.event_id == Event.id, Artifact.role == "original_video",
                              Artifact.available.is_(True))
    due = and_(Event.expires_at.is_not(None), Event.expires_at < now, ~any_clip)
    flagged = Event.completeness["expired"].as_boolean()
    changed = 0
    for ev, is_due in session.execute(
            select(Event, due.label("due")).where(
                Event.device_pk == device.id,
                or_(and_(due, flagged.is_not(True)), and_(flagged.is_(True), ~due)))):
        ev.completeness = {**(ev.completeness or {}), "expired": bool(is_due)}
        changed += 1
    return changed


def index_device(session: Session, s3: S3, device: Device, full_scan: bool = False,
                 now: Optional[datetime] = None) -> IndexStats:
    """Index one device's `production_<site>/` and `dataset_<site>/` prefixes; commits once at the end."""
    return _Run(session, s3, device, full_scan, now or datetime.now(timezone.utc)).run()


def index_all(session: Session, s3: S3, full_scan: bool = False) -> dict[str, IndexStats]:
    """Index every enrolled device, one commit per device; a failing device does not stop the rest."""
    out: dict[str, IndexStats] = {}
    for device_pk, site in session.execute(select(Device.id, Device.site).order_by(Device.id)).all():
        device = session.get(Device, device_pk)
        try:
            out[site] = index_device(session, s3, device, full_scan=full_scan)
        except Exception:
            session.rollback()
            log.exception("indexing site %s failed", site)
    return out


