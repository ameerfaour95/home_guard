"""Index a box's legacy S3 prefixes into Events, Artifacts, AI runs and Feedback.

One clip can exist twice: the production copy (`production_<site>/`, what the owner was sent) and the
training copy (`dataset_<site>/`, with the teacher inputs and answer). Both copies merge into one Event
keyed by (device, camera, stem); every object keeps its own Artifact row and every JSON body fetched is
saved as a RawRevision before merging, so differing metadata is never lost.

Each run lists each prefix once, fetches only JSON whose (key, ETag) is new, never downloads media, and
rebuilds only the events touched by a change -- from the stored revisions, not from S3.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from home_guard_project.fleet_contract._time import parse_utc
from home_guard_project.fleet_contract.keys import KeyInfo, parse_key, stem_kind
from home_guard_project.fleet_contract.legacy import ClipRecord, parse_feedback, parse_heartbeat, parse_meta

from .models import AiRun, Artifact, Device, Event, Feedback, IndexProblem, RawRevision, S3Cursor
from .s3 import S3, ObjInfo

log = logging.getLogger(__name__)

ROOTS = ("production", "dataset")
COPY_NAME = {"production": "production", "dataset": "training"}
INDEXED_AREAS = {"meta", "clips", "feedback", "status", "responses", "vlm_crops", "yolo_images", "yolo_labels"}
EVENT_ROLES = ("original_video", "meta", "raw_answer", "teacher_frame", "crop_video", "yolo_image", "yolo_label")
AI_RANK = {"none": 0, "fallback": 1, "failed": 2, "real": 3}
FULL_SCAN_EVERY = timedelta(minutes=30)
PRODUCTION_RETENTION = timedelta(days=14)
INVALID_JSON = "_invalid_json"  # RawRevision body marker for an object that is not JSON
HEARTBEAT_SUFFIX = "/_status/heartbeat.json"
MIME = {".mp4": "video/mp4", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".json": "application/json",
        ".txt": "text/plain"}


@dataclass
class IndexStats:
    new_events: int = 0
    updated_events: int = 0
    artifacts: int = 0
    feedback: int = 0
    problems: int = 0


def artifact_role(info: KeyInfo) -> Optional[str]:
    """Artifact role for an indexed key, or None when the key is not indexed."""
    area, key = info.area, info.key
    if area == "clips":
        return "original_video"
    if area == "meta":
        return "meta" if key.endswith(".meta.json") else None
    if area == "responses":
        return "raw_answer"
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


def _cut(value, n: int):
    return value[:n] if isinstance(value, str) else value


def _prefix(root: str, site: str) -> str:
    return f"{root}_{site}/"


class _Run:
    """State of one index pass over one device."""

    def __init__(self, session: Session, s3: S3, device: Device, full_scan: bool, now: datetime):
        self.session, self.s3, self.device = session, s3, device
        self.full_scan, self.now = full_scan, now
        self.site = device.site
        self.stats = IndexStats()
        self.dirty: set[tuple[str, str]] = set()  # (camera, stem) of events to rebuild
        self.dirty_ids: set[int] = set()  # event ids to rebuild
        self.fetch: list[tuple[KeyInfo, ObjInfo, str]] = []

    # ------------------------------------------------------------ problems

    def problem(self, key: str, reason: str) -> None:
        """One IndexProblem per key (its reasons joined); unchanged reasons are not re-counted."""
        row = self.session.get(IndexProblem, key)
        if row is not None and row.reason == reason:
            return
        if row is None:
            self.session.add(IndexProblem(s3_key=key, reason=reason, seen_at=self.now))
        else:
            row.reason, row.seen_at = reason, self.now
        self.stats.problems += 1

    def clear_problem(self, key: str) -> None:
        row = self.session.get(IndexProblem, key)
        if row is not None:
            self.session.delete(row)

    # ------------------------------------------------------------ listing

    def list_prefix(self, root: str) -> None:
        prefix = _prefix(root, self.site)
        session = self.session
        known = {a.s3_key: a for a in session.scalars(
            select(Artifact).where(Artifact.s3_key.startswith(prefix, autoescape=True)))}
        revisions = set(session.execute(select(RawRevision.s3_key, RawRevision.etag).where(
            RawRevision.s3_key.startswith(prefix, autoescape=True))).all())
        seen: set[str] = set()
        for obj in self.s3.list(prefix):
            info = parse_key(obj.key)
            if info is None or info.site != self.site or info.area not in INDEXED_AREAS:
                continue
            role = artifact_role(info)
            if role is None:
                continue
            seen.add(obj.key)
            camera, stem = info.camera, info.stem
            if (camera and len(camera) > 128) or (stem and len(stem) > 255):
                self.problem(obj.key, "camera or stem name too long")
                continue
            art = known.get(obj.key)
            changed = False
            if art is None:
                art = Artifact(role=role, s3_key=obj.key, etag=obj.etag, bytes=obj.size,
                               mime=MIME.get(info.ext.lower()), available=True, provenance="box",
                               detail={"copy": COPY_NAME[root]}, camera=camera, stem=stem,
                               last_modified=obj.last_modified)
                session.add(art)
                changed = True
            elif art.etag != obj.etag or not art.available:
                art.etag, art.bytes, art.last_modified, art.available = obj.etag, obj.size, obj.last_modified, True
                changed = True
                if art.event_id is not None:
                    self.dirty_ids.add(art.event_id)
            if changed:
                self.stats.artifacts += 1
                if role in EVENT_ROLES and camera and stem:
                    self.dirty.add((camera, stem))
            is_heartbeat = role == "status" and obj.key.endswith(HEARTBEAT_SUFFIX)
            if (role in ("meta", "feedback") or is_heartbeat) and (obj.key, obj.etag) not in revisions:
                self.fetch.append((info, obj, role))
        self.disappearance_check(prefix, known, seen)

    def disappearance_check(self, prefix: str, known: dict[str, Artifact], seen: set[str]) -> None:
        cursor = self.session.get(S3Cursor, prefix)
        if cursor is None:
            cursor = S3Cursor(prefix=prefix)
            self.session.add(cursor)
        last = cursor.last_full_scan
        if not (self.full_scan or last is None or self.now - last >= FULL_SCAN_EVERY):
            return
        for key, art in known.items():
            if key not in seen and art.available:
                art.available = False
                if art.event_id is not None:
                    self.dirty_ids.add(art.event_id)
        cursor.last_full_scan = self.now

    # ------------------------------------------------------------ JSON bodies

    def fetch_bodies(self) -> None:
        for info, obj, role in self.fetch:
            try:
                text = self.s3.get_text(obj.key)
            except Exception as e:  # deleted between listing and get, throttled, ...
                self.problem(obj.key, f"fetch failed: {type(e).__name__}")
                continue
            try:
                body: Any = json.loads(text)
                invalid = False
            except ValueError:
                body, invalid = {INVALID_JSON: text[:100_000]}, True
            self.session.add(RawRevision(s3_key=obj.key, etag=obj.etag, fetched_at=self.now, body=body))
            if invalid:
                self.problem(obj.key, "invalid json")
                continue
            if role == "meta":
                if info.camera and info.stem:
                    self.dirty.add((info.camera, info.stem))
                else:
                    self.problem(obj.key, "meta key outside camera/day layout")
            elif role == "feedback":
                self.upsert_feedback(obj.key, body)
            else:
                self.store_heartbeat(obj.key, body)

    def upsert_feedback(self, key: str, body: Any) -> None:
        if not isinstance(body, dict):
            self.problem(key, "invalid feedback body")
            return
        rec = parse_feedback(key, body)
        session = self.session
        fb = session.scalars(select(Feedback).where(Feedback.s3_key == key)).first()
        if fb is None:
            fb = Feedback(s3_key=key, device_pk=self.device.id)
            session.add(fb)
        alert_stem = rec.alert_id if rec.alert_id and len(rec.alert_id) <= 255 else None
        fb.alert_stem = alert_stem
        fb.verdict, fb.action = _cut(rec.verdict, 64), _cut(rec.action, 64)
        fb.note, fb.raw_text, fb.source = rec.note, rec.raw_text, _cut(rec.source, 64)
        fb.scope_camera = _cut(rec.scope_camera, 128)
        fb.received_at = rec.time_utc
        fb.event_id = None
        if alert_stem:
            ev = session.scalars(select(Event).where(Event.device_pk == self.device.id,
                                                     Event.stem == alert_stem)).first()
            if ev is not None:
                fb.event_id = ev.id
                self.dirty_ids.add(ev.id)
        self.stats.feedback += 1

    def store_heartbeat(self, key: str, body: Any) -> None:
        if not isinstance(body, dict):
            self.problem(key, "invalid heartbeat body")
            return
        hb = parse_heartbeat(body)
        self.device.last_heartbeat = body
        self.device.last_heartbeat_at = hb.time_utc

    # ------------------------------------------------------------ events

    def rebuild_all(self) -> None:
        targets = set(self.dirty)
        if self.dirty_ids:
            targets |= set(self.session.execute(
                select(Event.camera, Event.stem).where(Event.id.in_(self.dirty_ids))).all())
        for camera, stem in sorted(targets):
            self.rebuild(camera, stem)

    def _event_artifacts(self, camera: str, stem: str, roles=EVENT_ROLES) -> list[Artifact]:
        prefixes = [_prefix(root, self.site) for root in ROOTS]
        return list(self.session.scalars(select(Artifact).where(
            Artifact.camera == camera, Artifact.stem == stem, Artifact.role.in_(roles),
            or_(*(Artifact.s3_key.startswith(p, autoescape=True) for p in prefixes)))))

    def _copy_records(self, metas: list[Artifact]) -> dict[str, tuple[ClipRecord, Any, KeyInfo]]:
        """Parse the current revision of each copy's meta; record or clear its problems."""
        records = {}
        for art in metas:
            info = parse_key(art.s3_key)
            rev = self.session.scalars(select(RawRevision).where(
                RawRevision.s3_key == art.s3_key, RawRevision.etag == art.etag)).first()
            if rev is None:
                rev = self.session.scalars(select(RawRevision).where(RawRevision.s3_key == art.s3_key)
                                           .order_by(RawRevision.id.desc())).first()
            if rev is None or (isinstance(rev.body, dict) and INVALID_JSON in rev.body):
                continue
            rec = parse_meta(art.s3_key, rev.body)
            problems = list(rec.problems)
            usable = rec.start_ts is not None or rec.trigger_ts is not None
            if not usable:
                problems.append("no start time")
            if problems:
                self.problem(art.s3_key, "; ".join(problems))
            else:
                self.clear_problem(art.s3_key)
            if usable:
                records[info.root] = (rec, rev.body, info)
        return records

    def rebuild(self, camera: str, stem: str) -> None:
        session, device = self.session, self.device
        ev = session.scalars(select(Event).where(Event.device_pk == device.id, Event.camera == camera,
                                                 Event.stem == stem)).first()
        artifacts = self._event_artifacts(camera, stem)
        metas = sorted((a for a in artifacts if a.role == "meta"), key=lambda a: ROOTS.index(parse_key(a.s3_key).root))
        records = self._copy_records(metas)
        if not records and ev is None:
            return  # orphan media: stays with event_id NULL until its meta arrives
        if ev is None:
            ev = Event(device_pk=device.id, site=self.site, camera=camera, stem=stem, created_at=self.now)
            session.add(ev)
            self.stats.new_events += 1
        else:
            self.stats.updated_events += 1
        if records:
            self.merge(ev, records)
        ev.updated_at = self.now
        session.flush()

        for art in artifacts:
            if art.event_id != ev.id:
                art.event_id = ev.id
        feedback = list(session.scalars(select(Feedback).where(Feedback.device_pk == device.id,
                                                               Feedback.alert_stem == stem)))
        fb_keys = [fb.s3_key for fb in feedback]
        for fb in feedback:
            fb.event_id = ev.id
        if fb_keys:
            for art in session.scalars(select(Artifact).where(Artifact.s3_key.in_(fb_keys))):
                art.event_id = ev.id

        run = self.upsert_ai_run(ev, records)
        meta_feedback = _union_owner_feedback([r for r, _, _ in records.values()])
        ev.owner_verdicts = _owner_verdicts(feedback, meta_feedback)
        prod_prefix = _prefix("production", self.site)
        prod_clip = any(a.role == "original_video" and a.available and a.s3_key.startswith(prod_prefix)
                        for a in artifacts)
        copies = sorted(COPY_NAME[parse_key(a.s3_key).root] for a in metas)
        started = datetime.fromtimestamp(ev.start_ts, timezone.utc) if ev.start_ts is not None else None
        sampled = any(r.sampled_frames for r, _, _ in records.values())
        ev.completeness = {
            "video": any(a.role == "original_video" and a.available for a in artifacts),
            "boxes": "sampled" if sampled and any(a.role == "yolo_label" and a.available for a in artifacts) else "none",
            "ai": run.status if run is not None else "none",
            "owner_feedback": bool(feedback) or bool(meta_feedback),
            "expired": ("production" in copies and not prod_clip and started is not None
                        and self.now - started > PRODUCTION_RETENTION),
            "copies": copies,
        }

    def merge(self, ev: Event, records: dict[str, tuple[ClipRecord, Any, KeyInfo]]) -> None:
        ordered = [records[root] for root in ROOTS if root in records]  # production first
        best = max(ordered, key=lambda r: AI_RANK.get(r[0].ai.status, 0))  # ties keep production
        chosen = best if best[0].ai.status != "none" else ordered[0]
        rec_c = chosen[0]
        recs = [r for r, _, _ in ordered]
        first = lambda attr: next((getattr(r, attr) for r in recs if getattr(r, attr) is not None), None)  # noqa: E731

        ev.kind = _cut(next((r.kind for r in recs if r.kind and r.kind != "owner_feedback"), None)
                       or stem_kind(ev.stem)[2] or "unknown", 32)
        ev.trigger_ts = first("trigger_ts")
        start = first("start_ts")
        ev.start_ts = start if start is not None else float(ev.trigger_ts)
        ev.end_ts = first("end_ts")
        ev.day = ordered[0][2].day
        ev.summary = rec_c.summary or ""
        ev.label = _cut(rec_c.label, 64)
        ev.alert_command = _cut(rec_c.alert_command, 32)
        ev.alert_reason = rec_c.alert_reason or ""
        prod = records.get("production")
        train = records.get("dataset")
        dispatch = prod[0].dispatch if prod is not None else None
        ev.dispatch = dispatch if dispatch is not None else (train[0].dispatch if train is not None else None)
        ev.detected = next((r.detected for r in recs if r.detected), [])
        ev.class_max_conf = next((r.class_max_conf for r in recs if r.class_max_conf), {})
        ev.duration_sec = first("duration_sec")
        ev.fps = first("fps")
        ev.frame_size = first("frame_size")
        ev.clip_start_local = next((_cut(body.get("clip_start_local"), 64) for _, body, _ in ordered
                                    if isinstance(body, dict) and isinstance(body.get("clip_start_local"), str)), None)
        ev.expires_at = (datetime.fromtimestamp(ev.start_ts, timezone.utc) + PRODUCTION_RETENTION
                         if prod is not None else None)

    def upsert_ai_run(self, ev: Event, records) -> Optional[AiRun]:
        """One guard run per event; replaced only by an equal or better status (real > failed > fallback > none)."""
        session = self.session
        run = session.scalars(select(AiRun).where(AiRun.event_id == ev.id, AiRun.purpose == "guard")
                              .order_by(AiRun.id)).first()
        if not records:
            return run
        ordered = [(root, records[root]) for root in ROOTS if root in records]
        root, (rec, _, _) = max(ordered, key=lambda item: AI_RANK.get(item[1][0].ai.status, 0))
        ai = rec.ai
        if run is not None and AI_RANK.get(ai.status, 0) < AI_RANK.get(run.status, 0):
            return run  # keep the better answer we already have
        if run is None:
            run = AiRun(event_id=ev.id, purpose="guard")
            session.add(run)
        run.status = ai.status
        run.model = _cut(ai.model, 128)
        run.prompt_version = _cut(ai.prompt_version, 64)
        run.prompt = ai.prompt
        run.parsed = ai.parsed
        prefix = _prefix(root, self.site)
        wanted = ([prefix + ai.raw_rel] if ai.raw_rel else []) + [prefix + rel for rel in ai.input_frames_rel]
        ids = dict(session.execute(select(Artifact.s3_key, Artifact.id).where(Artifact.s3_key.in_(wanted))).all()) \
            if wanted else {}
        run.raw_artifact_id = ids.get(prefix + ai.raw_rel) if ai.raw_rel else None
        run.input_artifact_ids = [ids[prefix + rel] for rel in ai.input_frames_rel if prefix + rel in ids]
        return run

    # ------------------------------------------------------------ driver

    def run(self) -> IndexStats:
        for root in ROOTS:
            self.list_prefix(root)
        self.session.flush()
        self.fetch_bodies()
        self.session.flush()
        self.rebuild_all()
        self.session.commit()
        return self.stats


def _union_owner_feedback(records: list[ClipRecord]) -> list[dict]:
    """owner_feedback entries across copies, deduplicated by (time_utc, verdict)."""
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


def run_once(engine) -> int:
    """`manage index-once`: one pass over every device using env AWS credentials."""
    import boto3

    from .db import session_scope

    client = boto3.client("s3", region_name=os.environ.get("HG_CLOUD_REGION", "us-east-1"))
    s3 = S3(client, os.environ.get("HG_CLOUD_BUCKET", "security-camera-project-v1"))
    with session_scope(engine) as session:
        results = index_all(session, s3)
    for site, stats in results.items():
        print(f"{site}: {stats}")
    return 0
