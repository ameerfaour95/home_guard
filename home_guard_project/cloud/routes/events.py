"""Event history: list, detail, sampled YOLO boxes, review state, density and review counts.

Labelers see only events of customers who gave training consent, with the customer's identity replaced by
pseudonyms (`customer-<6 hex>` for the customer and site, `cam-<6 hex>` for each camera), and without delivery
details or the owner's words. Admin and support see everything.
"""
from __future__ import annotations

import base64
import binascii
import math
import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Literal, Optional, get_args

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import Float, Integer, and_, case, cast, false, func, or_, select, true, tuple_, type_coerce
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from home_guard_project.fleet_contract.classes import COCO_NAMES
from home_guard_project.fleet_contract.keys import parse_key
from home_guard_project.fleet_contract.legacy import parse_heartbeat, parse_meta

from .. import audit, pseudonym
from ..deps import current_staff, get_session
from ..models import (AiRun, Artifact, AuditLog, Camera, Customer, Device, Event, Feedback, RawRevision, ReviewState,
                      Staff)
from ..schemas import (AiRunOut, ArtifactOut, Box, DensityOut, DensityRow, DetectionsOut, DispatchOut, EventDetail,
                       EventKind, EventPage, EventSummary, FeedbackOut, FrameBoxes, ReviewCount, ReviewUpdate)

router = APIRouter(tags=["events"], dependencies=[Depends(current_staff)])

VIEW_AUDIT_WINDOW = timedelta(minutes=10)
MAX_DENSITY_BUCKETS = 5000
SAMPLED_MODEL = "yolo (collection export)"
_KINDS = set(get_args(EventKind))
_COMPLETENESS_DEFAULT = {"video": False, "boxes": "none", "ai": "none", "owner_feedback": False, "expired": False,
                         "copies": []}
_FEEDBACK_PRIVATE = ("raw_text", "note", "from", "chat_id")


# ---------------------------------------------------------------- helpers: time, viewer, visibility

def _now(request: Request) -> datetime:
    return request.app.state.clock()


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _ts_utc(ts: Optional[float]) -> Optional[datetime]:
    return datetime.fromtimestamp(ts, timezone.utc) if ts is not None else None


class _Viewer:
    """Who is asking, and how identities are shown to them."""

    def __init__(self, staff: Staff, request: Request):
        self.staff = staff
        self.labeler = staff.role == "labeler"
        self.secret = request.app.state.settings.jwt_secret

    def visible(self):
        """Condition over a query joined to Customer: what this viewer may see."""
        return Customer.consent_training.is_(True) if self.labeler else true()

    def customer_name(self, customer_id: int, name: str) -> str:
        return pseudonym.customer(self.secret, customer_id) if self.labeler else name

    def site(self, customer_id: int, site: str) -> str:
        return pseudonym.customer(self.secret, customer_id) if self.labeler else site

    def camera(self, site: str, camera: str) -> str:
        return pseudonym.camera(self.secret, site, camera) if self.labeler else camera


def _scoped(stmt):
    """Join Device and Customer onto a statement selecting from Event."""
    return stmt.join(Device, Device.id == Event.device_pk).join(Customer, Customer.id == Device.customer_id)


def _site_condition(session: Session, viewer: _Viewer, site: str):
    """Labelers filter by the customer pseudonym they see as the site; others by the real site."""
    if not viewer.labeler:
        return Device.site == site
    ids = [cid for cid in session.scalars(select(Customer.id).where(Customer.consent_training.is_(True)))
           if pseudonym.customer(viewer.secret, cid) == site]
    return Device.customer_id.in_(ids) if ids else false()


def _camera_condition(session: Session, viewer: _Viewer, camera: str):
    """Labelers filter by the camera pseudonym they see; others by the real camera name."""
    if not viewer.labeler:
        return Event.camera == camera
    if not camera.startswith("cam-"):
        return false()
    known = set(session.execute(_scoped(select(Event.site, Event.camera).distinct()).where(viewer.visible())).all())
    pairs = sorted(p for p in known if pseudonym.camera(viewer.secret, p[0], p[1]) == camera)
    return tuple_(Event.site, Event.camera).in_(pairs) if pairs else false()


def _has_verdict(verdict: str):
    return type_coerce(Event.owner_verdicts, JSONB).contains([verdict])


def _max_class_conf():
    conf = type_coerce(Event.class_max_conf, JSONB)
    each = func.jsonb_each_text(conf).table_valued("key", "value")
    best = select(func.max(cast(each.c.value, Float))).select_from(each).scalar_subquery()
    return case((func.jsonb_typeof(conf) == "object", best), else_=None)


_reviewed = func.coalesce(ReviewState.reviewed, false())
_flagged = func.coalesce(ReviewState.flagged, false())

# Built-in saved filters. Task 13 moves them into studio.BUILTIN_FILTERS with these same meanings.
BUILTIN_FILTERS = {
    "false_alarm": lambda: _has_verdict("false_alarm"),
    "ai_dismissed_person": lambda: and_(Event.kind == "false_positive",
                                        type_coerce(Event.detected, JSONB).contains(["person"])),
    "ai_failed": lambda: Event.completeness["ai"].as_string().in_(["failed", "fallback"]),
    "real_but_wrong": lambda: _has_verdict("real_but_wrong"),
    "low_conf": lambda: _max_class_conf() < 0.45,
    "paused": lambda: Event.kind == "paused",
}


# ---------------------------------------------------------------- cursor

def _encode_cursor(start_ts: float, event_id: int) -> str:
    return base64.urlsafe_b64encode(f"{start_ts!r}:{event_id}".encode("ascii")).decode("ascii")


def _decode_cursor(cursor: str) -> tuple[float, int]:
    try:
        raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True).decode("ascii")
        ts_text, _, id_text = raw.rpartition(":")
        ts, event_id = float(ts_text), int(id_text)
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid cursor")
    if not math.isfinite(ts):
        raise HTTPException(status_code=400, detail="Invalid cursor")
    return ts, event_id


# ---------------------------------------------------------------- summaries

def _event_select():
    return _scoped(select(Event, Customer.id, Customer.name, Customer.timezone, ReviewState.reviewed,
                          ReviewState.flagged)).outerjoin(ReviewState, ReviewState.event_id == Event.id)


def _summary_fields(viewer: _Viewer, ev: Event, customer_id: int, customer_name: str, tz: Optional[str],
                    reviewed: Optional[bool], flagged: Optional[bool], has_thumbnail: bool) -> dict[str, Any]:
    completeness = {**_COMPLETENESS_DEFAULT, **(ev.completeness if isinstance(ev.completeness, dict) else {})}
    return dict(
        id=ev.id, site=viewer.site(customer_id, ev.site), customer_id=customer_id,
        customer_name=viewer.customer_name(customer_id, customer_name), camera=viewer.camera(ev.site, ev.camera),
        kind=ev.kind if ev.kind in _KINDS else "unknown",
        start_utc=_ts_utc(ev.start_ts), end_utc=_ts_utc(ev.end_ts), summary=ev.summary or "", label=ev.label,
        alert_command=ev.alert_command,
        detected=[d for d in (ev.detected or []) if isinstance(d, str)],
        owner_verdicts=[v for v in (ev.owner_verdicts or []) if isinstance(v, str)],
        completeness=completeness, reviewed=bool(reviewed), flagged=bool(flagged),
        thumbnail_url=f"/v1/events/{ev.id}/thumbnail" if has_thumbnail else None,
        timezone=tz or "UTC",
    )


def _thumbnail_ids(session: Session, ids: list[int]) -> set[int]:
    if not ids:
        return set()
    return set(session.scalars(select(Artifact.event_id).where(
        Artifact.event_id.in_(ids), Artifact.role == "thumbnail", Artifact.available.is_(True)).distinct()))


def _load_one(session: Session, viewer: _Viewer, event_id: int):
    """(event, customer id, customer name, timezone, reviewed, flagged); 404 when missing or not visible."""
    row = session.execute(_event_select().where(Event.id == event_id, viewer.visible())).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return row


def _summary_of(session: Session, viewer: _Viewer, row) -> dict[str, Any]:
    ev, cid, name, tz, reviewed, flagged = row
    return _summary_fields(viewer, ev, cid, name, tz, reviewed, flagged, ev.id in _thumbnail_ids(session, [ev.id]))


def _device_id(session: Session, ev: Event) -> Optional[str]:
    return session.scalar(select(Device.device_id).where(Device.id == ev.device_pk))


# ---------------------------------------------------------------- list

@router.get("/events", response_model=EventPage)
def list_events(
    request: Request,
    site: Optional[str] = None,
    customer_id: Optional[int] = None,
    camera: Optional[str] = None,
    kind: Optional[str] = None,
    ai: Optional[str] = None,
    verdict: Optional[str] = None,
    q: Optional[str] = None,
    from_utc: Optional[datetime] = None,
    to_utc: Optional[datetime] = None,
    reviewed: Optional[bool] = None,
    flagged: Optional[bool] = None,
    filter: Optional[str] = None,
    cursor: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
    staff: Staff = Depends(current_staff),
    session: Session = Depends(get_session),
):
    viewer = _Viewer(staff, request)
    conds = [viewer.visible()]
    if site is not None:
        conds.append(_site_condition(session, viewer, site))
    if customer_id is not None:
        conds.append(Device.customer_id == customer_id)
    if camera is not None:
        conds.append(_camera_condition(session, viewer, camera))
    if kind is not None:
        conds.append(Event.kind == kind)
    if ai is not None:
        conds.append(Event.completeness["ai"].as_string() == ai)
    if verdict is not None:
        conds.append(_has_verdict(verdict))
    if q:
        conds.append(Event.search.op("@@")(func.websearch_to_tsquery("simple", q)))
    if from_utc is not None:
        conds.append(Event.start_ts >= _utc(from_utc).timestamp())
    if to_utc is not None:
        conds.append(Event.start_ts < _utc(to_utc).timestamp())
    if reviewed is not None:
        conds.append(_reviewed.is_(reviewed))
    if flagged is not None:
        conds.append(_flagged.is_(flagged))
    if filter is not None:
        builder = BUILTIN_FILTERS.get(filter)
        if builder is None:
            raise HTTPException(status_code=400, detail=f"Unknown filter: {filter}")
        conds.append(builder())
    if cursor:
        ts, last_id = _decode_cursor(cursor)
        conds.append(or_(Event.start_ts < ts, and_(Event.start_ts == ts, Event.id < last_id)))
    rows = session.execute(_event_select().where(*conds)
                           .order_by(Event.start_ts.desc(), Event.id.desc()).limit(limit + 1)).all()
    more = len(rows) > limit
    rows = rows[:limit]
    thumbs = _thumbnail_ids(session, [r[0].id for r in rows])
    items = [EventSummary(**_summary_fields(viewer, ev, cid, name, tz, rv, fl, ev.id in thumbs))
             for ev, cid, name, tz, rv, fl in rows]
    next_cursor = _encode_cursor(rows[-1][0].start_ts, rows[-1][0].id) if more and rows else None
    return EventPage(items=items, next_cursor=next_cursor)


# ---------------------------------------------------------------- density and review counts

def _bucket_counts(session: Session, start: float, step: int, n: int, conds: list,
                   by: tuple = ()) -> dict[tuple, list[list[int]]]:
    """{group key: [events, alerts, false_alarms] lists} for events starting in [start, start + n*step)."""
    idx = cast(func.floor((Event.start_ts - start) / step), Integer)
    stmt = (select(*by, idx, func.count(), func.count().filter(Event.kind == "alert"),
                   func.count().filter(_has_verdict("false_alarm")))
            .where(Event.start_ts >= start, Event.start_ts < start + n * step, *conds)
            .group_by(*by, idx))
    out: dict[tuple, list[list[int]]] = {}
    for row in session.execute(stmt).all():
        key, (i, events, alerts, false_alarms) = tuple(row[:len(by)]), tuple(row[len(by):])
        if not 0 <= i < n:
            continue
        lists = out.setdefault(key, [[0] * n, [0] * n, [0] * n])
        lists[0][i], lists[1][i], lists[2][i] = events, alerts, false_alarms
    return out


def _starts(start: float, step: int, n: int) -> list[datetime]:
    return [datetime.fromtimestamp(start + i * step, timezone.utc) for i in range(n)]


@router.get("/events/density", response_model=DensityOut)
def events_density(
    request: Request,
    from_utc: datetime,
    to_utc: datetime,
    site: Optional[str] = None,
    customer_id: Optional[int] = None,
    camera: Optional[str] = None,
    kind: Optional[str] = None,
    bucket: Literal["hour", "day"] = "hour",
    staff: Staff = Depends(current_staff),
    session: Session = Depends(get_session),
):
    # Counts per camera per hour/day bucket covering [from_utc, to_utc). Buckets are aligned to the hour/day in
    # UTC (not the customer's timezone); `timezone` is the customer's when exactly one customer is in scope, else
    # "UTC". Every known camera of the devices in scope gets a row (Camera rows, cameras seen in events, heartbeat
    # cameras), even with zero counts. With more than one device in scope an admin/support row is named
    # `<site>/<camera>`; a labeler's row is the camera pseudonym. (Comments, not a docstring: the OpenAPI is frozen.)
    viewer = _Viewer(staff, request)
    step = 3600 if bucket == "hour" else 86400
    start = math.floor(_utc(from_utc).timestamp() / step) * step
    end = math.ceil(_utc(to_utc).timestamp() / step) * step
    if end <= start:
        raise HTTPException(status_code=400, detail="to_utc must be after from_utc")
    n = int((end - start) // step)
    if n > MAX_DENSITY_BUCKETS:
        raise HTTPException(status_code=400, detail=f"Too many buckets (max {MAX_DENSITY_BUCKETS})")

    dev_q = (select(Device.id, Device.site, Device.customer_id, Device.last_heartbeat, Customer.timezone)
             .join(Customer, Customer.id == Device.customer_id).where(viewer.visible()))
    if site is not None:
        dev_q = dev_q.where(_site_condition(session, viewer, site))
    if customer_id is not None:
        dev_q = dev_q.where(Device.customer_id == customer_id)
    devices = session.execute(dev_q.order_by(Device.site)).all()
    timezones = {d.customer_id: d.timezone for d in devices}
    tz = (next(iter(timezones.values())) or "UTC") if len(timezones) == 1 else "UTC"
    if not devices:
        return DensityOut(bucket=bucket, starts_utc=_starts(start, step, n), timezone=tz, rows=[])

    pks = [d.id for d in devices]
    known: set[tuple[int, str]] = set(session.execute(
        select(Camera.device_pk, Camera.name).where(Camera.device_pk.in_(pks))).all())
    known |= set(session.execute(select(Event.device_pk, Event.camera).where(Event.device_pk.in_(pks))
                                 .distinct()).all())
    for d in devices:
        if isinstance(d.last_heartbeat, dict):
            known |= {(d.id, name) for name in parse_heartbeat(d.last_heartbeat).cameras}
    site_of = {d.id: d.site for d in devices}
    multi = len(devices) > 1

    def label(pk: int, cam: str) -> str:
        if viewer.labeler:
            return pseudonym.camera(viewer.secret, site_of[pk], cam)
        return f"{site_of[pk]}/{cam}" if multi else cam

    conds = [Event.device_pk.in_(pks)]
    if camera is not None:
        known = {(pk, cam) for pk, cam in known if (label(pk, cam) if viewer.labeler else cam) == camera}
        conds.append(tuple_(Event.device_pk, Event.camera).in_(sorted(known)) if known else false())
    if kind is not None:
        conds.append(Event.kind == kind)
    counts = _bucket_counts(session, start, step, n, conds, by=(Event.device_pk, Event.camera))
    rows = []
    for pk, cam in known:
        events, alerts, false_alarms = counts.get((pk, cam), [[0] * n, [0] * n, [0] * n])
        rows.append(DensityRow(camera=label(pk, cam), events=events, alerts=alerts, false_alarms=false_alarms))
    rows.sort(key=lambda r: r.camera)
    return DensityOut(bucket=bucket, starts_utc=_starts(start, step, n), timezone=tz, rows=rows)


def fleet_activity_density(session: Session, now: datetime, hours: int) -> DensityOut:
    """One "fleet" row of hourly counts over every device; the last bucket is the hour containing `now`."""
    end = math.floor(_utc(now).timestamp() / 3600) * 3600 + 3600
    start = end - hours * 3600
    empty = [[0] * hours, [0] * hours, [0] * hours]
    events, alerts, false_alarms = _bucket_counts(session, start, 3600, hours, []).get((), empty)
    return DensityOut(bucket="hour", starts_utc=_starts(start, 3600, hours), timezone="UTC",
                      rows=[DensityRow(camera="fleet", events=events, alerts=alerts, false_alarms=false_alarms)])


@router.get("/events/review-count", response_model=ReviewCount)
def review_count(request: Request, staff: Staff = Depends(current_staff), session: Session = Depends(get_session)):
    # unreviewed_24h: events of the last 24 h not reviewed; flagged_open: flagged and not reviewed.
    viewer = _Viewer(staff, request)
    since = (_now(request) - timedelta(hours=24)).timestamp()
    base = _scoped(select(func.count()).select_from(Event)).outerjoin(
        ReviewState, ReviewState.event_id == Event.id).where(viewer.visible(), _reviewed.is_(False))
    return ReviewCount(unreviewed_24h=session.scalar(base.where(Event.start_ts >= since)) or 0,
                       flagged_open=session.scalar(base.where(_flagged.is_(True))) or 0)


# ---------------------------------------------------------------- detail

def _drop_keys(node: Any, names: frozenset) -> Any:
    """A copy of `node` without any dict key in `names`, at any depth."""
    if isinstance(node, dict):
        return {k: _drop_keys(v, names) for k, v in node.items() if k not in names}
    if isinstance(node, list):
        return [_drop_keys(v, names) for v in node]
    return node


def _redact_meta(body: dict, camera_pseudonym: str) -> dict:
    """A meta document as a labeler sees it: no delivery details, no owner words, pseudonymous camera."""
    body = _drop_keys(body, frozenset({"dispatch"}))
    feedback = body.get("owner_feedback")
    if isinstance(feedback, list):
        body["owner_feedback"] = [{k: v for k, v in e.items() if k not in _FEEDBACK_PRIVATE}
                                  if isinstance(e, dict) else e for e in feedback]
    if "camera_name" in body:
        body["camera_name"] = camera_pseudonym
    return body


def _mask_key(key: str, ev: Event, site_p: str, cam_p: str) -> str:
    """An S3 key with the site and camera replaced by the labeler's pseudonyms."""
    head, _, rest = key.partition("/")
    for root in ("dataset", "production"):
        if head == f"{root}_{ev.site}":
            head = f"{root}_{site_p}"
            break
    else:
        head = "hidden"
    return f"{head}/{rest.replace(ev.camera, cam_p)}" if rest else head


def _newest_meta_body(session: Session, arts: Iterable[Artifact]) -> dict:
    """Newest raw revision of the training copy's meta, else of the production copy's."""
    metas = sorted((a for a in arts if a.role == "meta"), key=lambda a: 0 if a.s3_key.startswith("dataset_") else 1)
    for art in metas:
        body = session.scalar(select(RawRevision.body).where(RawRevision.s3_key == art.s3_key)
                              .order_by(RawRevision.fetched_at.desc(), RawRevision.id.desc()).limit(1))
        if body is not None:
            return body if isinstance(body, dict) else {"value": body}
    return {}


def _dispatch_sent(detail: dict) -> Optional[bool]:
    """True when any Telegram result is ok or WhatsApp reports sent; False when they say otherwise; else None."""
    found = False

    def walk(node: Any, in_whatsapp: bool) -> bool:
        nonlocal found
        if isinstance(node, list):
            return any(walk(v, in_whatsapp) for v in node)
        if not isinstance(node, dict):
            return False
        results = node.get("results")
        if isinstance(results, list):
            found = True
            if any(isinstance(r, dict) and r.get("ok") is True for r in results):
                return True
        if in_whatsapp and isinstance(node.get("sent"), bool):
            found = True
            if node["sent"]:
                return True
        return any(walk(v, in_whatsapp or k == "whatsapp") for k, v in node.items() if k != "results")

    if walk(detail, False):
        return True
    return False if found else None


def _dispatch_out(dispatch: Any) -> Optional[DispatchOut]:
    if not isinstance(dispatch, dict):
        return None
    channel = dispatch.get("channel")
    return DispatchOut(channel=channel if isinstance(channel, str) else None, sent=_dispatch_sent(dispatch),
                       detail=dispatch)


def _audit_view(session: Session, request: Request, staff: Staff, ev: Event, customer_id: int) -> None:
    """One `event_view` row per (staff, event) per 10 minutes."""
    target = f"event/{ev.id}"
    last = session.scalar(select(AuditLog.ts).where(
        AuditLog.action == "event_view", AuditLog.staff_id == staff.id, AuditLog.target == target)
        .order_by(AuditLog.ts.desc()).limit(1))
    if last is not None and last > _now(request) - VIEW_AUDIT_WINDOW:
        return
    audit.record(session, staff.id, "event_view", target=target, customer_id=customer_id,
                 device_id=_device_id(session, ev))


@router.get("/events/{event_id}", response_model=EventDetail)
def get_event(event_id: int, request: Request, staff: Staff = Depends(current_staff),
              session: Session = Depends(get_session)):
    viewer = _Viewer(staff, request)
    row = _load_one(session, viewer, event_id)
    ev, customer_id = row[0], row[1]
    fields = _summary_of(session, viewer, row)
    runs = session.scalars(select(AiRun).where(AiRun.event_id == ev.id, AiRun.purpose == "guard")
                           .order_by(AiRun.id)).all()
    feedback = session.scalars(select(Feedback).where(Feedback.event_id == ev.id)
                               .order_by(Feedback.received_at.asc().nulls_last(), Feedback.id)).all()
    arts = session.scalars(select(Artifact).where(Artifact.event_id == ev.id).order_by(Artifact.id)).all()
    raw_meta = _newest_meta_body(session, arts)
    if viewer.labeler:
        raw_meta = _redact_meta(raw_meta, fields["camera"])
    detail = EventDetail(
        **fields,
        clip_start_local=ev.clip_start_local, duration_sec=ev.duration_sec, fps=ev.fps,
        frame_size=ev.frame_size if isinstance(ev.frame_size, list) else None,
        alert_reason=ev.alert_reason or "",
        dispatch=None if viewer.labeler else _dispatch_out(ev.dispatch),
        ai_runs=[AiRunOut(id=r.id, purpose="guard", status=r.status, model=r.model, prompt_version=r.prompt_version,
                          prompt=r.prompt, parsed=r.parsed if isinstance(r.parsed, dict) else None,
                          raw_text_artifact_id=r.raw_artifact_id,
                          input_frame_artifact_ids=[i for i in (r.input_artifact_ids or []) if isinstance(i, int)])
                 for r in runs],
        feedback=[FeedbackOut(id=f.id, verdict=f.verdict, action=f.action,
                              note="" if viewer.labeler else f.note, raw_text="" if viewer.labeler else f.raw_text,
                              source=f.source, received_utc=f.received_at or fields["start_utc"])
                  for f in feedback],
        artifacts=[ArtifactOut(id=a.id, role=a.role,
                               s3_key=_mask_key(a.s3_key, ev, fields["site"], fields["camera"]) if viewer.labeler
                               else a.s3_key,
                               bytes=a.bytes, available=a.available, provenance=a.provenance,
                               detail=a.detail if isinstance(a.detail, dict) else None)
                   for a in arts],
        raw_meta=raw_meta,
    )
    _audit_view(session, request, staff, ev, customer_id)
    return detail


# ---------------------------------------------------------------- detections

class _LRU:
    """A small thread-safe LRU map."""

    def __init__(self, size: int):
        self.size = size
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._items:
                return None
            self._items.move_to_end(key)
            return self._items[key]

    def put(self, key, value) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


LABEL_CACHE = _LRU(512)  # (s3_key, etag) -> label file text


def _sampled_frames(session: Session, arts: Iterable[Artifact]) -> tuple[list[dict], str]:
    """Sampled frames from the event's applied meta (training copy first) and the prefix their paths are under."""
    metas = sorted((a for a in arts if a.role == "meta"), key=lambda a: 0 if a.s3_key.startswith("dataset_") else 1)
    for art in metas:
        if not art.applied_etag:
            continue
        body = session.scalar(select(RawRevision.body).where(RawRevision.s3_key == art.s3_key,
                                                             RawRevision.etag == art.applied_etag))
        info = parse_key(art.s3_key)
        if not isinstance(body, dict) or info is None:
            continue
        frames = parse_meta(art.s3_key, body).sampled_frames
        if frames:
            return frames, f"{info.root}_{info.site}/"
    return [], ""


def _clamp(v: float) -> float:
    return min(1.0, max(0.0, v))


def _parse_label(text: str) -> list[Box]:
    """YOLO `cls xc yc w h` lines (COCO ids, normalised) -> boxes with normalised xyxy; bad lines are skipped."""
    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = (float(p) for p in parts[1:5])
        except (ValueError, OverflowError):
            continue
        if not all(math.isfinite(v) for v in (xc, yc, w, h)):
            continue
        boxes.append(Box(cls=cls, label=COCO_NAMES.get(cls, str(cls)), conf=None,
                         xyxy=[_clamp(xc - w / 2), _clamp(yc - h / 2), _clamp(xc + w / 2), _clamp(yc + h / 2)]))
    return boxes


def _read_label(s3, key: str, etag: Optional[str]) -> Optional[str]:
    cached = LABEL_CACHE.get((key, etag))
    if cached is not None:
        return cached
    try:
        text = s3.get_text(key)
    except Exception:  # deleted since indexing, throttled, ...: the frame shows as not run
        return None
    LABEL_CACHE.put((key, etag), text)
    return text


@router.get("/events/{event_id}/detections", response_model=DetectionsOut)
def get_detections(event_id: int, request: Request, staff: Staff = Depends(current_staff),
                   session: Session = Depends(get_session)):
    viewer = _Viewer(staff, request)
    ev = _load_one(session, viewer, event_id)[0]
    completeness = ev.completeness if isinstance(ev.completeness, dict) else {}
    if completeness.get("boxes") != "sampled":
        return DetectionsOut(provenance="none", model=None, frames=[])
    s3 = request.app.state.s3
    if s3 is None:
        raise HTTPException(status_code=503, detail="Storage is not configured")
    spec, prefix = _sampled_frames(session, session.scalars(select(Artifact).where(Artifact.event_id == ev.id)))
    keys = sorted({prefix + f["label_path"] for f in spec if isinstance(f.get("label_path"), str)})
    labels = {a.s3_key: a for a in session.scalars(select(Artifact).where(
        Artifact.s3_key.in_(keys), Artifact.role == "yolo_label"))} if keys else {}
    frames: dict[int, FrameBoxes] = {}
    for f in spec:
        index = f.get("frame_index")
        if type(index) is not int or index < 0 or index in frames:
            continue
        offset = f.get("approx_time_offset_sec")
        if ev.fps and ev.fps > 0:
            t_sec = index / ev.fps
        elif isinstance(offset, (int, float)) and not isinstance(offset, bool) and math.isfinite(offset):
            t_sec = float(offset)
        else:
            t_sec = 0.0
        key = prefix + f["label_path"] if isinstance(f.get("label_path"), str) else None
        art = labels.get(key) if key else None
        text = _read_label(s3, key, art.etag) if art is not None and art.available else None
        if text is None:
            frames[index] = FrameBoxes(frame_index=index, t_sec=t_sec, status="not_run", boxes=[])
            continue
        boxes = _parse_label(text)
        frames[index] = FrameBoxes(frame_index=index, t_sec=t_sec, status="ran" if boxes else "ran_empty",
                                   boxes=boxes)
    return DetectionsOut(provenance="sampled", model=SAMPLED_MODEL, frames=[frames[i] for i in sorted(frames)])


# ---------------------------------------------------------------- review

@router.patch("/events/{event_id}/review", response_model=EventSummary)
def review_event(event_id: int, body: ReviewUpdate, request: Request, staff: Staff = Depends(current_staff),
                 session: Session = Depends(get_session)):
    viewer = _Viewer(staff, request)
    row = _load_one(session, viewer, event_id)
    ev, customer_id = row[0], row[1]
    changes = body.model_dump(exclude_none=True)
    if changes:
        values = {"event_id": ev.id, "by": staff.id, "at": _now(request), **changes}
        session.execute(pg_insert(ReviewState).values(**values).on_conflict_do_update(
            index_elements=[ReviewState.event_id], set_={k: v for k, v in values.items() if k != "event_id"}))
        audit.record(session, staff.id, "review", target=f"event/{ev.id}", customer_id=customer_id,
                     device_id=_device_id(session, ev), detail=changes)
        session.expire_all()
        row = _load_one(session, viewer, event_id)
    return EventSummary(**_summary_of(session, viewer, row))
