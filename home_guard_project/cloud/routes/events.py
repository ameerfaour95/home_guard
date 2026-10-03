"""Event history: list, detail, sampled YOLO boxes, review state, density and review counts.

Labelers see only events of customers who gave training consent, with the customer's identity replaced by
pseudonyms (`customer-<6 hex>` for the customer and site, `cam-<6 hex>` for each camera), and without delivery
details or the owner's words. Their responses are built from allowlisted projections, never by deleting fields
from real data: free text (summaries, reasons, prompts, AI output) passes through `redact`, storage keys become
opaque `artifact-<id>` references, `raw_meta` keeps only a fixed set of non-identifying fields, enum fields
(`alert_command`, `label`) outside their value sets are null, and `clip_start_local` is never shown. Labelers get
no prompt text and cannot search (any `q` is one uniform 400): capabilities they do not need are removed, and
redaction is only defence in depth. Admin and support see everything.

Labelers never get a real customer id: `customer_id` is 0 in every labeler response, and a labeler's
`customer_id` filter is one uniform 400 (sequential ids would tell which clips come from a known household).
"""
from __future__ import annotations

import base64
import binascii
import math
import re
import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Literal, Optional, get_args

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import BigInteger, Integer, and_, cast, false, func, literal, or_, select, text, true, tuple_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from home_guard_project.fleet_contract.classes import COCO_NAMES
from home_guard_project.fleet_contract.keys import parse_key
from home_guard_project.fleet_contract.legacy import parse_heartbeat, parse_meta
from home_guard_project.fleet_contract.yolo_labels import parse_label

from .. import audit, pseudonym, redact
from .. import studio as studio_logic
from ..access import may_see_thumbnail
from ..deps import TEXT_MAX, SessionDep, check_length, current_staff, id_in_range, require_id
from ..models import (AiRun, Artifact, AuditLog, Camera, Collection, CollectionItem, Customer, Device, Event, Feedback,
                      IndexProblem, RawRevision, ReviewState, Staff)
from ..s3 import ETagMismatch
from ..schemas import (AiRunOut, ArtifactOut, Box, DensityOut, DensityRow, DetectionsOut, DispatchOut, EventDetail,
                       AiStatus, EventKind, EventPage, EventSummary, FeedbackOut, FrameBoxes, ReviewCount, ReviewUpdate)

router = APIRouter(tags=["events"], dependencies=[Depends(current_staff)])

VIEW_AUDIT_WINDOW = timedelta(minutes=10)
MAX_DENSITY_BUCKETS = 5000
SAMPLED_MODEL = "yolo (collection export)"
_KINDS = set(get_args(EventKind))
_COMPLETENESS_DEFAULT = {"video": False, "boxes": "none", "ai": "none", "owner_feedback": False, "expired": False,
                         "copies": []}
# What a labeler may see of an artifact's detail and of the raw meta document (allowlists, not blocklists).
_LABELER_ARTIFACT_DETAIL = ("fps", "tile_w", "tile_h", "count", "copy")
_LABELER_META = {
    "kind": None, "duration_sec": None, "fps_estimated": None, "frames_written": None, "codec": None,
    "buffer": ("store_size",), "yolo": ("class_counts", "class_max_conf", "trigger_classes"),
    "model_response": None, "teacher": ("model", "prompt_version", "temperature"),
}


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
        self._identities: dict[int, redact.Identity] = {}

    def identity(self, session: Session, device_pk: int) -> redact.Identity:
        """The device's identity terms mapped to this labeler's pseudonyms (cached for the request)."""
        if device_pk not in self._identities:
            self._identities[device_pk] = redact.identity(session, session.get(Device, device_pk), self.secret)
        return self._identities[device_pk]

    def text(self, session: Session, device_pk: int, value: Optional[str]) -> str:
        """Free text as this viewer may read it."""
        return self.identity(session, device_pk).text(value) if self.labeler else (value or "")

    def visible(self):
        """Condition over a query joined to Customer: what this viewer may see."""
        return Customer.consent_training.is_(True) if self.labeler else true()

    def customer_name(self, customer_id: int, name: str) -> str:
        return pseudonym.customer(self.secret, customer_id) if self.labeler else name

    def site(self, customer_id: int, site: str) -> str:
        return pseudonym.customer(self.secret, customer_id) if self.labeler else site

    def camera(self, site: str, camera: str) -> str:
        return pseudonym.camera(self.secret, site, camera) if self.labeler else camera

    def customer_id(self, customer_id: int) -> int:
        return 0 if self.labeler else customer_id


CUSTOMER_FILTER_REFUSED = "Filtering by customer is not available for this role"


def _customer_condition(viewer: _Viewer, customer_id: Optional[int]):
    """The `customer_id` filter: refused for labelers (one 400, whatever the id); an id that cannot exist matches
    nothing."""
    if viewer.labeler:
        raise HTTPException(status_code=400, detail=CUSTOMER_FILTER_REFUSED)
    return Device.customer_id == customer_id if id_in_range(customer_id) else false()


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


_has_verdict = studio_logic.has_verdict

# The only values `kind`, `ai` and `verdict` can take; anything else is a 400 for every role before any query.
_AI_STATUSES = frozenset(get_args(AiStatus))


def _check_vocabulary(kind: Optional[str] = None, ai: Optional[str] = None, verdict: Optional[str] = None) -> None:
    for name, value, allowed in (("kind", kind, _KINDS), ("ai", ai, _AI_STATUSES),
                                 ("verdict", verdict, redact.VERDICTS | {redact.UNKNOWN_VERDICT})):
        if value is not None and value not in allowed:
            raise HTTPException(status_code=400, detail=f"Unknown {name}")


_reviewed = func.coalesce(ReviewState.reviewed, false())
_flagged = func.coalesce(ReviewState.flagged, false())

# ---------------------------------------------------------------- cursor

def _encode_cursor(start_ts: float, event_id: int) -> str:
    return base64.urlsafe_b64encode(f"{start_ts!r}:{event_id}".encode("ascii")).decode("ascii")


_CURSOR = re.compile(r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?):([1-9]\d{0,18})")
MAX_EVENT_ID = 2 ** 63 - 1  # the database's integer domain for ids


def _decode_cursor(cursor: str) -> tuple[float, int]:
    """(start_ts, id) of a cursor; anything else -- empty, bad base64, extra parts, a non-finite time, an id
    outside 1..2^63-1 -- is a 400 before any query runs."""
    bad = HTTPException(status_code=400, detail="Invalid cursor")
    if not cursor or len(cursor) > 128:
        raise bad
    try:
        raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True).decode("ascii")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise bad
    match = _CURSOR.fullmatch(raw)
    if match is None:
        raise bad
    ts, event_id = float(match.group(1)), int(match.group(2))
    if not math.isfinite(ts) or not 1 <= event_id <= MAX_EVENT_ID:
        raise bad
    return ts, event_id


# ---------------------------------------------------------------- summaries

def _event_select():
    return _scoped(select(Event, Customer.id, Customer.name, Customer.timezone, ReviewState.reviewed,
                          ReviewState.flagged)).outerjoin(ReviewState, ReviewState.event_id == Event.id)


def _summary_fields(session: Session, viewer: _Viewer, ev: Event, customer_id: int, customer_name: str,
                    tz: Optional[str], reviewed: Optional[bool], flagged: Optional[bool],
                    has_thumbnail: bool) -> dict[str, Any]:
    completeness = {**_COMPLETENESS_DEFAULT, **(ev.completeness if isinstance(ev.completeness, dict) else {})}
    shown = lambda value: viewer.text(session, ev.device_pk, value)  # noqa: E731
    return dict(
        id=ev.id, site=viewer.site(customer_id, ev.site), customer_id=viewer.customer_id(customer_id),
        customer_name=viewer.customer_name(customer_id, customer_name), camera=viewer.camera(ev.site, ev.camera),
        kind=ev.kind if ev.kind in _KINDS else "unknown",
        start_utc=_ts_utc(ev.start_ts), end_utc=_ts_utc(ev.end_ts), summary=shown(ev.summary),
        label=redact.label(ev.label) if viewer.labeler else ev.label,
        alert_command=redact.alert_command(ev.alert_command) if viewer.labeler else ev.alert_command,
        detected=[shown(d) for d in (ev.detected or []) if isinstance(d, str)],
        owner_verdicts=[(redact.verdict(v) if viewer.labeler else shown(v))
                        for v in (ev.owner_verdicts or []) if isinstance(v, str)],
        completeness=completeness, reviewed=bool(reviewed), flagged=bool(flagged),
        thumbnail_url=f"/v1/events/{ev.id}/thumbnail" if has_thumbnail else None,
        timezone=tz or "UTC",
    )


def _thumbnail_ids(session: Session, viewer: _Viewer, ids: list[int]) -> set[int]:
    """Events among `ids` with a thumbnail this viewer may open (access.may_see_thumbnail)."""
    if not ids:
        return set()
    has_thumb = select(Artifact.event_id).where(Artifact.event_id.in_(ids), Artifact.role == "thumbnail",
                                                Artifact.available.is_(True))
    rows = session.execute(_scoped(select(Event.id, Customer.consent_recordings, Customer.consent_training)
                                   .select_from(Event)).where(Event.id.in_(ids), Event.id.in_(has_thumb))).all()
    return {eid for eid, rec, train in rows if may_see_thumbnail(viewer.staff.role, bool(rec), bool(train))}


def _load_one(session: Session, viewer: _Viewer, event_id: int):
    """(event, customer id, customer name, timezone, reviewed, flagged); 404 when missing or not visible."""
    row = session.execute(_event_select().where(Event.id == event_id, viewer.visible())).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return row


def _summary_of(session: Session, viewer: _Viewer, row) -> dict[str, Any]:
    ev, cid, name, tz, reviewed, flagged = row
    return _summary_fields(session, viewer, ev, cid, name, tz, reviewed, flagged,
                           ev.id in _thumbnail_ids(session, viewer, [ev.id]))


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
    collection_id: Optional[int] = None,
    with_total: bool = False,
    cursor: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
    staff: Staff = Depends(current_staff),
    session: Session = SessionDep,
):
    """Event history, newest first. `with_total=true` also fills `total` (counted up to 10000; beyond that
    `total` is 10000 and `total_capped` is true). `collection_id` keeps only events in that collection."""
    viewer = _Viewer(staff, request)
    # Everything a request can be refused for is decided here, before any query: the answer never depends on
    # what is stored. Labelers cannot search at all (any query text, whatever it says, is the same 400).
    for field, value in (("site", site), ("camera", camera), ("q", q), ("filter", filter)):
        check_length(field, value, TEXT_MAX)
    if q and viewer.labeler:
        raise HTTPException(status_code=400, detail="Search is not available for this role")
    customer_cond = _customer_condition(viewer, customer_id) if customer_id is not None else None
    if filter is not None and studio_logic.builtin_condition(filter) is None:  # studio.BUILTIN_FILTERS
        raise HTTPException(status_code=400, detail=f"Unknown filter: {filter}")
    if cursor is not None:
        _decode_cursor(cursor)
    _check_vocabulary(kind=kind, ai=ai, verdict=verdict)
    conds = [viewer.visible()]
    if site is not None:
        conds.append(_site_condition(session, viewer, site))
    if customer_cond is not None:
        conds.append(customer_cond)
    if camera is not None:
        conds.append(_camera_condition(session, viewer, camera))
    if kind is not None:
        conds.append(Event.kind == kind)
    if ai is not None:
        conds.append(Event.completeness["ai"].as_string() == ai)
    if verdict is not None:
        conds.append(_has_verdict(verdict))
    if q:  # admin and support only (labelers were refused above)
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
        conds.append(studio_logic.builtin_condition(filter))
    if collection_id is not None:  # a collection this viewer may not see filters like one that does not exist
        conds.append(_in_collection(collection_id, staff))
    return event_page(session, viewer, conds, cursor, limit, with_total)


TOTAL_CAP = 10_000


def visible_collections(staff: Staff):
    """Condition over Collection: the collections `staff` may see. Non-labelers see all; a labeler sees the
    public ones and the ones private to them. Privacy is fixed at creation (`private_to_staff_id`), so a later
    role change never exposes a collection. Every route that takes a collection id applies this rule."""
    if staff.role != "labeler":
        return true()
    return or_(Collection.private_to_staff_id.is_(None), Collection.private_to_staff_id == staff.id)


def collection_visible(staff: Staff, col: Collection) -> bool:
    """`visible_collections` for one loaded collection."""
    return staff.role != "labeler" or col.private_to_staff_id is None or col.private_to_staff_id == staff.id


def _in_collection(collection_id: int, staff: Staff):
    """Condition over Event: in that collection, when `staff` may see it (else nothing, as for no collection)."""
    if not id_in_range(collection_id):
        return false()
    return Event.id.in_(select(CollectionItem.event_id)
                        .join(Collection, Collection.id == CollectionItem.collection_id)
                        .where(CollectionItem.collection_id == collection_id, visible_collections(staff)))


def event_page(session: Session, viewer: _Viewer, conds: list, cursor: Optional[str], limit: int,
               with_total: bool = False) -> EventPage:
    """One keyset page of events matching `conds` (which already include viewer.visible()), newest first.
    Shared by /events and the studio collection items route so both show identical rows and pseudonyms."""
    after = _decode_cursor(cursor) if cursor is not None else None  # absent: the first page; bad: 400, no query
    total, capped = None, False
    if with_total:
        counted = session.scalar(select(func.count()).select_from(
            _event_select().where(*conds).order_by(None).limit(TOTAL_CAP + 1).subquery()))
        capped = counted > TOTAL_CAP
        total = TOTAL_CAP if capped else counted
    if after is not None:
        ts, last_id = after
        last = literal(last_id, BigInteger)  # compared as bigint: any id of the cursor domain is a valid bound
        conds = [*conds, or_(Event.start_ts < ts, and_(Event.start_ts == ts, Event.id < last))]
    rows = session.execute(_event_select().where(*conds)
                           .order_by(Event.start_ts.desc(), Event.id.desc()).limit(limit + 1)).all()
    more = len(rows) > limit
    rows = rows[:limit]
    thumbs = _thumbnail_ids(session, viewer, [r[0].id for r in rows])
    items = [EventSummary(**_summary_fields(session, viewer, ev, cid, name, tz, rv, fl, ev.id in thumbs))
             for ev, cid, name, tz, rv, fl in rows]
    next_cursor = _encode_cursor(rows[-1][0].start_ts, rows[-1][0].id) if more and rows else None
    return EventPage(items=items, next_cursor=next_cursor, total=total, total_capped=capped)


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
    session: Session = SessionDep,
):
    # Counts per camera per hour/day bucket covering [from_utc, to_utc). Buckets are aligned to the hour/day in
    # UTC (not the customer's timezone); `timezone` is the customer's when exactly one customer is in scope, else
    # "UTC". Every known camera of the devices in scope gets a row (Camera rows, cameras seen in events, heartbeat
    # cameras), even with zero counts -- except for labelers, who get only cameras with events in the interval. With more than one device in scope an admin/support row is named
    # `<site>/<camera>`; a labeler's row is the camera pseudonym. (Comments, not a docstring: the OpenAPI is frozen.)
    viewer = _Viewer(staff, request)
    _check_vocabulary(kind=kind)
    for field, value in (("site", site), ("camera", camera)):
        check_length(field, value, TEXT_MAX)
    customer_cond = _customer_condition(viewer, customer_id) if customer_id is not None else None
    step = 3600 if bucket == "hour" else 86400
    begin, finish = _utc(from_utc), _utc(to_utc)  # naive endpoints are UTC
    if finish <= begin:  # the requested interval, before any rounding to buckets
        raise HTTPException(status_code=400, detail="to_utc must be after from_utc")
    start = math.floor(begin.timestamp() / step) * step
    end = math.ceil(finish.timestamp() / step) * step
    n = int((end - start) // step)
    if n > MAX_DENSITY_BUCKETS:
        raise HTTPException(status_code=400, detail=f"Too many buckets (max {MAX_DENSITY_BUCKETS})")

    dev_q = (select(Device.id, Device.site, Device.customer_id, Device.last_heartbeat, Customer.timezone)
             .join(Customer, Customer.id == Device.customer_id).where(viewer.visible()))
    if site is not None:
        dev_q = dev_q.where(_site_condition(session, viewer, site))
    if customer_cond is not None:
        dev_q = dev_q.where(customer_cond)
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
        if viewer.labeler and (pk, cam) not in counts:
            continue  # a labeler sees only cameras with events here: the row list is not the camera inventory
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
def review_count(request: Request, staff: Staff = Depends(current_staff), session: Session = SessionDep):
    # unreviewed_24h: events of the last 24 h not reviewed; flagged_open: flagged and not reviewed.
    viewer = _Viewer(staff, request)
    since = (_now(request) - timedelta(hours=24)).timestamp()
    base = _scoped(select(func.count()).select_from(Event)).outerjoin(
        ReviewState, ReviewState.event_id == Event.id).where(viewer.visible(), _reviewed.is_(False))
    return ReviewCount(unreviewed_24h=session.scalar(base.where(Event.start_ts >= since)) or 0,
                       flagged_open=session.scalar(base.where(_flagged.is_(True))) or 0)


# ---------------------------------------------------------------- detail

def _labeler_meta(body: dict, ident: redact.Identity) -> dict:
    """The raw meta document as a labeler sees it: only the allowlisted fields, free text redacted. No paths of
    any kind, no delivery details, no owner feedback, no timestamps beyond what the summary already shows."""
    out: dict[str, Any] = {}
    for key, children in _LABELER_META.items():
        if key not in body:
            continue
        value = body[key]
        if children is None:
            out[key] = value
        elif isinstance(value, dict):
            picked = {k: value[k] for k in children if k in value}
            if picked:
                out[key] = picked
    return ident.json(out)


def _labeler_artifact_detail(detail: Any, ident: redact.Identity) -> Optional[dict]:
    if not isinstance(detail, dict):
        return None
    picked = {k: detail[k] for k in _LABELER_ARTIFACT_DETAIL
              if k in detail and isinstance(detail[k], (str, int, float, bool))}
    return ident.json(picked) or None


def labeler_artifact_ref(artifact_id: int) -> str:
    """What a labeler sees instead of a storage key: an opaque reference, no path at all."""
    return f"artifact-{artifact_id}"


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
    """One `event_view` row per (staff, event) per 10 minutes.

    The transaction-scoped advisory lock on (staff, event) serialises concurrent views, so the check and the
    insert act as one step: a second request waits, then sees the first one's row. One clock (the app's) both
    decides the window and stamps the row."""
    target = f"event/{ev.id}"
    session.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"),
                    {"k": f"event_view:{staff.id}:{ev.id}"})
    now = _now(request)
    # bounded by the window (index staff_id, action, target, ts): only the last 10 minutes are ever read
    seen = session.scalar(select(AuditLog.id).where(
        AuditLog.staff_id == staff.id, AuditLog.action == "event_view", AuditLog.target == target,
        AuditLog.ts > now - VIEW_AUDIT_WINDOW).limit(1))
    if seen is not None:
        return
    audit.record(session, staff.id, "event_view", target=target, customer_id=customer_id,
                 device_id=_device_id(session, ev), ts=now)


@router.get("/events/{event_id}", response_model=EventDetail)
def get_event(event_id: int, request: Request, staff: Staff = Depends(current_staff),
              session: Session = SessionDep):
    require_id(event_id, "Event not found")
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
    parsed = lambda r: r.parsed if isinstance(r.parsed, dict) else None  # noqa: E731
    frame_ids = lambda r: [i for i in (r.input_artifact_ids or []) if isinstance(i, int)]  # noqa: E731
    if viewer.labeler:
        ident = viewer.identity(session, ev.device_pk)
        optional = lambda value: ident.text(value) if value is not None else None  # noqa: E731
        ai_runs = [AiRunOut(id=r.id, purpose="guard", status=r.status, model=optional(r.model),
                            prompt_version=optional(r.prompt_version), prompt=None,  # never prompt text
                            parsed=ident.json(parsed(r)), raw_text_artifact_id=r.raw_artifact_id,
                            input_frame_artifact_ids=frame_ids(r)) for r in runs]
        feedback_out = [FeedbackOut(id=f.id, verdict=redact.verdict(f.verdict) if f.verdict else "", action=ident.text(f.action), note="",
                                    raw_text="", source=ident.text(f.source),
                                    received_utc=f.received_at or fields["start_utc"]) for f in feedback]
        artifacts = [ArtifactOut(id=a.id, role=a.role, s3_key=labeler_artifact_ref(a.id), bytes=a.bytes,
                                 available=a.available, provenance=a.provenance,
                                 detail=_labeler_artifact_detail(a.detail, ident))
                     for a in arts if a.role != "opaque_copy"]
        raw_meta, alert_reason = _labeler_meta(raw_meta, ident), ident.text(ev.alert_reason)
    else:
        ai_runs = [AiRunOut(id=r.id, purpose="guard", status=r.status, model=r.model,
                            prompt_version=r.prompt_version, prompt=r.prompt, parsed=parsed(r),
                            raw_text_artifact_id=r.raw_artifact_id, input_frame_artifact_ids=frame_ids(r))
                   for r in runs]
        feedback_out = [FeedbackOut(id=f.id, verdict=f.verdict, action=f.action, note=f.note, raw_text=f.raw_text,
                                    source=f.source, received_utc=f.received_at or fields["start_utc"])
                        for f in feedback]
        artifacts = [ArtifactOut(id=a.id, role=a.role, s3_key=a.s3_key, bytes=a.bytes, available=a.available,
                                 provenance=a.provenance, detail=a.detail if isinstance(a.detail, dict) else None)
                     for a in arts]
        alert_reason = ev.alert_reason or ""
    detail = EventDetail(
        **fields,
        clip_start_local=None if viewer.labeler else ev.clip_start_local, duration_sec=ev.duration_sec, fps=ev.fps,
        frame_size=ev.frame_size if isinstance(ev.frame_size, list) else None,
        alert_reason=alert_reason,
        dispatch=None if viewer.labeler else _dispatch_out(ev.dispatch),
        ai_runs=ai_runs, feedback=feedback_out, artifacts=artifacts, raw_meta=raw_meta,
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


def _boxes(text: str) -> Optional[list[Box]]:
    """YOLO `cls xc yc w h` lines (COCO ids, normalised) -> boxes with normalised xyxy; None when the file is
    malformed (the one strict validator of fleet_contract.yolo_labels, shared with the training exporter)."""
    rows, problem = parse_label(text)
    if problem is not None:
        return None
    return [Box(cls=r.cls, label=COCO_NAMES.get(r.cls, str(r.cls)), conf=None,
                xyxy=[_clamp(r.xc - r.w / 2), _clamp(r.yc - r.h / 2), _clamp(r.xc + r.w / 2), _clamp(r.yc + r.h / 2)])
            for r in rows]


BAD_LABEL = "invalid yolo label"


def _label_problem(session: Session, key: str, etag: Optional[str], now: datetime, bad: bool) -> None:
    """Record (or, once the file reads cleanly, clear) the problem of a malformed label file."""
    if bad:
        session.execute(pg_insert(IndexProblem).values(s3_key=key, reason=BAD_LABEL, seen_at=now, etag=etag)
                        .on_conflict_do_update(index_elements=[IndexProblem.s3_key],
                                               set_={"reason": BAD_LABEL, "seen_at": now, "etag": etag}))
    else:
        row = session.get(IndexProblem, key)
        if row is not None and row.reason == BAD_LABEL:
            session.delete(row)


def _read_label(s3, key: str, etag: Optional[str]) -> Optional[str]:
    """The label file's text of the indexed revision (`etag`, a conditional read), cached by (key, etag); None when
    it cannot be read or was replaced since indexing (the next index pass picks the new revision up)."""
    cached = LABEL_CACHE.get((key, etag))
    if cached is not None:
        return cached
    try:
        text = s3.get_text(key, if_match=etag) if etag else s3.get_text(key)
    except ETagMismatch:
        return None
    except Exception:  # deleted since indexing, throttled, ...: the frame shows as not run
        return None
    if etag:
        LABEL_CACHE.put((key, etag), text)
    return text


@router.get("/events/{event_id}/detections", response_model=DetectionsOut)
def get_detections(event_id: int, request: Request, staff: Staff = Depends(current_staff),
                   session: Session = SessionDep):
    require_id(event_id, "Event not found")
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
        boxes = _boxes(text) if text is not None else None
        if text is not None:
            _label_problem(session, key, art.etag, _now(request), bad=boxes is None)
        if boxes is None:  # unreadable, replaced since indexing, or malformed: never shown as an empty scene
            frames[index] = FrameBoxes(frame_index=index, t_sec=t_sec, status="not_run", boxes=[])
            continue
        frames[index] = FrameBoxes(frame_index=index, t_sec=t_sec, status="ran" if boxes else "ran_empty",
                                   boxes=boxes)
    return DetectionsOut(provenance="sampled", model=SAMPLED_MODEL, frames=[frames[i] for i in sorted(frames)])


# ---------------------------------------------------------------- review

@router.patch("/events/{event_id}/review", response_model=EventSummary)
def review_event(event_id: int, body: ReviewUpdate, request: Request, staff: Staff = Depends(current_staff),
                 session: Session = SessionDep):
    require_id(event_id, "Event not found")
    viewer = _Viewer(staff, request)
    row = _load_one(session, viewer, event_id)
    ev, customer_id = row[0], row[1]
    changes = body.model_dump(exclude_none=True)
    if changes:
        values = {"event_id": ev.id, "by": staff.id, "at": _now(request), **changes}
        session.execute(pg_insert(ReviewState).values(**values).on_conflict_do_update(
            index_elements=[ReviewState.event_id], set_={k: v for k, v in values.items() if k != "event_id"}))
        audit.record(session, staff.id, "review", target=f"event/{ev.id}", customer_id=customer_id,
                     device_id=_device_id(session, ev), detail=changes, ts=_now(request))
        session.expire_all()
        row = _load_one(session, viewer, event_id)
    return EventSummary(**_summary_of(session, viewer, row))
