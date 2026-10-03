import functools
import math
import threading
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from .. import audit
from .. import studio as studio_logic
from ..deps import SessionDep, current_staff, require_role
from ..models import Collection, CollectionItem, Customer, Device, Event, Export, Staff
from ..schemas import (
    CollectionIn,
    CollectionItems,
    CollectionOut,
    EventPage,
    ExportOut,
    ExportPreview,
    ExportRequest,
    SavedFilter,
)
from .events import _decode_cursor, _Viewer, _in_collection, event_page

router = APIRouter(prefix="/studio", tags=["studio"], dependencies=[Depends(current_staff)])

_studio_staff = require_role("admin", "labeler")

# Route functions below carry no docstrings on purpose: a docstring becomes the operation description in the
# frozen OpenAPI document (docs/admin/openapi.json).


@router.get("/filters", response_model=list[SavedFilter])
def list_filters():
    return studio_logic.BUILTIN_FILTERS


# ---------------------------------------------------------------- collections

MAX_ITEMS_PER_CALL = 5000
MAX_DB_ID = 2 ** 31 - 1  # ids are 32-bit integers in the database
_EVENT_NOT_FOUND = "Event not found"


def _staff_name(session: Session, staff_id: Optional[int]) -> str:
    if staff_id is None:
        return ""
    return session.scalar(select(Staff.name).where(Staff.id == staff_id)) or ""


def _event_counts(session: Session, labeler: bool, collection_ids: list[int]) -> dict[int, int]:
    """Events per collection that this viewer may see."""
    if not collection_ids:
        return {}
    stmt = (select(CollectionItem.collection_id, func.count())
            .join(Event, Event.id == CollectionItem.event_id)
            .join(Device, Device.id == Event.device_pk).join(Customer, Customer.id == Device.customer_id)
            .where(CollectionItem.collection_id.in_(collection_ids)).group_by(CollectionItem.collection_id))
    if labeler:
        stmt = stmt.where(Customer.consent_training.is_(True))
    return dict(session.execute(stmt).all())


def _collection_out(session: Session, staff: Staff, col: Collection, count: Optional[int] = None) -> CollectionOut:
    if count is None:
        count = _event_counts(session, staff.role == "labeler", [col.id]).get(col.id, 0)
    return CollectionOut(id=col.id, name=col.name, description=col.description or "", event_count=count,
                         created_by=_staff_name(session, col.created_by),
                         created_utc=col.created_at or datetime.fromtimestamp(0, timezone.utc))


def _load_collection(session: Session, collection_id: int) -> Collection:
    col = session.get(Collection, collection_id) if 1 <= collection_id <= MAX_DB_ID else None
    if col is None:
        raise HTTPException(status_code=404, detail="Collection not found")
    return col


def _visible_events(session: Session, staff: Staff, event_ids: list[int]) -> list[int]:
    """The distinct ids, every one of which must exist and be visible to `staff`; otherwise one uniform 404, so
    a labeler cannot tell an event they may not see from one that does not exist."""
    ids = sorted(set(event_ids))
    if len(ids) > MAX_ITEMS_PER_CALL:
        raise HTTPException(status_code=400, detail=f"At most {MAX_ITEMS_PER_CALL} events per call")
    if any(not 1 <= i <= MAX_DB_ID for i in ids):
        raise HTTPException(status_code=404, detail=_EVENT_NOT_FOUND)
    if not ids:
        return []
    stmt = (select(Event.id).join(Device, Device.id == Event.device_pk)
            .join(Customer, Customer.id == Device.customer_id).where(Event.id.in_(ids)))
    if staff.role == "labeler":
        stmt = stmt.where(Customer.consent_training.is_(True))
    if set(session.scalars(stmt)) != set(ids):
        raise HTTPException(status_code=404, detail=_EVENT_NOT_FOUND)
    return ids


@router.get("/collections", response_model=list[CollectionOut], dependencies=[Depends(_studio_staff)])
def list_collections(staff: Staff = Depends(current_staff), session: Session = SessionDep):
    cols = session.scalars(select(Collection).order_by(Collection.id.desc())).all()
    counts = _event_counts(session, staff.role == "labeler", [c.id for c in cols])
    return [_collection_out(session, staff, c, counts.get(c.id, 0)) for c in cols]


@router.post("/collections", response_model=CollectionOut, dependencies=[Depends(_studio_staff)])
def create_collection(body: CollectionIn, request: Request, staff: Staff = Depends(current_staff),
                      session: Session = SessionDep):
    name = body.name.strip()
    if not name or len(name) > 255:
        raise HTTPException(status_code=400, detail="A collection needs a name of at most 255 characters")
    col = Collection(name=name, description=body.description or "", created_by=staff.id,
                     created_at=request.app.state.clock())
    session.add(col)
    session.flush()
    audit.record(session, staff.id, "collection_create", target=f"collection/{col.id}", detail={"name": name})
    return _collection_out(session, staff, col, 0)


@router.post("/collections/{collection_id}/items", response_model=CollectionOut,
             dependencies=[Depends(_studio_staff)])
def add_collection_items(collection_id: int, body: CollectionItems, request: Request,
                         staff: Staff = Depends(current_staff), session: Session = SessionDep):
    col = _load_collection(session, collection_id)
    ids = _visible_events(session, staff, body.event_ids)
    if ids:
        now = request.app.state.clock()
        session.execute(pg_insert(CollectionItem).values(
            [{"collection_id": col.id, "event_id": i, "added_by": staff.id, "added_at": now} for i in ids])
            .on_conflict_do_nothing(index_elements=[CollectionItem.collection_id, CollectionItem.event_id]))
        audit.record(session, staff.id, "collection_add", target=f"collection/{col.id}",
                     detail={"event_ids": ids})
    return _collection_out(session, staff, col)


@router.delete("/collections/{collection_id}/items", response_model=CollectionOut,
               dependencies=[Depends(_studio_staff)])
def remove_collection_items(collection_id: int, body: CollectionItems, staff: Staff = Depends(current_staff),
                            session: Session = SessionDep):
    col = _load_collection(session, collection_id)
    ids = _visible_events(session, staff, body.event_ids)
    if ids:
        session.execute(delete(CollectionItem).where(CollectionItem.collection_id == col.id,
                                                     CollectionItem.event_id.in_(ids)))
        audit.record(session, staff.id, "collection_remove", target=f"collection/{col.id}",
                     detail={"event_ids": ids})
    return _collection_out(session, staff, col)


# ---------------------------------------------------------------- exports

MANIFEST_URL_TTL = 3600


def _export_out(session: Session, request: Request, export: Export) -> ExportOut:
    s3 = request.app.state.s3
    url = None
    if export.state in ("ready", "partial") and s3 is not None:
        url = s3.presign(export.s3_prefix + "manifest.json", ttl=MANIFEST_URL_TTL)  # only ever the manifest
    return ExportOut(id=export.id, name=export.name, version=export.version, state=export.state,
                     item_count=export.item_count, s3_prefix=export.s3_prefix, manifest_url=url, error=export.error,
                     created_utc=export.created_at, created_by=_staff_name(session, export.created_by))


def _check_request(body: ExportRequest) -> None:
    if not body.formats:
        raise HTTPException(status_code=400, detail="Choose at least one format")
    if len(body.name) > 100:
        raise HTTPException(status_code=400, detail="The export name is too long (at most 100 characters)")
    values = list(body.split.values())
    if not values or any(not math.isfinite(v) or v < 0 for v in values) or sum(values) <= 0:
        raise HTTPException(status_code=400, detail="Split fractions must be non-negative and not all zero")


def default_export_runner(job) -> None:
    """Runs an export job on a background thread (tests inject a synchronous runner on app.state)."""
    threading.Thread(target=job, name="training-export", daemon=True).start()


@router.get("/exports", response_model=list[ExportOut], dependencies=[Depends(_studio_staff)])
def list_exports(request: Request, session: Session = SessionDep):
    exports = session.scalars(select(Export).order_by(Export.id.desc()).limit(500)).all()
    return [_export_out(session, request, e) for e in exports]


@router.post("/exports", response_model=ExportOut, dependencies=[Depends(_studio_staff)])
def create_export(body: ExportRequest, request: Request, staff: Staff = Depends(current_staff),
                  session: Session = SessionDep):
    _check_request(body)
    _load_collection(session, body.collection_id)
    s3 = request.app.state.s3
    if s3 is None:
        raise HTTPException(status_code=503, detail="Storage is not configured")
    version = studio_logic.next_version(session, s3, body.name)
    export = Export(name=body.name, version=version, state="queued",
                    s3_prefix=studio_logic.export_prefix(body.name, version),
                    request={**body.model_dump(), "as_labeler": staff.role == "labeler"},
                    created_by=staff.id, created_at=request.app.state.clock())
    session.add(export)
    session.flush()
    audit.record(session, staff.id, "export_create", target=f"export/{export.id}",
                 detail={"name": body.name, "version": version, "collection_id": body.collection_id,
                         "formats": list(body.formats)})
    session.commit()  # the job reads the row from its own session
    runner = getattr(request.app.state, "export_runner", None) or default_export_runner
    runner(functools.partial(studio_logic.run_export_job, request.app.state.sessionmaker, s3, export.id,
                             request.app.state.settings.jwt_secret))
    session.refresh(export)
    return _export_out(session, request, export)


@router.get("/collections/{collection_id}/items", response_model=EventPage,
            dependencies=[Depends(_studio_staff)])
def list_collection_items(
    collection_id: int,
    request: Request,
    cursor: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
    staff: Staff = Depends(current_staff),
    session: Session = SessionDep,
):
    """Events in a collection, newest first, same rows and pseudonyms as /events. Labelers only see events
    of customers who gave training consent."""
    if cursor is not None:
        _decode_cursor(cursor)  # a bad cursor is a 400 before any query
    if session.get(Collection, collection_id) is None:
        raise HTTPException(status_code=404, detail="Collection not found")
    viewer = _Viewer(staff, request)
    return event_page(session, viewer, [viewer.visible(), _in_collection(collection_id)], cursor, limit)


@router.post("/exports/preview", response_model=ExportPreview, dependencies=[Depends(_studio_staff)])
def preview_export(body: ExportRequest, request: Request, staff: Staff = Depends(current_staff),
                   session: Session = SessionDep):
    """What an export of this request would contain, using the same selection and split code as the builder.

    `include_fallback_ai=false` excludes only fallback/failed AI from vlm.jsonl; such events still contribute
    clips and YOLO labels, so `no_real_ai` is reported only for VLM-only exports (formats == ["vlm_jsonl"]).
    Events of customers without training consent are always excluded (`no_training_consent`).
    """
    # For a labeler, who may not know those events exist, they are silently left out instead (no exclusion
    # entry, no count). (A comment, not part of the docstring: the docstring is in the frozen OpenAPI.)
    if session.get(Collection, body.collection_id) is None:
        raise HTTPException(status_code=404, detail="Collection not found")
    included, excluded = studio_logic.select_export_items(session, body.collection_id, body,
                                                          labeler=staff.role == "labeler")
    splits = studio_logic.assign_splits(included, body.name, body.split,
                                        secret=request.app.state.settings.jwt_secret)
    counts = {n: 0 for n in body.split}
    for s in splits.values():
        counts[s] = counts.get(s, 0) + 1
    warnings = []
    if abs(sum(body.split.values()) - 1.0) > 1e-6:
        warnings.append("Split fractions do not add up to 1; they were normalised.")
    if not included:
        warnings.append("No events would be exported.")
    else:
        empty = [n for n, c in counts.items() if c == 0 and body.split[n] > 0]
        if empty:
            warnings.append("Empty split: " + ", ".join(sorted(empty)))
    if excluded:
        warnings.append(f"{len(excluded)} event(s) excluded.")
    return ExportPreview(included_ids=[e.id for e in included], excluded=excluded, split_counts=counts,
                         groups=studio_logic.group_count(included), warnings=warnings)


@router.get("/exports/{export_id}", response_model=ExportOut, dependencies=[Depends(_studio_staff)])
def get_export(export_id: int, request: Request, session: Session = SessionDep):
    export = session.get(Export, export_id) if 1 <= export_id <= MAX_DB_ID else None
    if export is None:
        raise HTTPException(status_code=404, detail="Export not found")
    return _export_out(session, request, export)
