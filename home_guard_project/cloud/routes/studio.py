import functools
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from .. import audit
from .. import studio as studio_logic
from ..deps import MAX_DB_ID, NAME_MAX, NOTES_MAX, SessionDep, check_length, current_staff, require_role
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
#
# Collections and exports a labeler creates are visible only to that labeler and to admins, and their names are
# not checked against household identities (that check answered differently for hidden households, an oracle).
# Names chosen by admins, which every labeler may see, keep the identity check.


@router.get("/filters", response_model=list[SavedFilter])
def list_filters():
    return studio_logic.BUILTIN_FILTERS


# ---------------------------------------------------------------- collections

MAX_ITEMS_PER_CALL = 5000
_EVENT_NOT_FOUND = "Event not found"
_COLLECTION_NOT_FOUND = "Collection not found"


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


def _labeler_ids(session: Session) -> set[int]:
    return set(session.scalars(select(Staff.id).where(Staff.role == "labeler")))


def _collection_visible(staff: Staff, col: Collection, labelers: set[int]) -> bool:
    """Admins see every collection; a labeler sees their own and those made by non-labelers (admins)."""
    if staff.role != "labeler":
        return True
    return col.created_by == staff.id or col.created_by not in labelers


def _load_collection(session: Session, collection_id: int, staff: Staff) -> Collection:
    """The collection when it exists and `staff` may see it; otherwise one 404, the same for hidden and absent."""
    col = session.get(Collection, collection_id) if 1 <= collection_id <= MAX_DB_ID else None
    if col is None or not _collection_visible(staff, col, _labeler_ids(session)):
        raise HTTPException(status_code=404, detail=_COLLECTION_NOT_FOUND)
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
    labelers = _labeler_ids(session)
    cols = [c for c in session.scalars(select(Collection).order_by(Collection.id.desc())).all()
            if _collection_visible(staff, c, labelers)]
    counts = _event_counts(session, staff.role == "labeler", [c.id for c in cols])
    return [_collection_out(session, staff, c, counts.get(c.id, 0)) for c in cols]


@router.post("/collections", response_model=CollectionOut, dependencies=[Depends(_studio_staff)])
def create_collection(body: CollectionIn, request: Request, staff: Staff = Depends(current_staff),
                      session: Session = SessionDep):
    check_length("name", body.name, NAME_MAX)
    check_length("description", body.description, NOTES_MAX)
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail=f"A collection needs a name of at most {NAME_MAX} characters")
    if staff.role != "labeler":  # a labeler's collection is private to them: no identity check, no oracle
        _checked(studio_logic.validate_collection_text, session, name, body.description)
    now = request.app.state.clock()
    col = Collection(name=name, description=body.description or "", created_by=staff.id, created_at=now)
    session.add(col)
    session.flush()
    audit.record(session, staff.id, "collection_create", target=f"collection/{col.id}", detail={"name": name},
                 ts=now)
    return _collection_out(session, staff, col, 0)


@router.post("/collections/{collection_id}/items", response_model=CollectionOut,
             dependencies=[Depends(_studio_staff)])
def add_collection_items(collection_id: int, body: CollectionItems, request: Request,
                         staff: Staff = Depends(current_staff), session: Session = SessionDep):
    col = _load_collection(session, collection_id, staff)
    ids = _visible_events(session, staff, body.event_ids)
    if ids:
        now = request.app.state.clock()
        session.execute(pg_insert(CollectionItem).values(
            [{"collection_id": col.id, "event_id": i, "added_by": staff.id, "added_at": now} for i in ids])
            .on_conflict_do_nothing(index_elements=[CollectionItem.collection_id, CollectionItem.event_id]))
        audit.record(session, staff.id, "collection_add", target=f"collection/{col.id}",
                     detail={"event_ids": ids}, ts=now)
    return _collection_out(session, staff, col)


@router.delete("/collections/{collection_id}/items", response_model=CollectionOut,
               dependencies=[Depends(_studio_staff)])
def remove_collection_items(collection_id: int, body: CollectionItems, request: Request,
                            staff: Staff = Depends(current_staff), session: Session = SessionDep):
    col = _load_collection(session, collection_id, staff)
    ids = _visible_events(session, staff, body.event_ids)
    if ids:
        session.execute(delete(CollectionItem).where(CollectionItem.collection_id == col.id,
                                                     CollectionItem.event_id.in_(ids)))
        audit.record(session, staff.id, "collection_remove", target=f"collection/{col.id}",
                     detail={"event_ids": ids}, ts=request.app.state.clock())
    return _collection_out(session, staff, col)


# ---------------------------------------------------------------- exports

MANIFEST_URL_TTL = 3600


_EXPORT_NOT_FOUND = "Export not found"


def _checked(fn, *args):
    """Run a studio validator; its refusal becomes a 400 with the validator's message."""
    try:
        return fn(*args)
    except studio_logic.RequestError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _export_out(session: Session, request: Request, staff: Staff, export: Export) -> ExportOut:
    s3 = request.app.state.s3
    url = None
    if export.state in ("ready", "partial") and s3 is not None:
        url = s3.presign(export.s3_prefix + "manifest.json", ttl=MANIFEST_URL_TTL)  # only ever the manifest
    error = export.error
    if error is None and staff.role == "admin" and studio_logic.consent_withdrawn(session, export):
        error = studio_logic.CONSENT_WARNING  # computed when read; never stored
    return ExportOut(id=export.id, name=export.name, version=export.version, state=export.state,
                     item_count=export.item_count, s3_prefix=export.s3_prefix, manifest_url=url, error=error,
                     created_utc=export.created_at, created_by=_staff_name(session, export.created_by))


def _run_job(request: Request, job) -> None:
    """Hand an export job to the injected runner (tests run it synchronously), else to the bounded executor."""
    runner = getattr(request.app.state, "export_runner", None)
    if runner is None:
        runner = request.app.state.export_executor.submit
    runner(job)


@router.get("/exports", response_model=list[ExportOut], dependencies=[Depends(_studio_staff)])
def list_exports(request: Request, staff: Staff = Depends(current_staff), session: Session = SessionDep):
    stmt = select(Export).order_by(Export.id.desc()).limit(500)
    if staff.role != "admin":
        stmt = stmt.where(Export.created_by == staff.id)
    exports = [e for e in session.scalars(stmt).all() if studio_logic.export_visible_to(session, staff, e)]
    return [_export_out(session, request, staff, e) for e in exports]


@router.post("/exports", response_model=ExportOut, dependencies=[Depends(_studio_staff)])
def create_export(body: ExportRequest, request: Request, staff: Staff = Depends(current_staff),
                  session: Session = SessionDep):
    labeler = staff.role == "labeler"
    _checked(studio_logic.validate_export_request, session, body, labeler)
    _load_collection(session, body.collection_id, staff)
    _checked(studio_logic.check_selection_size, session, body.collection_id, labeler)
    s3 = request.app.state.s3
    if s3 is None:
        raise HTTPException(status_code=503, detail="Storage is not configured")
    secret = request.app.state.settings.jwt_secret
    snapshot = studio_logic.take_snapshot(session, body, labeler=labeler, secret=secret)
    version = studio_logic.next_version(session, s3, body.name)
    export = Export(name=body.name, version=version, state="queued",
                    s3_prefix=studio_logic.export_prefix(body.name, version),
                    request={**body.model_dump(), "as_labeler": labeler, "snapshot": snapshot},
                    created_by=staff.id, created_at=request.app.state.clock())
    session.add(export)
    session.flush()
    audit.record(session, staff.id, "export_create", target=f"export/{export.id}",
                 detail={"name": body.name, "version": version, "collection_id": body.collection_id,
                         "formats": list(body.formats)}, ts=request.app.state.clock())
    session.commit()  # the job reads the row from its own session
    _run_job(request, functools.partial(studio_logic.run_export_job, request.app.state.sessionmaker, s3,
                                        export.id, secret))
    session.refresh(export)
    return _export_out(session, request, staff, export)


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
    _load_collection(session, collection_id, staff)
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
    labeler = staff.role == "labeler"
    warnings = list(_checked(studio_logic.validate_export_request, session, body, labeler))
    _load_collection(session, body.collection_id, staff)
    _checked(studio_logic.check_selection_size, session, body.collection_id, labeler)
    included, excluded = studio_logic.select_export_items(session, body.collection_id, body, labeler=labeler)
    splits = studio_logic.assign_splits(included, body.name, body.split,
                                        secret=request.app.state.settings.jwt_secret)
    counts = {n: 0 for n in body.split}
    for s in splits.values():
        counts[s] = counts.get(s, 0) + 1
    if not included:
        warnings.append("No events would be exported.")
    else:
        empty = [n for n, c in counts.items() if c == 0 and body.split[n] > 0]
        if empty:
            warnings.append("Empty split: " + ", ".join(sorted(empty)))
        warnings += studio_logic.small_set_warnings(studio_logic.group_count(included))
    if excluded:
        warnings.append(f"{len(excluded)} event(s) excluded.")
    return ExportPreview(included_ids=[e.id for e in included], excluded=excluded, split_counts=counts,
                         groups=studio_logic.group_count(included), warnings=warnings)


@router.get("/exports/{export_id}", response_model=ExportOut, dependencies=[Depends(_studio_staff)])
def get_export(export_id: int, request: Request, staff: Staff = Depends(current_staff),
               session: Session = SessionDep):
    export = session.get(Export, export_id) if 1 <= export_id <= MAX_DB_ID else None
    if export is None or not studio_logic.export_visible_to(session, staff, export):
        raise HTTPException(status_code=404, detail=_EXPORT_NOT_FOUND)  # the same for hidden and absent
    return _export_out(session, request, staff, export)
