from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from .. import studio as studio_logic
from ..deps import SessionDep, current_staff, require_role
from ..models import Collection, Staff
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


@router.get("/filters", response_model=list[SavedFilter])
def list_filters():
    raise HTTPException(status_code=501)


@router.get("/collections", response_model=list[CollectionOut])
def list_collections():
    raise HTTPException(status_code=501)


@router.post("/collections", response_model=CollectionOut)
def create_collection(body: CollectionIn):
    raise HTTPException(status_code=501)


@router.post("/collections/{collection_id}/items", response_model=CollectionOut)
def add_collection_items(collection_id: int, body: CollectionItems):
    raise HTTPException(status_code=501)


@router.delete("/collections/{collection_id}/items", response_model=CollectionOut)
def remove_collection_items(collection_id: int, body: CollectionItems):
    raise HTTPException(status_code=501)


@router.get("/exports", response_model=list[ExportOut])
def list_exports():
    raise HTTPException(status_code=501)


@router.post("/exports", response_model=ExportOut)
def create_export(body: ExportRequest):
    raise HTTPException(status_code=501)


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


@router.get("/exports/{export_id}", response_model=ExportOut)
def get_export(export_id: int):
    raise HTTPException(status_code=501)
