from fastapi import APIRouter, Depends, HTTPException

from ..deps import current_staff, require_role
from ..models import Staff
from ..schemas import AnnotationIn, AnnotationOut, AnnotationVersion, PublishOut, PublishRequest, ReviewDecision

router = APIRouter(tags=["annotations"], dependencies=[Depends(current_staff)])

# Contract 2f stubs: the shapes and roles are frozen here; the bodies land with the server implementation.
# Route functions carry no docstrings on purpose (they would become the OpenAPI description).

_NOT_BUILT = "Not implemented yet"


@router.get("/events/{event_id}/annotation", response_model=AnnotationOut,
            dependencies=[Depends(require_role("admin", "labeler", "support"))])
def get_annotation(event_id: int):
    raise HTTPException(status_code=501, detail=_NOT_BUILT)


@router.put("/events/{event_id}/annotation", response_model=AnnotationOut,
            dependencies=[Depends(require_role("admin", "labeler"))])
def save_annotation(event_id: int, body: AnnotationIn):
    raise HTTPException(status_code=501, detail=_NOT_BUILT)


@router.post("/events/{event_id}/annotation/review", response_model=AnnotationOut)
def review_annotation(event_id: int, body: ReviewDecision, staff: Staff = Depends(require_role("admin"))):
    raise HTTPException(status_code=501, detail=_NOT_BUILT)


@router.get("/events/{event_id}/annotation/history", response_model=list[AnnotationVersion],
            dependencies=[Depends(require_role("admin", "labeler", "support"))])
def annotation_history(event_id: int):
    raise HTTPException(status_code=501, detail=_NOT_BUILT)


@router.post("/studio/collections/{collection_id}/publish", response_model=PublishOut,
             dependencies=[Depends(require_role("admin"))])
def publish_collection(collection_id: int, body: PublishRequest):
    raise HTTPException(status_code=501, detail=_NOT_BUILT)


@router.get("/studio/publishes", response_model=list[PublishOut], dependencies=[Depends(require_role("admin"))])
def list_publishes():
    raise HTTPException(status_code=501, detail=_NOT_BUILT)
