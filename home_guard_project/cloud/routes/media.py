from fastapi import APIRouter, Depends, HTTPException

from ..deps import bearer
from ..schemas import MediaAccess, MediaAccessRequest

router = APIRouter(tags=["media"], dependencies=[Depends(bearer)])


@router.post("/artifacts/{artifact_id}/access", response_model=MediaAccess)
def artifact_access(artifact_id: int, body: MediaAccessRequest):
    raise HTTPException(status_code=501)


@router.get(
    "/events/{event_id}/thumbnail",
    status_code=307,
    responses={307: {"description": "Redirect to a presigned thumbnail URL"}},
)
def event_thumbnail(event_id: int):
    raise HTTPException(status_code=501)
