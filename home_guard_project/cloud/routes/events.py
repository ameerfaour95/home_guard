from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException

from ..deps import current_staff
from ..schemas import DensityOut, DetectionsOut, EventDetail, EventPage, EventSummary, ReviewCount, ReviewUpdate

router = APIRouter(tags=["events"], dependencies=[Depends(current_staff)])


@router.get("/events", response_model=EventPage)
def list_events(
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
    limit: int = 50,
):
    raise HTTPException(status_code=501)


@router.get("/events/density", response_model=DensityOut)
def events_density(
    from_utc: datetime,
    to_utc: datetime,
    site: Optional[str] = None,
    customer_id: Optional[int] = None,
    camera: Optional[str] = None,
    kind: Optional[str] = None,
    bucket: Literal["hour", "day"] = "hour",
):
    raise HTTPException(status_code=501)


@router.get("/events/review-count", response_model=ReviewCount)
def review_count():
    raise HTTPException(status_code=501)


@router.get("/events/{event_id}", response_model=EventDetail)
def get_event(event_id: int):
    raise HTTPException(status_code=501)


@router.get("/events/{event_id}/detections", response_model=DetectionsOut)
def get_detections(event_id: int):
    raise HTTPException(status_code=501)


@router.patch("/events/{event_id}/review", response_model=EventSummary)
def review_event(event_id: int, body: ReviewUpdate):
    raise HTTPException(status_code=501)
