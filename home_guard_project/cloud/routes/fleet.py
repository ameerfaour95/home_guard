from fastapi import APIRouter, Depends, HTTPException

from ..deps import bearer
from ..schemas import DeviceSummary, EnrollRequest, FleetResponse

router = APIRouter(tags=["fleet"], dependencies=[Depends(bearer)])


@router.get("/fleet", response_model=FleetResponse)
def fleet():
    raise HTTPException(status_code=501)


@router.post("/devices/enroll", response_model=DeviceSummary)
def enroll(body: EnrollRequest):
    raise HTTPException(status_code=501)
