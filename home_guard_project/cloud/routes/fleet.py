from fastapi import APIRouter, Depends, HTTPException

from ..deps import bearer, require_role
from ..schemas import DeviceSummary, EnrollRequest, FleetResponse

router = APIRouter(tags=["fleet"], dependencies=[Depends(bearer), Depends(require_role("admin", "support"))])


@router.get("/fleet", response_model=FleetResponse)
def fleet():
    raise HTTPException(status_code=501)


@router.post("/devices/enroll", response_model=DeviceSummary)
def enroll(body: EnrollRequest):
    raise HTTPException(status_code=501)
