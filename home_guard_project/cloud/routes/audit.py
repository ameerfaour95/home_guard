from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ..deps import bearer
from ..schemas import AuditPage, IndexProblem

router = APIRouter(tags=["audit"], dependencies=[Depends(bearer)])


@router.get("/audit", response_model=AuditPage)
def list_audit(
    staff: Optional[str] = None,
    customer_id: Optional[int] = None,
    action: Optional[str] = None,
    cursor: Optional[str] = None,
):
    raise HTTPException(status_code=501)


@router.get("/index/problems", response_model=list[IndexProblem])
def index_problems():
    raise HTTPException(status_code=501)
