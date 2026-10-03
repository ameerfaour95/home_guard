import base64
import binascii
import re
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from ..deps import SessionDep, current_staff, require_role
from ..models import AuditLog, IndexProblem as IndexProblemRow, Staff
from ..schemas import AuditEntry, AuditPage, IndexProblem

router = APIRouter(tags=["audit"], dependencies=[Depends(current_staff)])

PAGE_SIZE = 50
PROBLEM_LIMIT = 500
_CURSOR = re.compile(r"(\S+):([1-9]\d{0,18})")


def _encode(ts: datetime, row_id: int) -> str:
    return base64.urlsafe_b64encode(f"{ts.isoformat()}:{row_id}".encode("ascii")).decode("ascii")


def _decode(cursor: str) -> tuple[datetime, int]:
    bad = HTTPException(status_code=400, detail="Invalid cursor")
    if not cursor or len(cursor) > 128:
        raise bad
    try:
        raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True).decode("ascii")
        match = _CURSOR.fullmatch(raw)
        if match is None:
            raise bad
        ts = datetime.fromisoformat(match.group(1))
        row_id = int(match.group(2))
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise bad
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    if row_id > 2 ** 63 - 1:
        raise bad
    return ts, row_id


@router.get("/audit", response_model=AuditPage, dependencies=[Depends(require_role("admin"))])
def list_audit(
    staff: Optional[str] = None,
    customer_id: Optional[int] = None,
    action: Optional[str] = None,
    cursor: Optional[str] = None,
    session: Session = SessionDep,
):
    after = _decode(cursor) if cursor is not None else None
    q = select(AuditLog, Staff.email).outerjoin(Staff, Staff.id == AuditLog.staff_id)
    if staff:
        if staff.isdigit() and len(staff) <= 18:
            q = q.where(AuditLog.staff_id == int(staff))
        else:
            q = q.where(AuditLog.staff_id.in_(select(Staff.id).where(Staff.email == staff)))
    if customer_id is not None:
        q = q.where(AuditLog.customer_id == customer_id)
    if action:
        q = q.where(AuditLog.action == action)
    if after is not None:
        ts, rid = after
        q = q.where(or_(AuditLog.ts < ts, and_(AuditLog.ts == ts, AuditLog.id < rid)))
    rows = session.execute(q.order_by(AuditLog.ts.desc(), AuditLog.id.desc()).limit(PAGE_SIZE + 1)).all()
    more = len(rows) > PAGE_SIZE
    rows = rows[:PAGE_SIZE]
    items = [AuditEntry(id=r.id, ts=r.ts, staff=r.staff_name or email or "deleted staff", action=r.action,
                        customer_id=r.customer_id, device_id=r.device_id, target=r.target or "",
                        reason=r.reason or "", detail=r.detail) for r, email in rows]
    return AuditPage(items=items, next_cursor=_encode(rows[-1][0].ts, rows[-1][0].id) if more and rows else None)


@router.get("/index/problems", response_model=list[IndexProblem], dependencies=[Depends(require_role("admin"))])
def index_problems(session: Session = SessionDep):
    rows = session.scalars(select(IndexProblemRow).order_by(IndexProblemRow.seen_at.desc(),
                                                            IndexProblemRow.s3_key).limit(PROBLEM_LIMIT))
    return [IndexProblem(s3_key=r.s3_key, reason=r.reason, seen_utc=r.seen_at) for r in rows]
