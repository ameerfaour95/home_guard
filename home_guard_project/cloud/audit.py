"""Append-only audit trail. The DB rejects UPDATE/DELETE on audit_log."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import AuditLog, Staff


def record(session: Session, staff_id: Optional[int], action: str, target: str = "", reason: str = "",
           customer_id: Optional[int] = None, device_id: Optional[str] = None,
           detail: Optional[dict[str, Any]] = None) -> AuditLog:
    staff_name = session.scalar(select(Staff.name).where(Staff.id == staff_id)) if staff_id is not None else None
    row = AuditLog(ts=datetime.now(timezone.utc), staff_id=staff_id, staff_name=staff_name, action=action, target=target,
                   reason=reason, customer_id=customer_id, device_id=device_id, detail=detail)
    session.add(row)
    session.flush()
    return row
