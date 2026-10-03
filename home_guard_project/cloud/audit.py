"""Append-only audit trail. The DB rejects UPDATE/DELETE on audit_log."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from .models import AuditLog


def record(session: Session, staff_id: Optional[int], action: str, target: str = "", reason: str = "",
           customer_id: Optional[int] = None, device_id: Optional[str] = None,
           detail: Optional[dict[str, Any]] = None) -> AuditLog:
    row = AuditLog(ts=datetime.now(timezone.utc), staff_id=staff_id, action=action, target=target,
                   reason=reason, customer_id=customer_id, device_id=device_id, detail=detail)
    session.add(row)
    session.flush()
    return row
