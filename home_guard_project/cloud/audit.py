"""Append-only audit trail. The DB rejects UPDATE/DELETE on audit_log."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from home_guard_project.fleet_contract import health, notices

from .models import AuditLog, Camera, Customer, Device, OwnerNotice, Staff

log = logging.getLogger(__name__)

NOTICE_WINDOW = timedelta(minutes=30)


def record(session: Session, staff_id: Optional[int], action: str, target: str = "", reason: str = "",
           customer_id: Optional[int] = None, device_id: Optional[str] = None,
           detail: Optional[dict[str, Any]] = None, ts: Optional[datetime] = None,
           staff_name: Optional[str] = None) -> AuditLog:
    """`staff_name` names a non-staff actor (the box's setup program) when `staff_id` is None."""
    if staff_id is not None:
        staff_name = session.scalar(select(Staff.name).where(Staff.id == staff_id))
    row = AuditLog(ts=ts or datetime.now(timezone.utc), staff_id=staff_id, staff_name=staff_name, action=action, target=target,
                   reason=reason, customer_id=customer_id, device_id=device_id, detail=detail)
    session.add(row)
    session.flush()
    return row


# ---------------------------------------------------------------- owner notices

def camera_label(session: Session, device_pk: int, camera: str) -> str:
    """What the owner calls a camera: the box's camera_list name or a Camera.display_name (an old id by its channel),
    else "Camera N" (fleet_contract.health.camera_label). Never the raw id: this text reaches the owner."""
    names = {cam: shown.strip() for cam, shown in session.execute(
        select(Camera.name, Camera.display_name).where(Camera.device_pk == device_pk)).all() if shown and shown.strip()}
    device = session.get(Device, device_pk)
    names.update({c["id"]: c["name"] for c in health.camera_list(device.last_heartbeat if device else None) or ()
                  if c["name"]})
    return health.camera_label(camera, names)


def _notice_body(session: Session, row: OwnerNotice, device: Device, staff_name: str) -> dict[str, Any]:
    tz_name = session.scalar(select(Customer.timezone).where(Customer.id == device.customer_id)) or "UTC"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
    first = datetime.fromtimestamp(row.first_ts, timezone.utc)
    last = datetime.fromtimestamp(row.last_ts, timezone.utc)
    cameras = list(row.cameras or [])
    t1, t2 = f"{first.astimezone(tz):%H:%M}", f"{last.astimezone(tz):%H:%M}"
    when = t1 if t1 == t2 else f"{t1}–{t2}"
    message = notices.notice_message(row.kind, cameras, when)
    return {"schema_version": 1, "id": row.id, "kind": row.kind, "staff_name": staff_name, "cameras": cameras,
            "from_utc": first.strftime("%Y-%m-%dT%H:%M:%SZ"), "to_utc": last.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "message": message}


def _upload_notice(session: Session, s3, row: OwnerNotice, device: Device, staff_name: str) -> None:
    """Write the notice object; a failure leaves the row `pending_upload` instead of failing the caller."""
    try:
        s3.put_json(row.s3_key, _notice_body(session, row, device, staff_name))
        row.pending_upload = False
    except Exception as e:
        log.warning("owner notice %s could not be written to S3 (%s); kept for retry", row.id, type(e).__name__)
        row.pending_upload = True


def owner_notice(session: Session, s3, device: Device, staff: Staff, kind: str, cameras: list[str],
                 now: Optional[datetime] = None) -> OwnerNotice:
    """Tell the owner (in his app, via fleet/<device>/notices/) that staff looked at his recordings.

    Views by the same staff on the same device within 30 minutes share one notice object. The advisory lock
    (held until the caller's transaction ends) serialises concurrent views of one (staff, device, kind), so two
    views can never both decide to start a notice. The DB row is decided first; the S3 write comes after, and its
    failure only marks the row `pending_upload`. Every DB write that could fail is flushed before the put. If the
    surrounding transaction still rolls back after a successful put, that is harmless: the object only names the
    time window, and the next view rewrites it from DB state (same key while the window is open, a new key after).
    """
    notices.check_kind(kind)  # fleet_contract.notices.NOTICE_KINDS: kinds are data the owner's side branches on
    now = now or datetime.now(timezone.utc)
    ts = now.timestamp()
    session.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"),
                    {"k": f"owner_notice:{staff.id}:{device.id}:{kind}"})
    row = session.scalar(select(OwnerNotice).where(
        OwnerNotice.device_pk == device.id, OwnerNotice.staff_id == staff.id, OwnerNotice.kind == kind)
        .order_by(OwnerNotice.last_ts.desc(), OwnerNotice.id.desc()).limit(1))
    if row is not None and row.last_ts is not None and row.last_ts >= (now - NOTICE_WINDOW).timestamp():
        row.cameras = list(row.cameras or []) + [c for c in cameras if c not in (row.cameras or [])]
        row.last_ts = max(row.last_ts, ts)
    else:
        row = OwnerNotice(device_pk=device.id, staff_id=staff.id, kind=kind, cameras=list(dict.fromkeys(cameras)),
                          first_ts=ts, last_ts=ts)
        session.add(row)
        session.flush()
        row.s3_key = (f"fleet/{device.device_id}/notices/"
                      f"{datetime.fromtimestamp(ts, timezone.utc):%Y%m%dT%H%M%S}_{row.id}.json")
    session.flush()
    _upload_notice(session, s3, row, device, staff.name)
    session.flush()
    return row


def retry_pending_notices(session: Session, s3, now: Optional[datetime] = None) -> int:
    """Upload notices whose first write failed; returns how many succeeded."""
    done = 0
    for row in session.scalars(select(OwnerNotice).where(OwnerNotice.pending_upload.is_(True))
                               .order_by(OwnerNotice.id).with_for_update()).all():
        device = session.get(Device, row.device_pk)
        staff_name = (session.scalar(select(Staff.name).where(Staff.id == row.staff_id))
                      if row.staff_id is not None else None) or "Home Guard support"
        _upload_notice(session, s3, row, device, staff_name)
        done += 0 if row.pending_upload else 1
    session.flush()
    return done
