from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Iterable, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from home_guard_project.fleet_contract import health
from home_guard_project.fleet_contract.legacy import parse_heartbeat

from .. import audit, redact
from ..deps import TEXT_MAX, SessionDep, check_length, require_id, require_role
from ..models import Camera, Customer, Device, Event, Feedback, Staff
from ..schemas import DensityOut, DeviceSummary, EnrollRequest, FleetResponse, HealthReason
from .events import fleet_activity_density

router = APIRouter(tags=["fleet"])

SITE_MAX = 64  # the column's width (sites and ssh users)
_SEVERITY_RANK = {"offline": 0, "critical": 1, "warning": 2, "unknown": 3, "healthy": 4}


def now_of(request: Request) -> datetime:
    return request.app.state.clock()


def build_summaries(session: Session, now: datetime, customer_id: Optional[int] = None,
                    device_pks: Optional[Iterable[int]] = None) -> list[DeviceSummary]:
    """One DeviceSummary per device, sorted by severity, customer name, site. Grouped queries, no per-device loops."""
    q = select(Device, Customer.name, Customer.name_source).join(Customer, Customer.id == Device.customer_id)
    if customer_id is not None:
        q = q.where(Device.customer_id == customer_id)
    if device_pks is not None:
        q = q.where(Device.id.in_(list(device_pks)))
    rows = session.execute(q).all()
    if not rows:
        return []
    pks = [r[0].id for r in rows]
    since24 = (now - timedelta(hours=24)).timestamp()
    events = dict(session.execute(select(Event.device_pk, func.count()).where(
        Event.device_pk.in_(pks), Event.start_ts >= since24).group_by(Event.device_pk)).all())
    alerts = dict(session.execute(select(Event.device_pk, func.count()).where(
        Event.device_pk.in_(pks), Event.start_ts >= since24, Event.kind == "alert").group_by(Event.device_pk)).all())
    false_alarms = dict(session.execute(select(Feedback.device_pk, func.count()).where(
        Feedback.device_pk.in_(pks), Feedback.verdict == "false_alarm",
        Feedback.received_at >= now - timedelta(days=7)).group_by(Feedback.device_pk)).all())
    cams: dict[int, set[str]] = {}
    for pk, name in session.execute(select(Camera.device_pk, Camera.name).where(Camera.device_pk.in_(pks))).all():
        cams.setdefault(pk, set()).add(name)

    out: list[DeviceSummary] = []
    for dev, cust_name, name_source in rows:
        hb = parse_heartbeat(dev.last_heartbeat) if isinstance(dev.last_heartbeat, dict) else None
        v, reasons = health.verdict(hb, now)
        names = set(cams.get(dev.id, ())) | (set(hb.cameras) if hb else set())
        stale = sum(1 for newest in hb.cameras.values() if health.camera_stale(newest, now)) if hb else 0
        out.append(DeviceSummary(
            device_id=dev.device_id, site=dev.site, customer_id=dev.customer_id, customer_name=cust_name,
            verdict=v, reasons=[HealthReason(**r) for r in reasons],
            last_seen_utc=hb.time_utc if hb else None, mode=hb.mode if hb else None, host=hb.host if hb else None,
            cameras_total=len(names), cameras_stale=stale,
            disk_free_gb=hb.disk_free_gb if hb else None,
            collector_running=hb.collector_running if hb else None, stopped=hb.stopped if hb else None,
            newest_clip_utc=hb.newest_clip_utc if hb else None,
            events_24h=events.get(dev.id, 0), alerts_24h=alerts.get(dev.id, 0),
            false_alarms_7d=false_alarms.get(dev.id, 0),
            needs_details=(dev.enrolled_by == "discovered" and name_source != "admin"),
            enrolled_by=dev.enrolled_by, app_version=dev.app_version))
    out.sort(key=lambda d: (_SEVERITY_RANK[d.verdict], d.customer_name, d.site))
    return out


@router.get("/fleet", response_model=FleetResponse, dependencies=[Depends(require_role("admin", "support"))])
def fleet(request: Request, session: Session = SessionDep):
    now = now_of(request)
    return FleetResponse(devices=build_summaries(session, now), generated_utc=now)


@router.get("/fleet/activity", response_model=DensityOut, dependencies=[Depends(require_role("admin", "support"))])
def fleet_activity(request: Request, hours: int = Query(24, ge=1, le=168), session: Session = SessionDep):
    # Hourly event, alert and false-alarm counts over the whole fleet; the last bucket is the current hour.
    return fleet_activity_density(session, now_of(request), hours)


@router.post("/devices/enroll", response_model=DeviceSummary)
def enroll(body: EnrollRequest, request: Request, staff: Staff = Depends(require_role("admin")),
           session: Session = SessionDep):
    check_length("site", body.site, SITE_MAX)
    check_length("ssh_user", body.ssh_user, SITE_MAX)
    check_length("tailscale_host", body.tailscale_host, TEXT_MAX)
    customer = session.get(Customer, require_id(body.customer_id, "Customer not found"))
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    if session.scalar(select(Device.id).where(Device.site == body.site)) is not None:
        raise HTTPException(status_code=409, detail="That site is already assigned to a device")
    dev = Device(device_id=str(uuid.uuid4()), site=body.site, tailscale_host=body.tailscale_host,
                 ssh_user=body.ssh_user, customer_id=customer.id, enrolled_at=now_of(request))
    session.add(dev)
    try:
        session.flush()
    except IntegrityError:  # lost a race on the unique site
        session.rollback()
        raise HTTPException(status_code=409, detail="That site is already assigned to a device")
    redact.remember(session, dev, now=now_of(request))  # the household's names, for labeler redaction
    audit.record(session, staff.id, "device_enroll", target=dev.site, customer_id=customer.id, device_id=dev.device_id,
                 ts=now_of(request))
    return build_summaries(session, now_of(request), device_pks=[dev.id])[0]
