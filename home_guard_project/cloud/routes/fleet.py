from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from home_guard_project.fleet_contract import health
from home_guard_project.fleet_contract.legacy import parse_heartbeat

from .. import audit, redact
from ..boxes import box_identity, box_lineage  # noqa: F401  (box_identity: imported from here by tests)
from ..deps import TEXT_MAX, SessionDep, check_length, require_id, require_role
from ..models import Camera, Customer, Device, Event, Feedback, Staff
from ..schemas import CameraOut, DensityOut, DeviceSummary, EnrollRequest, FleetResponse, HealthReason
from .events import fleet_activity_density

router = APIRouter(tags=["fleet"])

SITE_MAX = 64  # the column's width (sites and ssh users)
_SEVERITY_RANK = {"offline": 0, "critical": 1, "warning": 2, "unknown": 3, "healthy": 4}


def now_of(request: Request) -> datetime:
    return request.app.state.clock()


@dataclass
class KnownCamera:
    camera: str
    owner_name: str                     # the family's name when the box sent one, else ""
    current: bool
    in_heartbeat: bool
    heartbeat_newest: Optional[datetime]
    last_event_utc: Optional[datetime]
    enabled: bool = True                # false: in the box's camera_list but switched off (never warned about)

    @property
    def newest_clip_utc(self) -> Optional[datetime]:
        times = [t for t in (self.heartbeat_newest, self.last_event_utc) if t is not None]
        return max(times) if times else None


def camera_inventory(session: Session, now: datetime, devices: list[Device]) -> dict[int, list[KnownCamera]]:
    """Every camera ever known per device pk (heartbeat, Camera rows, events), split into current and retired by
    fleet_contract.health.split_cameras: the box's camera_list when its heartbeat carries one (switched-on cameras are
    current), else a clip in the last 48 h. Owner names come from camera_list or Camera.display_name."""
    pks = [d.id for d in devices]
    out: dict[int, list[KnownCamera]] = {pk: [] for pk in pks}
    if not pks:
        return out
    names: dict[tuple[int, str], str] = {}
    known: set[tuple[int, str]] = set()
    for pk, cam, shown in session.execute(select(Camera.device_pk, Camera.name, Camera.display_name)
                                          .where(Camera.device_pk.in_(pks))).all():
        known.add((pk, cam))
        if shown:
            names[(pk, cam)] = shown
    last_event = {(pk, cam): ts for pk, cam, ts in session.execute(
        select(Event.device_pk, Event.camera, func.max(Event.start_ts)).where(Event.device_pk.in_(pks))
        .group_by(Event.device_pk, Event.camera)).all()}
    known |= set(last_event)
    since48 = (now - health.CURRENT_WINDOW).timestamp()
    for dev in devices:
        body = dev.last_heartbeat if isinstance(dev.last_heartbeat, dict) else None
        hb = parse_heartbeat(body) if body is not None else None
        newest = dict(hb.cameras) if hb else {}
        configured = health.camera_list(body)
        listed = health.listed_cameras(body)  # switched-on cameras; the box's list decides when it sends one
        switched_off = {c["id"] for c in configured or () if not c["enabled"]}
        for cam in configured or ():
            if cam["name"]:
                names[(dev.id, cam["id"])] = cam["name"]
        ids = {cam for pk, cam in known if pk == dev.id} | set(newest) | {c["id"] for c in configured or ()}
        events = {cam: datetime.fromtimestamp(ts, timezone.utc) for (pk, cam), ts in last_event.items()
                  if pk == dev.id and ts is not None}
        recent = [cam for (pk, cam), ts in last_event.items() if pk == dev.id and ts is not None and ts >= since48]
        current, _ = health.split_cameras({cam: newest.get(cam) for cam in ids}, now,
                                          site=(hb.site if hb and hb.site else dev.site), recent=recent,
                                          listed=list(listed) if listed is not None else None)
        current = set(current)
        reported = set(newest) | {c["id"] for c in configured or ()}  # the box judges these (no clip yet: None)
        own = {cam: name for (pk, cam), name in names.items() if pk == dev.id}  # this house only
        out[dev.id] = sorted((KnownCamera(cam, health.owner_name(cam, own), cam in current, cam in reported,
                                          newest.get(cam), events.get(cam), cam not in switched_off) for cam in ids),
                             key=lambda c: (not c.current, c.camera))
    return out


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
    inventory = camera_inventory(session, now, [r[0] for r in rows])
    replaced, old_sites = box_lineage(session)

    out: list[DeviceSummary] = []
    for dev, cust_name, name_source in rows:
        hb = parse_heartbeat(dev.last_heartbeat) if isinstance(dev.last_heartbeat, dict) else None
        cams = inventory[dev.id]
        if hb:
            # warnings only over the cameras the house has now; renamed or removed ids are retired, never warned about
            judged = {c.camera: c.heartbeat_newest for c in cams if c.in_heartbeat}
            current = [c.camera for c in cams if c.current and c.in_heartbeat]
            v, reasons = health.verdict(replace(hb, cameras=judged), now, cameras=current,
                                        names={c.camera: c.owner_name for c in cams if c.owner_name})
        else:
            v, reasons = health.verdict(hb, now)
        current_cams = [c for c in cams if c.current]
        stale = sum(1 for c in current_cams if c.in_heartbeat and health.camera_stale(c.heartbeat_newest, now))
        out.append(DeviceSummary(
            device_id=dev.device_id, site=dev.site, customer_id=dev.customer_id, customer_name=cust_name,
            verdict=v, reasons=[HealthReason(**r) for r in reasons],
            last_seen_utc=hb.time_utc if hb else None, mode=hb.mode if hb else None, host=hb.host if hb else None,
            cameras_total=len(current_cams), cameras_stale=stale,
            disk_free_gb=hb.disk_free_gb if hb else None,
            collector_running=hb.collector_running if hb else None, stopped=hb.stopped if hb else None,
            newest_clip_utc=hb.newest_clip_utc if hb else None,
            events_24h=events.get(dev.id, 0), alerts_24h=alerts.get(dev.id, 0),
            false_alarms_7d=false_alarms.get(dev.id, 0),
            needs_details=(dev.enrolled_by == "discovered" and name_source != "admin"),
            enrolled_by=dev.enrolled_by, app_version=dev.app_version,
            replaced_by=replaced.get(dev.id, (None, None))[0], replaced_by_site=replaced.get(dev.id, (None, None))[1],
            old_sites=old_sites.get(dev.id, [])))
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


@router.get("/cameras", response_model=list[CameraOut], dependencies=[Depends(require_role("admin", "support"))])
def cameras(request: Request, customer_id: Optional[int] = None, session: Session = SessionDep):
    # Every known camera of the houses in scope (one customer, or all), current ones first, with the name staff read.
    q = select(Device).order_by(Device.site)
    if customer_id is not None:
        if session.get(Customer, require_id(customer_id, "Customer not found")) is None:
            raise HTTPException(status_code=404, detail="Customer not found")
        q = q.where(Device.customer_id == customer_id)
    devices = list(session.scalars(q))
    inventory = camera_inventory(session, now_of(request), devices)
    return [CameraOut(customer_id=dev.customer_id, device_id=dev.device_id, site=dev.site, camera=c.camera,
                      name=health.camera_label(c.camera, {c.camera: c.owner_name} if c.owner_name else None),
                      owner_named=bool(c.owner_name), current=c.current, newest_clip_utc=c.newest_clip_utc,
                      enabled=c.enabled)
            for dev in devices for c in inventory[dev.id]]


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
