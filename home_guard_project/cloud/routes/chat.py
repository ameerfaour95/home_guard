"""The owner's Telegram conversation with the box, read-only, for the customer page's Chat tab.

The box uploads it as ``production_<site>/chat/<YYYY-MM-DD>.jsonl`` (one ChatFeed line per message: ts, who, name,
kind, text, camera, alert_id, image, delivered, error; 14-day retention) with the alert pictures under
``production_<site>/chat/images/``. It is read from S3 on demand (a day is small), never indexed or stored here.

The conversation is customer content: like a recording it needs the customer's recordings consent, every view is
audited (``chat_view``; a picture is a ``media_view``) and the owner is told staff looked (an owner notice of kind
"chat"). Camera ids in the lines are shown by the owner's names (the heartbeat's camera_list), never raw.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from home_guard_project.fleet_contract import health
from home_guard_project.fleet_contract.camera_names import replace_ids

from .. import audit, notice_delivery
from ..access import media_refusal
from ..deps import TEXT_MAX, SessionDep, check_length, require_id, require_role
from ..models import Camera, Customer, Device, Staff
from ..schemas import ChatDay, ChatImageRequest, ChatLine, MediaAccess
from .media import URL_TTL_SECONDS, _deny, _s3

router = APIRouter(tags=["chat"])

_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_IMAGE = re.compile(r"^[A-Za-z0-9_-]{1,160}\.jpg$")  # ChatFeed names pictures <alert id, unsafe chars as _>.jpg
_CAMERA_IN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*?_ch\d+(?=_\d|\b|_)")  # a box camera id, e.g. ameer_tes2_ch6
_FILE = re.compile(r"^[A-Za-z0-9_.-]{1,200}\.(?:jpg|jpeg|png|mp4)$")
SEARCH_DAYS = 14  # the box keeps 14 days; a search reads at most that many day files per box
_NOT_FOUND = "Customer not found"


def _customer(session: Session, customer_id: int) -> Customer:
    customer = session.get(Customer, require_id(customer_id, _NOT_FOUND))
    if customer is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return customer


def _check_consent(session: Session, staff: Staff, customer: Customer, target: str) -> None:
    refusal = media_refusal(staff.role, "review", customer.consent_recordings, customer.consent_training)
    if refusal is not None:
        _deny(session, staff, target, customer, None, "review", refusal)


def _names(session: Session, dev: Device) -> dict[str, str]:
    """Camera id -> the owner's name: the box's camera_list, else a Camera.display_name."""
    names = {cam: shown for cam, shown in session.execute(
        select(Camera.name, Camera.display_name).where(Camera.device_pk == dev.id)).all() if shown}
    names.update({c["id"]: c["name"] for c in health.camera_list(dev.last_heartbeat) or () if c["name"]})
    return names


def _days(s3, site: str) -> list[str]:
    prefix = f"production_{site}/chat/"
    out = []
    for obj in s3.list(prefix):
        name = obj.key[len(prefix):]
        if name.endswith(".jsonl") and _DAY.match(name[:-6]):
            out.append(name[:-6])
    return out


def _camera_of(entry: dict, text: str) -> str:
    """The camera a line is about: its own field, else the alert id's camera, else a sent file's name
    (``<camera>_<epoch>_<...>.jpg|mp4``, what the assistant's photo and video lines carry)."""
    for value in (entry.get("camera"), entry.get("alert_id"), text):
        if isinstance(value, str) and value:
            found = _CAMERA_IN.search(value)
            if found:
                return found.group(0)
    return ""


def _lines(s3, dev: Device, day: str, names: dict[str, str]) -> list[ChatLine]:
    key = f"production_{dev.site}/chat/{day}.jsonl"
    if s3.head(key) is None:
        return []
    aliases = {cam: [name] for cam, name in names.items() if name}
    out = []
    for raw in s3.get_text(key).splitlines():
        try:
            entry = json.loads(raw)
            ts = datetime.fromtimestamp(float(entry["ts"]), timezone.utc)
        except (ValueError, TypeError, KeyError, OverflowError, OSError):
            continue  # a damaged line is skipped, the rest of the day still shows
        if not isinstance(entry, dict):
            continue

        def text(field):
            value = entry.get(field)
            return value if isinstance(value, str) else ""
        kind, body = text("kind"), text("text")
        camera = _camera_of(entry, body)
        if kind in ("photo", "video") and _FILE.match(body):
            body = ""  # the assistant's photo / video line carries only a file name (with the camera id)
        # every camera id in the text (old site ids too: ameer_tes2_ch6) becomes the owner's name by its channel
        ids = set(_CAMERA_IN.findall(body)) | ({camera} if camera else set())
        image = text("image").rsplit("/", 1)[-1]
        out.append(ChatLine(ts=ts, site=dev.site, who=text("who") or "unknown", name=text("name"), kind=kind,
                            text=replace_ids(body, sorted(ids), "en", aliases), camera=camera,
                            camera_name=health.camera_label(camera, names) if camera else "",
                            alert_id=text("alert_id"), image=image if _IMAGE.match(image) else "",
                            delivered=entry.get("delivered") is not False, error=text("error")))
    return out


@router.get("/customers/{customer_id}/chat", response_model=ChatDay)
def chat(customer_id: int, request: Request, background_tasks: BackgroundTasks, day: Optional[str] = None,
         q: Optional[str] = None,
         staff: Staff = Depends(require_role("admin", "support")), session: Session = SessionDep):
    # One day of the conversation (default: the newest day uploaded), or with `q` the matching lines of every
    # uploaded day (newest first, up to 14 per box). `days` lists the uploaded days, newest first.
    check_length("q", q, TEXT_MAX)
    if day is not None and not _DAY.match(day):
        raise HTTPException(status_code=400, detail="day must be YYYY-MM-DD")
    customer = _customer(session, customer_id)
    _check_consent(session, staff, customer, f"chat/customer/{customer.id}")
    s3 = _s3(request)
    devices = list(session.scalars(select(Device).where(Device.customer_id == customer.id).order_by(Device.site)))
    days_of = {dev.id: sorted(_days(s3, dev.site), reverse=True) for dev in devices}
    days = sorted({d for ds in days_of.values() for d in ds}, reverse=True)
    needle = (q or "").strip().casefold()
    shown_day = None if needle else (day or (days[0] if days else None))
    messages: list[ChatLine] = []
    for dev in devices:
        names = _names(session, dev)
        for d in (days_of[dev.id][:SEARCH_DAYS] if needle else [shown_day] if shown_day in days_of[dev.id] else []):
            for line in _lines(s3, dev, d, names):
                if not needle or needle in " ".join((line.text, line.name, line.camera_name, line.error)).casefold():
                    messages.append(line)
    messages.sort(key=lambda m: m.ts, reverse=bool(needle))
    now: datetime = request.app.state.clock()
    # the search text is not audited (it may hold names); what was opened is
    audit.record(session, staff.id, "chat_view", target=f"chat/customer/{customer.id}", reason="review",
                 customer_id=customer.id, detail={"day": shown_day, "search": bool(needle), "messages": len(messages)},
                 ts=now)
    notices = [audit.owner_notice(session, s3, dev, staff, kind="chat", cameras=[], now=now)
               for dev in devices if days_of[dev.id]]
    notice_delivery.schedule(request, background_tasks, notices)  # pushed to the boxes after the commit
    return ChatDay(day=shown_day, days=days, messages=messages)


@router.post("/customers/{customer_id}/chat/images/access", response_model=MediaAccess)
def chat_image(customer_id: int, body: ChatImageRequest, request: Request, background_tasks: BackgroundTasks,
               staff: Staff = Depends(require_role("admin", "support")), session: Session = SessionDep):
    # A presigned URL for a picture an alert sent in the chat (`image` as a chat line names it).
    customer = _customer(session, customer_id)
    dev = session.scalar(select(Device).where(Device.customer_id == customer.id, Device.site == body.site))
    if dev is None or not _IMAGE.match(body.image):
        raise HTTPException(status_code=404, detail="Picture not found")
    key = f"production_{dev.site}/chat/images/{body.image}"
    _check_consent(session, staff, customer, key)
    s3 = _s3(request)
    if s3.head(key) is None:
        raise HTTPException(status_code=404, detail="Picture not found")
    now: datetime = request.app.state.clock()
    audit.record(session, staff.id, "media_view", target=key, reason="review", customer_id=customer.id,
                 device_id=dev.device_id, detail={"role": "chat_image", "camera": None}, ts=now)
    url = s3.presign(key, URL_TTL_SECONDS)
    notice = audit.owner_notice(session, s3, dev, staff, kind="chat", cameras=[], now=now)
    notice_delivery.schedule(request, background_tasks, [notice])
    return MediaAccess(url=url, expires_utc=now + timedelta(seconds=URL_TTL_SECONDS), mime="image/jpeg")
