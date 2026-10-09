"""One physical box, many site rows: which device rows are one box, and which row is the box now."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from home_guard_project.fleet_contract.legacy import parse_heartbeat

from .models import Device, RawRevision


def _host(value) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def box_identity(heartbeat: Optional[dict], registration: Optional[dict]) -> tuple[str, list[str]]:
    """(box_id, hosts) of one device row. ``box_id`` is the box's own id (32 hex, heartbeat ``box_id``, else the
    registration's); ``hosts`` are its host names in the order the reviewer set: the registration's tailscale_host,
    its box_host, then the heartbeat's host."""
    hb = heartbeat if isinstance(heartbeat, dict) else {}
    reg = registration if isinstance(registration, dict) else {}
    box_id = _host(hb.get("box_id")) or _host(reg.get("box_id"))
    hosts = [h for h in (_host(reg.get("tailscale_host")), _host(reg.get("box_host")), _host(hb.get("host"))) if h]
    return box_id, list(dict.fromkeys(hosts))


def box_lineage(session: Session) -> tuple[dict[int, tuple[str, str]], dict[int, list[str]]]:
    """One physical box, many site names: a box renamed its site (ameer_tes2 -> ameer_week_0_1) or was re-registered,
    and each site became its own device row. Which rows are one box, in order: the same ``box_id``; else the same
    registration tailscale_host, else registration box_host, else heartbeat host (box_identity). A different box_id
    always means a different box, even on the same host. A row from before box_id existed (an old site's last
    heartbeat) joins the box_id box that has its host, when exactly one does. The row heard from last is the box,
    the others are its old site names.

    Returns ({old device pk: (current device_id, current site)}, {current device pk: [old sites, newest first]})."""
    devices = list(session.scalars(select(Device)))
    keys = {f"dataset_{d.site}/_status/registration.json": d.id for d in devices}
    registrations: dict[int, dict] = {}
    for key, body in session.execute(select(RawRevision.s3_key, RawRevision.body)
                                     .where(RawRevision.s3_key.in_(list(keys))).order_by(RawRevision.id)).all():
        registrations[keys[key]] = body  # the newest revision wins
    oldest = datetime.min.replace(tzinfo=timezone.utc)
    rows = []
    for dev in devices:
        hb = parse_heartbeat(dev.last_heartbeat) if isinstance(dev.last_heartbeat, dict) else None
        reg = dict(registrations.get(dev.id) or {})
        reg.setdefault("tailscale_host", dev.tailscale_host)  # discovery copies it from the registration
        box_id, hosts = box_identity(dev.last_heartbeat, reg)
        rows.append((hb.time_utc if hb and hb.time_utc else oldest, dev, box_id, hosts))
    boxes: dict[tuple[str, str], list] = {}
    for row in rows:
        if row[2]:
            boxes.setdefault(("box_id", row[2]), []).append(row)
    for row in rows:
        if row[2] or not row[3]:
            continue
        owners = [key for key, members in boxes.items() if key[0] == "box_id"
                  and any(set(row[3]) & set(m[3]) for m in members)]
        key = owners[0] if len(owners) == 1 else ("host", row[3][0])
        boxes.setdefault(key, []).append(row)
    replaced: dict[int, tuple[str, str]] = {}
    old_sites: dict[int, list[str]] = {}
    for members in boxes.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda m: (m[0], m[1].id), reverse=True)
        current = members[0][1]
        for _, dev, _, _ in members[1:]:
            replaced[dev.id] = (current.device_id, current.site)
        old_sites[current.id] = [dev.site for _, dev, _, _ in members[1:]]
    return replaced, old_sites


def current_site_for(session: Session, device: Device) -> str:
    """The site the box of *device* is heard from now (box_lineage): an old site row names its live site, a current
    or ungrouped row its own."""
    replaced, _ = box_lineage(session)
    return replaced.get(device.id, (None, device.site))[1]
