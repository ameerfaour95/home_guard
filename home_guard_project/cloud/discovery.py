"""Auto-enrolment: boxes publish dataset_<site>/_status/registration.json (installer-recorded owner, consents,
host) next to heartbeat.json; this pass turns them into a Customer + Device so the setup program adds the household
to the admin panel by itself. Read-only toward S3 (HEAD / GET / list only).

Registrations are deduplicated by ETag in raw_revisions. An admin's choices win: a customer name an admin typed
(`name_source == "admin"`) is never overwritten, and consents change only when the registration carries a newer
`consent.recorded_utc` than the one applied before (the owner's latest answer wins). The owner phone is stored on
the customer (never returned by any route) and never written to the audit log.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from . import audit, redact
from .deps import NAME_MAX
from .models import Customer, Device, IndexProblem, RawRevision
from .s3 import ETagMismatch, S3

log = logging.getLogger(__name__)

POOLS = {"uca", "smarthome", "multi"}  # dataset_uca, dataset_smarthome, dataset_multi: not households
IGNORE_ENV = "HG_CLOUD_DISCOVERY_IGNORE"
SITE_MAX = 64
CONSENTS = (("live", "consent_live"), ("recordings", "consent_recordings"), ("training", "consent_training"))


def ignored_sites(extra: Optional[Iterable[str]] = None) -> set[str]:
    """Excluded pools plus the comma separated HG_CLOUD_DISCOVERY_IGNORE list ("site" or "dataset_site")."""
    raw = extra if extra is not None else os.environ.get(IGNORE_ENV, "").split(",")
    out = {s.strip().removeprefix("dataset_") for s in raw if s and s.strip()}
    return POOLS | out


def humanise(site: str) -> str:
    return " ".join(site.replace("_", " ").replace("-", " ").split()).title() or site


def _utc(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _text(value: Any, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


FUTURE_SLACK = timedelta(minutes=5)
ACTOR_SETUP, ACTOR_DISCOVERY = "system:setup", "system:discovery"
FUTURE_REASON = "registration time in the future"


def _problem(session, key: str, etag: str, reason: str, now: datetime) -> None:
    session.execute(pg_insert(IndexProblem).values(s3_key=key, reason=reason, seen_at=now, etag=etag)
                    .on_conflict_do_update(index_elements=[IndexProblem.s3_key],
                                           set_={"reason": reason, "seen_at": now, "etag": etag}))


def _redacted(body: dict) -> dict:
    """The stored copy of a registration: everything but the owner's phone (that lives on the customer only)."""
    return {k: v for k, v in body.items() if k != "owner_phone"}


def _fetch_registration(session, s3: S3, key: str, now: datetime) -> tuple[Optional[dict], bool]:
    """(registration body or None, is_new_revision). None when absent, unreadable, not a v1 registration or
    recorded in the future. A revision already in raw_revisions is read from there, never fetched again (the stored
    copy has no owner_phone)."""
    info = s3.head(key)
    if info is None:
        return None, False
    stored = session.scalar(select(RawRevision).where(RawRevision.s3_key == key, RawRevision.etag == info.etag))
    if stored is not None:
        body, fresh = stored.body, False
    else:
        try:
            body = json.loads(s3.get_text(key, if_match=info.etag))
        except ETagMismatch:
            return None, False  # replaced while we looked; the next pass reads the new one
        except ValueError:
            log.warning("registration %s is not JSON; skipped", key)
            return None, False
        fresh = True
    if not isinstance(body, dict) or body.get("schema_version") != 1:
        log.warning("registration %s has an unknown shape; skipped", key)
        return None, False
    if fresh:
        session.execute(pg_insert(RawRevision).values(s3_key=key, etag=info.etag, fetched_at=now, body=_redacted(body))
                        .on_conflict_do_nothing(index_elements=[RawRevision.s3_key, RawRevision.etag]))
    consent = body.get("consent")
    recorded = _utc(consent.get("recorded_utc")) if isinstance(consent, dict) else None
    if recorded is not None and recorded > now + FUTURE_SLACK:
        _problem(session, key, info.etag, FUTURE_REASON, now)
        return None, False
    return body, fresh


def _consents_of(reg: dict) -> Optional[tuple[dict, datetime]]:
    consent = reg.get("consent")
    if not isinstance(consent, dict):
        return None
    recorded = _utc(consent.get("recorded_utc"))
    if recorded is None:
        return None
    return {key: bool(consent.get(key)) for key, _ in CONSENTS}, recorded


def _proposal(values: dict, recorded: datetime, reg: dict) -> dict:
    return {**values, "recorded_utc": recorded.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "installer": _text(reg.get("installer"), 255)}


def _apply_consents(session, cust: Customer, dev: Device, reg: dict, now: datetime, first: bool = False) -> None:
    """Box consent is a proposal, never a grant. Owner revocations (true -> false) apply on their own except on an
    admin-enrolled device; grants, and every consent on first enrolment, wait for an admin to confirm."""
    parsed = _consents_of(reg)
    if parsed is None:
        return
    values, recorded = parsed
    prior = cust.consent_proposed.get("recorded_utc") if isinstance(cust.consent_proposed, dict) else None
    baseline = max([t for t in (cust.consent_recorded_utc, _utc(prior)) if t is not None], default=None)
    if baseline is not None and recorded <= baseline:
        return
    revoked, granted = [], []
    for key, field in CONSENTS:
        have = bool(getattr(cust, field))
        if have and not values[key]:
            revoked.append(field)
        elif values[key] and not have:
            granted.append(field)
    admin_owned = dev.enrolled_by == "admin"
    if not admin_owned and not first:
        for field in revoked:
            setattr(cust, field, False)
        if revoked:
            audit.record(session, None, "consent_revoked_by_owner", target=dev.site, customer_id=cust.id,
                         device_id=dev.device_id, detail={"fields": sorted(revoked)}, ts=now,
                         staff_name=ACTOR_DISCOVERY)
    if first or granted or (admin_owned and revoked):
        cust.consent_proposed = _proposal(values, recorded, reg)
        fields = sorted(f for _, f in CONSENTS) if first else sorted(granted + (revoked if admin_owned else []))
        audit.record(session, None, "consent_proposed", target=dev.site, customer_id=cust.id, device_id=dev.device_id,
                     detail={"fields": fields}, ts=now, staff_name=ACTOR_SETUP)
    elif cust.consent_proposed is not None:
        cust.consent_proposed = None  # the owner's newer answer asks for nothing beyond what is confirmed


def _enroll(session, site: str, reg: Optional[dict], now: datetime) -> str:
    owner = _text(reg.get("owner_name"), NAME_MAX) if reg else ""
    if owner:
        cust = Customer(name=owner, name_source="setup", owner_phone=_text(reg.get("owner_phone"), 64) or None)
        enrolled_by = "setup"
    else:
        cust = Customer(name=humanise(site)[:NAME_MAX], name_source="discovered")
        enrolled_by = "discovered"
    session.add(cust)
    session.flush()
    dev = Device(device_id=str(uuid.uuid4()), site=site, customer_id=cust.id, enrolled_at=now, enrolled_by=enrolled_by,
                 tailscale_host=_text(reg.get("tailscale_host"), 255) if reg else "",
                 app_version=(_text(reg.get("app_version"), 64) or None) if reg else None)
    session.add(dev)
    session.flush()
    extra = [("customer", owner)] if owner else []
    redact.remember(session, dev, extra=extra, now=now)  # the owner name is an identity term for labelers
    action = "auto_enroll" if enrolled_by == "setup" else "auto_discover"
    audit.record(session, None, action, target=site, customer_id=cust.id, device_id=dev.device_id, ts=now,
                 staff_name=ACTOR_SETUP if enrolled_by == "setup" else ACTOR_DISCOVERY)
    if reg:
        _apply_consents(session, cust, dev, reg, now, first=True)
    return action


def _update(session, dev: Device, reg: dict, now: datetime) -> list[str]:
    cust = session.get(Customer, dev.customer_id)
    done: list[str] = []
    host = _text(reg.get("tailscale_host"), 255)
    if host and host != dev.tailscale_host:
        dev.tailscale_host = host
        done.append("host")
    version = _text(reg.get("app_version"), 64)
    if version and version != dev.app_version:
        dev.app_version = version
        done.append("app_version")
    owner = _text(reg.get("owner_name"), NAME_MAX)
    phone = _text(reg.get("owner_phone"), 64)
    if phone and phone != cust.owner_phone:
        cust.owner_phone = phone
    if dev.enrolled_by == "discovered" and owner:  # the registration finally arrived
        dev.enrolled_by = "setup"
        if cust.name_source != "admin":
            cust.name, cust.name_source = owner, "setup"
        done.append("registered")
    state = lambda: (tuple(getattr(cust, f) for _, f in CONSENTS), cust.consent_proposed)  # noqa: E731
    before = state()
    _apply_consents(session, cust, dev, reg, now, first="registered" in done)
    session.flush()
    if owner or host:
        extra = [("customer", owner)] if owner else []
        redact.remember(session, dev, extra=extra, now=now)
    if state() != before:
        done.append("consents")
    return done


def discover(session, s3: S3, now: Optional[datetime] = None, ignore: Optional[Iterable[str]] = None) -> dict[str, str]:
    """One pass over the dataset_* folders; returns {site: what happened} for the sites that changed."""
    now = now or datetime.now(timezone.utc)
    skip = ignored_sites(ignore)
    out: dict[str, str] = {}
    for prefix in s3.list_dirs("dataset_"):
        site = prefix.rstrip("/").removeprefix("dataset_")
        if not site or len(site) > SITE_MAX or site in skip:
            continue
        try:
            with session.begin_nested():
                known = session.scalar(select(Device).where(Device.site == site))
                reg, fresh = _fetch_registration(session, s3, f"{prefix}_status/registration.json", now)
                if known is None:
                    if reg is None and s3.head(f"{prefix}_status/heartbeat.json") is None:
                        continue  # a folder of clips with no status object is not a box we can name
                    out[site] = _enroll(session, site, reg, now)
                elif reg is not None and fresh:
                    done = _update(session, known, reg, now)
                    if done:
                        out[site] = "updated: " + ", ".join(done)
        except Exception:  # noqa: BLE001 -- one bad site never stops the rest
            log.exception("discovery of %s failed", site)
    return out
