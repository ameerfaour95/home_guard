from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import audit, redact
from ..deps import NAME_MAX, NOTES_MAX, SessionDep, check_length, require_id, require_role
from ..models import Customer, Device, Staff
from ..schemas import ConsentConfirm, CustomerIn, CustomerOut
from .fleet import build_summaries, now_of

router = APIRouter(tags=["customers"])

_FIELDS = tuple(CustomerIn.model_fields)
_NOT_FOUND = "Customer not found"
_CONSENTS = ("consent_live", "consent_recordings", "consent_training")
TIMEZONE_MAX = 64  # the column's width
# Consent comes from the customer, through the box's setup: an admin settles what the box recorded and may withdraw
# consent, but never grants what the customer did not give.
NO_PROPOSAL = "No consent was recorded at this customer's setup: consent can only come from the customer"
GRANT_REFUSED = ("Consent comes from the customer: confirm the consent this customer gave at setup instead of "
                 "granting it here")
STALE_PROPOSAL = "The box recorded a newer answer from this customer: reload and check it again"


def _out(session: Session, request: Request, c: Customer) -> CustomerOut:
    return CustomerOut(id=c.id, consent_proposed=c.consent_proposed, **{f: getattr(c, f) for f in _FIELDS},
                       devices=build_summaries(session, now_of(request), customer_id=c.id))


def _get(session: Session, customer_id: int) -> Customer:
    c = session.get(Customer, require_id(customer_id, _NOT_FOUND))
    if c is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return c


def _check(body: CustomerIn) -> None:
    check_length("name", body.name, NAME_MAX)
    check_length("notes", body.notes, NOTES_MAX)
    check_length("timezone", body.timezone, TIMEZONE_MAX)


@router.get("/customers", response_model=list[CustomerOut], dependencies=[Depends(require_role("admin", "support"))])
def list_customers(request: Request, session: Session = SessionDep):
    return [_out(session, request, c) for c in session.scalars(select(Customer).order_by(Customer.name, Customer.id))]


@router.post("/customers", response_model=CustomerOut)
def create_customer(body: CustomerIn, request: Request, staff: Staff = Depends(require_role("admin")),
                    session: Session = SessionDep):
    _check(body)
    if any(getattr(body, f) for f in _CONSENTS):  # a new customer has given nothing yet
        raise HTTPException(status_code=422, detail=GRANT_REFUSED)
    c = Customer(**body.model_dump())
    session.add(c)
    session.flush()
    audit.record(session, staff.id, "customer_create", target=c.name, customer_id=c.id, ts=now_of(request))
    return _out(session, request, c)


@router.get("/customers/{customer_id}", response_model=CustomerOut,
            dependencies=[Depends(require_role("admin", "support"))])
def get_customer(customer_id: int, request: Request, session: Session = SessionDep):
    return _out(session, request, _get(session, customer_id))


@router.patch("/customers/{customer_id}", response_model=CustomerOut)
def update_customer(customer_id: int, body: CustomerIn, request: Request,
                    staff: Staff = Depends(require_role("admin")), session: Session = SessionDep):
    # Only the fields present in the request change (a PATCH of {"name": ...} keeps consents, timezone and
    # notes). The audit row names the changed fields only, never their values; a no-op writes no row.
    _check(body)
    c = _get(session, customer_id)
    proposed = c.consent_proposed if isinstance(c.consent_proposed, dict) else {}
    for f in sorted(body.model_fields_set & set(_CONSENTS)):
        if getattr(body, f) and not getattr(c, f) and not proposed.get(f.removeprefix("consent_")):
            raise HTTPException(status_code=422, detail=GRANT_REFUSED)
    old_name = c.name
    changed = []
    for f in sorted(body.model_fields_set & set(_FIELDS)):
        new = getattr(body, f)
        if getattr(c, f) != new:
            changed.append(f)
            setattr(c, f, new)
    if "name" in changed:
        c.name_source = "admin"  # a person named it: discovery never renames it again
    set_consents = sorted(set(body.model_fields_set) & set(_CONSENTS))
    confirmed = bool(set_consents) and c.consent_proposed is not None
    if set_consents:  # an admin decision on consents: the box's proposal is settled, later answers must be newer
        c.consent_proposed = None
        c.consent_recorded_utc = now_of(request)
    session.flush()
    if "name" in changed:
        # the old name stays an identity term of every device of this customer (labeler redaction), and the
        # labelers' stored search text is recomputed with both names
        for device in session.scalars(select(Device).where(Device.customer_id == c.id).order_by(Device.id)).all():
            redact.remember(session, device, extra=[("customer", old_name), ("customer", c.name)],
                            now=now_of(request))
            redact.backfill(session, device, everything=True)
    if confirmed:
        audit.record(session, staff.id, "consent_confirmed", target=c.name, customer_id=c.id,
                     detail={"fields": set_consents}, ts=now_of(request))
    if changed:
        audit.record(session, staff.id, "customer_update", target=c.name, customer_id=c.id,
                     detail={"changed": changed}, ts=now_of(request))
    return _out(session, request, c)


@router.post("/customers/{customer_id}/consent/confirm", response_model=CustomerOut)
def confirm_consent(customer_id: int, body: ConsentConfirm, request: Request,
                    staff: Staff = Depends(require_role("admin")), session: Session = SessionDep):
    # Settles the consent the customer gave at setup: exactly what the box recorded, nothing more.
    c = _get(session, customer_id)
    proposed = c.consent_proposed if isinstance(c.consent_proposed, dict) else None
    if proposed is None:
        raise HTTPException(status_code=409, detail=NO_PROPOSAL)
    from ..discovery import _utc

    recorded = _utc(proposed.get("recorded_utc"))
    if recorded is None or recorded != _utc(body.recorded_utc.isoformat()):
        raise HTTPException(status_code=409, detail=STALE_PROPOSAL)
    changed = []
    for f in _CONSENTS:
        value = bool(proposed.get(f.removeprefix("consent_")))
        if getattr(c, f) != value:
            changed.append(f)
            setattr(c, f, value)
    c.consent_proposed = None
    c.consent_recorded_utc = recorded  # the customer's own answer is the newest one applied
    audit.record(session, staff.id, "consent_confirmed", target=c.name, customer_id=c.id,
                 detail={"fields": changed, "proposal": {k: proposed.get(k) for k in
                                                         ("live", "recordings", "training", "recorded_utc", "installer")}},
                 ts=now_of(request))
    session.flush()
    return _out(session, request, c)
