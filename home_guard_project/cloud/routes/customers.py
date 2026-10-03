from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import audit, redact
from ..deps import NAME_MAX, NOTES_MAX, SessionDep, check_length, require_id, require_role
from ..models import Customer, Device, Staff
from ..schemas import CustomerIn, CustomerOut
from .fleet import build_summaries, now_of

router = APIRouter(tags=["customers"])

_FIELDS = tuple(CustomerIn.model_fields)
_NOT_FOUND = "Customer not found"
TIMEZONE_MAX = 64  # the column's width


def _out(session: Session, request: Request, c: Customer) -> CustomerOut:
    return CustomerOut(id=c.id, **{f: getattr(c, f) for f in _FIELDS},
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
    old_name = c.name
    changed = []
    for f in sorted(body.model_fields_set & set(_FIELDS)):
        new = getattr(body, f)
        if getattr(c, f) != new:
            changed.append(f)
            setattr(c, f, new)
    session.flush()
    if "name" in changed:
        # the old name stays an identity term of every device of this customer (labeler redaction), and the
        # labelers' stored search text is recomputed with both names
        for device in session.scalars(select(Device).where(Device.customer_id == c.id).order_by(Device.id)).all():
            redact.remember(session, device, extra=[("customer", old_name), ("customer", c.name)],
                            now=now_of(request))
            redact.backfill(session, device, everything=True)
    if changed:
        audit.record(session, staff.id, "customer_update", target=c.name, customer_id=c.id,
                     detail={"changed": changed}, ts=now_of(request))
    return _out(session, request, c)
