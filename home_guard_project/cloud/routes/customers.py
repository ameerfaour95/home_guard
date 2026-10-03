from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import audit
from ..deps import get_session, require_role
from ..models import Customer, Staff
from ..schemas import CustomerIn, CustomerOut
from .fleet import build_summaries, now_of

router = APIRouter(tags=["customers"])

_FIELDS = tuple(CustomerIn.model_fields)


def _out(session: Session, request: Request, c: Customer) -> CustomerOut:
    return CustomerOut(id=c.id, **{f: getattr(c, f) for f in _FIELDS},
                       devices=build_summaries(session, now_of(request), customer_id=c.id))


def _get(session: Session, customer_id: int) -> Customer:
    c = session.get(Customer, customer_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return c


@router.get("/customers", response_model=list[CustomerOut], dependencies=[Depends(require_role("admin", "support"))])
def list_customers(request: Request, session: Session = Depends(get_session)):
    return [_out(session, request, c) for c in session.scalars(select(Customer).order_by(Customer.name, Customer.id))]


@router.post("/customers", response_model=CustomerOut)
def create_customer(body: CustomerIn, request: Request, staff: Staff = Depends(require_role("admin")),
                    session: Session = Depends(get_session)):
    c = Customer(**body.model_dump())
    session.add(c)
    session.flush()
    audit.record(session, staff.id, "customer_create", target=c.name, customer_id=c.id)
    return _out(session, request, c)


@router.get("/customers/{customer_id}", response_model=CustomerOut,
            dependencies=[Depends(require_role("admin", "support"))])
def get_customer(customer_id: int, request: Request, session: Session = Depends(get_session)):
    return _out(session, request, _get(session, customer_id))


@router.patch("/customers/{customer_id}", response_model=CustomerOut)
def update_customer(customer_id: int, body: CustomerIn, request: Request,
                    staff: Staff = Depends(require_role("admin")), session: Session = Depends(get_session)):
    c = _get(session, customer_id)
    changed = {}
    for f, new in body.model_dump().items():
        old = getattr(c, f)
        if old != new:
            changed[f] = ["<redacted>", "<redacted>"] if f == "notes" else [old, new]  # never log notes text
            setattr(c, f, new)
    session.flush()
    if changed:
        audit.record(session, staff.id, "customer_update", target=c.name, customer_id=c.id, detail={"changed": changed})
    return _out(session, request, c)
