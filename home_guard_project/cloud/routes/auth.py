from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from .. import audit, auth
from ..deps import SessionDep, current_staff
from ..models import AuditLog, Staff
from ..schemas import LoginRequest, RefreshRequest, StaffOut, TokenPair

router = APIRouter(tags=["auth"])


def _recent_failures(session: Session, email: str) -> int:
    since = datetime.now(timezone.utc) - auth.LOCKOUT_WINDOW
    return session.scalar(select(func.count()).select_from(AuditLog).where(
        AuditLog.action == "login_failed", AuditLog.ts >= since,
        AuditLog.detail["email"].as_string() == email)) or 0


@router.post("/auth/login", response_model=TokenPair)
def login(body: LoginRequest, request: Request, session: Session = SessionDep):
    # scope="function": the session commits before the response is sent
    settings = request.app.state.settings
    email = body.email.strip().lower()
    # serialise attempts per email so concurrent guesses cannot all pass the failure count
    session.execute(text("SELECT pg_advisory_xact_lock(hashtext(:email))"), {"email": email})
    staff = session.scalars(select(Staff).where(func.lower(Staff.email) == email).with_for_update()).first()
    staff_id = staff.id if staff else None
    if _recent_failures(session, email) >= auth.MAX_FAILURES:
        audit.record(session, staff_id, "login_locked", detail={"email": email})
        session.commit()
        raise HTTPException(status_code=429, detail="Too many attempts. Try again in 15 minutes.")
    counter = None
    if len(body.password) > auth.MAX_PASSWORD_LEN or len(body.totp) > auth.MAX_TOTP_LEN:
        ok = False  # absurd input: same answer, no hashing
    elif staff is None:
        auth.verify_dummy(body.password)
        ok = False
    else:
        # evaluate both factors so timing does not reveal which one was wrong
        pw_ok = auth.verify_password(staff.password_hash, body.password)
        counter = auth.totp_counter(staff.totp_secret, body.totp)
        fresh = counter is not None and (staff.totp_last_counter is None or counter > staff.totp_last_counter)
        ok = pw_ok and fresh and not staff.disabled
    if not ok:
        audit.record(session, staff_id, "login_failed", detail={"email": email})
        session.commit()
        raise HTTPException(status_code=401, detail=auth.BAD_LOGIN)
    staff.totp_last_counter = counter  # a code works once
    pair = auth.issue_tokens(staff, settings, session)
    audit.record(session, staff.id, "login", detail={"email": email})
    session.commit()
    return pair


@router.post("/auth/local", response_model=TokenPair)
def local_signin(request: Request, session: Session = SessionDep):
    """One-click sign-in for the founder's laptop: only with HG_CLOUD_LOCAL_TRUST=1 and a loopback client."""
    settings = request.app.state.settings
    host = request.client.host if request.client else None
    staff = None
    if settings.local_trust and host in ("127.0.0.1", "::1") and settings.local_admin_email:
        staff = session.scalars(select(Staff).where(
            func.lower(Staff.email) == settings.local_admin_email.strip().lower())).first()
    if staff is None or staff.disabled:
        raise HTTPException(status_code=404, detail="Not Found")
    pair = auth.issue_tokens(staff, settings, session)
    audit.record(session, staff.id, "local_signin", detail={"email": staff.email})
    session.commit()
    return pair


@router.post("/auth/refresh", response_model=TokenPair)
def refresh(body: RefreshRequest, request: Request, session: Session = SessionDep):
    pair = auth.rotate(session, body.refresh_token, request.app.state.settings)
    if pair is None:
        session.commit()
        raise HTTPException(status_code=401, detail="Sign in again")
    session.commit()
    return pair


@router.get("/me", response_model=StaffOut)
def me(staff: Staff = Depends(current_staff)):
    return staff
