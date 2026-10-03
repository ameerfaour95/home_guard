from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import audit, auth
from ..deps import current_staff, get_session
from ..models import AuditLog, Staff
from ..schemas import LoginRequest, RefreshRequest, StaffOut, TokenPair

router = APIRouter(tags=["auth"])


def _recent_failures(session: Session, email: str) -> int:
    since = datetime.now(timezone.utc) - auth.LOCKOUT_WINDOW
    return session.scalar(select(func.count()).select_from(AuditLog).where(
        AuditLog.action == "login_failed", AuditLog.ts >= since,
        AuditLog.detail["email"].as_string() == email)) or 0


@router.post("/auth/login", response_model=TokenPair)
def login(body: LoginRequest, request: Request, session: Session = Depends(get_session)):
    settings = request.app.state.settings
    email = body.email.strip().lower()
    if _recent_failures(session, email) >= auth.MAX_FAILURES:
        audit.record(session, None, "login_locked", detail={"email": email})
        session.commit()
        raise HTTPException(status_code=429, detail="Too many attempts. Try again in 15 minutes.")
    staff = session.scalars(select(Staff).where(func.lower(Staff.email) == email)).first()
    if staff is None:
        auth.verify_dummy(body.password)
        ok = False
    else:
        # evaluate both factors so timing does not reveal which one was wrong
        pw_ok = auth.verify_password(staff.password_hash, body.password)
        code_ok = auth.verify_totp(staff.totp_secret, body.totp)
        ok = pw_ok and code_ok and not staff.disabled
    if not ok:
        audit.record(session, staff.id if staff else None, "login_failed", detail={"email": email})
        session.commit()
        raise HTTPException(status_code=401, detail=auth.BAD_LOGIN)
    pair = auth.issue_tokens(staff, settings, session)
    audit.record(session, staff.id, "login", detail={"email": email})
    return pair


@router.post("/auth/refresh", response_model=TokenPair)
def refresh(body: RefreshRequest, request: Request, session: Session = Depends(get_session)):
    pair = auth.rotate(session, body.refresh_token, request.app.state.settings)
    if pair is None:
        raise HTTPException(status_code=401, detail="Sign in again")
    return pair


@router.get("/me", response_model=StaffOut)
def me(staff: Staff = Depends(current_staff)):
    return staff
