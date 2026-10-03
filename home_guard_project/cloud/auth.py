"""Passwords (argon2id), TOTP, JWT access tokens and rotating refresh tokens."""
from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
import pyotp
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .models import RefreshToken, Staff
from .schemas import StaffOut, TokenPair
from .settings import Settings

_hasher = PasswordHasher()  # argon2id by default
BAD_LOGIN = "Email, password or code is wrong"
MAX_FAILURES = 5
LOCKOUT_WINDOW = timedelta(minutes=15)
_DUMMY_HASH = _hasher.hash("not-a-real-password")


def hash_password(pw: str) -> str:
    return _hasher.hash(pw)


def verify_password(hashed: str, pw: str) -> bool:
    try:
        return _hasher.verify(hashed, pw)
    except (VerificationError, InvalidHashError):
        return False


def verify_dummy(pw: str) -> None:
    """Burn the same time as a real check when the email is unknown."""
    verify_password(_DUMMY_HASH, pw)


def verify_totp(secret: str, code: str) -> bool:
    return bool(code) and pyotp.TOTP(secret).verify(code.strip(), valid_window=1)


def make_access_token(staff: Staff, settings: Settings) -> str:
    now = datetime.now(timezone.utc)
    claims = {"sub": str(staff.id), "role": staff.role, "typ": "access",
              "iat": now, "exp": now + timedelta(seconds=settings.access_ttl)}
    return jwt.encode(claims, settings.jwt_secret, algorithm="HS256")


def decode_access_token(token: str, settings: Settings) -> Optional[dict]:
    try:
        claims = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"], options={"require": ["exp", "sub"]})
    except jwt.PyJWTError:
        return None
    return claims if claims.get("typ") == "access" else None


def _hash_refresh(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _new_refresh(session: Session, staff: Staff, settings: Settings, family: str) -> str:
    raw = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    session.add(RefreshToken(staff_id=staff.id, token_hash=_hash_refresh(raw), family=family,
                             expires_at=now + timedelta(seconds=settings.refresh_ttl), created_at=now))
    session.flush()
    return raw


def issue_tokens(staff: Staff, settings: Settings, session: Session, family: Optional[str] = None) -> TokenPair:
    refresh = _new_refresh(session, staff, settings, family or uuid.uuid4().hex)
    return TokenPair(access_token=make_access_token(staff, settings), refresh_token=refresh,
                     expires_in=settings.access_ttl,
                     staff=StaffOut(id=staff.id, email=staff.email, name=staff.name, role=staff.role))


def revoke_all(session: Session, staff_id: int) -> None:
    session.execute(update(RefreshToken).where(RefreshToken.staff_id == staff_id,
                                               RefreshToken.revoked_at.is_(None))
                    .values(revoked_at=datetime.now(timezone.utc)))


def rotate(session: Session, raw: str, settings: Settings) -> Optional[TokenPair]:
    """Exchange a refresh token for a new pair. None means reject (caller answers 401).
    Reuse of an already-rotated token revokes every refresh token of that staff."""
    row = session.scalars(select(RefreshToken).where(RefreshToken.token_hash == _hash_refresh(raw))
                          .with_for_update()).first()
    if row is None:
        return None
    if row.revoked_at is not None:
        revoke_all(session, row.staff_id)
        session.commit()
        return None
    if row.expires_at <= datetime.now(timezone.utc):
        return None
    staff = session.get(Staff, row.staff_id)
    if staff is None or staff.disabled:
        return None
    row.revoked_at = datetime.now(timezone.utc)
    return issue_tokens(staff, settings, session, family=row.family or None)
