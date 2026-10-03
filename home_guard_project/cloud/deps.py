"""Shared FastAPI dependencies: DB session, current staff, role gate."""
from __future__ import annotations

from typing import Callable, Iterator, Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from . import auth
from .models import Staff

bearer = HTTPBearer(auto_error=False)  # auto_error off so we answer 401 (not 403) ourselves


def get_session(request: Request) -> Iterator[Session]:
    """One session per request: commit on success, rollback on error."""
    session: Session = request.app.state.sessionmaker()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# The ONLY way routes get a session. scope="function": the dependency exits (commits) before the response is sent,
# so a failed commit is a 500 and never a delivered success. Do not use Depends(get_session) directly.
SessionDep = Depends(get_session, scope="function")


def _unauthorized() -> HTTPException:
    return HTTPException(status_code=401, detail="Not signed in", headers={"WWW-Authenticate": "Bearer"})


def current_staff(request: Request,
                  creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
                  session: Session = SessionDep) -> Staff:
    if creds is None:
        raise _unauthorized()
    claims = auth.decode_access_token(creds.credentials, request.app.state.settings)
    if claims is None:
        raise _unauthorized()
    try:
        staff = session.get(Staff, int(claims["sub"]))
    except (ValueError, TypeError):
        raise _unauthorized()
    if staff is None or staff.disabled:
        raise _unauthorized()
    return staff


def require_role(*roles: str) -> Callable[..., Staff]:
    def dep(staff: Staff = Depends(current_staff)) -> Staff:
        if staff.role not in roles:
            raise HTTPException(status_code=403, detail="Your role cannot do this")
        return staff

    return dep
