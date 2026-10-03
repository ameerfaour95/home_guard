"""Fix round for Task 7: auth hardening."""
import concurrent.futures as cf
import inspect
from datetime import datetime, timedelta, timezone

import pyotp
import pytest
from sqlalchemy import select

from home_guard_project.cloud.db import session_scope
from home_guard_project.cloud.models import AuditLog, RefreshToken, Staff


def _login(client, staff, password, secret, code=None):
    return client.post("/v1/auth/login", json={
        "email": staff.email, "password": password, "totp": code or pyotp.TOTP(secret).now()})


def _audit(client, action):
    with session_scope(client.app.state.engine) as s:
        return list(s.scalars(select(AuditLog).where(AuditLog.action == action).order_by(AuditLog.id)))


def _at(secret, offset):
    return pyotp.TOTP(secret).at(datetime.now(timezone.utc).timestamp() + offset)


def test_login_visible_to_a_new_session_and_committed_in_handler(client, staff_factory):
    staff, pw, secret, _ = staff_factory("admin")
    assert _login(client, staff, pw, secret).status_code == 200
    with session_scope(client.app.state.engine) as s:  # brand-new session
        assert [a.staff_id for a in s.scalars(select(AuditLog).where(AuditLog.action == "login"))] == [staff.id]
        assert s.scalars(select(RefreshToken)).one().staff_id == staff.id
    from home_guard_project.cloud.routes import auth as routes_auth
    for fn in (routes_auth.login, routes_auth.refresh):  # commit happens before the response is sent
        assert getattr(inspect.signature(fn).parameters["session"].default, "scope", None) == "function"


def test_concurrent_wrong_attempts_cannot_beat_lockout(client, staff_factory):
    staff, pw, secret, _ = staff_factory("admin")
    body = {"email": staff.email, "password": "wrong-password", "totp": "000000"}
    with cf.ThreadPoolExecutor(10) as ex:
        codes = list(ex.map(lambda _: client.post("/v1/auth/login", json=body).status_code, range(10)))
    assert set(codes) <= {401, 429}
    assert codes.count(401) <= 5 and codes.count(429) >= 5


def test_totp_code_cannot_be_replayed(client, staff_factory):
    staff, pw, secret, _ = staff_factory("admin")
    code = pyotp.TOTP(secret).now()
    assert _login(client, staff, pw, secret, code=code).status_code == 200
    assert _login(client, staff, pw, secret, code=code).status_code == 401
    assert _login(client, staff, pw, secret, code=_at(secret, -30)).status_code == 401  # older counter
    assert _login(client, staff, pw, secret, code=_at(secret, 30)).status_code == 200  # newer counter


def test_failed_password_does_not_burn_totp_counter(client, staff_factory):
    staff, pw, secret, _ = staff_factory("admin")
    code = pyotp.TOTP(secret).now()
    assert _login(client, staff, "wrong-password", secret, code=code).status_code == 401
    assert _login(client, staff, pw, secret, code=code).status_code == 200


def test_refresh_reuse_writes_audit_row(client, staff_factory):
    staff, pw, secret, _ = staff_factory("admin")
    first = _login(client, staff, pw, secret).json()
    assert client.post("/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}).status_code == 200
    assert client.post("/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}).status_code == 401
    rows = _audit(client, "refresh_reuse")
    assert len(rows) == 1 and rows[0].staff_id == staff.id and rows[0].detail["family"]


def test_session_cap_family_started_at(client, staff_factory):
    staff, pw, secret, _ = staff_factory("admin")
    ttl = client.app.state.settings.refresh_ttl
    first = _login(client, staff, pw, secret).json()
    started = datetime.now(timezone.utc) - timedelta(seconds=ttl - 100)
    with session_scope(client.app.state.engine) as s:
        tok = s.scalars(select(RefreshToken)).one()
        assert tok.family_started_at is not None
        tok.family_started_at = started
    second = client.post("/v1/auth/refresh", json={"refresh_token": first["refresh_token"]})
    assert second.status_code == 200
    with session_scope(client.app.state.engine) as s:
        newest = s.scalars(select(RefreshToken).order_by(RefreshToken.id.desc())).first()
        assert newest.family_started_at == started
        assert newest.expires_at <= started + timedelta(seconds=ttl)
        newest.family_started_at = datetime.now(timezone.utc) - timedelta(seconds=ttl + 1)
    assert client.post("/v1/auth/refresh", json={"refresh_token": second.json()["refresh_token"]}).status_code == 401


def test_every_v1_route_requires_token(client):
    open_paths = {"/v1/auth/login", "/v1/auth/refresh"}
    checked = 0
    for path, ops in client.app.openapi()["paths"].items():  # every registered route
        if not path.startswith("/v1") or path in open_paths:
            continue
        concrete = path.replace("{", "").replace("}", "")
        for method in ops:
            r = client.request(method.upper(), concrete)
            assert r.status_code == 401, (method, path, r.status_code)
            checked += 1
    assert checked >= 10


def test_jwt_secret_must_be_32_bytes(monkeypatch):
    from home_guard_project.cloud.settings import Settings

    monkeypatch.setenv("HG_CLOUD_DB_URL", "postgresql://x")
    monkeypatch.setenv("HG_CLOUD_JWT_SECRET", "short")
    with pytest.raises(ValueError, match="HG_CLOUD_JWT_SECRET"):
        Settings.from_env()
    monkeypatch.setenv("HG_CLOUD_JWT_SECRET", "s" * 32)
    assert Settings.from_env().jwt_secret == "s" * 32
    assert len(Settings.for_tests("postgresql://x").jwt_secret.encode()) >= 32


def test_login_locked_records_staff_id(client, staff_factory):
    staff, pw, secret, _ = staff_factory("admin")
    for _ in range(5):
        _login(client, staff, pw, secret, code="000000")
    assert _login(client, staff, pw, secret).status_code == 429
    locked = _audit(client, "login_locked")
    assert len(locked) == 1 and locked[0].staff_id == staff.id


def test_oversized_credentials_rejected_without_hashing(client, staff_factory, monkeypatch):
    from home_guard_project.cloud import auth as auth_mod

    staff, pw, secret, _ = staff_factory("admin")
    called = []
    monkeypatch.setattr(auth_mod, "verify_password", lambda *a: called.append(1) or False)
    monkeypatch.setattr(auth_mod, "verify_dummy", lambda *a: called.append(1))
    r1 = client.post("/v1/auth/login", json={"email": staff.email, "password": "p" * 1025, "totp": "123456"})
    r2 = client.post("/v1/auth/login", json={"email": staff.email, "password": pw, "totp": "1" * 17})
    r3 = client.post("/v1/auth/login", json={"email": "nobody@example.com", "password": "p" * 2000, "totp": "1"})
    assert r1.status_code == r2.status_code == r3.status_code == 401
    assert r1.json()["detail"] == auth_mod.BAD_LOGIN and not called
    assert len(_audit(client, "login_failed")) == 3


def test_audit_rows_snapshot_staff_name(client, staff_factory):
    staff, pw, secret, _ = staff_factory("admin")
    _login(client, staff, pw, secret)
    assert _audit(client, "login")[0].staff_name == staff.name
    client.post("/v1/auth/login", json={"email": "nobody@example.com", "password": "x", "totp": "1"})
    assert _audit(client, "login_failed")[-1].staff_name is None


def test_padded_uppercase_email_shares_lockout(client, staff_factory):
    staff, pw, secret, _ = staff_factory("admin")
    sloppy = f"  {staff.email.upper()} "
    for _ in range(5):
        r = client.post("/v1/auth/login", json={"email": sloppy, "password": pw, "totp": "000000"})
        assert r.status_code == 401
    assert _login(client, staff, pw, secret).status_code == 429


def test_refresh_for_disabled_staff_401(client, staff_factory):
    staff, pw, secret, _ = staff_factory("admin")
    tokens = _login(client, staff, pw, secret).json()
    with session_scope(client.app.state.engine) as s:
        s.get(Staff, staff.id).disabled = True
    assert client.post("/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}).status_code == 401


def test_role_change_applies_to_issued_access_token(client, staff_factory):
    staff, _, _, headers = staff_factory("admin")
    assert client.get("/v1/fleet", headers=headers).status_code == 501
    with session_scope(client.app.state.engine) as s:
        s.get(Staff, staff.id).role = "labeler"
    assert client.get("/v1/fleet", headers=headers).status_code == 403
