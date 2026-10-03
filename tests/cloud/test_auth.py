import pyotp
from sqlalchemy import select

from home_guard_project.cloud.db import session_scope
from home_guard_project.cloud.models import AuditLog, RefreshToken


def _login(client, staff, password, secret, code=None):
    return client.post("/v1/auth/login", json={
        "email": staff.email, "password": password, "totp": code or pyotp.TOTP(secret).now()})


def _audit(client, action):
    with session_scope(client.app.state.engine) as s:
        return list(s.scalars(select(AuditLog).where(AuditLog.action == action)))


def test_login_success(client, staff_factory):
    staff, pw, secret, _ = staff_factory("admin")
    r = _login(client, staff, pw, secret)
    assert r.status_code == 200
    body = r.json()
    assert body["staff"]["email"] == staff.email and body["staff"]["role"] == "admin"
    assert body["expires_in"] == client.app.state.settings.access_ttl
    me = client.get("/v1/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200 and me.json()["email"] == staff.email
    rows = _audit(client, "login")
    assert len(rows) == 1 and rows[0].staff_id == staff.id


def test_bad_totp_and_bad_password_same_401(client, staff_factory):
    staff, pw, secret, _ = staff_factory("admin")
    r1 = _login(client, staff, pw, secret, code="000000")
    r2 = _login(client, staff, "wrong-password", secret)
    r3 = client.post("/v1/auth/login", json={"email": "nobody@example.com", "password": "x", "totp": "123456"})
    assert r1.status_code == r2.status_code == r3.status_code == 401
    assert r1.json() == r2.json() == r3.json()
    assert r1.json()["detail"] == "Email, password or code is wrong"
    rows = _audit(client, "login_failed")
    assert len(rows) == 3
    assert all("password" not in (r.detail or {}) and "totp" not in (r.detail or {}) for r in rows)
    assert rows[0].detail["email"] == staff.email


def test_lockout_on_sixth_attempt(client, staff_factory):
    staff, pw, secret, _ = staff_factory("admin")
    for _ in range(5):
        assert _login(client, staff, pw, secret, code="000000").status_code == 401
    assert _login(client, staff, pw, secret).status_code == 429  # correct creds, still locked


def test_disabled_staff_401(client, staff_factory):
    staff, pw, secret, headers = staff_factory("admin")
    from home_guard_project.cloud.models import Staff
    with session_scope(client.app.state.engine) as s:
        s.get(Staff, staff.id).disabled = True
    assert _login(client, staff, pw, secret).status_code == 401
    assert client.get("/v1/me", headers=headers).status_code == 401


def test_me_requires_valid_token(client, staff_factory):
    assert client.get("/v1/me").status_code == 401
    assert client.get("/v1/me", headers={"Authorization": "Bearer garbage"}).status_code == 401


def test_expired_access_token_401(client, staff_factory):
    staff, pw, secret, _ = staff_factory("admin")
    client.app.state.settings.access_ttl = -1
    body = _login(client, staff, pw, secret).json()
    r = client.get("/v1/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert r.status_code == 401


def test_refresh_rotates_and_reuse_revokes(client, staff_factory):
    staff, pw, secret, _ = staff_factory("admin")
    first = _login(client, staff, pw, secret).json()
    r = client.post("/v1/auth/refresh", json={"refresh_token": first["refresh_token"]})
    assert r.status_code == 200
    second = r.json()
    assert second["refresh_token"] != first["refresh_token"]
    # reuse of the rotated token: 401 and everything revoked
    assert client.post("/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}).status_code == 401
    assert client.post("/v1/auth/refresh", json={"refresh_token": second["refresh_token"]}).status_code == 401
    with session_scope(client.app.state.engine) as s:
        rows = list(s.scalars(select(RefreshToken)))
        assert rows and all(t.revoked_at is not None for t in rows)
        assert all(len(t.token_hash) == 64 and t.token_hash != first["refresh_token"] for t in rows)


def test_refresh_unknown_token_401(client):
    assert client.post("/v1/auth/refresh", json={"refresh_token": "nope"}).status_code == 401


def test_labeler_forbidden_on_fleet(client, staff_factory):
    _, _, _, headers = staff_factory("labeler")
    assert client.get("/v1/fleet", headers=headers).status_code == 403
    assert client.get("/v1/fleet").status_code == 401
    _, _, _, admin = staff_factory("admin")
    assert client.get("/v1/fleet", headers=admin).status_code == 501  # stub body, role passed
