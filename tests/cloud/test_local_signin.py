from fastapi.testclient import TestClient
from sqlalchemy import select

from home_guard_project.cloud.db import session_scope
from home_guard_project.cloud.models import AuditLog, Staff

LOOP = ("127.0.0.1", 50000)


def _enable(client, staff, trust=True):
    client.app.state.settings.local_trust = trust
    client.app.state.settings.local_admin_email = staff.email


def _post(client, host=LOOP):
    with TestClient(client.app, client=host) as c:
        return c.post("/v1/auth/local")


def test_local_signin_on_loopback(client, staff_factory):
    staff, *_ = staff_factory("admin")
    _enable(client, staff)
    for host in (LOOP, ("::1", 50000)):
        r = _post(client, host)
        assert r.status_code == 200, r.text
        assert r.json()["staff"]["email"] == staff.email and r.json()["access_token"]
    with session_scope(client.app.state.engine) as s:
        assert s.scalars(select(AuditLog).where(AuditLog.action == "local_signin")).first() is not None
    me = client.get("/v1/me", headers={"Authorization": f"Bearer {r.json()['access_token']}"})
    assert me.status_code == 200


def test_local_signin_404_when_trust_off(client, staff_factory):
    staff, *_ = staff_factory("admin")
    _enable(client, staff, trust=False)
    r = _post(client)
    assert r.status_code == 404
    assert r.json() == client.get("/v1/no-such-route").json()


def test_local_signin_404_from_non_loopback(client, staff_factory):
    staff, *_ = staff_factory("admin")
    _enable(client, staff)
    assert _post(client, ("10.0.0.5", 1234)).status_code == 404


def test_local_signin_404_without_staff_row(client, staff_factory):
    staff, *_ = staff_factory("admin")
    _enable(client, staff)
    client.app.state.settings.local_admin_email = "nobody@example.com"
    assert _post(client).status_code == 404


def test_local_signin_404_for_disabled_staff(client, staff_factory):
    staff, *_ = staff_factory("admin")
    _enable(client, staff)
    with session_scope(client.app.state.engine) as s:
        s.get(Staff, staff.id).disabled = True
    assert _post(client).status_code == 404
