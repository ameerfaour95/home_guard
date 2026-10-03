"""Fix round for Task 6: migration 0002_hardening, manage.py behaviour."""
import importlib
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import insert, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError, IntegrityError

from home_guard_project.cloud import models as m
from home_guard_project.cloud.db import make_engine, session_scope

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def _chain(s, site="acme"):
    cust = m.Customer(name="Acme")
    s.add(cust)
    s.flush()
    dev = m.Device(device_id=str(uuid.uuid4()), site=site, customer_id=cust.id, enrolled_at=NOW)
    s.add(dev)
    s.flush()
    return cust, dev


def test_audit_log_truncate_blocked(db_engine):
    with session_scope(db_engine) as s:
        s.add(m.AuditLog(ts=NOW, action="login", target="x", reason=""))
    with pytest.raises(DBAPIError, match="append-only"):
        with session_scope(db_engine) as s:
            s.execute(text("TRUNCATE audit_log"))
    with session_scope(db_engine) as s:
        assert s.scalars(select(m.AuditLog)).one().action == "login"


def test_keyset_indexes(db_engine):
    with session_scope(db_engine) as s:
        defs = dict(s.execute(text(
            "SELECT indexname, indexdef FROM pg_indexes WHERE tablename IN ('events', 'audit_log')")).all())
    assert "(device_pk, start_ts DESC, id DESC)" in defs["ix_events_device_start"]
    assert "(camera, start_ts DESC, id DESC)" in defs["ix_events_camera_start"]
    assert "(action, ts)" in defs["ix_audit_log_action_ts"]


def test_device_site_unique(db_engine):
    with pytest.raises(IntegrityError):
        with session_scope(db_engine) as s:
            cust, _ = _chain(s)
            s.add(m.Device(device_id=str(uuid.uuid4()), site="acme", customer_id=cust.id))


def test_bulk_insert_uses_server_defaults(db_engine):
    with session_scope(db_engine) as s:
        s.execute(insert(m.Customer).values(name="Bulk"))
        s.execute(insert(m.Staff).values(email="b@x.io", name="B", role="admin", password_hash="h", totp_secret="t"))
        _, dev = _chain(s)
        s.execute(insert(m.Event).values(device_pk=dev.id, site="acme", camera="c", stem="s", start_ts=1.0))
        s.flush()
        ev = s.scalars(select(m.Event)).one()
        assert (ev.kind, ev.summary, ev.alert_reason, ev.completeness) == ("unknown", "", "", {})
        s.execute(insert(m.Artifact).values(role="meta", s3_key="k"))
        s.execute(insert(m.AuditLog).values(ts=NOW, action="a"))
        s.execute(insert(m.AiRun).values(event_id=ev.id))
        s.execute(insert(m.OwnerNotice).values(device_pk=dev.id, kind="k"))
        s.flush()
        c = s.scalars(select(m.Customer).where(m.Customer.name == "Bulk")).one()
        assert c.timezone == "Asia/Jerusalem" and c.consent_live is False and c.notes == ""
        assert s.scalars(select(m.Staff).where(m.Staff.email == "b@x.io")).one().disabled is False
        a = s.scalars(select(m.Artifact)).one()
        assert a.available is True and a.provenance == "box"
        assert s.scalars(select(m.AiRun)).one().input_artifact_ids == []
        assert s.scalars(select(m.OwnerNotice)).one().cameras == []


def _env(url):
    env = {k: v for k, v in os.environ.items() if k != "HG_CLOUD_DB_URL"}
    env["HG_CLOUD_JWT_SECRET"] = "x" * 32
    if url:
        env["HG_CLOUD_DB_URL"] = url
    return env


def _manage(env, *args):
    return subprocess.run([sys.executable, "-m", "home_guard_project.cloud.manage", *args], env=env,
                          capture_output=True, text=True)


def test_manage_without_db_url_is_a_clean_error():
    for cmd in (["init-db"], ["create-staff", "--email", "a@x.io", "--name", "A", "--role", "admin"]):
        r = _manage(_env(None), *cmd)
        assert r.returncode != 0
        assert "KeyError" not in r.stderr and "HG_CLOUD_DB_URL" in r.stderr


def test_manage_enroll_existing_site_leaves_no_customer(pg_url, pg_template):
    name = "t_" + uuid.uuid4().hex[:12]
    with pg_template.connect() as c:
        c.execute(text(f"CREATE DATABASE {name}"))
    url = make_url(pg_url).set(database=name).render_as_string(hide_password=False)
    try:
        env = _env(url)
        assert _manage(env, "init-db").returncode == 0
        assert _manage(env, "enroll", "--customer", "First", "--site", "dup").returncode == 0
        r = _manage(env, "enroll", "--customer", "Second", "--site", "dup")
        assert r.returncode == 1 and "already enrolled" in r.stderr
        eng = make_engine(url)
        with session_scope(eng) as s:
            assert [c.name for c in s.scalars(select(m.Customer))] == ["First"]
        eng.dispose()
    finally:
        with pg_template.connect() as c:
            c.execute(text(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)"))


def test_manage_swallows_only_the_expected_missing_module(monkeypatch):
    from home_guard_project.cloud import manage

    def missing(expected):
        def fake(name, package=None):
            raise ModuleNotFoundError(f"No module named {expected!r}", name=expected)
        return fake

    monkeypatch.setattr(importlib, "import_module", missing("boto3"))
    with pytest.raises(ModuleNotFoundError):
        manage.cmd_index_once(None)
    with pytest.raises(ModuleNotFoundError):
        manage.cmd_serve(type("A", (), {"port": 1})())
    monkeypatch.setattr(importlib, "import_module", missing("home_guard_project.cloud.indexer"))
    assert manage.cmd_index_once(None) == 0
