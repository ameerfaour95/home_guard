import json
import subprocess
import sys
from datetime import datetime, timezone

import pytest
from sqlalchemy import delete, inspect, select, text, update
from sqlalchemy.exc import DBAPIError, IntegrityError

from home_guard_project.cloud import models as m
from home_guard_project.cloud.db import make_engine, session_scope

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def _chain(s):
    cust = m.Customer(name="Acme")
    s.add(cust)
    s.flush()
    dev = m.Device(device_id="11111111-2222-3333-4444-555555555555", site="acme", customer_id=cust.id, enrolled_at=NOW)
    s.add(dev)
    s.flush()
    return cust, dev


def _event(dev, stem="s1", camera="front"):
    return m.Event(device_pk=dev.id, site="acme", camera=camera, stem=stem, kind="alert",
                   start_ts=1.0, trigger_ts=2, day="2026-10-03", completeness={"video": True})


def test_make_engine_rewrites_driver():
    e = make_engine("postgresql://u:p@h/db")
    assert e.url.drivername == "postgresql+psycopg"


def test_all_tables_created_by_migration(db_engine):
    names = set(inspect(db_engine).get_table_names())
    expected = {"staff", "refresh_tokens", "customers", "devices", "cameras", "events", "artifacts",
                "raw_revisions", "ai_runs", "feedback", "review_state", "collections", "collection_items",
                "exports", "audit_log", "index_problems", "s3_cursors", "owner_notices"}
    assert expected <= names


def test_insert_chain_and_fulltext(db_engine):
    with session_scope(db_engine) as s:
        cust, dev = _chain(s)
        ev = _event(dev)
        ev.summary = "A person walks past the gate"
        s.add(ev)
        s.flush()
        s.add(m.Artifact(event_id=ev.id, role="original_video", s3_key="k/1.mp4", available=True, provenance="box",
                         camera="front", stem="s1"))
        s.add(m.AiRun(event_id=ev.id, status="real", parsed={"a": 1}, input_artifact_ids=[]))
        s.add(m.ReviewState(event_id=ev.id, reviewed=True))
    with session_scope(db_engine) as s:
        hits = s.scalars(select(m.Event).where(m.Event.search.op("@@")(text("plainto_tsquery('simple','gate')")))).all()
        assert len(hits) == 1
        assert hits[0].alert_reason == "" and hits[0].detected in (None, [])
        assert s.scalars(select(m.Artifact)).one().provenance == "box"


def test_event_unique_device_camera_stem(db_engine):
    with pytest.raises(IntegrityError):
        with session_scope(db_engine) as s:
            _, dev = _chain(s)
            s.add(_event(dev))
            s.add(_event(dev))


def test_artifact_s3_key_unique(db_engine):
    with pytest.raises(IntegrityError):
        with session_scope(db_engine) as s:
            s.add(m.Artifact(role="meta", s3_key="same"))
            s.add(m.Artifact(role="meta", s3_key="same"))


def test_raw_revision_and_export_unique(db_engine):
    with pytest.raises(IntegrityError):
        with session_scope(db_engine) as s:
            for _ in range(2):
                s.add(m.RawRevision(s3_key="k", etag="e", fetched_at=NOW, body={}))
    with pytest.raises(IntegrityError):
        with session_scope(db_engine) as s:
            s.add(m.Staff(email="a@x", name="A", role="admin", password_hash="h", totp_secret="t"))
            s.flush()
            sid = s.scalars(select(m.Staff.id)).one()
            for _ in range(2):
                s.add(m.Export(name="n", version=1, state="queued", created_by=sid, request={}, created_at=NOW))


def test_audit_log_append_only(db_engine):
    with session_scope(db_engine) as s:
        s.add(m.AuditLog(ts=NOW, action="login", target="x", reason=""))
    with pytest.raises(DBAPIError, match="append-only"):
        with session_scope(db_engine) as s:
            s.execute(update(m.AuditLog).values(action="tampered"))
    with pytest.raises(DBAPIError, match="append-only"):
        with session_scope(db_engine) as s:
            s.execute(delete(m.AuditLog))
    with session_scope(db_engine) as s:
        assert s.scalars(select(m.AuditLog)).one().action == "login"


def test_manage_create_staff_and_enroll(pg_url, pg_template):
    import os
    import uuid

    from sqlalchemy.engine import make_url

    name = "t_" + uuid.uuid4().hex[:12]
    with pg_template.connect() as c:
        c.execute(text(f"CREATE DATABASE {name}"))
    url = make_url(pg_url).set(database=name).render_as_string(hide_password=False)
    try:
        env = {**os.environ, "HG_CLOUD_DB_URL": url, "HG_CLOUD_JWT_SECRET": "x" * 32}
        r = subprocess.run([sys.executable, "-m", "home_guard_project.cloud.manage", "init-db"], env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        r = subprocess.run([sys.executable, "-m", "home_guard_project.cloud.manage", "create-staff", "--email", "a@x.io",
                            "--name", "Ann", "--role", "admin"], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert "otpauth://totp/" in r.stdout and "password" in r.stdout.lower()
        eng = make_engine(url)
        with session_scope(eng) as s:
            st = s.scalars(select(m.Staff)).one()
            assert st.password_hash.startswith("$argon2") and st.email == "a@x.io" and st.totp_secret
            s.add(m.Customer(name="Acme"))
        r = subprocess.run([sys.executable, "-m", "home_guard_project.cloud.manage", "enroll", "--customer", "Acme",
                            "--site", "acme", "--tailscale-host", "h1"], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        with session_scope(eng) as s:
            d = s.scalars(select(m.Device)).one()
            assert d.site == "acme" and d.tailscale_host == "h1" and len(d.device_id) == 36
        eng.dispose()
    finally:
        with pg_template.connect() as c:
            c.execute(text(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)"))
