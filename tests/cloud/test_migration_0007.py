"""Migration 0007 on a populated database: upgrade backfills identity aliases, downgrade keeps every earlier row,
re-upgrade works, and the models match the migrated schema (no drift)."""
import json
import uuid

import pytest
from sqlalchemy import create_engine, inspect, text

from .conftest import _db_url


@pytest.fixture()
def fresh_url(pg_url, pg_template):
    name = "mig_" + uuid.uuid4().hex[:10]
    with pg_template.connect() as c:
        c.execute(text(f"CREATE DATABASE {name}"))
    yield _db_url(pg_url, name)
    with pg_template.connect() as c:
        c.execute(text(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)"))


def _engine(url):
    return create_engine(url.replace("postgresql://", "postgresql+psycopg://", 1))


def _populate(conn):
    heartbeat = {"host": "oak-laptop", "time_utc": "2026-10-03T11:00:00Z",
                 "cameras": {"oak_ch1": {"newest_clip_utc": "2026-10-03T10:00:00Z"}}}
    conn.execute(text("INSERT INTO customers (id, name) VALUES (1, 'Oakhaven Family')"))
    conn.execute(text("INSERT INTO devices (id, device_id, site, tailscale_host, customer_id, last_heartbeat) "
                      "VALUES (1, :d, 'oak', 'oak-box', 1, CAST(:hb AS jsonb))"),
                 {"d": str(uuid.uuid4()), "hb": json.dumps(heartbeat)})
    conn.execute(text("INSERT INTO cameras (device_pk, name, display_name) VALUES (1, 'oak_ch2', 'Oak Porch')"))
    conn.execute(text("INSERT INTO events (device_pk, site, camera, stem, start_ts) "
                      "VALUES (1, 'oak', 'oak_ch3', 'oak_ch3_1791020177_alert', 1791020177)"))
    conn.execute(text("INSERT INTO index_problems (s3_key, reason, seen_at) VALUES ('k', 'media: x', now())"))
    conn.execute(text("INSERT INTO audit_log (ts, staff_id, action, target) VALUES (now(), 1, 'event_view', 'e/1')"))


COUNTS = ("customers", "devices", "cameras", "events", "index_problems", "audit_log")


def _counts(conn):
    return {t: conn.execute(text(f"SELECT count(*) FROM {t}")).scalar() for t in COUNTS}


def test_0007_upgrade_downgrade_on_a_populated_database(fresh_url):
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from home_guard_project.cloud.manage import alembic_config
    from home_guard_project.cloud.models import Base

    cfg = alembic_config(fresh_url)
    command.upgrade(cfg, "0006")
    engine = _engine(fresh_url)
    try:
        with engine.begin() as conn:
            _populate(conn)
            before = _counts(conn)
        command.upgrade(cfg, "0007")
        with engine.connect() as conn:
            aliases = set(conn.execute(text("SELECT kind, value FROM identity_aliases WHERE device_pk = 1")).all())
            assert {("customer", "Oakhaven Family"), ("site", "oak"), ("host", "oak-box"), ("host", "oak-laptop"),
                    ("camera", "oak_ch1"), ("camera", "oak_ch2"), ("camera", "oak_ch3"),
                    ("display_name", "Oak Porch")} <= aliases
            assert conn.execute(text("SELECT attempts FROM index_problems")).scalar() == 0
            assert _counts(conn) == before
        command.downgrade(cfg, "0006")
        with engine.connect() as conn:
            assert "identity_aliases" not in inspect(conn).get_table_names()
            assert "attempts" not in {c["name"] for c in inspect(conn).get_columns("index_problems")}
            assert _counts(conn) == before
        command.upgrade(cfg, "head")
        with engine.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM identity_aliases")).scalar() >= 8
            assert _counts(conn) == before
            diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)  # head (0008) = the models
            assert diff == [], diff
    finally:
        engine.dispose()
