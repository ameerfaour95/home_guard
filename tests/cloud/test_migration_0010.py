"""Migration 0010 (customers.consent_proposed, phone scrub of stored registrations): 0010 -> 0008 -> head."""
import json
import uuid

from sqlalchemy import inspect, text

from .test_migration_0007 import _engine, fresh_url  # noqa: F401 (fixture)


def test_0010_downgrade_to_0008_and_back_to_head(fresh_url):
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from home_guard_project.cloud.manage import alembic_config
    from home_guard_project.cloud.models import Base

    cfg = alembic_config(fresh_url)
    command.upgrade(cfg, "0009")
    engine = _engine(fresh_url)
    try:
        body = {"schema_version": 1, "owner_name": "Dana", "owner_phone": "+972501234567"}
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO customers (id, name, owner_phone) VALUES (1, 'Dana', '+972501234567')"))
            conn.execute(text("INSERT INTO devices (id, device_id, site, tailscale_host, customer_id) "
                              "VALUES (1, :d, 'dana', 'h', 1)"), {"d": str(uuid.uuid4())})
            conn.execute(text("INSERT INTO raw_revisions (s3_key, etag, fetched_at, body) "
                              "VALUES ('dataset_dana/_status/registration.json', 'e1', now(), CAST(:b AS jsonb))"),
                         {"b": json.dumps(body)})
        command.upgrade(cfg, "0010")
        with engine.connect() as conn:
            assert "consent_proposed" in {c["name"] for c in inspect(conn).get_columns("customers")}
            stored = conn.execute(text("SELECT body FROM raw_revisions")).scalar()
            assert "owner_phone" not in stored and stored["owner_name"] == "Dana"
            assert conn.execute(text("SELECT owner_phone FROM customers")).scalar() == "+972501234567"
        command.downgrade(cfg, "0008")
        with engine.connect() as conn:
            cols = {c["name"] for c in inspect(conn).get_columns("customers")}
            assert "consent_proposed" not in cols and "owner_phone" not in cols
            assert conn.execute(text("SELECT count(*) FROM customers")).scalar() == 1
            assert conn.execute(text("SELECT count(*) FROM raw_revisions")).scalar() == 1
        command.upgrade(cfg, "head")
        with engine.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM customers")).scalar() == 1
            diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
            assert diff == [], diff
    finally:
        engine.dispose()
