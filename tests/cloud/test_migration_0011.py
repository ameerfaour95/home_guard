"""Migration 0011 (in-app labeling: annotations, annotation_heads, annotation_reviews, tagging_publishes):
0010 -> 0011 -> 0010 -> head, earlier rows kept, and the models match the migrated schema (no drift)."""
import uuid

from sqlalchemy import inspect, text

from .test_migration_0007 import _engine, fresh_url  # noqa: F401 (fixture)

TABLES = {"annotations", "annotation_heads", "annotation_reviews", "tagging_publishes"}


def test_0011_upgrade_downgrade_and_back_to_head(fresh_url):
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from home_guard_project.cloud.manage import alembic_config
    from home_guard_project.cloud.models import Base

    cfg = alembic_config(fresh_url)
    command.upgrade(cfg, "0010")
    engine = _engine(fresh_url)
    try:
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO customers (id, name) VALUES (1, 'Dana')"))
            conn.execute(text("INSERT INTO devices (id, device_id, site, tailscale_host, customer_id) "
                              "VALUES (1, :d, 'dana', 'h', 1)"), {"d": str(uuid.uuid4())})
            conn.execute(text("INSERT INTO events (id, device_pk, site, camera, stem, start_ts) "
                              "VALUES (1, 1, 'dana', 'cam', 'cam_1791020177_alert', 1791020177)"))
        command.upgrade(cfg, "0011")
        with engine.begin() as conn:
            assert TABLES <= set(inspect(conn).get_table_names())
            conn.execute(text("INSERT INTO annotations (event_id, version, status, tracks, description, "
                              "ai_description, created_at) VALUES (1, 1, 'edited', '[]'::jsonb, 'd', 'a', now())"))
            dup = conn.begin_nested()
            try:
                conn.execute(text("INSERT INTO annotations (event_id, version, status, tracks, description, "
                                  "ai_description, created_at) VALUES (1, 1, 'edited', '[]'::jsonb, 'd', 'a', now())"))
                raise AssertionError("a second row for the same (event, version) must be refused")
            except Exception as e:  # noqa: BLE001
                if isinstance(e, AssertionError):
                    raise
                dup.rollback()
        command.downgrade(cfg, "0010")
        with engine.connect() as conn:
            assert not (TABLES & set(inspect(conn).get_table_names()))
            assert conn.execute(text("SELECT count(*) FROM events")).scalar() == 1
        command.upgrade(cfg, "head")
        with engine.connect() as conn:
            assert TABLES <= set(inspect(conn).get_table_names())
            diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
            assert diff == [], diff
    finally:
        engine.dispose()
