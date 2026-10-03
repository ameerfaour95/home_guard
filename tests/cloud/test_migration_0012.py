"""Migration 0012 (precomputed weak-label suggestions: annotation_suggestions): 0011 -> 0012 -> 0011 -> head, the
row goes with its event, and the models match the migrated schema (no drift)."""
import uuid

from sqlalchemy import inspect, text

from .test_migration_0007 import _engine, fresh_url  # noqa: F401 (fixture)


def test_0012_upgrade_downgrade_and_back_to_head(fresh_url):
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from home_guard_project.cloud.manage import alembic_config
    from home_guard_project.cloud.models import Base

    cfg = alembic_config(fresh_url)
    command.upgrade(cfg, "0011")
    engine = _engine(fresh_url)
    try:
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO customers (id, name) VALUES (1, 'Dana')"))
            conn.execute(text("INSERT INTO devices (id, device_id, site, tailscale_host, customer_id) "
                              "VALUES (1, :d, 'dana', 'h', 1)"), {"d": str(uuid.uuid4())})
            conn.execute(text("INSERT INTO events (id, device_pk, site, camera, stem, start_ts) "
                              "VALUES (1, 1, 'dana', 'cam', 'cam_1791020177_alert', 1791020177)"))
        command.upgrade(cfg, "0012")
        with engine.begin() as conn:
            assert "annotation_suggestions" in inspect(conn).get_table_names()
            conn.execute(text("INSERT INTO annotation_suggestions (event_id, tracks, sources, computed_at) "
                              "VALUES (1, '[]'::jsonb, '{}'::jsonb, now())"))
            conn.execute(text("DELETE FROM events WHERE id = 1"))
            assert conn.execute(text("SELECT count(*) FROM annotation_suggestions")).scalar() == 0
        command.downgrade(cfg, "0011")
        with engine.connect() as conn:
            assert "annotation_suggestions" not in inspect(conn).get_table_names()
        command.upgrade(cfg, "head")
        with engine.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
            assert diff == [], diff
    finally:
        engine.dispose()
