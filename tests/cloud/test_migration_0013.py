"""Migration 0013 (a tagging batch name is used once, forever): tagging_publishes.batch_name becomes unique across
all states. Existing duplicates keep the newest row under the name (older rows get `#<id>` appended), 0012 -> 0013
-> 0012 -> head, and the models match the migrated schema (no drift)."""
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from .test_migration_0007 import _engine, fresh_url  # noqa: F401 (fixture)


def _publish(conn, pid, name, state):
    conn.execute(text("INSERT INTO tagging_publishes (id, batch_name, state, created_by, created_at) "
                      "VALUES (:id, :n, :s, 1, now())"), {"id": pid, "n": name, "s": state})


def test_0013_unique_batch_names(fresh_url):
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from home_guard_project.cloud.manage import alembic_config
    from home_guard_project.cloud.models import Base

    cfg = alembic_config(fresh_url)
    command.upgrade(cfg, "0012")
    engine = _engine(fresh_url)
    try:
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO staff (id, email, name, role, password_hash, totp_secret, disabled) "
                              "VALUES (1, 'a@x', 'Admin', 'admin', '', '', false)"))
            _publish(conn, 1, "night", "failed")
            _publish(conn, 2, "night", "ready")
            _publish(conn, 3, "other", "ready")
        command.upgrade(cfg, "0013")
        with engine.begin() as conn:
            rows = dict(conn.execute(text("SELECT id, batch_name FROM tagging_publishes")).all())
            assert rows == {1: "night#1", 2: "night", 3: "other"}
        with pytest.raises(IntegrityError):
            with engine.begin() as conn:
                _publish(conn, 4, "other", "failed")
        command.downgrade(cfg, "0012")
        with engine.begin() as conn:
            _publish(conn, 4, "other", "failed")  # no longer unique
            conn.execute(text("DELETE FROM tagging_publishes WHERE id = 4"))
        command.upgrade(cfg, "head")
        with engine.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
            assert diff == [], diff
    finally:
        engine.dispose()
