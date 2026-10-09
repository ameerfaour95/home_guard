"""Migration 0019 (owner_notices.delivery_* and last_delivery_error): 0018 -> 0019 -> 0018 -> head, no drift."""
from sqlalchemy import inspect

from .test_migration_0007 import _engine, fresh_url  # noqa: F401 (fixture)


def test_0019_owner_notice_delivery(fresh_url):
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from home_guard_project.cloud.manage import alembic_config
    from home_guard_project.cloud.models import Base

    cfg = alembic_config(fresh_url)
    command.upgrade(cfg, "0018")
    engine = _engine(fresh_url)
    try:
        columns = lambda: {c["name"] for c in inspect(engine).get_columns("owner_notices")}  # noqa: E731
        added = {"delivery_state", "delivery_since", "delivery_attempts", "delivery_heartbeat_at", "last_delivery_error",
                 "delivery_body_sha"}
        assert not columns() & added
        command.upgrade(cfg, "0019")
        assert added <= columns()
        command.downgrade(cfg, "0018")
        assert not columns() & added
        command.upgrade(cfg, "head")
        with engine.connect() as conn:
            assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []
    finally:
        engine.dispose()
