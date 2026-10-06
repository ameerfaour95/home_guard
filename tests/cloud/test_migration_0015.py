"""Migration 0015 (consent comes from the sales contract): every existing customer is switched on with one audit row
"consent from the sales contract", new customers default to on, consent_source is recorded; 0014 -> 0015 -> 0014 ->
head, and the models match the migrated schema (no drift)."""
from sqlalchemy import text

from .test_migration_0007 import _engine, fresh_url  # noqa: F401 (fixture)


def test_0015_consent_from_contract(fresh_url):
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from home_guard_project.cloud.manage import alembic_config
    from home_guard_project.cloud.models import Base

    cfg = alembic_config(fresh_url)
    command.upgrade(cfg, "0014")
    engine = _engine(fresh_url)
    try:
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO customers (id, name, consent_live, consent_recordings, consent_training) "
                              "VALUES (1, 'Dana', false, false, false), (2, 'Eli', true, true, false)"))
        command.upgrade(cfg, "0015")
        with engine.begin() as conn:
            rows = conn.execute(text("SELECT id, consent_live, consent_recordings, consent_training, consent_source "
                                     "FROM customers ORDER BY id")).all()
            assert [tuple(r) for r in rows] == [(1, True, True, True, "contract"), (2, True, True, True, "contract")]
            audit = conn.execute(text("SELECT customer_id, staff_name, reason, detail FROM audit_log "
                                      "WHERE action = 'consent_from_contract' ORDER BY customer_id")).all()
            assert [(a[0], a[1], a[2]) for a in audit] == [(1, "migration 0015", "consent from the sales contract"),
                                                         (2, "migration 0015", "consent from the sales contract")]
            assert audit[1][3] == {"before": {"live": True, "recordings": True, "training": False}}
            conn.execute(text("INSERT INTO customers (id, name) VALUES (3, 'New')"))
            new = conn.execute(text("SELECT consent_live, consent_recordings, consent_training, consent_source "
                                    "FROM customers WHERE id = 3")).one()
            assert tuple(new) == (True, True, True, "contract")
        command.downgrade(cfg, "0014")
        command.upgrade(cfg, "head")
        with engine.connect() as conn:
            assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []
    finally:
        engine.dispose()
