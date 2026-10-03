"""identity history, media retries, bounded view de-duplication

- identity_aliases: every name a device's household has been known by (customer, site, camera, display name,
  host); labeler redaction uses their union. Backfilled from the current customers, devices, cameras, event
  cameras and heartbeats.
- index_problems.attempts / next_retry_at: transient media failures retry with backoff.
- ix_audit_log_staff_action_target_ts: the "viewed in the last 10 minutes?" lookups use a bounded index scan.

Revision ID: 0007
Revises: 0006
"""
from datetime import datetime, timezone

import sqlalchemy as sa
from alembic import op

revision = '0007'
down_revision = '0006'
branch_labels = None
depends_on = None


def _backfill(conn) -> None:
    from home_guard_project.fleet_contract.legacy import parse_heartbeat

    now = datetime.now(timezone.utc)
    rows: set[tuple[int, str, str]] = set()
    for pk, site, host, customer, heartbeat in conn.execute(sa.text(
            "SELECT d.id, d.site, d.tailscale_host, c.name, d.last_heartbeat "
            "FROM devices d JOIN customers c ON c.id = d.customer_id")):
        rows |= {(pk, "site", site), (pk, "host", host or ""), (pk, "customer", customer or "")}
        if isinstance(heartbeat, dict):
            hb = parse_heartbeat(heartbeat)
            rows.add((pk, "host", hb.host or ""))
            rows |= {(pk, "camera", name) for name in hb.cameras}
    for pk, name, shown in conn.execute(sa.text("SELECT device_pk, name, display_name FROM cameras")):
        rows |= {(pk, "camera", name or ""), (pk, "display_name", (shown or "").strip())}
    for pk, camera in conn.execute(sa.text("SELECT DISTINCT device_pk, camera FROM events")):
        rows.add((pk, "camera", camera or ""))
    values = [{"d": pk, "k": kind, "v": value, "t": now} for pk, kind, value in sorted(rows) if value]
    if values:
        conn.execute(sa.text("INSERT INTO identity_aliases (device_pk, kind, value, first_seen) "
                             "VALUES (:d, :k, :v, :t) ON CONFLICT DO NOTHING"), values)


def upgrade() -> None:
    op.create_table(
        'identity_aliases',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('device_pk', sa.Integer(), nullable=False),
        sa.Column('kind', sa.String(length=16), nullable=False),
        sa.Column('value', sa.Text(), nullable=False),
        sa.Column('first_seen', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['device_pk'], ['devices.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('device_pk', 'kind', 'value'),
    )
    op.create_index('ix_identity_aliases_device_pk', 'identity_aliases', ['device_pk'])
    op.add_column('index_problems', sa.Column('attempts', sa.Integer(), server_default=sa.text('0'), nullable=False))
    op.add_column('index_problems', sa.Column('next_retry_at', sa.DateTime(timezone=True), nullable=True))
    op.create_index('ix_audit_log_staff_action_target_ts', 'audit_log', ['staff_id', 'action', 'target', 'ts'])
    _backfill(op.get_bind())


def downgrade() -> None:
    op.drop_index('ix_audit_log_staff_action_target_ts', table_name='audit_log')
    op.drop_column('index_problems', 'next_retry_at')
    op.drop_column('index_problems', 'attempts')
    op.drop_index('ix_identity_aliases_device_pk', table_name='identity_aliases')
    op.drop_table('identity_aliases')
