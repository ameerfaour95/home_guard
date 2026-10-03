"""hardening: truncate guard, keyset indexes, server defaults, staff_name, totp counter, session cap

Revision ID: 0002
Revises: 0001
"""
from alembic import op
import sqlalchemy as sa

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None

JSON_OBJ = sa.text("'{}'::jsonb")
JSON_ARR = sa.text("'[]'::jsonb")

# (table, column, server default, type) for NOT NULL columns that have Python-side defaults
DEFAULTS = [
    ("staff", "disabled", sa.false(), sa.Boolean()),
    ("refresh_tokens", "family", "", sa.String(64)),
    ("customers", "timezone", "Asia/Jerusalem", sa.String(64)),
    ("customers", "consent_live", sa.false(), sa.Boolean()),
    ("customers", "consent_recordings", sa.false(), sa.Boolean()),
    ("customers", "consent_training", sa.false(), sa.Boolean()),
    ("customers", "notes", "", sa.Text()),
    ("devices", "tailscale_host", "", sa.String(255)),
    ("devices", "ssh_user", "ameer", sa.String(64)),
    ("events", "kind", "unknown", sa.String(32)),
    ("events", "summary", "", sa.Text()),
    ("events", "alert_reason", "", sa.Text()),
    ("events", "completeness", JSON_OBJ, None),
    ("artifacts", "available", sa.true(), sa.Boolean()),
    ("artifacts", "provenance", "box", sa.String(16)),
    ("ai_runs", "purpose", "guard", sa.String(32)),
    ("ai_runs", "status", "none", sa.String(16)),
    ("ai_runs", "input_artifact_ids", JSON_ARR, None),
    ("feedback", "verdict", "", sa.String(64)),
    ("feedback", "action", "", sa.String(64)),
    ("feedback", "note", "", sa.Text()),
    ("feedback", "raw_text", "", sa.Text()),
    ("feedback", "source", "", sa.String(64)),
    ("review_state", "reviewed", sa.false(), sa.Boolean()),
    ("review_state", "flagged", sa.false(), sa.Boolean()),
    ("collections", "description", "", sa.Text()),
    ("exports", "version", sa.text("1"), sa.Integer()),
    ("exports", "state", "queued", sa.String(16)),
    ("exports", "item_count", sa.text("0"), sa.Integer()),
    ("exports", "s3_prefix", "", sa.String(1024)),
    ("exports", "request", JSON_OBJ, None),
    ("audit_log", "target", "", sa.String(1024)),
    ("audit_log", "reason", "", sa.Text()),
    ("owner_notices", "cameras", JSON_ARR, None),
]


def upgrade() -> None:
    op.execute("""
    CREATE TRIGGER audit_log_no_truncate BEFORE TRUNCATE ON audit_log
    FOR EACH STATEMENT EXECUTE FUNCTION audit_log_reject()
    """)
    op.drop_index('ix_events_device_start', table_name='events')
    op.drop_index('ix_events_camera_start', table_name='events')
    op.create_index('ix_events_device_start', 'events', ['device_pk', sa.text('start_ts DESC'), sa.text('id DESC')])
    op.create_index('ix_events_camera_start', 'events', ['camera', sa.text('start_ts DESC'), sa.text('id DESC')])
    op.add_column('audit_log', sa.Column('staff_name', sa.Text(), nullable=True))
    op.create_index('ix_audit_log_action_ts', 'audit_log', ['action', 'ts'])
    op.add_column('staff', sa.Column('totp_last_counter', sa.BigInteger(), nullable=True))
    op.add_column('refresh_tokens', sa.Column('family_started_at', sa.DateTime(timezone=True), nullable=True))
    for table, col, default, _ in DEFAULTS:
        op.alter_column(table, col, server_default=default)


def downgrade() -> None:
    for table, col, _, _ in DEFAULTS:
        op.alter_column(table, col, server_default=None)
    op.drop_column('refresh_tokens', 'family_started_at')
    op.drop_column('staff', 'totp_last_counter')
    op.drop_index('ix_audit_log_action_ts', table_name='audit_log')
    op.drop_column('audit_log', 'staff_name')
    op.drop_index('ix_events_camera_start', table_name='events')
    op.drop_index('ix_events_device_start', table_name='events')
    op.create_index('ix_events_device_start', 'events', ['device_pk', 'start_ts'])
    op.create_index('ix_events_camera_start', 'events', ['camera', 'start_ts'])
    op.execute("DROP TRIGGER IF EXISTS audit_log_no_truncate ON audit_log")
