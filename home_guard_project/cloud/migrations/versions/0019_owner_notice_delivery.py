"""owner notices are pushed to the box over Tailscale SSH (cloud/notice_delivery.py): the delivery state of each

Additive, all nullable on owner_notices: delivery_state (pending | delivered | failed | gave_up; NULL: a notice from
before push delivery, never pushed), delivery_since (when the current body started waiting: 7 days, then gave_up),
delivery_attempts, delivery_heartbeat_at (the box's newest heartbeat at the last attempt: a retry waits for a newer
one), last_delivery_error and delivery_body_sha (sha256 of the encoded body the state is about: a body the box
refused is never re-sent, one already delivered is not re-pushed).

Revision ID: 0019
Revises: 0018
"""
import sqlalchemy as sa
from alembic import op

revision = '0019'
down_revision = '0018'
branch_labels = None
depends_on = None

_TS = sa.DateTime(timezone=True)
_COLUMNS = (
    ('delivery_state', sa.String(length=16)),
    ('delivery_since', _TS),
    ('delivery_attempts', sa.Integer()),
    ('delivery_heartbeat_at', _TS),
    ('last_delivery_error', sa.String(length=255)),
    ('delivery_body_sha', sa.String(length=64)),
)


def upgrade() -> None:
    for name, type_ in _COLUMNS:
        op.add_column('owner_notices', sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    for name, _ in reversed(_COLUMNS):
        op.drop_column('owner_notices', name)
