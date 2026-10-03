"""export leases: exports.worker_id and exports.heartbeat_at

A worker claims a queued export atomically (state queued -> running, worker_id set) and refreshes heartbeat_at
while it builds; the periodic sweep fails running exports whose heartbeat stopped.

Revision ID: 0006
Revises: 0005
"""
from alembic import op
import sqlalchemy as sa

revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('exports', sa.Column('worker_id', sa.String(length=64), nullable=True))
    op.add_column('exports', sa.Column('heartbeat_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column('exports', 'heartbeat_at')
    op.drop_column('exports', 'worker_id')
