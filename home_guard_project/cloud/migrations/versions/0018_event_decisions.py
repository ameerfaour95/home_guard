"""events, not only clips: each clip's event-layer decision (session, sent / held and why, baseline shadow)

Additive: events.session_id (the box's event session, box/events.py), events.decision (fleet_contract.event_outcome
decision_of: what the box did with the clip, JSONB) and events.would_raise (the baseline in shadow mode would have
raised it). NULL decision means "not parsed yet": the indexer backfills those from stored meta revisions (no S3 GET).

Numbered after admin-studio's 0017 (inbox). Until that lands on this branch it revises 0016; when both are merged,
down_revision becomes '0017' (one line) so the chain stays linear.

Revision ID: 0018
Revises: 0017
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '0018'
down_revision = '0017'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('events', sa.Column('session_id', sa.String(length=64), nullable=True))
    op.add_column('events', sa.Column('decision', postgresql.JSONB(), nullable=True))
    op.add_column('events', sa.Column('would_raise', sa.Boolean(), nullable=True))
    op.create_index('ix_events_device_session', 'events', ['device_pk', 'session_id'])


def downgrade() -> None:
    op.drop_index('ix_events_device_session', table_name='events')
    op.drop_column('events', 'would_raise')
    op.drop_column('events', 'decision')
    op.drop_column('events', 'session_id')
