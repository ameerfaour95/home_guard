"""tagging studio: tag_events, one append-only row per save (category, observation fields, description, flags)

Revision ID: 0014
Revises: 0013
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '0014'
down_revision = '0013'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'tag_events',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('clip_key', sa.String(length=512), nullable=False),
        sa.Column('clip_id', sa.String(length=255), server_default='', nullable=False),
        sa.Column('fields', postgresql.JSONB(), nullable=False),
        sa.Column('staff_id', sa.Integer(), nullable=True),
        sa.Column('staff_name', sa.Text(), server_default='', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f('ix_tag_events_clip_key'), 'tag_events', ['clip_key'])


def downgrade() -> None:
    op.drop_index(op.f('ix_tag_events_clip_key'), table_name='tag_events')
    op.drop_table('tag_events')
