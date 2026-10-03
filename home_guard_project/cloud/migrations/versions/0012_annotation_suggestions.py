"""precomputed weak-label suggestions: annotation_suggestions

Revision ID: 0012
Revises: 0011
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '0012'
down_revision = '0011'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'annotation_suggestions',
        sa.Column('event_id', sa.Integer(), sa.ForeignKey('events.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('tracks', postgresql.JSONB(), nullable=False),
        sa.Column('sources', postgresql.JSONB(), nullable=False),
        sa.Column('computed_at', sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('annotation_suggestions')
