"""tagging studio: clip_annotations, append-only box versions of dataset clips (clips that are not indexed events)

Revision ID: 0016
Revises: 0015
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '0016'
down_revision = '0015'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'clip_annotations',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('clip_key', sa.String(length=512), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('tracks', postgresql.JSONB(), nullable=False),
        sa.Column('description', sa.Text(), server_default='', nullable=False),
        sa.Column('drop_clip', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('needs_review', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('author_id', sa.Integer(), nullable=True),
        sa.Column('author_name', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('clip_key', 'version'),
    )
    op.create_index(op.f('ix_clip_annotations_clip_key'), 'clip_annotations', ['clip_key'])


def downgrade() -> None:
    op.drop_index(op.f('ix_clip_annotations_clip_key'), table_name='clip_annotations')
    op.drop_table('clip_annotations')
