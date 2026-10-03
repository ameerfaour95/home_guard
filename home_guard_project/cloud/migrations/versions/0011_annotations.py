"""in-app labeling: annotations (append-only versions), annotation_heads, annotation_reviews, tagging_publishes

Revision ID: 0011
Revises: 0010
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '0011'
down_revision = '0010'
branch_labels = None
depends_on = None

TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        'annotations',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('event_id', sa.Integer(), sa.ForeignKey('events.id', ondelete='CASCADE'), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('tracks', postgresql.JSONB(), nullable=False),
        sa.Column('description', sa.Text(), server_default='', nullable=False),
        sa.Column('ai_description', sa.Text(), server_default='', nullable=False),
        sa.Column('ai_run_id', sa.Integer(), sa.ForeignKey('ai_runs.id', ondelete='SET NULL'), nullable=True),
        sa.Column('drop_clip', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('needs_review', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('suggestions_used', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('author_id', sa.Integer(), nullable=True),
        sa.Column('author_name', sa.Text(), nullable=True),
        sa.Column('created_at', TS, nullable=False),
        sa.UniqueConstraint('event_id', 'version'),
    )
    op.create_index(op.f('ix_annotations_event_id'), 'annotations', ['event_id'])
    op.create_table(
        'annotation_heads',
        sa.Column('event_id', sa.Integer(), sa.ForeignKey('events.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('needs_review', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('drop_clip', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('updated_at', TS, nullable=False),
    )
    op.create_index('ix_annotation_heads_status', 'annotation_heads', ['status'])
    op.create_table(
        'annotation_reviews',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('event_id', sa.Integer(), sa.ForeignKey('events.id', ondelete='CASCADE'), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('decision', sa.String(length=16), nullable=False),
        sa.Column('note', sa.Text(), server_default='', nullable=False),
        sa.Column('frame', sa.Integer(), nullable=True),
        sa.Column('reviewer_id', sa.Integer(), nullable=True),
        sa.Column('reviewer_name', sa.Text(), nullable=True),
        sa.Column('created_at', TS, nullable=False),
    )
    op.create_index(op.f('ix_annotation_reviews_event_id'), 'annotation_reviews', ['event_id'])
    op.create_table(
        'tagging_publishes',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('batch_name', sa.String(length=128), nullable=False),
        sa.Column('collection_id', sa.Integer(), sa.ForeignKey('collections.id', ondelete='SET NULL'),
                  nullable=True),
        sa.Column('state', sa.String(length=16), server_default='queued', nullable=False),
        sa.Column('s3_prefix', sa.String(length=1024), server_default='', nullable=False),
        sa.Column('tasks', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('yolo_frames', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('vlm_lines', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('missing', postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column('snapshot', postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_by', sa.Integer(), sa.ForeignKey('staff.id'), nullable=False),
        sa.Column('created_at', TS, nullable=False),
        sa.Column('worker_id', sa.String(length=64), nullable=True),
        sa.Column('heartbeat_at', TS, nullable=True),
    )
    op.create_index('ix_tagging_publishes_batch_name', 'tagging_publishes', ['batch_name'])


def downgrade() -> None:
    op.drop_index('ix_tagging_publishes_batch_name', table_name='tagging_publishes')
    op.drop_table('tagging_publishes')
    op.drop_index(op.f('ix_annotation_reviews_event_id'), table_name='annotation_reviews')
    op.drop_table('annotation_reviews')
    op.drop_index('ix_annotation_heads_status', table_name='annotation_heads')
    op.drop_table('annotation_heads')
    op.drop_index(op.f('ix_annotations_event_id'), table_name='annotations')
    op.drop_table('annotations')
