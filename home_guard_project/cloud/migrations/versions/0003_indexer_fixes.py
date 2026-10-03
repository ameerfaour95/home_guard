"""indexer fixes: production-only muted, AI source revision, applied revision per artifact, problem etag, lookups

Revision ID: 0003
Revises: 0002
"""
from alembic import op
import sqlalchemy as sa

revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Event.muted: from the production copy only; NULL = unknown (no production copy).
    op.add_column('events', sa.Column('muted', sa.Boolean(), nullable=True))
    # The meta revision that supplied the AI run (and the event's summary/label/command).
    op.add_column('ai_runs', sa.Column('ai_source_key', sa.String(length=1024), nullable=True))
    op.add_column('ai_runs', sa.Column('ai_source_etag', sa.String(length=128), nullable=True))
    # Artifact.etag is what S3 lists now; applied_etag is the revision whose content is in effect.
    op.add_column('artifacts', sa.Column('applied_etag', sa.String(length=128), nullable=True))
    op.add_column('artifacts', sa.Column('etag_mismatches', sa.Integer(), nullable=False,
                                         server_default=sa.text('0')))
    # The revision a problem is about; it clears only when that key's current revision applies.
    op.add_column('index_problems', sa.Column('etag', sa.String(length=128), nullable=True))
    # Device-scoped bulk loads select by key prefix; camera/stem joins attach media to events.
    op.create_index('ix_artifacts_s3_key_prefix', 'artifacts', ['s3_key'],
                    postgresql_ops={'s3_key': 'text_pattern_ops'})
    op.create_index('ix_artifacts_camera_stem', 'artifacts', ['camera', 'stem'])
    op.create_index('ix_index_problems_s3_key_prefix', 'index_problems', ['s3_key'],
                    postgresql_ops={'s3_key': 'text_pattern_ops'})


def downgrade() -> None:
    op.drop_index('ix_index_problems_s3_key_prefix', table_name='index_problems')
    op.drop_index('ix_artifacts_camera_stem', table_name='artifacts')
    op.drop_index('ix_artifacts_s3_key_prefix', table_name='artifacts')
    op.drop_column('index_problems', 'etag')
    op.drop_column('artifacts', 'etag_mismatches')
    op.drop_column('artifacts', 'applied_etag')
    op.drop_column('ai_runs', 'ai_source_etag')
    op.drop_column('ai_runs', 'ai_source_key')
    op.drop_column('events', 'muted')
