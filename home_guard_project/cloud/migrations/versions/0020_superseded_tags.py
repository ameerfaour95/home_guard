"""superseded tags: a retag supersedes the clip's earlier tag (box feedback.supersede_tags writes superseded_by, the
new record's file name, and superseded_utc into the older file and its kept copy): feedback.superseded_by / _at

Revision ID: 0020
Revises: 0019
"""
import sqlalchemy as sa
from alembic import op

revision = '0020'
down_revision = '0019'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('feedback', sa.Column('superseded_by', sa.String(length=255), server_default='', nullable=False))
    op.add_column('feedback', sa.Column('superseded_at', sa.DateTime(timezone=True), nullable=True))
    # answers indexed before 0020: from the stored body of the newest revision
    op.execute("""
        UPDATE feedback AS f SET superseded_by = LEFT(COALESCE(r.body->>'superseded_by', ''), 255)
        FROM (SELECT DISTINCT ON (s3_key) s3_key, body FROM raw_revisions
              WHERE s3_key LIKE '%.feedback.json' ORDER BY s3_key, fetched_at DESC, id DESC) AS r
        WHERE r.s3_key = f.s3_key AND jsonb_typeof(r.body) = 'object' AND COALESCE(r.body->>'superseded_by', '') <> ''
    """)


def downgrade() -> None:
    op.drop_column('feedback', 'superseded_at')
    op.drop_column('feedback', 'superseded_by')
