"""inbox: the owner's Telegram tag, words, transcript and tagger on feedback; inbox_decisions (what an admin did with
each answer)

Revision ID: 0017
Revises: 0016
"""
import sqlalchemy as sa
from alembic import op

revision = '0017'
down_revision = '0016'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('feedback', sa.Column('owner_label', sa.String(length=32), server_default='', nullable=False))
    op.add_column('feedback', sa.Column('owner_text', sa.Text(), server_default='', nullable=False))
    op.add_column('feedback', sa.Column('transcript', sa.Text(), server_default='', nullable=False))
    op.add_column('feedback', sa.Column('tagged_by', sa.String(length=128), server_default='', nullable=False))
    op.create_table(
        'inbox_decisions',
        sa.Column('feedback_id', sa.Integer(), sa.ForeignKey('feedback.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('decision', sa.String(length=16), nullable=False),
        sa.Column('note', sa.Text(), server_default='', nullable=False),
        sa.Column('staff_id', sa.Integer(), sa.ForeignKey('staff.id', ondelete='SET NULL'), nullable=True),
        sa.Column('staff_name', sa.Text(), nullable=True),
        sa.Column('decided_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('prompt_version', sa.String(length=128), nullable=True),
    )
    # answers indexed before 0017: their fields from the stored body of the newest revision (the indexer re-applies a
    # feedback file only when it changes)
    op.execute("""
        UPDATE feedback AS f SET
            owner_label = LEFT(COALESCE(r.body->>'owner_label', ''), 32),
            owner_text = COALESCE(r.body->>'owner_text', ''),
            transcript = COALESCE(r.body->>'transcript', ''),
            tagged_by = LEFT(COALESCE(NULLIF(r.body->>'tagged_by', ''),
                                      CASE jsonb_typeof(r.body->'from') WHEN 'string' THEN r.body->>'from'
                                           WHEN 'object' THEN r.body->'from'->>'name' END, ''), 128)
        FROM (SELECT DISTINCT ON (s3_key) s3_key, body FROM raw_revisions
              WHERE s3_key LIKE '%.feedback.json' ORDER BY s3_key, fetched_at DESC, id DESC) AS r
        WHERE r.s3_key = f.s3_key AND jsonb_typeof(r.body) = 'object'
    """)


def downgrade() -> None:
    op.drop_table('inbox_decisions')
    for name in ('tagged_by', 'transcript', 'owner_text', 'owner_label'):
        op.drop_column('feedback', name)
