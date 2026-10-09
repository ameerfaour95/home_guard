"""superseded tags: a retag supersedes the clip's earlier tag (box feedback.supersede_tags writes superseded_by, the
new record's file name, and superseded_utc into the older file and its kept copy): feedback.superseded_by / _at

It first makes sure 0017's schema is there, idempotently: a database that went 0016 -> 0018 -> 0019 before 0017
existed on admin-console was stamped 0019 without ever running 0017 (the founder's laptop DB, repaired by hand on
2026-10-09: /tagging/queue and /inbox answered 500 "column feedback.owner_label does not exist"). Safe to re-run.

Revision ID: 0020
Revises: 0019
"""
import sqlalchemy as sa
from alembic import op

revision = '0020'
down_revision = '0019'
branch_labels = None
depends_on = None


def ensure_0017() -> None:
    """0017 (the inbox) once more, only what is missing: its four feedback columns, inbox_decisions as 0017 made it,
    and the backfill of the rows whose owner_label is still empty."""
    for column in ("owner_label VARCHAR(32) NOT NULL DEFAULT ''", "owner_text TEXT NOT NULL DEFAULT ''",
                   "transcript TEXT NOT NULL DEFAULT ''", "tagged_by VARCHAR(128) NOT NULL DEFAULT ''"):
        op.execute(f"ALTER TABLE feedback ADD COLUMN IF NOT EXISTS {column}")
    op.execute("""
        CREATE TABLE IF NOT EXISTS inbox_decisions (
            feedback_id INTEGER NOT NULL PRIMARY KEY REFERENCES feedback (id) ON DELETE CASCADE,
            decision VARCHAR(16) NOT NULL,
            note TEXT NOT NULL DEFAULT '',
            staff_id INTEGER REFERENCES staff (id) ON DELETE SET NULL,
            staff_name TEXT,
            decided_at TIMESTAMP WITH TIME ZONE NOT NULL,
            prompt_version VARCHAR(128)
        )
    """)
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
        WHERE r.s3_key = f.s3_key AND jsonb_typeof(r.body) = 'object' AND f.owner_label = ''
    """)


def upgrade() -> None:
    ensure_0017()
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
