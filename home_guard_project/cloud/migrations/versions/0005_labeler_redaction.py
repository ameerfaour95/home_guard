"""labeler redaction: events.summary_redacted and its search vector

The redacted text needs each device's identity terms, which only the application knows, so the migration adds
empty columns; `manage redact-backfill` (and every indexer pass) fills them.

Revision ID: 0005
Revises: 0004
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0005'
down_revision = '0004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('events', sa.Column('summary_redacted', sa.Text(), nullable=True))
    op.add_column('events', sa.Column(
        'search_redacted', postgresql.TSVECTOR(),
        sa.Computed("to_tsvector('simple', coalesce(summary_redacted,''))", persisted=True), nullable=True))
    op.create_index('ix_events_search_redacted', 'events', ['search_redacted'], unique=False,
                    postgresql_using='gin')


def downgrade() -> None:
    op.drop_index('ix_events_search_redacted', table_name='events', postgresql_using='gin')
    op.drop_column('events', 'search_redacted')
    op.drop_column('events', 'summary_redacted')
