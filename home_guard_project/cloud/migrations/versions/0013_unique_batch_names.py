"""a tagging batch name is used once, forever: tagging_publishes.batch_name unique across all states

Existing duplicates (from the retired resume flow) keep the newest row under the name; older rows get `#<id>`
appended (a character a batch name can never contain), so no history is lost.

Revision ID: 0013
Revises: 0012
"""
import sqlalchemy as sa
from alembic import op

revision = '0013'
down_revision = '0012'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text(
        "UPDATE tagging_publishes t SET batch_name = t.batch_name || '#' || t.id "
        "WHERE EXISTS (SELECT 1 FROM tagging_publishes o WHERE o.batch_name = t.batch_name AND o.id > t.id)"))
    op.drop_index('ix_tagging_publishes_batch_name', table_name='tagging_publishes')
    op.create_index('ix_tagging_publishes_batch_name', 'tagging_publishes', ['batch_name'], unique=True)


def downgrade() -> None:
    op.drop_index('ix_tagging_publishes_batch_name', table_name='tagging_publishes')
    op.create_index('ix_tagging_publishes_batch_name', 'tagging_publishes', ['batch_name'])
