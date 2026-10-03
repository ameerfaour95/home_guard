"""owner notices: pending_upload marks a notice whose S3 object could not be written yet

Revision ID: 0004
Revises: 0003
"""
from alembic import op
import sqlalchemy as sa

revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('owner_notices', sa.Column('pending_upload', sa.Boolean(), nullable=True))


def downgrade() -> None:
    op.drop_column('owner_notices', 'pending_upload')
