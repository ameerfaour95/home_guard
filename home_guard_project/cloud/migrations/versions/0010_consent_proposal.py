"""box-written consent is a proposal until an admin confirms it

customers.consent_proposed (jsonb): {live, recordings, training, recorded_utc, installer}. Stored registration
bodies in raw_revisions lose their owner_phone (the phone lives on the customer only).

Revision ID: 0010
Revises: 0009
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '0010'
down_revision = '0009'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('customers', sa.Column('consent_proposed', postgresql.JSONB(), nullable=True))
    op.execute("UPDATE raw_revisions SET body = body - 'owner_phone' "
               "WHERE s3_key LIKE '%/_status/registration.json' AND body ? 'owner_phone'")


def downgrade() -> None:
    op.drop_column('customers', 'consent_proposed')
