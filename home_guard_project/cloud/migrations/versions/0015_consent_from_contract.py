"""consent comes from the sales contract: on for every customer, existing and new

customers.consent_source ('contract' | 'withdrawn'); the consent columns default to true; every existing customer
is switched on, with one audit row each ("consent from the sales contract").

Revision ID: 0015
Revises: 0014
"""
import sqlalchemy as sa
from alembic import op

revision = '0015'
down_revision = '0014'
branch_labels = None
depends_on = None

FIELDS = ('consent_live', 'consent_recordings', 'consent_training')


def upgrade() -> None:
    op.add_column('customers', sa.Column('consent_source', sa.String(length=16), server_default='contract',
                                         nullable=False))
    for field in FIELDS:
        op.alter_column('customers', field, server_default=sa.true())
    op.execute(sa.text(
        "INSERT INTO audit_log (ts, staff_id, staff_name, action, target, reason, customer_id, device_id, detail) "
        "SELECT now(), NULL, 'migration 0015', 'consent_from_contract', name, 'consent from the sales contract', id, "
        "NULL, jsonb_build_object('before', jsonb_build_object('live', consent_live, 'recordings', consent_recordings, "
        "'training', consent_training)) FROM customers"))
    op.execute(sa.text("UPDATE customers SET consent_live = true, consent_recordings = true, consent_training = true, "
                       "consent_source = 'contract'"))


def downgrade() -> None:
    for field in FIELDS:
        op.alter_column('customers', field, server_default=sa.false())
    op.drop_column('customers', 'consent_source')
