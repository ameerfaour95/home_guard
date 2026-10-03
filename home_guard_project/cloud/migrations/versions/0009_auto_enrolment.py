"""auto-enrolment of customer boxes from their S3 registration

customers.name_source ("admin" | "setup" | "discovered"), customers.owner_phone, customers.consent_recorded_utc;
devices.enrolled_by ("admin" | "setup" | "discovered"), devices.app_version. Existing rows were all made by an
admin.

Revision ID: 0009
Revises: 0008
"""
import sqlalchemy as sa
from alembic import op

revision = '0009'
down_revision = '0008'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('customers', sa.Column('name_source', sa.String(16), nullable=False, server_default='admin'))
    op.add_column('customers', sa.Column('owner_phone', sa.String(64), nullable=True))
    op.add_column('customers', sa.Column('consent_recorded_utc', sa.DateTime(timezone=True), nullable=True))
    op.add_column('devices', sa.Column('enrolled_by', sa.String(16), nullable=False, server_default='admin'))
    op.add_column('devices', sa.Column('app_version', sa.String(64), nullable=True))


def downgrade() -> None:
    op.drop_column('devices', 'app_version')
    op.drop_column('devices', 'enrolled_by')
    op.drop_column('customers', 'consent_recorded_utc')
    op.drop_column('customers', 'owner_phone')
    op.drop_column('customers', 'name_source')
