"""older identity history, private collections

- identity_aliases: backfilled from the history that predates 0007 -- customer names in the audit log
  (customer_create/customer_update targets, the old and new names of a logged rename), enrolment sites, raw JSON
  revisions (camera_name, site, prompt_camera_name, host, feedback and heartbeat cameras), indexed object cameras
  and a `camera_aliases` table if one exists (redact.legacy_names). Each device gets a `_scanned` marker row, so
  the runtime safety net (redact.ensure_history) does not read it again. Events of a device that gained names get
  their stored labeler search text cleared (refilled by the next index pass).
- collections.private_to_staff_id: set at creation for a collection a labeler made (their id), never changed
  later, so a role change cannot expose it. Backfilled from the creators who are labelers now.

Revision ID: 0008
Revises: 0007
"""
import sqlalchemy as sa
from alembic import op

revision = '0008'
down_revision = '0007'
branch_labels = None
depends_on = None


def upgrade() -> None:
    from home_guard_project.cloud.redact import scan_legacy

    op.add_column('collections', sa.Column('private_to_staff_id', sa.Integer(), nullable=True))
    conn = op.get_bind()
    conn.execute(sa.text("UPDATE collections SET private_to_staff_id = created_by "
                         "WHERE created_by IN (SELECT id FROM staff WHERE role = 'labeler')"))
    for (device_pk,) in conn.execute(sa.text("SELECT id FROM devices ORDER BY id")).all():
        scan_legacy(conn, device_pk)


def downgrade() -> None:
    op.execute("DELETE FROM identity_aliases WHERE kind = '_scanned'")  # a re-upgrade reads the history again
    op.drop_column('collections', 'private_to_staff_id')
