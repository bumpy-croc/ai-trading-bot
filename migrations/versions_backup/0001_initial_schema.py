"""Initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2025-07-06
"""

# revision identifiers, used by Alembic.
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Initial schema creation.

    NOTE:
      This migration previously called models.Base.metadata.create_all, which bypasses Alembic's
      migration tracking and can lead to drift. Regenerate this migration using:

          alembic revision --autogenerate -m "initial schema"

      and replace this file with the generated `op.create_table` statements.
    """
    pass


def downgrade() -> None:
    """Schema downgrade for initial revision.

    This should mirror the upgrade operations by dropping created tables and constraints.
    Regenerate via autogenerate as described above.
    """
    pass
