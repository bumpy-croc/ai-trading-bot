"""update_positions_to_use_positionstatus

Revision ID: 5dd66bc421f8
Revises: 0013_add_order_table
Create Date: 2025-09-02 21:41:42.310634

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "5dd66bc421f8"
down_revision = "0013_add_order_table"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Update positions table to use positionstatus enum instead of orderstatus
    # Since we're nuking the database anyway, this is a clean migration

    # Drop existing foreign key constraints and columns if they exist
    op.execute("ALTER TABLE positions DROP CONSTRAINT IF EXISTS positions_status_fkey")

    # Change the status column to use positionstatus enum
    op.execute(
        """
        ALTER TABLE positions
        ALTER COLUMN status TYPE positionstatus
        USING CASE
            WHEN status = 'OPEN' THEN 'OPEN'::positionstatus
            WHEN status = 'CLOSED' THEN 'CLOSED'::positionstatus
            ELSE 'OPEN'::positionstatus
        END
    """
    )


def downgrade() -> None:
    # Revert positions table to use orderstatus enum
    op.execute("ALTER TABLE positions DROP CONSTRAINT IF EXISTS positions_status_fkey")

    # Change back to orderstatus enum
    op.execute(
        """
        ALTER TABLE positions
        ALTER COLUMN status TYPE orderstatus
        USING CASE
            WHEN status = 'OPEN' THEN 'OPEN'::orderstatus
            WHEN status = 'CLOSED' THEN 'CLOSED'::orderstatus
            ELSE 'OPEN'::orderstatus
        END
    """
    )
