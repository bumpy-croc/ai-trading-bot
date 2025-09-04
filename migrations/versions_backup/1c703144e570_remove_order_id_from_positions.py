"""remove_order_id_from_positions

Revision ID: 1c703144e570
Revises: b673f791db9f
Create Date: 2025-09-04 12:03:39.624101

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1c703144e570'
down_revision = 'b673f791db9f'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Remove the unique constraint first
    op.drop_constraint("uq_position_order_session", "positions", type_="unique")

    # Drop the order_id column
    op.drop_column("positions", "order_id")


def downgrade() -> None:
    # Add back the order_id column
    op.add_column("positions", sa.Column("order_id", sa.String(100)))

    # Recreate the unique constraint
    op.create_unique_constraint("uq_position_order_session", "positions", ["order_id", "session_id"])
