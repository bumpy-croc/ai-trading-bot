"""Add trailing stop fields to positions

Revision ID: 0008_trailing_stops
Revises: 0006_partial_operations
Create Date: 2025-08-21
"""

import sqlalchemy as sa  # type: ignore
from alembic import op  # type: ignore

revision = "0008_trailing_stops"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade():
    # Add columns to positions table
    op.add_column("positions", sa.Column("trailing_stop_activated", sa.Boolean(), nullable=False, server_default=sa.text("false")))
    op.add_column("positions", sa.Column("trailing_stop_price", sa.Numeric(18, 8), nullable=True))
    op.add_column("positions", sa.Column("breakeven_triggered", sa.Boolean(), nullable=False, server_default=sa.text("false")))


def downgrade():
    # Remove columns from positions table
    op.drop_column("positions", "breakeven_triggered")
    op.drop_column("positions", "trailing_stop_price")
    op.drop_column("positions", "trailing_stop_activated")