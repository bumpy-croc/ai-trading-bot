"""Add order tracking columns to positions table

Revision ID: 0006_position_order_tracking
Revises: 0005_enhanced_performance
Create Date: 2025-12-29 22:00:00.000000

Adds columns to track exchange order IDs for positions:
- entry_order_id: Exchange order ID for the entry order
- stop_loss_order_id: Exchange order ID for server-side stop-loss

These support real Binance order execution with server-side stop-losses (PR #451).
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0006_position_order_tracking"
down_revision = "0005_enhanced_performance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add order tracking columns to positions table."""
    op.add_column(
        "positions",
        sa.Column("entry_order_id", sa.String(100), nullable=True),
    )
    op.add_column(
        "positions",
        sa.Column("stop_loss_order_id", sa.String(100), nullable=True),
    )


def downgrade() -> None:
    """Remove order tracking columns from positions table."""
    op.drop_column("positions", "stop_loss_order_id")
    op.drop_column("positions", "entry_order_id")
