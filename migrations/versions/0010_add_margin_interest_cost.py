"""Add margin_interest_cost column to trades table

Revision ID: 0010_margin_interest_cost
Revises: 0009_widen_order_status
Create Date: 2026-03-31 22:00:00.000000

Tracks cumulative margin borrow interest deducted from realized PnL
when closing short positions. Defaults to 0 for spot/long trades.
"""

import sqlalchemy as sa
from alembic import op

revision = "0010_margin_interest_cost"
down_revision = "0009_widen_order_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "trades",
        sa.Column(
            "margin_interest_cost",
            sa.Numeric(precision=18, scale=8),
            server_default="0",
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("trades", "margin_interest_cost")
