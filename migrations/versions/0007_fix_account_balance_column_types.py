"""Fix account_balances column types from Float to Numeric(18,8)

Revision ID: 0007_fix_balance_types
Revises: 0006_position_order_tracking
Create Date: 2026-02-28 09:00:00.000000

The initial migration (0001) created account_balances columns as Float
(doubleprecision), but the model was later updated to Numeric(18,8) for
financial precision. This migration aligns the database with the model.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0007_fix_balance_types"
down_revision = "0006_position_order_tracking"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "account_balances",
        "total_balance",
        existing_type=sa.Float(),
        type_=sa.Numeric(18, 8),
        existing_nullable=False,
    )
    op.alter_column(
        "account_balances",
        "available_balance",
        existing_type=sa.Float(),
        type_=sa.Numeric(18, 8),
        existing_nullable=False,
    )
    op.alter_column(
        "account_balances",
        "reserved_balance",
        existing_type=sa.Float(),
        type_=sa.Numeric(18, 8),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "account_balances",
        "total_balance",
        existing_type=sa.Numeric(18, 8),
        type_=sa.Float(),
        existing_nullable=False,
    )
    op.alter_column(
        "account_balances",
        "available_balance",
        existing_type=sa.Numeric(18, 8),
        type_=sa.Float(),
        existing_nullable=False,
    )
    op.alter_column(
        "account_balances",
        "reserved_balance",
        existing_type=sa.Numeric(18, 8),
        type_=sa.Float(),
        existing_nullable=True,
    )
