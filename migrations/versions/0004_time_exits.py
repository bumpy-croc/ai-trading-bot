"""time exits core schema

Revision ID: 0004_time_exits
Revises: 0003_dynamic_risk_tables
Create Date: 2025-08-21 00:00:00.000000
"""

from __future__ import annotations

from alembic import op  # type: ignore
import sqlalchemy as sa  # type: ignore


# revision identifiers, used by Alembic.
revision = "0004_time_exits"
down_revision = "0003_dynamic_risk_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add columns to positions
    with op.batch_alter_table("positions") as batch_op:
        batch_op.add_column(sa.Column("max_holding_until", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("end_of_day_exit", sa.Boolean(), nullable=True, server_default=sa.false()))
        batch_op.add_column(sa.Column("weekend_exit", sa.Boolean(), nullable=True, server_default=sa.false()))
        batch_op.add_column(sa.Column("time_restriction_group", sa.String(length=50), nullable=True))

    # Add columns to trading_sessions
    with op.batch_alter_table("trading_sessions") as batch_op:
        batch_op.add_column(sa.Column("time_exit_config", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("market_timezone", sa.String(length=50), nullable=True))

    # Create market_sessions table
    op.create_table(
        "market_sessions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=50), nullable=False, unique=True, index=True),
        sa.Column("timezone", sa.String(length=50), nullable=False, server_default="UTC"),
        sa.Column("open_time", sa.Time(), nullable=True),
        sa.Column("close_time", sa.Time(), nullable=True),
        sa.Column("days_of_week", sa.JSON(), nullable=True),
        sa.Column("is_24h", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    # Drop market_sessions table
    op.drop_table("market_sessions")

    # Remove columns from trading_sessions
    with op.batch_alter_table("trading_sessions") as batch_op:
        batch_op.drop_column("market_timezone")
        batch_op.drop_column("time_exit_config")

    # Remove columns from positions
    with op.batch_alter_table("positions") as batch_op:
        batch_op.drop_column("time_restriction_group")
        batch_op.drop_column("weekend_exit")
        batch_op.drop_column("end_of_day_exit")
        batch_op.drop_column("max_holding_until")

