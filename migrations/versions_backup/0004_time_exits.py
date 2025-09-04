"""time exits core schema

Revision ID: 0004_time_exits
Revises: 0003_dynamic_risk_tables
Create Date: 2025-08-21 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa  # type: ignore
from alembic import op  # type: ignore
from sqlalchemy.engine import reflection  # type: ignore

# revision identifiers, used by Alembic.
revision = "0004_time_exits"
down_revision = "0003_dynamic_risk_tables"
branch_labels = None
depends_on = None


def _has_column(table: str, column: str) -> bool:
    """Check if a column exists in the given table."""
    bind = op.get_bind()
    insp = reflection.Inspector.from_engine(bind)
    cols = [c["name"] for c in insp.get_columns(table)]
    return column in cols


def upgrade() -> None:
    # Add columns to positions (idempotent checks)
    with op.batch_alter_table("positions") as batch_op:
        if not _has_column("positions", "max_holding_until"):
            batch_op.add_column(sa.Column("max_holding_until", sa.DateTime(), nullable=True))
        if not _has_column("positions", "end_of_day_exit"):
            batch_op.add_column(sa.Column("end_of_day_exit", sa.Boolean(), nullable=True, server_default=sa.false()))
        if not _has_column("positions", "weekend_exit"):
            batch_op.add_column(sa.Column("weekend_exit", sa.Boolean(), nullable=True, server_default=sa.false()))
        if not _has_column("positions", "time_restriction_group"):
            batch_op.add_column(sa.Column("time_restriction_group", sa.String(length=50), nullable=True))

    # Add columns to trading_sessions (idempotent checks)
    with op.batch_alter_table("trading_sessions") as batch_op:
        if not _has_column("trading_sessions", "time_exit_config"):
            batch_op.add_column(sa.Column("time_exit_config", sa.JSON(), nullable=True))
        if not _has_column("trading_sessions", "market_timezone"):
            batch_op.add_column(sa.Column("market_timezone", sa.String(length=50), nullable=True))

    # Create market_sessions table (idempotent)
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("market_sessions"):
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
    # Drop market_sessions table if it exists
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("market_sessions"):
        op.drop_table("market_sessions")

    # Remove columns from trading_sessions (idempotent checks)
    with op.batch_alter_table("trading_sessions") as batch_op:
        if _has_column("trading_sessions", "market_timezone"):
            batch_op.drop_column("market_timezone")
        if _has_column("trading_sessions", "time_exit_config"):
            batch_op.drop_column("time_exit_config")

    # Remove columns from positions (idempotent checks)
    with op.batch_alter_table("positions") as batch_op:
        if _has_column("positions", "time_restriction_group"):
            batch_op.drop_column("time_restriction_group")
        if _has_column("positions", "weekend_exit"):
            batch_op.drop_column("weekend_exit")
        if _has_column("positions", "end_of_day_exit"):
            batch_op.drop_column("end_of_day_exit")
        if _has_column("positions", "max_holding_until"):
            batch_op.drop_column("max_holding_until")

