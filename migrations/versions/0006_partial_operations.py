"""Add partial operations fields and partial_trades table

Revision ID: 0006_partial_operations
Revises: 0005_mfe_mae_tracking
Create Date: 2025-08-21
"""

import sqlalchemy as sa  # type: ignore
from alembic import op  # type: ignore
from sqlalchemy.engine import reflection  # type: ignore

revision = "0006_partial_operations"
down_revision = "0005_mfe_mae_tracking"
branch_labels = None
depends_on = None


def _has_column(table: str, column: str) -> bool:
    bind = op.get_bind()
    insp = reflection.Inspector.from_engine(bind)
    cols = [c["name"] for c in insp.get_columns(table)]
    return column in cols


def upgrade():
    # Add new columns to positions (idempotent checks)
    if not _has_column("positions", "original_size"):
        op.add_column("positions", sa.Column("original_size", sa.Numeric(18, 8)))
    if not _has_column("positions", "current_size"):
        op.add_column("positions", sa.Column("current_size", sa.Numeric(18, 8)))
    if not _has_column("positions", "partial_exits_taken"):
        op.add_column("positions", sa.Column("partial_exits_taken", sa.Integer(), nullable=True))
    if not _has_column("positions", "scale_ins_taken"):
        op.add_column("positions", sa.Column("scale_ins_taken", sa.Integer(), nullable=True))
    if not _has_column("positions", "last_partial_exit_price"):
        op.add_column("positions", sa.Column("last_partial_exit_price", sa.Numeric(18, 8)))
    if not _has_column("positions", "last_scale_in_price"):
        op.add_column("positions", sa.Column("last_scale_in_price", sa.Numeric(18, 8)))

    # Backfill existing positions: set original_size = size and current_size = size
    op.execute("UPDATE positions SET original_size = size WHERE original_size IS NULL")
    op.execute("UPDATE positions SET current_size = size WHERE current_size IS NULL")
    op.execute("UPDATE positions SET partial_exits_taken = COALESCE(partial_exits_taken, 0)")
    op.execute("UPDATE positions SET scale_ins_taken = COALESCE(scale_ins_taken, 0)")

    # Create enum for partial operation type
    op.execute("""
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'partialoperationtype') THEN
            CREATE TYPE partialoperationtype AS ENUM ('partial_exit', 'scale_in');
        END IF;
    END$$;
    """)

    # Create partial_trades table if not exists
    op.create_table(
        "partial_trades",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("position_id", sa.Integer(), sa.ForeignKey("positions.id"), nullable=False),
        sa.Column("operation_type", sa.Enum("partial_exit", "scale_in", name="partialoperationtype"), nullable=False),
        sa.Column("size", sa.Numeric(18, 8), nullable=False),
        sa.Column("price", sa.Numeric(18, 8), nullable=False),
        sa.Column("pnl", sa.Numeric(18, 8), nullable=True),
        sa.Column("target_level", sa.Integer(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_partial_trade_position", "partial_trades", ["position_id", "timestamp"])


def downgrade():
    op.drop_index("idx_partial_trade_position", table_name="partial_trades")
    op.drop_table("partial_trades")

    for col in [
        "last_scale_in_price",
        "last_partial_exit_price",
        "scale_ins_taken",
        "partial_exits_taken",
        "current_size",
        "original_size",
    ]:
        try:
            op.drop_column("positions", col)
        except Exception:
            pass