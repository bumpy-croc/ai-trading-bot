"""Add MFE/MAE tracking columns to positions and trades

Revision ID: 0005_mfe_mae_tracking
Revises: 0004_time_exits
Create Date: 2025-08-21
"""

import sqlalchemy as sa  # type: ignore
from alembic import op  # type: ignore
from sqlalchemy.engine import reflection  # type: ignore

revision = "0005_mfe_mae_tracking"
down_revision = "0004_time_exits"
branch_labels = None
depends_on = None


def _has_column(table: str, column: str) -> bool:
    """Check if a column exists in the given table."""
    bind = op.get_bind()
    insp = reflection.Inspector.from_engine(bind)
    cols = [c["name"] for c in insp.get_columns(table)]
    return column in cols


def upgrade():
    # Positions: add rolling MFE/MAE (idempotent checks)
    if not _has_column("positions", "mfe"):
        op.add_column("positions", sa.Column("mfe", sa.Numeric(18, 8), nullable=True))
    if not _has_column("positions", "mae"):
        op.add_column("positions", sa.Column("mae", sa.Numeric(18, 8), nullable=True))
    if not _has_column("positions", "mfe_price"):
        op.add_column("positions", sa.Column("mfe_price", sa.Numeric(18, 8), nullable=True))
    if not _has_column("positions", "mae_price"):
        op.add_column("positions", sa.Column("mae_price", sa.Numeric(18, 8), nullable=True))
    if not _has_column("positions", "mfe_time"):
        op.add_column("positions", sa.Column("mfe_time", sa.DateTime(), nullable=True))
    if not _has_column("positions", "mae_time"):
        op.add_column("positions", sa.Column("mae_time", sa.DateTime(), nullable=True))

    # Trades: add final MFE/MAE recorded at completion (idempotent checks)
    if not _has_column("trades", "mfe"):
        op.add_column("trades", sa.Column("mfe", sa.Numeric(18, 8), nullable=True))
    if not _has_column("trades", "mae"):
        op.add_column("trades", sa.Column("mae", sa.Numeric(18, 8), nullable=True))
    if not _has_column("trades", "mfe_price"):
        op.add_column("trades", sa.Column("mfe_price", sa.Numeric(18, 8), nullable=True))
    if not _has_column("trades", "mae_price"):
        op.add_column("trades", sa.Column("mae_price", sa.Numeric(18, 8), nullable=True))
    if not _has_column("trades", "mfe_time"):
        op.add_column("trades", sa.Column("mfe_time", sa.DateTime(), nullable=True))
    if not _has_column("trades", "mae_time"):
        op.add_column("trades", sa.Column("mae_time", sa.DateTime(), nullable=True))


def downgrade():
    # Trades (idempotent checks)
    if _has_column("trades", "mae_time"):
        op.drop_column("trades", "mae_time")
    if _has_column("trades", "mfe_time"):
        op.drop_column("trades", "mfe_time")
    if _has_column("trades", "mae_price"):
        op.drop_column("trades", "mae_price")
    if _has_column("trades", "mfe_price"):
        op.drop_column("trades", "mfe_price")
    if _has_column("trades", "mae"):
        op.drop_column("trades", "mae")
    if _has_column("trades", "mfe"):
        op.drop_column("trades", "mfe")

    # Positions (idempotent checks)
    if _has_column("positions", "mae_time"):
        op.drop_column("positions", "mae_time")
    if _has_column("positions", "mfe_time"):
        op.drop_column("positions", "mfe_time")
    if _has_column("positions", "mae_price"):
        op.drop_column("positions", "mae_price")
    if _has_column("positions", "mfe_price"):
        op.drop_column("positions", "mfe_price")
    if _has_column("positions", "mae"):
        op.drop_column("positions", "mae")
    if _has_column("positions", "mfe"):
        op.drop_column("positions", "mfe")