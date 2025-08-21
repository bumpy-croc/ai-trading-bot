"""Add MFE/MAE tracking columns to positions and trades

Revision ID: 0004_mfe_mae_tracking
Revises: 0003_dynamic_risk_tables
Create Date: 2025-08-21
"""

import sqlalchemy as sa  # type: ignore
from alembic import op  # type: ignore

revision = "0004_mfe_mae_tracking"
down_revision = "0003_dynamic_risk_tables"
branch_labels = None
depends_on = None


def upgrade():
    # Positions: add rolling MFE/MAE
    op.add_column("positions", sa.Column("mfe", sa.Numeric(18, 8), nullable=True))
    op.add_column("positions", sa.Column("mae", sa.Numeric(18, 8), nullable=True))
    op.add_column("positions", sa.Column("mfe_price", sa.Numeric(18, 8), nullable=True))
    op.add_column("positions", sa.Column("mae_price", sa.Numeric(18, 8), nullable=True))
    op.add_column("positions", sa.Column("mfe_time", sa.DateTime(), nullable=True))
    op.add_column("positions", sa.Column("mae_time", sa.DateTime(), nullable=True))

    # Trades: add final MFE/MAE recorded at completion
    op.add_column("trades", sa.Column("mfe", sa.Numeric(18, 8), nullable=True))
    op.add_column("trades", sa.Column("mae", sa.Numeric(18, 8), nullable=True))
    op.add_column("trades", sa.Column("mfe_price", sa.Numeric(18, 8), nullable=True))
    op.add_column("trades", sa.Column("mae_price", sa.Numeric(18, 8), nullable=True))
    op.add_column("trades", sa.Column("mfe_time", sa.DateTime(), nullable=True))
    op.add_column("trades", sa.Column("mae_time", sa.DateTime(), nullable=True))


def downgrade():
    # Trades
    op.drop_column("trades", "mae_time")
    op.drop_column("trades", "mfe_time")
    op.drop_column("trades", "mae_price")
    op.drop_column("trades", "mfe_price")
    op.drop_column("trades", "mae")
    op.drop_column("trades", "mfe")

    # Positions
    op.drop_column("positions", "mae_time")
    op.drop_column("positions", "mfe_time")
    op.drop_column("positions", "mae_price")
    op.drop_column("positions", "mfe_price")
    op.drop_column("positions", "mae")
    op.drop_column("positions", "mfe")