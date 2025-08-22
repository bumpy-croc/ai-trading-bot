"""Add correlation_matrix and portfolio_exposures tables

Revision ID: 0004_correlation_tables
Revises: 0003_dynamic_risk_tables
Create Date: 2025-08-21
"""

import sqlalchemy as sa  # type: ignore
from alembic import op  # type: ignore

revision = "0004_correlation_tables"
down_revision = "0003_dynamic_risk_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
	# correlation_matrix table
	op.create_table(
		"correlation_matrix",
		sa.Column("id", sa.Integer(), nullable=False),
		sa.Column("symbol_pair", sa.String(length=50), nullable=True, index=True),
		sa.Column("correlation_value", sa.Numeric(18, 8), nullable=True),
		sa.Column("p_value", sa.Numeric(18, 8), nullable=True),
		sa.Column("sample_size", sa.Integer(), nullable=True),
		sa.Column("last_updated", sa.DateTime(), nullable=True),
		sa.Column("window_days", sa.Integer(), nullable=True),
		sa.PrimaryKeyConstraint("id"),
	)
	op.create_index("idx_corr_pair_updated", "correlation_matrix", ["symbol_pair", "last_updated"])

	# portfolio_exposures table
	op.create_table(
		"portfolio_exposures",
		sa.Column("id", sa.Integer(), nullable=False),
		sa.Column("correlation_group", sa.String(length=100), nullable=True, index=True),
		sa.Column("total_exposure", sa.Numeric(18, 8), nullable=True),
		sa.Column("position_count", sa.Integer(), nullable=True),
		sa.Column("symbols", sa.JSON(), nullable=True),
		sa.Column("last_updated", sa.DateTime(), nullable=True),
		sa.PrimaryKeyConstraint("id"),
	)


def downgrade() -> None:
	op.drop_index("idx_corr_pair_updated", table_name="correlation_matrix")
	op.drop_table("portfolio_exposures")
	op.drop_table("correlation_matrix")