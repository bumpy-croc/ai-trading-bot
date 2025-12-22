"""Add correlation_matrix and portfolio_exposures tables

Revision ID: 0009_correlation_tables
Revises: 0008_trailing_stops
Create Date: 2025-08-21
"""

import sqlalchemy as sa  # type: ignore
from alembic import op  # type: ignore

revision = "0009_correlation_tables"
down_revision = "0008_trailing_stops"
branch_labels = None
depends_on = None


def _table_exists(table_name: str) -> bool:
    """Return True if the table exists in the current database schema."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return inspector.has_table(table_name)


def _index_exists(table_name: str, index_name: str) -> bool:
    """Return True if an index with the given name exists on the table."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    try:
        indexes = inspector.get_indexes(table_name)
    except Exception:
        return False
    return any(i.get("name") == index_name for i in indexes)


def upgrade() -> None:
    # correlation_matrix table (idempotent)
    if not _table_exists("correlation_matrix"):
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

    # index on correlation_matrix (idempotent)
    if _table_exists("correlation_matrix") and not _index_exists(
        "correlation_matrix", "idx_corr_pair_updated"
    ):
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_corr_pair_updated ON correlation_matrix (symbol_pair, last_updated)"
        )

    # portfolio_exposures table (idempotent)
    if not _table_exists("portfolio_exposures"):
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
    # drop index first
    op.execute("DROP INDEX IF EXISTS idx_corr_pair_updated")
    # drop tables if they exist
    if _table_exists("portfolio_exposures"):
        op.drop_table("portfolio_exposures")
    if _table_exists("correlation_matrix"):
        op.drop_table("correlation_matrix")
