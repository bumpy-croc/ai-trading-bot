"""Add trailing stop fields to positions

Revision ID: 0008_trailing_stops
Revises: 0007
Create Date: 2025-08-21
"""

import sqlalchemy as sa  # type: ignore
from alembic import op  # type: ignore
from sqlalchemy.engine import reflection  # type: ignore

revision = "0008_trailing_stops"
down_revision = "0007"
branch_labels = None
depends_on = None


def _has_column(table: str, column: str) -> bool:
    """Check if a column exists in the given table."""
    bind = op.get_bind()
    insp = reflection.Inspector.from_engine(bind)
    cols = [c["name"] for c in insp.get_columns(table)]
    return column in cols


def upgrade():
    # Add columns to positions table (idempotent checks)
    if not _has_column("positions", "trailing_stop_activated"):
        op.add_column("positions", sa.Column("trailing_stop_activated", sa.Boolean(), nullable=False, server_default=sa.text("false")))
    if not _has_column("positions", "trailing_stop_price"):
        op.add_column("positions", sa.Column("trailing_stop_price", sa.Numeric(18, 8), nullable=True))
    if not _has_column("positions", "breakeven_triggered"):
        op.add_column("positions", sa.Column("breakeven_triggered", sa.Boolean(), nullable=False, server_default=sa.text("false")))


def downgrade():
    # Remove columns from positions table (idempotent checks)
    if _has_column("positions", "breakeven_triggered"):
        op.drop_column("positions", "breakeven_triggered")
    if _has_column("positions", "trailing_stop_price"):
        op.drop_column("positions", "trailing_stop_price")
    if _has_column("positions", "trailing_stop_activated"):
        op.drop_column("positions", "trailing_stop_activated")