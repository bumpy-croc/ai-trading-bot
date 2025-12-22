"""Backfill alias for strategy management migration

Revision ID: 0002_strategy_management
Revises: 0002_add_strategy_management
Create Date: 2025-10-20 00:00:00.000000

"""

# revision identifiers, used by Alembic.
revision = "0002_strategy_management"
down_revision = "0002_add_strategy_management"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """No-op placeholder to keep revision history aligned."""
    # This revision existed in production with identical schema changes
    # to 0002_add_strategy_management. Keeping it as an explicit
    # alias ensures alembic lookups resolve correctly.
    pass


def downgrade() -> None:
    """Step back to 0002_add_strategy_management."""
    pass
