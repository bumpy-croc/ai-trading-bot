"""Store entry balance for recovered positions

Revision ID: 0004_preserve_entry_balance
Revises: 0003_signal_precision
Create Date: 2025-10-22 12:00:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0004_preserve_entry_balance"
down_revision = "0003_signal_precision"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "positions",
        sa.Column("entry_balance", sa.Numeric(precision=18, scale=8), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("positions", "entry_balance")
