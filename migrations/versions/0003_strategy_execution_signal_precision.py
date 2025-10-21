"""Promote strategy execution precision to float

Revision ID: 0003_strategy_execution_signal_precision
Revises: 0002_strategy_management
Create Date: 2025-10-20 15:12:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0003_strategy_execution_signal_precision"
down_revision = "0002_strategy_management"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "strategy_executions",
        "signal_strength",
        type_=sa.Float(asdecimal=False),
        existing_type=sa.Numeric(precision=18, scale=8),
        postgresql_using="signal_strength::double precision",
    )
    op.alter_column(
        "strategy_executions",
        "confidence_score",
        type_=sa.Float(asdecimal=False),
        existing_type=sa.Numeric(precision=18, scale=8),
        postgresql_using="confidence_score::double precision",
    )


def downgrade() -> None:
    op.alter_column(
        "strategy_executions",
        "signal_strength",
        type_=sa.Numeric(precision=18, scale=8),
        existing_type=sa.Float(asdecimal=False),
        postgresql_using="signal_strength::numeric(18, 8)",
    )
    op.alter_column(
        "strategy_executions",
        "confidence_score",
        type_=sa.Numeric(precision=18, scale=8),
        existing_type=sa.Float(asdecimal=False),
        postgresql_using="confidence_score::numeric(18, 8)",
    )
