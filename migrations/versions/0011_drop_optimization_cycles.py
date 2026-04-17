"""Drop optimization_cycles table

Revision ID: 0011_drop_optimization_cycles
Revises: 0010_margin_interest_cost
Create Date: 2026-04-17 12:00:00.000000

The optimizer framework that populated this table has been replaced by a
file-based experiment ledger under experiments/.history/. The table was
never populated in production.
"""

import sqlalchemy as sa
from alembic import op

try:  # JSONB is PostgreSQL-only; fall back to generic JSON on other backends.
    from sqlalchemy.dialects.postgresql import JSONB as JSON
except Exception:  # pragma: no cover - non-Postgres environments
    JSON = sa.JSON  # type: ignore[assignment]


revision = "0011_drop_optimization_cycles"
down_revision = "0010_margin_interest_cost"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Indexes must be dropped before the table on some backends.
    with op.get_context().autocommit_block():
        bind = op.get_bind()
        inspector = sa.inspect(bind)
        if "optimization_cycles" in inspector.get_table_names():
            for idx_name in (
                "ix_optimization_cycles_timestamp",
                "idx_opt_cycle_time",
                "idx_opt_cycle_strategy",
            ):
                try:
                    op.drop_index(idx_name, table_name="optimization_cycles")
                except Exception:
                    # Index may not exist on older databases; continue.
                    pass
            op.drop_table("optimization_cycles")


def downgrade() -> None:
    op.create_table(
        "optimization_cycles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("strategy_name", sa.String(length=100), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("timeframe", sa.String(length=10), nullable=False),
        sa.Column("baseline_metrics", JSON, nullable=True),
        sa.Column("candidate_params", JSON, nullable=True),
        sa.Column("candidate_metrics", JSON, nullable=True),
        sa.Column("validator_report", JSON, nullable=True),
        sa.Column("decision", sa.String(length=20), nullable=True),
        sa.Column("session_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["trading_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_opt_cycle_strategy",
        "optimization_cycles",
        ["strategy_name", "symbol", "timeframe"],
        unique=False,
    )
    op.create_index("idx_opt_cycle_time", "optimization_cycles", ["timestamp"], unique=False)
    op.create_index(
        "ix_optimization_cycles_timestamp",
        "optimization_cycles",
        ["timestamp"],
        unique=False,
    )
