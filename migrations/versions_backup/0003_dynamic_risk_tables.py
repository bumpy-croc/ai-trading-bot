"""Add dynamic performance metrics and risk adjustments tables

Revision ID: 0003_dynamic_risk_tables
Revises: 0002_numeric_jsonb
Create Date: 2025-01-20
"""

import sqlalchemy as sa  # type: ignore
from alembic import op  # type: ignore

revision = "0003_dynamic_risk_tables"
down_revision = "0002_numeric_jsonb"
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


def upgrade():
    """Create dynamic performance metrics and risk adjustments tables idempotently."""

    # Create dynamic_performance_metrics table if missing
    if not _table_exists("dynamic_performance_metrics"):
        op.create_table(
            'dynamic_performance_metrics',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('timestamp', sa.DateTime(), nullable=False),
            sa.Column('rolling_win_rate', sa.Numeric(18, 8), nullable=True),
            sa.Column('rolling_sharpe_ratio', sa.Numeric(18, 8), nullable=True),
            sa.Column('current_drawdown', sa.Numeric(18, 8), nullable=True),
            sa.Column('volatility_30d', sa.Numeric(18, 8), nullable=True),
            sa.Column('consecutive_losses', sa.Integer(), nullable=True, default=0),
            sa.Column('consecutive_wins', sa.Integer(), nullable=True, default=0),
            sa.Column('risk_adjustment_factor', sa.Numeric(18, 8), nullable=True, default=1.0),
            sa.Column('profit_factor', sa.Numeric(18, 8), nullable=True),
            sa.Column('expectancy', sa.Numeric(18, 8), nullable=True),
            sa.Column('avg_trade_duration_hours', sa.Numeric(18, 8), nullable=True),
            sa.Column('session_id', sa.Integer(), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['session_id'], ['trading_sessions.id'], ),
            sa.PrimaryKeyConstraint('id')
        )

    # Create indexes for dynamic_performance_metrics if missing (PostgreSQL-specific)
    if _table_exists("dynamic_performance_metrics") and not _index_exists("dynamic_performance_metrics", "idx_dynamic_perf_timestamp"):
        op.execute("CREATE INDEX IF NOT EXISTS idx_dynamic_perf_timestamp ON dynamic_performance_metrics (timestamp)")
    if _table_exists("dynamic_performance_metrics") and not _index_exists("dynamic_performance_metrics", "idx_dynamic_perf_session"):
        op.execute("CREATE INDEX IF NOT EXISTS idx_dynamic_perf_session ON dynamic_performance_metrics (session_id)")

    # Create risk_adjustments table if missing
    if not _table_exists("risk_adjustments"):
        op.create_table(
            'risk_adjustments',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('timestamp', sa.DateTime(), nullable=False),
            sa.Column('adjustment_type', sa.String(50), nullable=False),
            sa.Column('trigger_reason', sa.String(200), nullable=True),
            sa.Column('parameter_name', sa.String(100), nullable=False),
            sa.Column('original_value', sa.Numeric(18, 8), nullable=False),
            sa.Column('adjusted_value', sa.Numeric(18, 8), nullable=False),
            sa.Column('adjustment_factor', sa.Numeric(18, 8), nullable=False),
            sa.Column('current_drawdown', sa.Numeric(18, 8), nullable=True),
            sa.Column('performance_score', sa.Numeric(18, 8), nullable=True),
            sa.Column('volatility_level', sa.Numeric(18, 8), nullable=True),
            sa.Column('duration_minutes', sa.Integer(), nullable=True),
            sa.Column('trades_during_adjustment', sa.Integer(), nullable=True, default=0),
            sa.Column('pnl_during_adjustment', sa.Numeric(18, 8), nullable=True),
            sa.Column('session_id', sa.Integer(), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['session_id'], ['trading_sessions.id'], ),
            sa.PrimaryKeyConstraint('id')
        )

    # Create indexes for risk_adjustments if missing (PostgreSQL-specific)
    if _table_exists("risk_adjustments") and not _index_exists("risk_adjustments", "idx_risk_adj_timestamp"):
        op.execute("CREATE INDEX IF NOT EXISTS idx_risk_adj_timestamp ON risk_adjustments (timestamp)")
    if _table_exists("risk_adjustments") and not _index_exists("risk_adjustments", "idx_risk_adj_type"):
        op.execute("CREATE INDEX IF NOT EXISTS idx_risk_adj_type ON risk_adjustments (adjustment_type)")
    if _table_exists("risk_adjustments") and not _index_exists("risk_adjustments", "idx_risk_adj_session"):
        op.execute("CREATE INDEX IF NOT EXISTS idx_risk_adj_session ON risk_adjustments (session_id)")


def downgrade():
    """Drop dynamic risk tables and indexes if they exist."""

    # Drop indexes first (PostgreSQL-specific IF EXISTS)
    op.execute("DROP INDEX IF EXISTS idx_risk_adj_session")
    op.execute("DROP INDEX IF EXISTS idx_risk_adj_type")
    op.execute("DROP INDEX IF EXISTS idx_risk_adj_timestamp")
    op.execute("DROP INDEX IF EXISTS idx_dynamic_perf_session")
    op.execute("DROP INDEX IF EXISTS idx_dynamic_perf_timestamp")

    # Drop tables if they exist
    if _table_exists('risk_adjustments'):
        op.drop_table('risk_adjustments')
    if _table_exists('dynamic_performance_metrics'):
        op.drop_table('dynamic_performance_metrics')