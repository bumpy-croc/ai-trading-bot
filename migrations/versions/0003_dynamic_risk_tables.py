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


def upgrade():
    """Create dynamic performance metrics and risk adjustments tables"""
    
    # Create dynamic_performance_metrics table
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
    
    # Create indexes for dynamic_performance_metrics
    op.create_index('idx_dynamic_perf_timestamp', 'dynamic_performance_metrics', ['timestamp'])
    op.create_index('idx_dynamic_perf_session', 'dynamic_performance_metrics', ['session_id'])
    
    # Create risk_adjustments table
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
    
    # Create indexes for risk_adjustments
    op.create_index('idx_risk_adj_timestamp', 'risk_adjustments', ['timestamp'])
    op.create_index('idx_risk_adj_type', 'risk_adjustments', ['adjustment_type'])
    op.create_index('idx_risk_adj_session', 'risk_adjustments', ['session_id'])


def downgrade():
    """Drop dynamic risk tables and indexes"""
    
    # Drop indexes first
    op.drop_index('idx_risk_adj_session', table_name='risk_adjustments')
    op.drop_index('idx_risk_adj_type', table_name='risk_adjustments')
    op.drop_index('idx_risk_adj_timestamp', table_name='risk_adjustments')
    op.drop_index('idx_dynamic_perf_session', table_name='dynamic_performance_metrics')
    op.drop_index('idx_dynamic_perf_timestamp', table_name='dynamic_performance_metrics')
    
    # Drop tables
    op.drop_table('risk_adjustments')
    op.drop_table('dynamic_performance_metrics')