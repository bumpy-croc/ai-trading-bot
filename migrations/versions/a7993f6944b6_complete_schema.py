"""complete_schema_single_migration

Revision ID: a7993f6944b6
Revises:
Create Date: 2025-09-07 22:04:45.117023

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import JSON, text


# revision identifiers, used by Alembic.
revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create complete database schema that matches current SQLAlchemy models exactly"""

    # Create enum types
    enum_definitions = [
        ("positionside", "'LONG', 'SHORT'"),
        ("positionstatus", "'OPEN', 'CLOSED'"),
        ("ordertype", "'ENTRY', 'PARTIAL_EXIT', 'SCALE_IN', 'FULL_EXIT'"),
        ("orderstatus", "'PENDING', 'OPEN', 'FILLED', 'CANCELLED', 'FAILED'"),
        ("tradesource", "'LIVE', 'BACKTEST', 'PAPER'"),
        ("eventtype", "'ENGINE_START', 'ENGINE_STOP', 'STRATEGY_CHANGE', 'MODEL_UPDATE', 'ERROR', 'WARNING', 'ALERT', 'BALANCE_ADJUSTMENT', 'TEST'"),
        ("partialoperationtype", "'PARTIAL_EXIT', 'SCALE_IN'")
    ]

    for enum_name, enum_values in enum_definitions:
        # Check if enum exists first
        check_sql = text(f"""
            SELECT 1 FROM pg_type t
            JOIN pg_namespace n ON t.typnamespace = n.oid
            WHERE t.typtype = 'e'
            AND t.typname = '{enum_name}'
            AND n.nspname = 'public'
        """)
        result = op.get_bind().execute(check_sql).fetchone()

        if result is None:
            # Enum doesn't exist, create it
            create_sql = f"CREATE TYPE {enum_name} AS ENUM ({enum_values})"
            op.execute(create_sql)
            print(f"Created enum: {enum_name}")
        else:
            # Enum exists, skip creation
            print(f"Enum {enum_name} already exists - skipping")

    # Create all tables with correct schema
    op.create_table('correlation_matrix',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('symbol_pair', sa.String(length=50), nullable=True),
    sa.Column('correlation_value', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('p_value', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('sample_size', sa.Integer(), nullable=True),
    sa.Column('last_updated', sa.DateTime(), nullable=True),
    sa.Column('window_days', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_corr_pair_updated', 'correlation_matrix', ['symbol_pair', 'last_updated'], unique=False)
    op.create_index(op.f('ix_correlation_matrix_symbol_pair'), 'correlation_matrix', ['symbol_pair'], unique=False)

    op.create_table('market_sessions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=50), nullable=True),
    sa.Column('timezone', sa.String(length=50), nullable=True),
    sa.Column('open_time', sa.Time(), nullable=True),
    sa.Column('close_time', sa.Time(), nullable=True),
    sa.Column('days_of_week', JSON, nullable=True),
    sa.Column('is_24h', sa.Boolean(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_market_sessions_name'), 'market_sessions', ['name'], unique=True)

    op.create_table('portfolio_exposures',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('correlation_group', sa.String(length=100), nullable=True),
    sa.Column('total_exposure', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('position_count', sa.Integer(), nullable=True),
    sa.Column('symbols', JSON, nullable=True),
    sa.Column('last_updated', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_portfolio_exposures_correlation_group'), 'portfolio_exposures', ['correlation_group'], unique=False)

    op.create_table('prediction_cache',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('cache_key', sa.String(length=255), nullable=False),
    sa.Column('model_name', sa.String(length=100), nullable=False),
    sa.Column('features_hash', sa.String(length=64), nullable=False),
    sa.Column('predicted_price', sa.Numeric(precision=18, scale=8), nullable=False),
    sa.Column('confidence', sa.Numeric(precision=18, scale=8), nullable=False),
    sa.Column('direction', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=False),
    sa.Column('access_count', sa.Integer(), nullable=True),
    sa.Column('last_accessed', sa.DateTime(), nullable=True),
    sa.Column('config_hash', sa.String(length=64), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_pred_cache_access', 'prediction_cache', ['last_accessed'], unique=False)
    op.create_index('idx_pred_cache_expires', 'prediction_cache', ['expires_at'], unique=False)
    op.create_index('idx_pred_cache_model_config', 'prediction_cache', ['model_name', 'config_hash'], unique=False)
    op.create_index(op.f('ix_prediction_cache_cache_key'), 'prediction_cache', ['cache_key'], unique=True)
    op.create_index(op.f('ix_prediction_cache_expires_at'), 'prediction_cache', ['expires_at'], unique=False)
    op.create_index(op.f('ix_prediction_cache_features_hash'), 'prediction_cache', ['features_hash'], unique=False)
    op.create_index(op.f('ix_prediction_cache_model_name'), 'prediction_cache', ['model_name'], unique=False)

    op.create_table('prediction_performance',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=False),
    sa.Column('model_name', sa.String(length=100), nullable=False),
    sa.Column('horizon', sa.Integer(), nullable=True),
    sa.Column('mae', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('rmse', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('mape', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('ic', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('mean_pred', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('std_pred', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('mean_real', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('std_real', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('strategy_name', sa.String(length=100), nullable=False),
    sa.Column('symbol', sa.String(length=20), nullable=False),
    sa.Column('timeframe', sa.String(length=10), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_pred_perf_model', 'prediction_performance', ['model_name', 'horizon'], unique=False)
    op.create_index('idx_pred_perf_time', 'prediction_performance', ['timestamp'], unique=False)
    op.create_index(op.f('ix_prediction_performance_timestamp'), 'prediction_performance', ['timestamp'], unique=False)

    op.create_table('trading_sessions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('session_name', sa.String(length=100), nullable=True),
    sa.Column('start_time', sa.DateTime(), nullable=False),
    sa.Column('end_time', sa.DateTime(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('mode', sa.Enum('LIVE', 'BACKTEST', 'PAPER', name='tradesource', native_enum=False, create_type=False), nullable=False),
    sa.Column('initial_balance', sa.Numeric(precision=18, scale=8), nullable=False),
    sa.Column('final_balance', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('strategy_name', sa.String(length=100), nullable=False),
    sa.Column('strategy_config', JSON, nullable=True),
    sa.Column('symbol', sa.String(length=20), nullable=False),
    sa.Column('timeframe', sa.String(length=10), nullable=False),
    sa.Column('exchange', sa.String(length=50), nullable=True),  # This column should exist
    sa.Column('time_exit_config', JSON, nullable=True),
    sa.Column('market_timezone', sa.String(length=50), nullable=True),
    sa.Column('total_pnl', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('total_trades', sa.Integer(), nullable=True),
    sa.Column('win_rate', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('max_drawdown', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_trading_sessions_start_time'), 'trading_sessions', ['start_time'], unique=False)

    # Create remaining tables
    op.create_table('account_balances',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('base_currency', sa.String(length=10), nullable=False),
    sa.Column('total_balance', sa.Float(), nullable=False),
    sa.Column('available_balance', sa.Float(), nullable=False),
    sa.Column('reserved_balance', sa.Float(), nullable=True),
    sa.Column('asset_balances', JSON, nullable=True),
    sa.Column('last_updated', sa.DateTime(), nullable=False),
    sa.Column('updated_by', sa.String(length=50), nullable=True),
    sa.Column('update_reason', sa.String(length=200), nullable=True),
    sa.Column('session_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['session_id'], ['trading_sessions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_balance_session_updated', 'account_balances', ['session_id', 'last_updated'], unique=False)

    op.create_table('account_history',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=False),
    sa.Column('balance', sa.Numeric(precision=18, scale=8), nullable=False),
    sa.Column('equity', sa.Numeric(precision=18, scale=8), nullable=False),
    sa.Column('margin_used', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('margin_available', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('total_pnl', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('daily_pnl', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('drawdown', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('open_positions', sa.Integer(), nullable=True),
    sa.Column('total_exposure', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('session_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['session_id'], ['trading_sessions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_account_time', 'account_history', ['timestamp'], unique=False)
    op.create_index(op.f('ix_account_history_timestamp'), 'account_history', ['timestamp'], unique=False)

    op.create_table('positions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('symbol', sa.String(length=20), nullable=False),
    sa.Column('side', sa.Enum('LONG', 'SHORT', name='positionside', native_enum=False, create_type=False), nullable=False),
    sa.Column('status', sa.Enum('OPEN', 'CLOSED', name='positionstatus', native_enum=False, create_type=False), nullable=False),
    sa.Column('entry_price', sa.Numeric(precision=18, scale=8), nullable=False),
    sa.Column('size', sa.Numeric(precision=18, scale=8), nullable=False),
    sa.Column('quantity', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('original_size', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('current_size', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('partial_exits_taken', sa.Integer(), nullable=True),
    sa.Column('scale_ins_taken', sa.Integer(), nullable=True),
    sa.Column('last_partial_exit_price', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('last_scale_in_price', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('stop_loss', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('take_profit', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('trailing_stop', sa.Boolean(), nullable=True),  # This should be Boolean
    sa.Column('trailing_stop_activated', sa.Boolean(), nullable=True),
    sa.Column('trailing_stop_price', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('breakeven_triggered', sa.Boolean(), nullable=True),
    sa.Column('entry_time', sa.DateTime(), nullable=False),
    sa.Column('last_update', sa.DateTime(), nullable=True),
    sa.Column('current_price', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('unrealized_pnl', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('unrealized_pnl_percent', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('mfe', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('mae', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('mfe_price', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('mae_price', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('mfe_time', sa.DateTime(), nullable=True),
    sa.Column('mae_time', sa.DateTime(), nullable=True),
    sa.Column('strategy_name', sa.String(length=100), nullable=False),
    sa.Column('confidence_score', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('exchange', sa.String(length=50), nullable=True),
    sa.Column('session_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.Column('max_holding_until', sa.DateTime(), nullable=True),
    sa.Column('end_of_day_exit', sa.Boolean(), nullable=True),
    sa.Column('weekend_exit', sa.Boolean(), nullable=True),
    sa.Column('time_restriction_group', sa.String(length=50), nullable=True),
    sa.ForeignKeyConstraint(['session_id'], ['trading_sessions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_positions_entry_time'), 'positions', ['entry_time'], unique=False)
    op.create_index(op.f('ix_positions_symbol'), 'positions', ['symbol'], unique=False)

    # Add remaining critical tables with correct enum types
    op.create_table('orders',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('position_id', sa.Integer(), nullable=False),
    sa.Column('order_type', sa.Enum('ENTRY', 'PARTIAL_EXIT', 'SCALE_IN', 'FULL_EXIT', name='ordertype', native_enum=False, create_type=False), nullable=False),
    sa.Column('status', sa.Enum('PENDING', 'OPEN', 'FILLED', 'CANCELLED', 'FAILED', name='orderstatus', native_enum=False, create_type=False), nullable=False),
    sa.Column('exchange_order_id', sa.String(length=100), nullable=True),
    sa.Column('internal_order_id', sa.String(length=100), nullable=False),
    sa.Column('symbol', sa.String(length=20), nullable=False),
    sa.Column('side', sa.Enum('LONG', 'SHORT', name='positionside', native_enum=False, create_type=False), nullable=False),
    sa.Column('quantity', sa.Numeric(precision=18, scale=8), nullable=False),
    sa.Column('price', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('filled_quantity', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('filled_price', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('commission', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('filled_at', sa.DateTime(), nullable=True),
    sa.Column('cancelled_at', sa.DateTime(), nullable=True),
    sa.Column('last_update', sa.DateTime(), nullable=True),
    sa.Column('strategy_name', sa.String(length=100), nullable=False),
    sa.Column('session_id', sa.Integer(), nullable=True),
    sa.Column('target_level', sa.Integer(), nullable=True),
    sa.Column('size_fraction', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.ForeignKeyConstraint(['position_id'], ['positions.id'], ),
    sa.ForeignKeyConstraint(['session_id'], ['trading_sessions.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('internal_order_id', 'session_id', name='uq_order_internal_session')
    )
    op.create_index('idx_order_position_type', 'orders', ['position_id', 'order_type'], unique=False)
    op.create_index('idx_order_status_created', 'orders', ['status', 'created_at'], unique=False)
    op.create_index(op.f('ix_orders_created_at'), 'orders', ['created_at'], unique=False)
    op.create_index(op.f('ix_orders_exchange_order_id'), 'orders', ['exchange_order_id'], unique=True)
    op.create_index(op.f('ix_orders_internal_order_id'), 'orders', ['internal_order_id'], unique=False)
    op.create_index(op.f('ix_orders_position_id'), 'orders', ['position_id'], unique=False)

    op.create_table('partial_trades',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('position_id', sa.Integer(), nullable=False),
    sa.Column('operation_type', sa.Enum('PARTIAL_EXIT', 'SCALE_IN', name='partialoperationtype', native_enum=False, create_type=False), nullable=False),
    sa.Column('size', sa.Numeric(precision=18, scale=8), nullable=False),
    sa.Column('price', sa.Numeric(precision=18, scale=8), nullable=False),
    sa.Column('pnl', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('target_level', sa.Integer(), nullable=True),
    sa.Column('timestamp', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['position_id'], ['positions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_partial_trade_position', 'partial_trades', ['position_id', 'timestamp'], unique=False)
    op.create_index(op.f('ix_partial_trades_position_id'), 'partial_trades', ['position_id'], unique=False)
    op.create_index(op.f('ix_partial_trades_timestamp'), 'partial_trades', ['timestamp'], unique=False)

    op.create_table('system_events',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=False),
    sa.Column('event_type', sa.Enum('ENGINE_START', 'ENGINE_STOP', 'STRATEGY_CHANGE', 'MODEL_UPDATE', 'ERROR', 'WARNING', 'ALERT', 'BALANCE_ADJUSTMENT', 'TEST', name='eventtype', native_enum=False, create_type=False), nullable=False),
    sa.Column('severity', sa.String(length=20), nullable=True),
    sa.Column('message', sa.Text(), nullable=False),
    sa.Column('details', JSON, nullable=True),
    sa.Column('component', sa.String(length=100), nullable=True),
    sa.Column('error_code', sa.String(length=50), nullable=True),
    sa.Column('stack_trace', sa.Text(), nullable=True),
    sa.Column('alert_sent', sa.Boolean(), nullable=True),
    sa.Column('alert_method', sa.String(length=50), nullable=True),
    sa.Column('session_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['session_id'], ['trading_sessions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_event_severity', 'system_events', ['severity', 'timestamp'], unique=False)
    op.create_index('idx_event_time_type', 'system_events', ['timestamp', 'event_type'], unique=False)
    op.create_index(op.f('ix_system_events_timestamp'), 'system_events', ['timestamp'], unique=False)

    op.create_table('trades',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('symbol', sa.String(length=20), nullable=False),
    sa.Column('side', sa.Enum('LONG', 'SHORT', name='positionside', native_enum=False, create_type=False), nullable=False),
    sa.Column('source', sa.Enum('LIVE', 'BACKTEST', 'PAPER', name='tradesource', native_enum=False, create_type=False), nullable=False),
    sa.Column('entry_price', sa.Numeric(precision=18, scale=8), nullable=False),
    sa.Column('exit_price', sa.Numeric(precision=18, scale=8), nullable=False),
    sa.Column('size', sa.Numeric(precision=18, scale=8), nullable=False),
    sa.Column('quantity', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('entry_time', sa.DateTime(), nullable=False),
    sa.Column('exit_time', sa.DateTime(), nullable=False),
    sa.Column('pnl', sa.Numeric(precision=18, scale=8), nullable=False),
    sa.Column('pnl_percent', sa.Numeric(precision=18, scale=8), nullable=False),
    sa.Column('commission', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('stop_loss', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('take_profit', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('exit_reason', sa.String(length=100), nullable=True),
    sa.Column('strategy_name', sa.String(length=100), nullable=False),
    sa.Column('strategy_config', JSON, nullable=True),
    sa.Column('confidence_score', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('order_id', sa.String(length=100), nullable=True),
    sa.Column('exchange', sa.String(length=50), nullable=True),
    sa.Column('timeframe', sa.String(length=10), nullable=True),
    sa.Column('mfe', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('mae', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('mfe_price', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('mae_price', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('mfe_time', sa.DateTime(), nullable=True),
    sa.Column('mae_time', sa.DateTime(), nullable=True),
    sa.Column('position_id', sa.Integer(), nullable=True),
    sa.Column('session_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['position_id'], ['positions.id'], ),
    sa.ForeignKeyConstraint(['session_id'], ['trading_sessions.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('order_id', 'session_id', name='uq_trade_order_session')
    )
    op.create_index('idx_trade_performance', 'trades', ['pnl_percent', 'strategy_name'], unique=False)
    op.create_index('idx_trade_time', 'trades', ['entry_time', 'exit_time'], unique=False)
    op.create_index(op.f('ix_trades_entry_time'), 'trades', ['entry_time'], unique=False)
    op.create_index(op.f('ix_trades_exit_time'), 'trades', ['exit_time'], unique=False)
    op.create_index(op.f('ix_trades_strategy_name'), 'trades', ['strategy_name'], unique=False)
    op.create_index(op.f('ix_trades_symbol'), 'trades', ['symbol'], unique=False)

    # Add missing tables
    op.create_table('dynamic_performance_metrics',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=False),
    sa.Column('rolling_win_rate', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('rolling_sharpe_ratio', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('current_drawdown', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('volatility_30d', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('consecutive_losses', sa.Integer(), nullable=True),
    sa.Column('consecutive_wins', sa.Integer(), nullable=True),
    sa.Column('risk_adjustment_factor', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('profit_factor', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('expectancy', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('avg_trade_duration_hours', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('session_id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['session_id'], ['trading_sessions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_dynamic_perf_session', 'dynamic_performance_metrics', ['session_id'], unique=False)
    op.create_index('idx_dynamic_perf_timestamp', 'dynamic_performance_metrics', ['timestamp'], unique=False)
    op.create_index(op.f('ix_dynamic_performance_metrics_timestamp'), 'dynamic_performance_metrics', ['timestamp'], unique=False)

    op.create_table('optimization_cycles',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=False),
    sa.Column('strategy_name', sa.String(length=100), nullable=False),
    sa.Column('symbol', sa.String(length=20), nullable=False),
    sa.Column('timeframe', sa.String(length=10), nullable=False),
    sa.Column('baseline_metrics', JSON, nullable=True),
    sa.Column('candidate_params', JSON, nullable=True),
    sa.Column('candidate_metrics', JSON, nullable=True),
    sa.Column('validator_report', JSON, nullable=True),
    sa.Column('decision', sa.String(length=20), nullable=True),
    sa.Column('session_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['session_id'], ['trading_sessions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_opt_cycle_strategy', 'optimization_cycles', ['strategy_name', 'symbol', 'timeframe'], unique=False)
    op.create_index('idx_opt_cycle_time', 'optimization_cycles', ['timestamp'], unique=False)
    op.create_index(op.f('ix_optimization_cycles_timestamp'), 'optimization_cycles', ['timestamp'], unique=False)

    op.create_table('performance_metrics',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('period', sa.String(length=20), nullable=False),
    sa.Column('period_start', sa.DateTime(), nullable=False),
    sa.Column('period_end', sa.DateTime(), nullable=False),
    sa.Column('total_trades', sa.Integer(), nullable=True),
    sa.Column('winning_trades', sa.Integer(), nullable=True),
    sa.Column('losing_trades', sa.Integer(), nullable=True),
    sa.Column('win_rate', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('total_return', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('total_return_percent', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('max_drawdown', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('max_drawdown_duration', sa.Integer(), nullable=True),
    sa.Column('sharpe_ratio', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('sortino_ratio', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('calmar_ratio', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('avg_win', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('avg_loss', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('profit_factor', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('expectancy', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('best_trade_pnl', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('worst_trade_pnl', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('largest_win_streak', sa.Integer(), nullable=True),
    sa.Column('largest_loss_streak', sa.Integer(), nullable=True),
    sa.Column('strategy_breakdown', JSON, nullable=True),
    sa.Column('session_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['session_id'], ['trading_sessions.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('period', 'period_start', 'session_id')
    )
    op.create_index('idx_metrics_period', 'performance_metrics', ['period', 'period_start'], unique=False)
    op.create_index(op.f('ix_performance_metrics_period_start'), 'performance_metrics', ['period_start'], unique=False)

    op.create_table('risk_adjustments',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=False),
    sa.Column('adjustment_type', sa.String(length=50), nullable=False),
    sa.Column('trigger_reason', sa.String(length=200), nullable=True),
    sa.Column('parameter_name', sa.String(length=100), nullable=False),
    sa.Column('original_value', sa.Numeric(precision=18, scale=8), nullable=False),
    sa.Column('adjusted_value', sa.Numeric(precision=18, scale=8), nullable=False),
    sa.Column('adjustment_factor', sa.Numeric(precision=18, scale=8), nullable=False),
    sa.Column('current_drawdown', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('performance_score', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('volatility_level', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('duration_minutes', sa.Integer(), nullable=True),
    sa.Column('trades_during_adjustment', sa.Integer(), nullable=True),
    sa.Column('pnl_during_adjustment', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('session_id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['session_id'], ['trading_sessions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_risk_adj_session', 'risk_adjustments', ['session_id'], unique=False)
    op.create_index('idx_risk_adj_timestamp', 'risk_adjustments', ['timestamp'], unique=False)
    op.create_index('idx_risk_adj_type', 'risk_adjustments', ['adjustment_type'], unique=False)
    op.create_index(op.f('ix_risk_adjustments_timestamp'), 'risk_adjustments', ['timestamp'], unique=False)

    op.create_table('strategy_executions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=False),
    sa.Column('strategy_name', sa.String(length=100), nullable=False),
    sa.Column('symbol', sa.String(length=20), nullable=False),
    sa.Column('timeframe', sa.String(length=10), nullable=True),
    sa.Column('signal_type', sa.String(length=20), nullable=True),
    sa.Column('signal_strength', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('confidence_score', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('indicators', JSON, nullable=True),
    sa.Column('sentiment_data', JSON, nullable=True),
    sa.Column('ml_predictions', JSON, nullable=True),
    sa.Column('action_taken', sa.String(length=50), nullable=True),
    sa.Column('position_size', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('reasons', JSON, nullable=True),
    sa.Column('price', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('volume', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('volatility', sa.Numeric(precision=18, scale=8), nullable=True),
    sa.Column('session_id', sa.Integer(), nullable=True),
    sa.Column('trade_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['session_id'], ['trading_sessions.id'], ),
    sa.ForeignKeyConstraint(['trade_id'], ['trades.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_strategy_executions_timestamp'), 'strategy_executions', ['timestamp'], unique=False)


def downgrade() -> None:
    """Drop all tables and enums - complete teardown for fresh start"""
    # Drop tables in reverse dependency order
    op.drop_index(op.f('ix_strategy_executions_timestamp'), table_name='strategy_executions')
    op.drop_table('strategy_executions')
    op.drop_index(op.f('ix_risk_adjustments_timestamp'), table_name='risk_adjustments')
    op.drop_index('idx_risk_adj_type', table_name='risk_adjustments')
    op.drop_index('idx_risk_adj_timestamp', table_name='risk_adjustments')
    op.drop_index('idx_risk_adj_session', table_name='risk_adjustments')
    op.drop_table('risk_adjustments')
    op.drop_index(op.f('ix_performance_metrics_period_start'), table_name='performance_metrics')
    op.drop_index('idx_metrics_period', table_name='performance_metrics')
    op.drop_table('performance_metrics')
    op.drop_index(op.f('ix_optimization_cycles_timestamp'), table_name='optimization_cycles')
    op.drop_index('idx_opt_cycle_time', table_name='optimization_cycles')
    op.drop_index('idx_opt_cycle_strategy', table_name='optimization_cycles')
    op.drop_table('optimization_cycles')
    op.drop_index(op.f('ix_dynamic_performance_metrics_timestamp'), table_name='dynamic_performance_metrics')
    op.drop_index('idx_dynamic_perf_timestamp', table_name='dynamic_performance_metrics')
    op.drop_index('idx_dynamic_perf_session', table_name='dynamic_performance_metrics')
    op.drop_table('dynamic_performance_metrics')
    op.drop_index(op.f('ix_trades_symbol'), table_name='trades')
    op.drop_index(op.f('ix_trades_strategy_name'), table_name='trades')
    op.drop_index(op.f('ix_trades_exit_time'), table_name='trades')
    op.drop_index(op.f('ix_trades_entry_time'), table_name='trades')
    op.drop_index('idx_trade_time', table_name='trades')
    op.drop_index('idx_trade_performance', table_name='trades')
    op.drop_table('trades')
    op.drop_index(op.f('ix_system_events_timestamp'), table_name='system_events')
    op.drop_index('idx_event_time_type', table_name='system_events')
    op.drop_index('idx_event_severity', table_name='system_events')
    op.drop_table('system_events')
    op.drop_index(op.f('ix_partial_trades_timestamp'), table_name='partial_trades')
    op.drop_index(op.f('ix_partial_trades_position_id'), table_name='partial_trades')
    op.drop_index('idx_partial_trade_position', table_name='partial_trades')
    op.drop_table('partial_trades')
    op.drop_index(op.f('ix_orders_position_id'), table_name='orders')
    op.drop_index(op.f('ix_orders_internal_order_id'), table_name='orders')
    op.drop_index(op.f('ix_orders_exchange_order_id'), table_name='orders')
    op.drop_index(op.f('ix_orders_created_at'), table_name='orders')
    op.drop_index('idx_order_status_created', table_name='orders')
    op.drop_index('idx_order_position_type', table_name='orders')
    op.drop_table('orders')
    op.drop_index(op.f('ix_positions_symbol'), table_name='positions')
    op.drop_index(op.f('ix_positions_entry_time'), table_name='positions')
    op.drop_table('positions')
    op.drop_index('idx_balance_session_updated', table_name='account_balances')
    op.drop_table('account_balances')
    op.drop_index(op.f('ix_account_history_timestamp'), table_name='account_history')
    op.drop_index('idx_account_time', table_name='account_history')
    op.drop_table('account_history')
    op.drop_index(op.f('ix_trading_sessions_start_time'), table_name='trading_sessions')
    op.drop_table('trading_sessions')
    op.drop_index(op.f('ix_prediction_performance_timestamp'), table_name='prediction_performance')
    op.drop_index('idx_pred_perf_time', table_name='prediction_performance')
    op.drop_index('idx_pred_perf_model', table_name='prediction_performance')
    op.drop_table('prediction_performance')
    op.drop_index(op.f('ix_prediction_cache_model_name'), table_name='prediction_cache')
    op.drop_index(op.f('ix_prediction_cache_features_hash'), table_name='prediction_cache')
    op.drop_index(op.f('ix_prediction_cache_expires_at'), table_name='prediction_cache')
    op.drop_index(op.f('ix_prediction_cache_cache_key'), table_name='prediction_cache')
    op.drop_index('idx_pred_cache_model_config', table_name='prediction_cache')
    op.drop_index('idx_pred_cache_expires', table_name='prediction_cache')
    op.drop_index('idx_pred_cache_access', table_name='prediction_cache')
    op.drop_table('prediction_cache')
    op.drop_index(op.f('ix_portfolio_exposures_correlation_group'), table_name='portfolio_exposures')
    op.drop_table('portfolio_exposures')
    op.drop_index(op.f('ix_market_sessions_name'), table_name='market_sessions')
    op.drop_table('market_sessions')
    op.drop_index(op.f('ix_correlation_matrix_symbol_pair'), table_name='correlation_matrix')
    op.drop_index('idx_corr_pair_updated', table_name='correlation_matrix')
    op.drop_table('correlation_matrix')

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS partialoperationtype CASCADE")
    op.execute("DROP TYPE IF EXISTS eventtype CASCADE")
    op.execute("DROP TYPE IF EXISTS tradesource CASCADE")
    op.execute("DROP TYPE IF EXISTS orderstatus CASCADE")
    op.execute("DROP TYPE IF EXISTS ordertype CASCADE")
    op.execute("DROP TYPE IF EXISTS positionstatus CASCADE")
    op.execute("DROP TYPE IF EXISTS positionside CASCADE")
