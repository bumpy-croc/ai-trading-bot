"""fresh_schema_simplified

Revision ID: b673f791db9f
Revises: 5dd66bc421f8
Create Date: 2025-09-02 21:42:41.247000

"""
import sqlalchemy as sa
from alembic import op


def upgrade() -> None:
    # Use pure raw SQL to create all tables since we're nuking the database anyway
    op.execute("""
        CREATE TABLE trading_sessions (
            id SERIAL PRIMARY KEY,
            strategy_name VARCHAR(100) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            trading_mode VARCHAR(20) NOT NULL,
            initial_balance DECIMAL(18,8) NOT NULL,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            is_active BOOLEAN,
            total_pnl DECIMAL(18,8),
            total_trades INTEGER,
            win_rate DECIMAL(5,2),
            max_drawdown DECIMAL(18,8),
            sharpe_ratio DECIMAL(10,4),
            time_exit_config TEXT
        )
    """)

    op.execute("""
        CREATE TABLE positions (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            side positionside NOT NULL,
            status positionstatus NOT NULL,
            entry_price DECIMAL(18,8) NOT NULL,
            size DECIMAL(18,8) NOT NULL,
            quantity DECIMAL(18,8),
            original_size DECIMAL(18,8),
            current_size DECIMAL(18,8),
            partial_exits_taken INTEGER,
            scale_ins_taken INTEGER,
            last_partial_exit_price DECIMAL(18,8),
            last_scale_in_price DECIMAL(18,8),
            stop_loss DECIMAL(18,8),
            take_profit DECIMAL(18,8),
            trailing_stop DECIMAL(18,8),
            trailing_stop_activated BOOLEAN,
            trailing_stop_price DECIMAL(18,8),
            breakeven_triggered BOOLEAN,
            entry_time TIMESTAMP,
            last_update TIMESTAMP,
            current_price DECIMAL(18,8),
            unrealized_pnl DECIMAL(18,8),
            unrealized_pnl_percent DECIMAL(18,8),
            mfe DECIMAL(18,8),
            mae DECIMAL(18,8),
            mfe_price DECIMAL(18,8),
            mae_price DECIMAL(18,8),
            mfe_time TIMESTAMP,
            mae_time TIMESTAMP,
            strategy_name VARCHAR(100) NOT NULL,
            confidence_score DECIMAL(5,4),
            order_id VARCHAR(100) NOT NULL,
            exchange VARCHAR(50),
            session_id INTEGER REFERENCES trading_sessions(id),
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            max_holding_until TIMESTAMP,
            end_of_day_exit BOOLEAN,
            weekend_exit BOOLEAN,
            time_restriction_group VARCHAR(50),
            UNIQUE(order_id, session_id)
        )
    """)

    op.execute("""
        CREATE TABLE orders (
            id SERIAL PRIMARY KEY,
            position_id INTEGER NOT NULL REFERENCES positions(id),
            order_type ordertype NOT NULL,
            status orderstatus NOT NULL,
            exchange_order_id VARCHAR(100),
            internal_order_id VARCHAR(100) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            side positionside NOT NULL,
            quantity DECIMAL(18,8) NOT NULL,
            price DECIMAL(18,8),
            filled_quantity DECIMAL(18,8),
            filled_price DECIMAL(18,8),
            commission DECIMAL(18,8),
            created_at TIMESTAMP,
            filled_at TIMESTAMP,
            cancelled_at TIMESTAMP,
            last_update TIMESTAMP,
            strategy_name VARCHAR(100) NOT NULL,
            session_id INTEGER REFERENCES trading_sessions(id),
            target_level INTEGER,
            size_fraction DECIMAL(18,8),
            UNIQUE(internal_order_id, session_id)
        )
    """)

    op.execute("""
        CREATE TABLE trades (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            side positionside NOT NULL,
            entry_price DECIMAL(18,8) NOT NULL,
            exit_price DECIMAL(18,8) NOT NULL,
            size DECIMAL(18,8) NOT NULL,
            quantity DECIMAL(18,8),
            entry_time TIMESTAMP NOT NULL,
            exit_time TIMESTAMP NOT NULL,
            pnl DECIMAL(18,8) NOT NULL,
            exit_reason VARCHAR(100) NOT NULL,
            strategy_name VARCHAR(100) NOT NULL,
            source VARCHAR(20) NOT NULL,
            order_id VARCHAR(100),
            exchange VARCHAR(50),
            session_id INTEGER REFERENCES trading_sessions(id),
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            confidence_score DECIMAL(5,4),
            commission DECIMAL(18,8),
            fees DECIMAL(18,8),
            max_favorable_excursion DECIMAL(18,8),
            max_adverse_excursion DECIMAL(18,8),
            holding_time_seconds INTEGER,
            entry_signal_strength DECIMAL(5,4),
            exit_signal_strength DECIMAL(5,4),
            market_regime VARCHAR(50),
            volatility DECIMAL(10,6),
            trend_strength DECIMAL(5,4),
            liquidity_score DECIMAL(5,4),
            risk_adjustment_factor DECIMAL(5,4),
            position_id INTEGER REFERENCES positions(id)
        )
    """)

    # Create indexes
    op.execute("CREATE INDEX idx_order_position_type ON orders(position_id, order_type)")
    op.execute("CREATE INDEX idx_order_status_created ON orders(status, created_at)")
    op.execute("CREATE INDEX ix_orders_created_at ON orders(created_at)")
    op.execute("CREATE UNIQUE INDEX ix_orders_exchange_order_id ON orders(exchange_order_id)")
    op.execute("CREATE INDEX ix_orders_internal_order_id ON orders(internal_order_id)")
    op.execute("CREATE INDEX ix_orders_position_id ON orders(position_id)")


def downgrade() -> None:
    # Drop tables
    op.drop_table('orders')
    op.drop_table('trades')
    op.drop_table('positions')
    op.drop_table('trading_sessions')

    # Drop enums
    op.execute("DROP TYPE IF EXISTS positionstatus")
    op.execute("DROP TYPE IF EXISTS ordertype")
    op.execute("DROP TYPE IF EXISTS orderstatus")
    op.execute("DROP TYPE IF EXISTS positionside")


# revision identifiers, used by Alembic.
revision = 'b673f791db9f'
down_revision = '5dd66bc421f8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create enums only if they don't exist
    connection = op.get_bind()

    # Check and create orderstatus
    result = connection.execute(sa.text("SELECT 1 FROM pg_type WHERE typname = 'orderstatus'"))
    if not result.fetchone():
        op.execute("CREATE TYPE orderstatus AS ENUM ('PENDING', 'OPEN', 'FILLED', 'CANCELLED', 'FAILED')")

    # Check and create ordertype
    result = connection.execute(sa.text("SELECT 1 FROM pg_type WHERE typname = 'ordertype'"))
    if not result.fetchone():
        op.execute("CREATE TYPE ordertype AS ENUM ('ENTRY', 'PARTIAL_EXIT', 'SCALE_IN', 'FULL_EXIT')")

    # Check and create positionside
    result = connection.execute(sa.text("SELECT 1 FROM pg_type WHERE typname = 'positionside'"))
    if not result.fetchone():
        op.execute("CREATE TYPE positionside AS ENUM ('LONG', 'SHORT')")

    # Check and create positionstatus
    result = connection.execute(sa.text("SELECT 1 FROM pg_type WHERE typname = 'positionstatus'"))
    if not result.fetchone():
        op.execute("CREATE TYPE positionstatus AS ENUM ('OPEN', 'CLOSED')")

    # Create trading_sessions table
    op.create_table('trading_sessions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('strategy_name', sa.String(length=100), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('timeframe', sa.String(length=10), nullable=False),
        sa.Column('trading_mode', sa.String(length=20), nullable=False),
        sa.Column('initial_balance', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('start_time', sa.DateTime(), nullable=True),
        sa.Column('end_time', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('total_pnl', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('total_trades', sa.Integer(), nullable=True),
        sa.Column('win_rate', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('max_drawdown', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('sharpe_ratio', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('time_exit_config', sa.Text(), nullable=True),  # Simplified from JSONB
        sa.PrimaryKeyConstraint('id')
    )

    # Create positions table with PositionStatus
    op.create_table('positions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('side', sa.Enum('LONG', 'SHORT', name='positionside', create_type=False), nullable=False),
        sa.Column('status', sa.Enum('OPEN', 'CLOSED', name='positionstatus', create_type=False), nullable=False),
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
        sa.Column('trailing_stop', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('trailing_stop_activated', sa.Boolean(), nullable=True),
        sa.Column('trailing_stop_price', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('breakeven_triggered', sa.Boolean(), nullable=True),
        sa.Column('entry_time', sa.DateTime(), nullable=True),
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
        sa.Column('confidence_score', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('order_id', sa.String(length=100), nullable=False),
        sa.Column('exchange', sa.String(length=50), nullable=True),
        sa.Column('session_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('max_holding_until', sa.DateTime(), nullable=True),
        sa.Column('end_of_day_exit', sa.Boolean(), nullable=True),
        sa.Column('weekend_exit', sa.Boolean(), nullable=True),
        sa.Column('time_restriction_group', sa.String(length=50), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['trading_sessions.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('order_id', 'session_id', name='uq_position_order_session')
    )

    # Create orders table
    op.create_table('orders',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('position_id', sa.Integer(), nullable=False),
        sa.Column('order_type', sa.Enum('ENTRY', 'PARTIAL_EXIT', 'SCALE_IN', 'FULL_EXIT', name='ordertype', create_type=False), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'OPEN', 'FILLED', 'CANCELLED', 'FAILED', name='orderstatus', create_type=False), nullable=False),
        sa.Column('exchange_order_id', sa.String(length=100), nullable=True),
        sa.Column('internal_order_id', sa.String(length=100), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('side', sa.Enum('LONG', 'SHORT', name='positionside', create_type=False), nullable=False),
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

    # Create trades table
    op.create_table('trades',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('side', sa.Enum('LONG', 'SHORT', name='positionside', create_type=False), nullable=False),
        sa.Column('entry_price', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('exit_price', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('size', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('quantity', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('entry_time', sa.DateTime(), nullable=False),
        sa.Column('exit_time', sa.DateTime(), nullable=False),
        sa.Column('pnl', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('exit_reason', sa.String(length=100), nullable=False),
        sa.Column('strategy_name', sa.String(length=100), nullable=False),
        sa.Column('source', sa.String(length=20), nullable=False),
        sa.Column('order_id', sa.String(length=100), nullable=True),
        sa.Column('exchange', sa.String(length=50), nullable=True),
        sa.Column('session_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('confidence_score', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('commission', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('fees', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('max_favorable_excursion', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('max_adverse_excursion', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('holding_time_seconds', sa.Integer(), nullable=True),
        sa.Column('entry_signal_strength', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('exit_signal_strength', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('market_regime', sa.String(length=50), nullable=True),
        sa.Column('volatility', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('trend_strength', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('liquidity_score', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('risk_adjustment_factor', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('position_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['position_id'], ['positions.id'], ),
        sa.ForeignKeyConstraint(['session_id'], ['trading_sessions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes
    op.create_index('idx_order_position_type', 'orders', ['position_id', 'order_type'], unique=False)
    op.create_index('idx_order_status_created', 'orders', ['status', 'created_at'], unique=False)
    op.create_index('ix_orders_created_at', 'orders', ['created_at'], unique=False)
    op.create_index('ix_orders_exchange_order_id', 'orders', ['exchange_order_id'], unique=True)
    op.create_index('ix_orders_internal_order_id', 'orders', ['internal_order_id'], unique=False)
    op.create_index('ix_orders_position_id', 'orders', ['position_id'], unique=False)


def downgrade() -> None:
    # Drop tables
    op.drop_table('orders')
    op.drop_table('trades')
    op.drop_table('positions')
    op.drop_table('trading_sessions')

    # Drop enums
    op.execute("DROP TYPE IF EXISTS positionstatus")
    op.execute("DROP TYPE IF EXISTS ordertype")
    op.execute("DROP TYPE IF EXISTS orderstatus")
    op.execute("DROP TYPE IF EXISTS positionside")
