"""
Database models for trade logging and performance tracking
"""

from sqlalchemy import (
    Column, Integer, String, Numeric, DateTime, Boolean, Text, 
    Enum, ForeignKey, Index, UniqueConstraint, Float, JSON  # Added Float and JSON
)
from sqlalchemy.orm import declarative_base, relationship
# * JSONB is PostgreSQL specific. Provide graceful fallback & SQLite compilation
from sqlalchemy.types import JSON

try:
    from sqlalchemy.dialects.postgresql import JSONB  # type: ignore
except ImportError:  # pragma: no cover – fallback when dialect unavailable
    JSONB = JSON  # type: ignore

# Ensure SQLite can compile JSONB columns by mapping them to generic JSON.
from sqlalchemy.ext.compiler import compiles  # type: ignore


@compiles(JSONB, "sqlite")  # type: ignore
def _compile_jsonb_for_sqlite(_type, compiler, **kw):  # noqa: D401
    """Compile JSONB as JSON for SQLite fallback."""
    return "JSON"

from datetime import datetime
import enum

Base = declarative_base()


class PositionSide(enum.Enum):
    """Position side enumeration"""
    LONG = "long"
    SHORT = "short"


class OrderStatus(enum.Enum):
    """Order status enumeration"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TradeSource(enum.Enum):
    """Source of the trade"""
    LIVE = "live"
    BACKTEST = "backtest"
    PAPER = "paper"


class EventType(enum.Enum):
    """System event types"""
    ENGINE_START = "engine_start"
    ENGINE_STOP = "engine_stop"
    STRATEGY_CHANGE = "strategy_change"
    MODEL_UPDATE = "model_update"
    ERROR = "error"
    WARNING = "warning"
    ALERT = "alert"
    BALANCE_ADJUSTMENT = "balance_adjustment"
    TEST = "test"  # Added for verification scripts and development diagnostics


class Trade(Base):
    """Completed trades table"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(Enum(PositionSide), nullable=False)
    source = Column(Enum(TradeSource), nullable=False, default=TradeSource.LIVE)
    
    # Trade details
    entry_price = Column(Numeric(18, 8), nullable=False)
    exit_price = Column(Numeric(18, 8), nullable=False)
    size = Column(Numeric(18, 8), nullable=False)  # Position size as % of balance
    quantity = Column(Numeric(18, 8))  # Actual quantity traded
    
    # Timestamps
    entry_time = Column(DateTime, nullable=False, index=True)
    exit_time = Column(DateTime, nullable=False, index=True)
    
    # Performance
    pnl = Column(Numeric(18, 8), nullable=False)  # Dollar P&L
    pnl_percent = Column(Numeric(18, 8), nullable=False)  # Percentage P&L
    commission = Column(Numeric(18, 8), default=0.0)
    
    # Risk management
    stop_loss = Column(Numeric(18, 8))
    take_profit = Column(Numeric(18, 8))
    exit_reason = Column(String(100))
    
    # Strategy information
    strategy_name = Column(String(100), nullable=False, index=True)
    strategy_config = Column(JSONB)  # Store strategy parameters
    confidence_score = Column(Numeric(18, 8))  # ML model confidence if applicable
    
    # Additional metadata
    order_id = Column(String(100), unique=True)
    exchange = Column(String(50), default='binance')
    timeframe = Column(String(10))
    
    # Relationships
    position_id = Column(Integer, ForeignKey('positions.id'))
    session_id = Column(Integer, ForeignKey('trading_sessions.id'))
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_trade_time', 'entry_time', 'exit_time'),
        Index('idx_trade_performance', 'pnl_percent', 'strategy_name'),
    )
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Position(Base):
    """Active positions table"""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(Enum(PositionSide), nullable=False)
    status = Column(Enum(OrderStatus), nullable=False, default=OrderStatus.PENDING)
    
    # Position details
    entry_price = Column(Numeric(18, 8), nullable=False)
    size = Column(Numeric(18, 8), nullable=False)
    quantity = Column(Numeric(18, 8))
    
    # Risk management
    stop_loss = Column(Numeric(18, 8))
    take_profit = Column(Numeric(18, 8))
    trailing_stop = Column(Boolean, default=False)
    
    # Timestamps
    entry_time = Column(DateTime, nullable=False, index=True)
    last_update = Column(DateTime, default=datetime.utcnow)
    
    # Current state
    current_price = Column(Numeric(18, 8))
    unrealized_pnl = Column(Numeric(18, 8), default=0.0)
    unrealized_pnl_percent = Column(Numeric(18, 8), default=0.0)
    
    # Strategy information
    strategy_name = Column(String(100), nullable=False)
    confidence_score = Column(Numeric(18, 8))
    
    # Order information
    order_id = Column(String(100), unique=True)
    exchange = Column(String(50), default='binance')
    
    # Relationships
    trades = relationship("Trade", backref="position")
    session_id = Column(Integer, ForeignKey('trading_sessions.id'))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AccountHistory(Base):
    """Account balance history table"""
    __tablename__ = 'account_history'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Balances
    balance = Column(Numeric(18, 8), nullable=False)
    equity = Column(Numeric(18, 8), nullable=False)  # Balance + unrealized P&L
    margin_used = Column(Numeric(18, 8), default=0.0)
    margin_available = Column(Numeric(18, 8))
    
    # Performance metrics at this point
    total_pnl = Column(Numeric(18, 8), default=0.0)
    daily_pnl = Column(Numeric(18, 8), default=0.0)
    drawdown = Column(Numeric(18, 8), default=0.0)
    
    # Position summary
    open_positions = Column(Integer, default=0)
    total_exposure = Column(Numeric(18, 8), default=0.0)
    
    # Session reference
    session_id = Column(Integer, ForeignKey('trading_sessions.id'))
    
    # Index for efficient time-series queries
    __table_args__ = (
        Index('idx_account_time', 'timestamp'),
    )
    
    created_at = Column(DateTime, default=datetime.utcnow)


class PerformanceMetrics(Base):
    """Aggregated performance metrics table"""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True)
    period = Column(String(20), nullable=False)  # 'daily', 'weekly', 'monthly', 'all-time'
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False)
    
    # Trade statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Numeric(18, 8), default=0.0)
    
    # Returns
    total_return = Column(Numeric(18, 8), default=0.0)
    total_return_percent = Column(Numeric(18, 8), default=0.0)
    
    # Risk metrics
    max_drawdown = Column(Numeric(18, 8), default=0.0)
    max_drawdown_duration = Column(Integer)  # In hours
    sharpe_ratio = Column(Numeric(18, 8))
    sortino_ratio = Column(Numeric(18, 8))
    calmar_ratio = Column(Numeric(18, 8))
    
    # Trade analysis
    avg_win = Column(Numeric(18, 8), default=0.0)
    avg_loss = Column(Numeric(18, 8), default=0.0)
    profit_factor = Column(Numeric(18, 8), default=0.0)
    expectancy = Column(Numeric(18, 8), default=0.0)
    
    # Best/worst trades
    best_trade_pnl = Column(Numeric(18, 8))
    worst_trade_pnl = Column(Numeric(18, 8))
    largest_win_streak = Column(Integer, default=0)
    largest_loss_streak = Column(Integer, default=0)
    
    # By strategy breakdown
    strategy_breakdown = Column(JSONB)  # Dict of strategy_name: metrics
    
    # Session reference
    session_id = Column(Integer, ForeignKey('trading_sessions.id'))
    
    # Ensure unique metrics per period
    __table_args__ = (
        UniqueConstraint('period', 'period_start', 'session_id'),
        Index('idx_metrics_period', 'period', 'period_start'),
    )
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TradingSession(Base):
    """Trading session tracking table"""
    __tablename__ = 'trading_sessions'
    
    id = Column(Integer, primary_key=True)
    session_name = Column(String(100))
    
    # Session details
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Configuration
    mode = Column(Enum(TradeSource), nullable=False)
    initial_balance = Column(Numeric(18, 8), nullable=False)
    final_balance = Column(Numeric(18, 8))
    
    # Strategy information
    strategy_name = Column(String(100), nullable=False)
    strategy_config = Column(JSONB)
    
    # Environment
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    exchange = Column(String(50), default='binance')
    
    # Performance summary
    total_pnl = Column(Numeric(18, 8))
    total_trades = Column(Integer, default=0)
    win_rate = Column(Numeric(18, 8))
    max_drawdown = Column(Numeric(18, 8))
    
    # Relationships
    trades = relationship("Trade", backref="session")
    positions = relationship("Position", backref="session")
    account_history = relationship("AccountHistory", backref="session")
    metrics = relationship("PerformanceMetrics", backref="session")
    events = relationship("SystemEvent", backref="session")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SystemEvent(Base):
    """System events and alerts table"""
    __tablename__ = 'system_events'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    event_type = Column(Enum(EventType), nullable=False)
    severity = Column(String(20), default='info')  # 'info', 'warning', 'error', 'critical'
    
    # Event details
    message = Column(Text, nullable=False)
    details = Column(JSONB)  # Additional structured data
    
    # Context
    component = Column(String(100))  # Which part of the system
    error_code = Column(String(50))
    stack_trace = Column(Text)
    
    # Alert status
    alert_sent = Column(Boolean, default=False)
    alert_method = Column(String(50))  # 'telegram', 'email', 'slack'
    
    # Session reference
    session_id = Column(Integer, ForeignKey('trading_sessions.id'))
    
    # Index for efficient queries
    __table_args__ = (
        Index('idx_event_time_type', 'timestamp', 'event_type'),
        Index('idx_event_severity', 'severity', 'timestamp'),
    )
    
    created_at = Column(DateTime, default=datetime.utcnow)


class StrategyExecution(Base):
    """Detailed strategy execution logs"""
    __tablename__ = 'strategy_executions'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Strategy details
    strategy_name = Column(String(100), nullable=False)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10))
    
    # Signal information
    signal_type = Column(String(20))  # 'entry', 'exit', 'hold'
    signal_strength = Column(Numeric(18, 8))
    confidence_score = Column(Numeric(18, 8))
    
    # Decision factors
    indicators = Column(JSONB)  # Dict of indicator values
    sentiment_data = Column(JSONB)  # Sentiment scores if used
    ml_predictions = Column(JSONB)  # ML model outputs if used
    
    # Execution result
    action_taken = Column(String(50))  # 'opened_long', 'closed_position', 'no_action'
    position_size = Column(Numeric(18, 8))
    reasons = Column(JSONB)  # List of reasons for the decision
    
    # Market context
    price = Column(Numeric(18, 8))
    volume = Column(Numeric(18, 8))
    volatility = Column(Numeric(18, 8))
    
    # Session reference
    session_id = Column(Integer, ForeignKey('trading_sessions.id'))
    trade_id = Column(Integer, ForeignKey('trades.id'))
    
    created_at = Column(DateTime, default=datetime.utcnow)


class AccountBalance(Base):
    """Current account balance tracking table"""
    __tablename__ = 'account_balances'
    
    id = Column(Integer, primary_key=True)
    
    # Balance information
    base_currency = Column(String(10), nullable=False, default='USD')  # USD, BTC, ETH, etc.
    total_balance = Column(Float, nullable=False)  # Total balance in base currency
    available_balance = Column(Float, nullable=False)  # Available for trading
    reserved_balance = Column(Float, default=0.0)  # Reserved in open positions
    
    # Balance breakdown by asset (for multi-asset support)
    asset_balances = Column(JSON, default=lambda: {})  # {'BTC': 0.1, 'ETH': 2.5, 'USD': 1000}
    
    # Metadata
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_by = Column(String(50), default='system')  # 'system', 'user', 'admin'
    update_reason = Column(String(200))  # 'trade_pnl', 'manual_adjustment', 'deposit', etc.
    
    # Session reference
    session_id = Column(Integer, ForeignKey('trading_sessions.id'))
    
    # Ensure we have one current balance per session
    __table_args__ = (
        Index('idx_balance_session_updated', 'session_id', 'last_updated'),
    )
    
    created_at = Column(DateTime, default=datetime.utcnow)

    @classmethod
    def get_current_balance(cls, session_id: int, db_session) -> float:
        """Get the current balance for a session"""
        latest_balance = db_session.query(cls).filter(
            cls.session_id == session_id
        ).order_by(cls.last_updated.desc()).first()
        
        return latest_balance.total_balance if latest_balance else 0.0
    
    @classmethod
    def update_balance(cls, session_id: int, new_balance: float, 
                      update_reason: str, updated_by: str, db_session) -> 'AccountBalance':
        """Update the current balance for a session"""
        balance_record = cls(
            session_id=session_id,
            base_currency='USD',
            total_balance=new_balance,
            available_balance=new_balance,  # Simplified for now
            last_updated=datetime.utcnow(),
            updated_by=updated_by,
            update_reason=update_reason
        )
        
        db_session.add(balance_record)
        db_session.commit()
        return balance_record 