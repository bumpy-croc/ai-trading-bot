"""
Database models for trade logging and performance tracking
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, 
    Enum, ForeignKey, Index, JSON, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
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


class Trade(Base):
    """Completed trades table"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(Enum(PositionSide), nullable=False)
    source = Column(Enum(TradeSource), nullable=False, default=TradeSource.LIVE)
    
    # Trade details
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)  # Position size as % of balance
    quantity = Column(Float)  # Actual quantity traded
    
    # Timestamps
    entry_time = Column(DateTime, nullable=False, index=True)
    exit_time = Column(DateTime, nullable=False, index=True)
    
    # Performance
    pnl = Column(Float, nullable=False)  # Dollar P&L
    pnl_percent = Column(Float, nullable=False)  # Percentage P&L
    commission = Column(Float, default=0.0)
    
    # Risk management
    stop_loss = Column(Float)
    take_profit = Column(Float)
    exit_reason = Column(String(100))
    
    # Strategy information
    strategy_name = Column(String(100), nullable=False, index=True)
    strategy_config = Column(JSON)  # Store strategy parameters
    confidence_score = Column(Float)  # ML model confidence if applicable
    
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
    entry_price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)
    quantity = Column(Float)
    
    # Risk management
    stop_loss = Column(Float)
    take_profit = Column(Float)
    trailing_stop = Column(Boolean, default=False)
    
    # Timestamps
    entry_time = Column(DateTime, nullable=False, index=True)
    last_update = Column(DateTime, default=datetime.utcnow)
    
    # Current state
    current_price = Column(Float)
    unrealized_pnl = Column(Float, default=0.0)
    unrealized_pnl_percent = Column(Float, default=0.0)
    
    # Strategy information
    strategy_name = Column(String(100), nullable=False)
    confidence_score = Column(Float)
    
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
    balance = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)  # Balance + unrealized P&L
    margin_used = Column(Float, default=0.0)
    margin_available = Column(Float)
    
    # Performance metrics at this point
    total_pnl = Column(Float, default=0.0)
    daily_pnl = Column(Float, default=0.0)
    drawdown = Column(Float, default=0.0)
    
    # Position summary
    open_positions = Column(Integer, default=0)
    total_exposure = Column(Float, default=0.0)
    
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
    win_rate = Column(Float, default=0.0)
    
    # Returns
    total_return = Column(Float, default=0.0)
    total_return_percent = Column(Float, default=0.0)
    
    # Risk metrics
    max_drawdown = Column(Float, default=0.0)
    max_drawdown_duration = Column(Integer)  # In hours
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    calmar_ratio = Column(Float)
    
    # Trade analysis
    avg_win = Column(Float, default=0.0)
    avg_loss = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    expectancy = Column(Float, default=0.0)
    
    # Best/worst trades
    best_trade_pnl = Column(Float)
    worst_trade_pnl = Column(Float)
    largest_win_streak = Column(Integer, default=0)
    largest_loss_streak = Column(Integer, default=0)
    
    # By strategy breakdown
    strategy_breakdown = Column(JSON)  # Dict of strategy_name: metrics
    
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
    initial_balance = Column(Float, nullable=False)
    final_balance = Column(Float)
    
    # Strategy information
    strategy_name = Column(String(100), nullable=False)
    strategy_config = Column(JSON)
    
    # Environment
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    exchange = Column(String(50), default='binance')
    
    # Performance summary
    total_pnl = Column(Float)
    total_trades = Column(Integer, default=0)
    win_rate = Column(Float)
    max_drawdown = Column(Float)
    
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
    details = Column(JSON)  # Additional structured data
    
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
    signal_strength = Column(Float)
    confidence_score = Column(Float)
    
    # Decision factors
    indicators = Column(JSON)  # Dict of indicator values
    sentiment_data = Column(JSON)  # Sentiment scores if used
    ml_predictions = Column(JSON)  # ML model outputs if used
    
    # Execution result
    action_taken = Column(String(50))  # 'opened_long', 'closed_position', 'no_action'
    position_size = Column(Float)
    reasons = Column(JSON)  # List of reasons for the decision
    
    # Market context
    price = Column(Float)
    volume = Column(Float)
    volatility = Column(Float)
    
    # Session reference
    session_id = Column(Integer, ForeignKey('trading_sessions.id'))
    trade_id = Column(Integer, ForeignKey('trades.id'))
    
    created_at = Column(DateTime, default=datetime.utcnow) 