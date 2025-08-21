"""
Database models for trade logging and performance tracking
"""

import enum
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,  # Added Float and JSON
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.types import TypeDecorator
from sqlalchemy import Time


# Portable JSON that chooses JSONB on PostgreSQL and JSON elsewhere (e.g., SQLite)
class PortableJSON(TypeDecorator):
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            try:  # pragma: no cover - depends on installed dialect
                from sqlalchemy.dialects.postgresql import JSONB  # type: ignore

                return dialect.type_descriptor(JSONB())
            except Exception:
                return dialect.type_descriptor(JSON())
        return dialect.type_descriptor(JSON())


JSONType = PortableJSON

Base = declarative_base()


class PositionSide(enum.Enum):
    """Position side enumeration"""

    LONG = "long"
    SHORT = "short"


class OrderStatus(enum.Enum):
    """Order status enumeration"""

    # Values use lowercase to match existing PostgreSQL enum labels
    PENDING = "pending"
    OPEN = "open"
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

    __tablename__ = "trades"

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
    strategy_config = Column(JSONType)  # Store strategy parameters
    confidence_score = Column(Numeric(18, 8))  # ML model confidence if applicable

    # Additional metadata
    order_id = Column(String(100))
    exchange = Column(String(50), default="binance")
    timeframe = Column(String(10))

    # Relationships
    position_id = Column(Integer, ForeignKey("positions.id"))
    session_id = Column(Integer, ForeignKey("trading_sessions.id"))

    # Indexes for performance
    __table_args__ = (
        Index("idx_trade_time", "entry_time", "exit_time"),
        Index("idx_trade_performance", "pnl_percent", "strategy_name"),
        UniqueConstraint("order_id", "session_id", name="uq_trade_order_session"),
    )

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Position(Base):
    """Active positions table"""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(Enum(PositionSide), nullable=False)
    status = Column(Enum(OrderStatus), nullable=False, default=OrderStatus.OPEN)

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
    order_id = Column(String(100))
    exchange = Column(String(50), default="binance")

    # Relationships
    trades = relationship("Trade", backref="position")
    session_id = Column(Integer, ForeignKey("trading_sessions.id"))

    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint("order_id", "session_id", name="uq_position_order_session"),)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Time-based exit fields
    max_holding_until = Column(DateTime)  # When position must be closed
    end_of_day_exit = Column(Boolean, default=False)
    weekend_exit = Column(Boolean, default=False)
    time_restriction_group = Column(String(50))


class MarketSession(Base):
    __tablename__ = "market_sessions"

    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, index=True)
    timezone = Column(String(50), default="UTC")
    open_time = Column(Time)
    close_time = Column(Time)
    days_of_week = Column(JSONType)  # e.g., [1,2,3,4,5]
    is_24h = Column(Boolean, default=False)


class AccountHistory(Base):
    """Account balance history table"""

    __tablename__ = "account_history"

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
    session_id = Column(Integer, ForeignKey("trading_sessions.id"))

    # Index for efficient time-series queries
    __table_args__ = (Index("idx_account_time", "timestamp"),)

    created_at = Column(DateTime, default=datetime.utcnow)


class PerformanceMetrics(Base):
    """Aggregated performance metrics table"""

    __tablename__ = "performance_metrics"

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
    strategy_breakdown = Column(JSONType)  # Dict of strategy_name: metrics

    # Session reference
    session_id = Column(Integer, ForeignKey("trading_sessions.id"))

    # Ensure unique metrics per period
    __table_args__ = (
        UniqueConstraint("period", "period_start", "session_id"),
        Index("idx_metrics_period", "period", "period_start"),
    )

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TradingSession(Base):
    """Trading session tracking table"""

    __tablename__ = "trading_sessions"

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
    strategy_config = Column(JSONType)

    # Environment
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    exchange = Column(String(50), default="binance")

    # Time-exit configuration
    time_exit_config = Column(JSONType)
    market_timezone = Column(String(50))

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

    __tablename__ = "system_events"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    event_type = Column(Enum(EventType), nullable=False)
    severity = Column(String(20), default="info")  # 'info', 'warning', 'error', 'critical'

    # Event details
    message = Column(Text, nullable=False)
    details = Column(JSONType)  # Additional structured data

    # Context
    component = Column(String(100))  # Which part of the system
    error_code = Column(String(50))
    stack_trace = Column(Text)

    # Alert status
    alert_sent = Column(Boolean, default=False)
    alert_method = Column(String(50))  # 'telegram', 'email', 'slack'

    # Session reference
    session_id = Column(Integer, ForeignKey("trading_sessions.id"))

    # Index for efficient queries
    __table_args__ = (
        Index("idx_event_time_type", "timestamp", "event_type"),
        Index("idx_event_severity", "severity", "timestamp"),
    )

    created_at = Column(DateTime, default=datetime.utcnow)


class StrategyExecution(Base):
    """Detailed strategy execution logs"""

    __tablename__ = "strategy_executions"

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
    indicators = Column(JSONType)  # Dict of indicator values
    sentiment_data = Column(JSONType)  # Sentiment scores if used
    ml_predictions = Column(JSONType)  # ML model outputs if used

    # Execution result
    action_taken = Column(String(50))  # 'opened_long', 'closed_position', 'no_action'
    position_size = Column(Numeric(18, 8))
    reasons = Column(JSONType)  # List of reasons for the decision

    # Market context
    price = Column(Numeric(18, 8))
    volume = Column(Numeric(18, 8))
    volatility = Column(Numeric(18, 8))

    # Session reference
    session_id = Column(Integer, ForeignKey("trading_sessions.id"))
    trade_id = Column(Integer, ForeignKey("trades.id"))

    created_at = Column(DateTime, default=datetime.utcnow)


class AccountBalance(Base):
    """Current account balance tracking table"""

    __tablename__ = "account_balances"

    id = Column(Integer, primary_key=True)

    # Balance information
    base_currency = Column(String(10), nullable=False, default="USD")  # USD, BTC, ETH, etc.
    total_balance = Column(Float, nullable=False)  # Total balance in base currency
    available_balance = Column(Float, nullable=False)  # Available for trading
    reserved_balance = Column(Float, default=0.0)  # Reserved in open positions

    # Balance breakdown by asset (for multi-asset support)
    asset_balances = Column(JSONType, default=lambda: {})  # {'BTC': 0.1, 'ETH': 2.5, 'USD': 1000}

    # Metadata
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_by = Column(String(50), default="system")  # 'system', 'user', 'admin'
    update_reason = Column(String(200))  # 'trade_pnl', 'manual_adjustment', 'deposit', etc.

    # Session reference
    session_id = Column(Integer, ForeignKey("trading_sessions.id"))

    # Ensure we have one current balance per session
    __table_args__ = (Index("idx_balance_session_updated", "session_id", "last_updated"),)

    created_at = Column(DateTime, default=datetime.utcnow)

    @classmethod
    def get_current_balance(cls, session_id: int, db_session) -> float:
        """Get the current balance for a session"""
        latest_balance = (
            db_session.query(cls)
            .filter(cls.session_id == session_id)
            .order_by(cls.last_updated.desc())
            .first()
        )

        return latest_balance.total_balance if latest_balance else 0.0

    @classmethod
    def update_balance(
        cls, session_id: int, new_balance: float, update_reason: str, updated_by: str, db_session
    ) -> "AccountBalance":
        """Update the current balance for a session"""
        balance_record = cls(
            session_id=session_id,
            base_currency="USD",
            total_balance=new_balance,
            available_balance=new_balance,  # Simplified for now
            last_updated=datetime.utcnow(),
            updated_by=updated_by,
            update_reason=update_reason,
        )

        db_session.add(balance_record)
        db_session.commit()
        return balance_record


class OptimizationCycle(Base):
    """Stores optimizer proposals, validations, and decisions per cycle."""

    __tablename__ = "optimization_cycles"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    strategy_name = Column(String(100), nullable=False)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)

    baseline_metrics = Column(JSONType)  # KPIs from baseline run
    candidate_params = Column(JSONType)  # Proposed parameter changes
    candidate_metrics = Column(JSONType)  # KPIs from candidate run

    validator_report = Column(JSONType)  # p-value, effect size, pass/fail
    decision = Column(String(20))  # 'propose', 'reject', 'apply'

    # Relationships
    session_id = Column(Integer, ForeignKey("trading_sessions.id"), nullable=True)

    __table_args__ = (
        Index("idx_opt_cycle_time", "timestamp"),
        Index("idx_opt_cycle_strategy", "strategy_name", "symbol", "timeframe"),
    )

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PredictionPerformance(Base):
    """Aggregated prediction calibration and accuracy metrics."""

    __tablename__ = "prediction_performance"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    horizon = Column(Integer, default=1)

    # Calibration/accuracy
    mae = Column(Numeric(18, 8))
    rmse = Column(Numeric(18, 8))
    mape = Column(Numeric(18, 8))
    ic = Column(Numeric(18, 8))  # information coefficient (rank correlation)

    # Distribution shift indicators
    mean_pred = Column(Numeric(18, 8))
    std_pred = Column(Numeric(18, 8))
    mean_real = Column(Numeric(18, 8))
    std_real = Column(Numeric(18, 8))

    # Model context
    strategy_name = Column(String(100), nullable=False)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)

    __table_args__ = (
        Index("idx_pred_perf_time", "timestamp"),
        Index("idx_pred_perf_model", "model_name", "horizon"),
    )

    created_at = Column(DateTime, default=datetime.utcnow)


class DynamicPerformanceMetrics(Base):
    """Dynamic performance metrics for adaptive risk management"""

    __tablename__ = "dynamic_performance_metrics"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Rolling performance metrics
    rolling_win_rate = Column(Numeric(18, 8))
    rolling_sharpe_ratio = Column(Numeric(18, 8))
    current_drawdown = Column(Numeric(18, 8))
    volatility_30d = Column(Numeric(18, 8))
    consecutive_losses = Column(Integer, default=0)
    consecutive_wins = Column(Integer, default=0)
    
    # Risk adjustment factor applied
    risk_adjustment_factor = Column(Numeric(18, 8), default=1.0)
    
    # Additional performance indicators
    profit_factor = Column(Numeric(18, 8))
    expectancy = Column(Numeric(18, 8))
    avg_trade_duration_hours = Column(Numeric(18, 8))
    
    # Session reference
    session_id = Column(Integer, ForeignKey("trading_sessions.id"), nullable=False)
    
    __table_args__ = (
        Index("idx_dynamic_perf_timestamp", "timestamp"),
        Index("idx_dynamic_perf_session", "session_id"),
    )

    created_at = Column(DateTime, default=datetime.utcnow)


class RiskAdjustment(Base):
    """Risk parameter adjustments tracking"""

    __tablename__ = "risk_adjustments"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Adjustment type and trigger
    adjustment_type = Column(String(50), nullable=False)  # 'drawdown', 'performance', 'volatility'
    trigger_reason = Column(String(200))  # Detailed reason for adjustment
    
    # Original and adjusted values
    parameter_name = Column(String(100), nullable=False)  # e.g., 'position_size_factor', 'stop_loss_multiplier'
    original_value = Column(Numeric(18, 8), nullable=False)
    adjusted_value = Column(Numeric(18, 8), nullable=False)
    adjustment_factor = Column(Numeric(18, 8), nullable=False)
    
    # Context for the adjustment
    current_drawdown = Column(Numeric(18, 8))
    performance_score = Column(Numeric(18, 8))
    volatility_level = Column(Numeric(18, 8))
    
    # Duration and effectiveness
    duration_minutes = Column(Integer)  # How long the adjustment was active
    trades_during_adjustment = Column(Integer, default=0)
    pnl_during_adjustment = Column(Numeric(18, 8))
    
    # Session reference
    session_id = Column(Integer, ForeignKey("trading_sessions.id"), nullable=False)
    
    __table_args__ = (
        Index("idx_risk_adj_timestamp", "timestamp"),
        Index("idx_risk_adj_type", "adjustment_type"),
        Index("idx_risk_adj_session", "session_id"),
    )

    created_at = Column(DateTime, default=datetime.utcnow)
