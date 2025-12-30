"""
Database models for trade logging and performance tracking
"""

import enum
from datetime import UTC, datetime
from typing import Any

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
    Time,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.types import TypeDecorator


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

# Type Base as Any to allow mypy to accept dynamic SQLAlchemy base class
Base: Any = declarative_base()


def utc_now() -> datetime:
    """Return an aware UTC timestamp for database defaults."""
    return datetime.now(UTC)


class PositionSide(enum.Enum):
    """Position side enumeration"""

    LONG = "LONG"
    SHORT = "SHORT"


class OrderStatus(enum.Enum):
    """Order status enumeration"""

    # Values use uppercase to match existing PostgreSQL enum labels
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class PositionStatus(enum.Enum):
    """Status of a trading position (distinct from order status)."""

    OPEN = "OPEN"  # Position is active and being held
    CLOSED = "CLOSED"  # Position has been closed/exited


class OrderType(enum.Enum):
    """Type of order in relation to its position."""

    ENTRY = "ENTRY"  # Initial order that creates the position
    PARTIAL_EXIT = "PARTIAL_EXIT"  # Order that partially closes position
    SCALE_IN = "SCALE_IN"  # Order that adds to existing position
    FULL_EXIT = "FULL_EXIT"  # Order that completely closes position


class TradeSource(enum.Enum):
    """Source of the trade"""

    LIVE = "LIVE"
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"


class EventType(enum.Enum):
    """System event types"""

    ENGINE_START = "ENGINE_START"
    ENGINE_STOP = "ENGINE_STOP"
    STRATEGY_CHANGE = "STRATEGY_CHANGE"
    MODEL_UPDATE = "MODEL_UPDATE"
    ERROR = "ERROR"
    WARNING = "WARNING"
    ALERT = "ALERT"
    BALANCE_ADJUSTMENT = "BALANCE_ADJUSTMENT"
    TEST = "TEST"  # Added for verification scripts and development diagnostics


class Trade(Base):
    """Completed trades table"""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(Enum(PositionSide, native_enum=False, create_type=False), nullable=False)
    source = Column(
        Enum(TradeSource, native_enum=False, create_type=False),
        nullable=False,
        default=TradeSource.LIVE,
    )

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

    # MFE/MAE for completed trades (percent decimals, e.g., 0.05 = +5%)
    mfe = Column(Numeric(18, 8), default=0.0)
    mae = Column(Numeric(18, 8), default=0.0)
    mfe_price = Column(Numeric(18, 8))
    mae_price = Column(Numeric(18, 8))
    mfe_time = Column(DateTime)
    mae_time = Column(DateTime)

    # Relationships
    position_id = Column(Integer, ForeignKey("positions.id"))
    session_id = Column(Integer, ForeignKey("trading_sessions.id"))

    # Indexes for performance
    __table_args__ = (
        Index("idx_trade_time", "entry_time", "exit_time"),
        Index("idx_trade_performance", "pnl_percent", "strategy_name"),
        UniqueConstraint("order_id", "session_id", name="uq_trade_order_session"),
    )

    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


class Position(Base):
    """Active positions table"""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(Enum(PositionSide, native_enum=False, create_type=False), nullable=False)
    status = Column(
        Enum(PositionStatus, name="positionstatus", native_enum=False, create_type=False),
        nullable=False,
        default=PositionStatus.OPEN,
    )

    # Position details
    entry_price = Column(Numeric(18, 8), nullable=False)
    size = Column(Numeric(18, 8), nullable=False)
    quantity = Column(Numeric(18, 8))
    entry_balance = Column(Numeric(18, 8))
    # Partial operations tracking
    original_size = Column(Numeric(18, 8))  # initial position size fraction
    current_size = Column(Numeric(18, 8))  # remaining size fraction
    partial_exits_taken = Column(Integer, default=0)
    scale_ins_taken = Column(Integer, default=0)
    last_partial_exit_price = Column(Numeric(18, 8))
    last_scale_in_price = Column(Numeric(18, 8))

    # Risk management
    stop_loss = Column(Numeric(18, 8))
    take_profit = Column(Numeric(18, 8))
    trailing_stop = Column(Boolean, default=False)
    trailing_stop_activated = Column(Boolean, default=False)
    trailing_stop_price = Column(Numeric(18, 8))
    breakeven_triggered = Column(Boolean, default=False)

    # Timestamps
    entry_time = Column(DateTime, nullable=False, index=True)
    last_update = Column(DateTime, default=utc_now)

    # Current state
    current_price = Column(Numeric(18, 8))
    unrealized_pnl = Column(Numeric(18, 8), default=0.0)
    unrealized_pnl_percent = Column(Numeric(18, 8), default=0.0)

    # Rolling MFE/MAE for active positions (percent decimals)
    mfe = Column(Numeric(18, 8), default=0.0)
    mae = Column(Numeric(18, 8), default=0.0)
    mfe_price = Column(Numeric(18, 8))
    mae_price = Column(Numeric(18, 8))
    mfe_time = Column(DateTime)
    mae_time = Column(DateTime)

    # Strategy information
    strategy_name = Column(String(100), nullable=False)
    confidence_score = Column(Numeric(18, 8))

    # Exchange information
    exchange = Column(String(50), default="binance")

    # Order tracking for live trading
    entry_order_id = Column(String(100))  # Exchange order ID for entry
    stop_loss_order_id = Column(String(100))  # Exchange order ID for server-side stop-loss

    # Relationships
    trades = relationship("Trade", backref="position")
    partial_trades = relationship("PartialTrade", backref="position")
    orders = relationship("Order", backref="position", cascade="all, delete-orphan")
    session_id = Column(Integer, ForeignKey("trading_sessions.id"))

    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Time-based exit fields
    max_holding_until = Column(DateTime)  # When position must be closed
    end_of_day_exit = Column(Boolean, default=False)
    weekend_exit = Column(Boolean, default=False)
    time_restriction_group = Column(String(50))


class Order(Base):
    """Individual orders table - tracks all orders associated with positions."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True)
    position_id = Column(Integer, ForeignKey("positions.id"), nullable=False, index=True)
    order_type = Column(Enum(OrderType, native_enum=False, create_type=False), nullable=False)
    status = Column(Enum(OrderStatus, native_enum=False, create_type=False), nullable=False)

    # Order identification
    exchange_order_id = Column(String(100), unique=True, index=True)  # From exchange
    internal_order_id = Column(String(100), nullable=False, index=True)  # Our reference

    # Order details
    symbol = Column(String(20), nullable=False)
    side = Column(
        Enum(PositionSide, native_enum=False, create_type=False), nullable=False
    )  # BUY/SELL
    quantity = Column(Numeric(18, 8), nullable=False)
    price = Column(Numeric(18, 8))  # Limit price (null for market orders)

    # Execution details
    filled_quantity = Column(Numeric(18, 8), default=0)
    filled_price = Column(Numeric(18, 8))
    commission = Column(Numeric(18, 8), default=0)

    # Timestamps
    created_at = Column(DateTime, default=utc_now, index=True)
    filled_at = Column(DateTime)
    cancelled_at = Column(DateTime)
    last_update = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Strategy context
    strategy_name = Column(String(100), nullable=False)
    session_id = Column(Integer, ForeignKey("trading_sessions.id"))

    # Partial operation context (if applicable)
    target_level = Column(Integer)  # For partial exits/scale-ins
    size_fraction = Column(Numeric(18, 8))  # Fraction of original position

    __table_args__ = (
        Index("idx_order_position_type", "position_id", "order_type"),
        Index("idx_order_status_created", "status", "created_at"),
        UniqueConstraint("internal_order_id", "session_id", name="uq_order_internal_session"),
    )


class PartialOperationType(enum.Enum):
    PARTIAL_EXIT = "PARTIAL_EXIT"
    SCALE_IN = "SCALE_IN"


class PartialTrade(Base):
    __tablename__ = "partial_trades"

    id = Column(Integer, primary_key=True)
    position_id = Column(Integer, ForeignKey("positions.id"), index=True, nullable=False)
    operation_type = Column(
        Enum(PartialOperationType, native_enum=False, create_type=False), nullable=False
    )
    size = Column(Numeric(18, 8), nullable=False)  # Fraction of original size executed
    price = Column(Numeric(18, 8), nullable=False)
    pnl = Column(Numeric(18, 8))  # Realized PnL in currency units
    target_level = Column(Integer)
    timestamp = Column(DateTime, default=utc_now, index=True)

    __table_args__ = (Index("idx_partial_trade_position", "position_id", "timestamp"),)


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

    # Enhanced risk metrics
    sharpe_ratio = Column(Numeric(18, 8))
    sortino_ratio = Column(Numeric(18, 8))
    calmar_ratio = Column(Numeric(18, 8))
    var_95 = Column(Numeric(18, 8))

    # Position summary
    open_positions = Column(Integer, default=0)
    total_exposure = Column(Numeric(18, 8), default=0.0)

    # Session reference
    session_id = Column(Integer, ForeignKey("trading_sessions.id"))

    # Index for efficient time-series queries
    __table_args__ = (Index("idx_account_time", "timestamp"),)

    created_at = Column(DateTime, default=utc_now)


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
    current_drawdown = Column(Numeric(18, 8), default=0.0)
    max_drawdown_duration = Column(Integer)  # In hours
    sharpe_ratio = Column(Numeric(18, 8))
    sortino_ratio = Column(Numeric(18, 8))
    calmar_ratio = Column(Numeric(18, 8))
    var_95 = Column(Numeric(18, 8))  # Value at Risk (95% confidence)

    # Trade analysis
    avg_win = Column(Numeric(18, 8), default=0.0)
    avg_loss = Column(Numeric(18, 8), default=0.0)
    profit_factor = Column(Numeric(18, 8), default=0.0)
    expectancy = Column(Numeric(18, 8), default=0.0)
    avg_trade_duration_hours = Column(Numeric(18, 8), default=0.0)

    # Best/worst trades
    best_trade_pnl = Column(Numeric(18, 8))
    worst_trade_pnl = Column(Numeric(18, 8))
    largest_win_streak = Column(Integer, default=0)
    largest_loss_streak = Column(Integer, default=0)
    consecutive_wins_current = Column(Integer, default=0)
    consecutive_losses_current = Column(Integer, default=0)

    # Costs
    total_fees_paid = Column(Numeric(18, 8), default=0.0)
    total_slippage_cost = Column(Numeric(18, 8), default=0.0)

    # By strategy breakdown
    strategy_breakdown = Column(JSONType)  # Dict of strategy_name: metrics

    # Session reference
    session_id = Column(Integer, ForeignKey("trading_sessions.id"))

    # Ensure unique metrics per period
    __table_args__ = (
        UniqueConstraint("period", "period_start", "session_id"),
        Index("idx_metrics_period", "period", "period_start"),
    )

    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


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
    mode = Column(Enum(TradeSource, native_enum=False, create_type=False), nullable=False)
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

    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


class SystemEvent(Base):
    """System events and alerts table"""

    __tablename__ = "system_events"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    event_type = Column(Enum(EventType, native_enum=False, create_type=False), nullable=False)
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

    created_at = Column(DateTime, default=utc_now)


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
    signal_strength = Column(Float(asdecimal=False))
    confidence_score = Column(Float(asdecimal=False))

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

    created_at = Column(DateTime, default=utc_now)


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
    last_updated = Column(DateTime, nullable=False, default=utc_now)
    updated_by = Column(String(50), default="system")  # 'system', 'user', 'admin'
    update_reason = Column(String(200))  # 'trade_pnl', 'manual_adjustment', 'deposit', etc.

    # Session reference
    session_id = Column(Integer, ForeignKey("trading_sessions.id"))

    # Ensure we have one current balance per session
    __table_args__ = (Index("idx_balance_session_updated", "session_id", "last_updated"),)

    created_at = Column(DateTime, default=utc_now)

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
            last_updated=datetime.now(UTC),
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
    timestamp = Column(DateTime, nullable=False, index=True, default=utc_now)
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

    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


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

    created_at = Column(DateTime, default=utc_now)


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

    created_at = Column(DateTime, default=utc_now)


class RiskAdjustment(Base):
    """Risk parameter adjustments tracking"""

    __tablename__ = "risk_adjustments"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Adjustment type and trigger
    adjustment_type = Column(String(50), nullable=False)  # 'drawdown', 'performance', 'volatility'
    trigger_reason = Column(String(200))  # Detailed reason for adjustment

    # Original and adjusted values
    parameter_name = Column(
        String(100), nullable=False
    )  # e.g., 'position_size_factor', 'stop_loss_multiplier'
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

    created_at = Column(DateTime, default=utc_now)


class CorrelationMatrix(Base):
    """Stores pairwise correlation values for symbol pairs."""

    __tablename__ = "correlation_matrix"

    id = Column(Integer, primary_key=True)
    symbol_pair = Column(String(50), index=True)  # e.g., "BTCUSDT-ETHUSDT" (sorted order)
    correlation_value = Column(Numeric(18, 8))
    p_value = Column(Numeric(18, 8))  # Optional statistical significance if computed
    sample_size = Column(Integer)
    last_updated = Column(DateTime)
    window_days = Column(Integer)

    __table_args__ = (Index("idx_corr_pair_updated", "symbol_pair", "last_updated"),)


class PortfolioExposure(Base):
    """Aggregated exposure per correlation group for portfolio-level limits."""

    __tablename__ = "portfolio_exposures"

    id = Column(Integer, primary_key=True)
    correlation_group = Column(String(100), index=True)
    total_exposure = Column(Numeric(18, 8))
    position_count = Column(Integer)
    symbols = Column(JSONType)  # List of symbols in group
    last_updated = Column(DateTime)


class PredictionCache(Base):
    """Cache for prediction results to avoid redundant inference"""

    __tablename__ = "prediction_cache"

    id = Column(Integer, primary_key=True)
    cache_key = Column(String(255), nullable=False, unique=True, index=True)
    model_name = Column(String(100), nullable=False, index=True)

    # Input features hash for cache key generation
    features_hash = Column(String(64), nullable=False, index=True)

    # Cached prediction results
    predicted_price = Column(Numeric(18, 8), nullable=False)
    confidence = Column(Numeric(18, 8), nullable=False)
    direction = Column(Integer, nullable=False)  # 1, 0, -1

    # Cache metadata
    created_at = Column(DateTime, default=utc_now, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Configuration context for cache invalidation
    config_hash = Column(String(64), nullable=False)  # Hash of model configuration

    __table_args__ = (
        Index("idx_pred_cache_expires", "expires_at"),
        Index("idx_pred_cache_model_config", "model_name", "config_hash"),
        Index("idx_pred_cache_access", "last_accessed"),
    )


class StrategyStatus(enum.Enum):
    """Strategy status enumeration"""

    EXPERIMENTAL = "EXPERIMENTAL"
    TESTING = "TESTING"
    PRODUCTION = "PRODUCTION"
    RETIRED = "RETIRED"
    DEPRECATED = "DEPRECATED"


class StrategyRegistry(Base):
    """Strategy registry with version control and metadata"""

    __tablename__ = "strategy_registry"

    id = Column(Integer, primary_key=True)
    strategy_id = Column(String(100), nullable=False, unique=True, index=True)
    name = Column(String(200), nullable=False, index=True)
    version = Column(String(20), nullable=False)

    # Lineage tracking
    parent_id = Column(String(100), index=True)
    lineage_path = Column(JSONType, default=lambda: [])  # Path from root ancestor
    branch_name = Column(String(100))
    merge_source = Column(String(100))

    # Metadata
    created_at = Column(DateTime, nullable=False, default=utc_now, index=True)
    created_by = Column(String(100), nullable=False)
    description = Column(Text)
    tags = Column(JSONType, default=lambda: [])
    status = Column(
        Enum(StrategyStatus, native_enum=False, create_type=False),
        nullable=False,
        default=StrategyStatus.EXPERIMENTAL,
    )

    # Component configurations
    signal_generator_config = Column(JSONType, nullable=False)
    risk_manager_config = Column(JSONType, nullable=False)
    position_sizer_config = Column(JSONType, nullable=False)
    regime_detector_config = Column(JSONType, nullable=False)

    # Additional parameters and metadata
    parameters = Column(JSONType, default=lambda: {})
    performance_summary = Column(JSONType)
    validation_results = Column(JSONType)

    # Integrity checksums
    config_hash = Column(String(64), nullable=False, index=True)
    component_hash = Column(String(64), nullable=False, index=True)

    # Relationships
    # Note: parent relationship removed due to foreign key constraint issues
    # The foreign key constraint is handled in the migration file
    versions = relationship("StrategyVersion", backref="strategy", cascade="all, delete-orphan")
    performance_records = relationship(
        "StrategyPerformance", backref="strategy", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_strategy_name_version", "name", "version"),
        Index("idx_strategy_status_created", "status", "created_at"),
        Index("idx_strategy_parent_created", "parent_id", "created_at"),
        UniqueConstraint("name", "version", name="uq_strategy_name_version"),
        UniqueConstraint("strategy_id", name="uq_strategy_id"),
    )

    created_at_db = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


class StrategyVersion(Base):
    """Strategy version tracking"""

    __tablename__ = "strategy_versions"

    id = Column(Integer, primary_key=True)
    strategy_id = Column(
        String(100), ForeignKey("strategy_registry.strategy_id"), nullable=False, index=True
    )
    version = Column(String(20), nullable=False)

    # Version metadata
    created_at = Column(DateTime, nullable=False, default=utc_now, index=True)
    changes = Column(JSONType, nullable=False)  # List of changes
    performance_delta = Column(JSONType)  # Performance comparison with previous version
    is_major = Column(Boolean, default=False)

    # Version-specific data
    component_changes = Column(JSONType)  # Which components changed
    parameter_changes = Column(JSONType)  # Parameter differences

    __table_args__ = (
        Index("idx_version_strategy_created", "strategy_id", "created_at"),
        UniqueConstraint("strategy_id", "version", name="uq_strategy_version"),
    )

    created_at_db = Column(DateTime, default=utc_now)


class StrategyPerformance(Base):
    """Strategy performance tracking and comparison"""

    __tablename__ = "strategy_performance"

    id = Column(Integer, primary_key=True)
    strategy_id = Column(
        String(100), ForeignKey("strategy_registry.strategy_id"), nullable=False, index=True
    )
    version = Column(String(20), nullable=False)

    # Performance period
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False, index=True)
    period_type = Column(String(20), nullable=False)  # 'backtest', 'paper', 'live'

    # Core performance metrics
    total_return = Column(Numeric(18, 8), nullable=False)
    total_return_pct = Column(Numeric(18, 8), nullable=False)
    sharpe_ratio = Column(Numeric(18, 8))
    max_drawdown = Column(Numeric(18, 8))
    win_rate = Column(Numeric(18, 8))

    # Trade statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    avg_trade_duration_hours = Column(Numeric(18, 8))

    # Risk metrics
    volatility = Column(Numeric(18, 8))
    sortino_ratio = Column(Numeric(18, 8))
    calmar_ratio = Column(Numeric(18, 8))
    var_95 = Column(Numeric(18, 8))  # Value at Risk 95%

    # Component attribution
    signal_generator_contribution = Column(Numeric(18, 8))
    risk_manager_contribution = Column(Numeric(18, 8))
    position_sizer_contribution = Column(Numeric(18, 8))

    # Regime-specific performance
    regime_performance = Column(JSONType)  # Performance by regime type

    # Additional metrics
    additional_metrics = Column(JSONType, default=lambda: {})

    # Test configuration
    test_symbol = Column(String(20))
    test_timeframe = Column(String(10))
    test_parameters = Column(JSONType)

    __table_args__ = (
        Index("idx_perf_strategy_period", "strategy_id", "period_start", "period_end"),
        Index("idx_perf_return_sharpe", "total_return_pct", "sharpe_ratio"),
        Index("idx_perf_period_type", "period_type", "period_start"),
    )

    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


class StrategyLineage(Base):
    """Strategy lineage and evolutionary tracking"""

    __tablename__ = "strategy_lineage"

    id = Column(Integer, primary_key=True)
    ancestor_id = Column(
        String(100), ForeignKey("strategy_registry.strategy_id"), nullable=False, index=True
    )
    descendant_id = Column(
        String(100), ForeignKey("strategy_registry.strategy_id"), nullable=False, index=True
    )

    # Relationship metadata
    relationship_type = Column(String(20), nullable=False)  # 'parent', 'branch', 'merge'
    generation_distance = Column(Integer, nullable=False)  # How many generations apart

    # Evolution tracking
    created_at = Column(DateTime, nullable=False, default=utc_now)
    evolution_reason = Column(String(200))  # Why this evolution was made
    change_impact = Column(JSONType)  # Impact analysis of changes

    # Performance comparison
    performance_improvement = Column(Numeric(18, 8))  # Performance delta
    risk_change = Column(Numeric(18, 8))  # Risk profile change

    __table_args__ = (
        Index("idx_lineage_ancestor", "ancestor_id", "generation_distance"),
        Index("idx_lineage_descendant", "descendant_id", "generation_distance"),
        UniqueConstraint("ancestor_id", "descendant_id", name="uq_lineage_pair"),
    )

    created_at_db = Column(DateTime, default=utc_now)
