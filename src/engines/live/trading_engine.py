from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from src.config import get_config
from src.config.constants import (
    DEFAULT_ACCOUNT_SNAPSHOT_INTERVAL,
    DEFAULT_CHECK_INTERVAL,
    DEFAULT_DATA_FRESHNESS_THRESHOLD,
    DEFAULT_DYNAMIC_RISK_ENABLED,
    DEFAULT_END_OF_DAY_FLAT,
    DEFAULT_ERROR_COOLDOWN,
    DEFAULT_INITIAL_BALANCE,
    DEFAULT_MARKET_TIMEZONE,
    DEFAULT_MAX_CHECK_INTERVAL,
    DEFAULT_MAX_HOLDING_HOURS,
    DEFAULT_MIN_CHECK_INTERVAL,
    DEFAULT_SLEEP_POLL_INTERVAL,
    DEFAULT_TIME_RESTRICTIONS,
    DEFAULT_WEEKEND_FLAT,
)
from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.coinbase_provider import CoinbaseProvider
from src.data_providers.data_provider import DataProvider
from src.data_providers.exchange_interface import OrderSide
from src.data_providers.exchange_interface import (
    OrderStatus as ExchangeOrderStatus,
)
from src.data_providers.sentiment_provider import SentimentDataProvider
from src.database.manager import DatabaseManager
from src.database.models import TradeSource

# Modular handlers (optional injection for testability)
from src.engines.live.data.market_data_handler import MarketDataHandler
from src.engines.live.execution.entry_handler import LiveEntryHandler, LiveEntrySignal
from src.engines.live.execution.execution_engine import LiveExecutionEngine
from src.engines.live.execution.exit_handler import LiveExitHandler
from src.engines.live.execution.position_tracker import (
    LivePosition,
    LivePositionTracker,
)
from src.engines.live.health.health_monitor import HealthMonitor
from src.engines.live.logging.event_logger import LiveEventLogger
from src.engines.live.strategy_manager import StrategyManager
from src.engines.shared.dynamic_risk_handler import DynamicRiskHandler
from src.engines.shared.models import (
    BaseTrade,
    PositionSide,
)
from src.engines.shared.partial_operations_manager import PartialOperationsManager
from src.engines.shared.policy_hydration import apply_policies_to_engine
from src.engines.shared.risk_configuration import (
    build_trailing_stop_policy,
    merge_dynamic_risk_config,
)
from src.infrastructure.logging.context import set_context, update_context
from src.infrastructure.logging.events import (
    log_data_event,
    log_engine_event,
    log_order_event,
    log_risk_event,
)
from src.position_management.correlation_engine import CorrelationConfig, CorrelationEngine
from src.position_management.dynamic_risk import DynamicRiskConfig, DynamicRiskManager
from src.position_management.partial_manager import PartialExitPolicy
from src.position_management.time_exits import TimeExitPolicy, TimeRestrictions
from src.position_management.trailing_stops import TrailingStopPolicy
from src.regime.detector import RegimeDetector
from src.risk.risk_manager import RiskManager, RiskParameters
from src.strategies.components import Position as ComponentPosition
from src.strategies.components import RuntimeContext, Signal, SignalDirection, StrategyRuntime
from src.strategies.components import Strategy as ComponentStrategy

from .account_sync import AccountSynchronizer
from .order_tracker import OrderTracker

logger = logging.getLogger(__name__)

# Type aliases for backward compatibility - use shared models
# Position uses LivePosition which has stop_loss_order_id for server-side stop tracking
Position = LivePosition
# Trade uses BaseTrade which has all required fields plus MFE/MAE tracking
Trade = BaseTrade


def _create_exchange_provider(provider: str, config: dict, testnet: bool = False):
    """Factory to create exchange provider and return (provider_instance, provider_name).

    Args:
        provider: Exchange provider name ('binance' or 'coinbase')
        config: Configuration dict containing API credentials
        testnet: If True, use testnet credentials and endpoint
    """
    if provider == "coinbase":
        api_key = config.get("COINBASE_API_KEY")
        api_secret = config.get("COINBASE_API_SECRET")
        if api_key and api_secret:
            return CoinbaseProvider(api_key, api_secret, testnet=testnet), "Coinbase"
        else:
            return None, "Coinbase (no credentials)"
    else:
        # Use testnet credentials if testnet mode is enabled, otherwise use production
        if testnet:
            api_key = config.get("BINANCE_TESTNET_API_KEY")
            api_secret = config.get("BINANCE_TESTNET_API_SECRET")
            provider_name = "Binance Testnet"
        else:
            api_key = config.get("BINANCE_API_KEY")
            api_secret = config.get("BINANCE_API_SECRET")
            provider_name = "Binance"

        if api_key and api_secret:
            return BinanceProvider(api_key, api_secret, testnet=testnet), provider_name
        else:
            return None, f"{provider_name} (no credentials)"


class LiveTradingEngine:
    """
    Advanced live trading engine that executes strategies in real-time.

    Features:
    - Real-time data streaming
    - Actual order execution
    - Position management
    - Risk management
    - Sentiment integration
    - Error handling & recovery
    - Performance monitoring
    - Graceful shutdown
    """

    def __init__(
        self,
        strategy: ComponentStrategy | StrategyRuntime,
        data_provider: DataProvider,
        sentiment_provider: SentimentDataProvider | None = None,
        risk_parameters: RiskParameters | None = None,
        check_interval: int = DEFAULT_CHECK_INTERVAL,  # seconds
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
        max_position_size: float = 0.1,  # 10% of balance per position
        enable_live_trading: bool = False,  # Safety flag - must be explicitly enabled
        log_trades: bool = True,
        alert_webhook_url: str | None = None,
        enable_hot_swapping: bool = True,  # Enable strategy hot-swapping
        resume_from_last_balance: bool = True,  # Resume balance from last account snapshot
        database_url: str | None = None,  # Database connection URL
        max_consecutive_errors: int = 10,  # Maximum consecutive errors before shutdown
        account_snapshot_interval: int = DEFAULT_ACCOUNT_SNAPSHOT_INTERVAL,  # Account snapshot interval in seconds (30 minutes)
        provider: str = "binance",  # 'binance' (default) or 'coinbase'
        testnet: bool = False,  # Use exchange testnet (separate credentials)
        # Dynamic risk management
        enable_dynamic_risk: bool = DEFAULT_DYNAMIC_RISK_ENABLED,
        dynamic_risk_config: DynamicRiskConfig | None = None,
        time_exit_policy: TimeExitPolicy | None = None,
        trailing_stop_policy: TrailingStopPolicy | None = None,
        partial_manager: PartialExitPolicy | None = None,
        enable_partial_operations: bool = False,
        # Execution realism parameters (parity with backtest engine)
        fee_rate: float = 0.001,  # 0.1% per trade (entry + exit)
        slippage_rate: float = 0.0005,  # 0.05% slippage per trade
        use_high_low_for_stops: bool = True,  # Check candle high/low for SL/TP detection
        max_filled_price_deviation: float = 0.5,  # Filled-price deviation threshold
        # Handler injection (all optional - defaults created if not provided)
        position_tracker: LivePositionTracker | None = None,
        execution_engine: LiveExecutionEngine | None = None,
        entry_handler: LiveEntryHandler | None = None,
        exit_handler: LiveExitHandler | None = None,
        market_data_handler: MarketDataHandler | None = None,
        event_logger: LiveEventLogger | None = None,
        health_monitor: HealthMonitor | None = None,
    ):
        """
        Initialize the live trading engine.

        Parameters
        ----------
        resume_from_last_balance : bool, optional
            If True, the engine attempts to fetch the most recent recorded
            account balance from the database and use it as the starting
            balance (`current_balance`). This is useful when restarting the
            engine so that equity is not reset to the `initial_balance` value.
            Defaults to True.
        account_snapshot_interval : int, optional
            How often to log account snapshots to database in seconds.
            Defaults to 1800 (30 minutes). Set to 0 to disable snapshots.
        """

        # Validate inputs
        if initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        if max_position_size <= 0 or max_position_size > 1:
            raise ValueError("Max position size must be between 0 and 1")
        if check_interval <= 0:
            raise ValueError("Check interval must be positive")
        if account_snapshot_interval < 0:
            raise ValueError("Account snapshot interval must be non-negative")

        self._runtime_dataset = None
        self._runtime_warmup = 0
        self._configure_strategy(strategy)
        self.data_provider = data_provider
        self.sentiment_provider = sentiment_provider

        component_risk = None
        component_risk_params = None
        if isinstance(self.strategy, ComponentStrategy):
            component_risk = getattr(self.strategy, "risk_manager", None)
            component_risk_params = self._extract_component_risk_parameters(component_risk)

        merged_risk_parameters = self._merge_risk_parameters(risk_parameters, component_risk_params)
        self.risk_manager = RiskManager(merged_risk_parameters)

        # Share the canonical risk manager with component strategies via the adapter.
        if isinstance(self.strategy, ComponentStrategy):
            if hasattr(component_risk, "bind_core_manager"):
                try:
                    component_risk.bind_core_manager(self.risk_manager)
                except Exception as bind_error:
                    logger.warning(
                        "Failed to bind core risk manager to component strategy: %s. "
                        "Component risk limits may not be enforced.",
                        bind_error,
                        exc_info=True,
                    )
            if hasattr(component_risk, "set_strategy_overrides"):
                overrides = getattr(self.strategy, "_risk_overrides", None)
                if overrides:
                    try:
                        component_risk.set_strategy_overrides(overrides)
                    except Exception as override_error:
                        logger.warning(
                            "Failed to propagate risk overrides to component manager: %s. "
                            "Strategy-specific risk parameters may not apply.",
                            override_error,
                            exc_info=True,
                        )

        # Trailing stop policy
        self.trailing_stop_policy = trailing_stop_policy or self._build_trailing_policy()
        self._trailing_stop_opt_in = self.trailing_stop_policy is not None

        # Dynamic risk management
        self.enable_dynamic_risk = enable_dynamic_risk
        self.dynamic_risk_manager = None
        self._component_dynamic_risk_config: DynamicRiskConfig | None = None
        if enable_dynamic_risk:
            config = dynamic_risk_config or DynamicRiskConfig()
            # Will be initialized after db_manager is available
            self._dynamic_risk_config = config

        # Cache component-provided correlation context to avoid repeated lookups per bar.
        self._component_risk_context_cache_key: tuple[str, int] | None = None
        self._component_risk_context_cache: dict[str, Any] | None = None

        # Timing configuration
        self.base_check_interval = check_interval
        self.check_interval = check_interval
        self.min_check_interval = DEFAULT_MIN_CHECK_INTERVAL
        self.max_check_interval = DEFAULT_MAX_CHECK_INTERVAL
        self.data_freshness_threshold = DEFAULT_DATA_FRESHNESS_THRESHOLD
        self.last_data_timestamp = None
        self.initial_balance = initial_balance
        self.current_balance = initial_balance  # Will be updated during startup
        self._balance_lock = threading.Lock()  # Protect concurrent balance modifications
        self.max_position_size = max_position_size
        self.enable_live_trading = enable_live_trading
        # Execution realism (parity with backtest)
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.use_high_low_for_stops = use_high_low_for_stops
        self.max_filled_price_deviation = max_filled_price_deviation
        self.log_trades = log_trades
        self.alert_webhook_url = alert_webhook_url
        self.enable_hot_swapping = enable_hot_swapping
        self.resume_from_last_balance = resume_from_last_balance
        self.account_snapshot_interval = account_snapshot_interval
        self.testnet = testnet
        # Partial operations policy (disabled by default for parity)
        self.enable_partial_operations = bool(enable_partial_operations)
        if partial_manager is not None:
            self.partial_manager = partial_manager
        elif enable_partial_operations:
            # Check strategy overrides first, then fall back to risk parameters
            strategy_overrides = (
                self.strategy.get_risk_overrides()
                if hasattr(self.strategy, "get_risk_overrides")
                else None
            )
            if strategy_overrides and "partial_operations" in strategy_overrides:
                partial_config = strategy_overrides["partial_operations"]
                self.partial_manager = PartialExitPolicy(
                    exit_targets=partial_config.get("exit_targets", []),
                    exit_sizes=partial_config.get("exit_sizes", []),
                    scale_in_thresholds=partial_config.get("scale_in_thresholds", []),
                    scale_in_sizes=partial_config.get("scale_in_sizes", []),
                    max_scale_ins=partial_config.get("max_scale_ins", 0),
                )
            else:
                rp = self.risk_manager.params if self.risk_manager else RiskParameters()
                self.partial_manager = PartialExitPolicy(
                    exit_targets=rp.partial_exit_targets or [],
                    exit_sizes=rp.partial_exit_sizes or [],
                    scale_in_thresholds=rp.scale_in_thresholds or [],
                    scale_in_sizes=rp.scale_in_sizes or [],
                    max_scale_ins=rp.max_scale_ins,
                )
        else:
            self.partial_manager = None
        self._partial_operations_opt_in = bool(
            self.enable_partial_operations or self.partial_manager is not None
        )

        # Correlation engine setup
        try:
            corr_cfg = CorrelationConfig(
                correlation_window_days=self.risk_manager.params.correlation_window_days,
                correlation_threshold=self.risk_manager.params.correlation_threshold,
                max_correlated_exposure=self.risk_manager.params.max_correlated_exposure,
                correlation_update_frequency_hours=self.risk_manager.params.correlation_update_frequency_hours,
            )
            self.correlation_engine = CorrelationEngine(config=corr_cfg)
        except Exception:
            self.correlation_engine = None

        # Initialize database manager
        try:
            self.db_manager = DatabaseManager(database_url)
        except (ConnectionError, OSError, ValueError) as e:
            print(
                f"❌ Could not connect to the PostgreSQL database: {e}\nThe trading engine cannot start without a database connection. Exiting..."
            )
            raise RuntimeError("Database connection required. Service stopped.") from e
        self.trading_session_id: int | None = None

        # Initialize dynamic risk manager after database is available
        if self.enable_dynamic_risk:
            try:
                # Merge strategy risk overrides with engine config
                final_config = self._merge_dynamic_risk_config(self._dynamic_risk_config)
                self.dynamic_risk_manager = DynamicRiskManager(
                    config=final_config, db_manager=self.db_manager
                )
                logger.info("Dynamic risk management enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize dynamic risk manager: {e}")
                self.dynamic_risk_manager = None
        self._dynamic_risk_handler = DynamicRiskHandler(self.dynamic_risk_manager)

        # Initialize exchange interface, account synchronizer, and order tracker
        self.exchange_interface = None
        self.account_synchronizer = None
        self.order_tracker: OrderTracker | None = None
        if enable_live_trading:
            try:
                config = get_config()
                self.exchange_interface, provider_name = _create_exchange_provider(
                    provider, config, testnet
                )
                if self.exchange_interface:
                    self.account_synchronizer = AccountSynchronizer(
                        self.exchange_interface, self.db_manager, self.trading_session_id
                    )
                    # Initialize order tracker for monitoring order fills
                    self.order_tracker = OrderTracker(
                        exchange=self.exchange_interface,
                        poll_interval=5,
                        on_fill=self._handle_order_fill,
                        on_partial_fill=self._handle_partial_fill,
                        on_cancel=self._handle_order_cancel,
                    )
                    logger.info(
                        f"{provider_name} exchange interface and account synchronizer initialized"
                    )
                else:
                    logger.warning(
                        f"{provider_name} API credentials not found - account sync disabled"
                    )
            except Exception as e:
                logger.warning(f"Failed to initialize exchange interface: {e}")

            # Fail fast if live trading requested but exchange interface unavailable
            if self.exchange_interface is None:
                raise ValueError(
                    "Cannot enable live trading without exchange interface. "
                    "Ensure valid API credentials are configured for the selected provider."
                )

        # Optionally resume balance from last snapshot (only in live trading mode)
        if self.resume_from_last_balance and self.enable_live_trading:
            try:
                # Get the latest active session ID
                active_session_id = self.db_manager.get_active_session_id()
                if active_session_id:
                    latest_balance = self.db_manager.get_current_balance(active_session_id)
                    if latest_balance and latest_balance > 0:
                        self.current_balance = latest_balance
                        self.initial_balance = latest_balance
                        logger.info(
                            f"Resumed from last recorded balance (account_balances): ${self.current_balance:,.2f}"
                        )
            except Exception as e:
                logger.warning(f"Could not resume from last balance: {e}")

        # Initialize strategy manager for hot-swapping
        self.strategy_manager = None
        if enable_hot_swapping:
            managed_strategy = (
                strategy.strategy if isinstance(strategy, StrategyRuntime) else strategy
            )
            # Support component-based Strategy
            if isinstance(managed_strategy, ComponentStrategy):
                self.strategy_manager = StrategyManager()
                self.strategy_manager.current_strategy = managed_strategy
                self.strategy_manager.on_strategy_change = self._handle_strategy_change
                self.strategy_manager.on_model_update = self._handle_model_update
                logger.info(f"Hot swapping enabled for {managed_strategy.__class__.__name__}")
            else:
                logger.info("Hot swapping disabled: provided strategy does not implement Strategy")

        # Set up strategy logging if database is available
        if self.db_manager:
            if hasattr(self.strategy, "set_database_manager"):
                self.strategy.set_database_manager(self.db_manager)

        # Trading state
        self.is_running = False
        self.completed_trades: list[Trade] = []
        self.last_data_update = None
        self.last_account_snapshot = None  # Track when we last logged account state
        self.timeframe: str | None = None  # Will be set when trading starts
        self._active_symbol: str | None = None

        # Performance tracker (unified with backtest engine)
        from src.performance.tracker import PerformanceTracker

        self.performance_tracker = PerformanceTracker(initial_balance)

        # Error handling
        self.max_consecutive_errors = max_consecutive_errors
        self.consecutive_errors = 0
        self.error_cooldown = DEFAULT_ERROR_COOLDOWN

        # Time exit policy (construct from overrides if not provided)
        self.time_exit_policy = time_exit_policy
        if self.time_exit_policy is None:
            overrides = None
            try:
                overrides = (
                    self.strategy.get_risk_overrides()
                    if hasattr(self.strategy, "get_risk_overrides")
                    else None
                )
            except Exception:
                overrides = None
            time_cfg = None
            if overrides and isinstance(overrides, dict):
                time_cfg = overrides.get("time_exits")
            if not time_cfg and self.risk_manager and getattr(self.risk_manager, "params", None):
                time_cfg = getattr(self.risk_manager.params, "time_exits", None)
            try:
                if time_cfg:
                    tr = time_cfg.get("time_restrictions") or DEFAULT_TIME_RESTRICTIONS
                    restrictions = TimeRestrictions(
                        no_overnight=bool(tr.get("no_overnight", False)),
                        no_weekend=bool(tr.get("no_weekend", False)),
                        trading_hours_only=bool(tr.get("trading_hours_only", False)),
                    )
                    self.time_exit_policy = TimeExitPolicy(
                        max_holding_hours=time_cfg.get(
                            "max_holding_hours", DEFAULT_MAX_HOLDING_HOURS
                        ),
                        end_of_day_flat=time_cfg.get("end_of_day_flat", DEFAULT_END_OF_DAY_FLAT),
                        weekend_flat=time_cfg.get("weekend_flat", DEFAULT_WEEKEND_FLAT),
                        market_timezone=time_cfg.get("market_timezone", DEFAULT_MARKET_TIMEZONE),
                        time_restrictions=restrictions,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to create time exit policy from config: %s. "
                    "Time-based exits will be disabled.",
                    e,
                    exc_info=True,
                )
                self.time_exit_policy = None

        # Threading
        self.main_thread = None
        self.stop_event = threading.Event()

        # Optional regime detector (feature-gated)
        self.regime_detector = None
        try:
            if os.getenv("FEATURE_ENABLE_REGIME_DETECTION", "").lower() == "true":
                self.regime_detector = RegimeDetector()
        except Exception:
            self.regime_detector = None

        # Setup graceful shutdown (main thread only)
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        else:
            logger.debug("Skipping signal handler registration outside main thread")

        # Initialize modular handlers (use injected or create defaults)
        self._init_modular_handlers(
            position_tracker=position_tracker,
            execution_engine=execution_engine,
            entry_handler=entry_handler,
            exit_handler=exit_handler,
            market_data_handler=market_data_handler,
            event_logger=event_logger,
            health_monitor=health_monitor,
        )

        logger.info(
            f"LiveTradingEngine initialized - Live Trading: {'ENABLED' if enable_live_trading else 'DISABLED'}"
        )

    @property
    def positions(self) -> dict[str, Position]:
        """Legacy view of active positions for backward compatibility."""
        return self.live_position_tracker._positions

    @positions.setter
    def positions(self, value: dict[str, Position]) -> None:
        """Replace tracked positions (legacy compatibility)."""
        self.live_position_tracker.reset()
        for order_id, position in value.items():
            if position.order_id is None:
                position.order_id = order_id
            self.live_position_tracker.track_recovered_position(position, db_id=None)

    def _merge_dynamic_risk_config(self, base_config: DynamicRiskConfig) -> DynamicRiskConfig:
        """Merge strategy risk overrides with base dynamic risk configuration.

        Uses shared risk configuration logic for consistency with backtest engine.
        """
        merged_config = merge_dynamic_risk_config(base_config, self.strategy)
        if merged_config != base_config:
            logger.info(f"Merged strategy dynamic risk overrides from {self._strategy_name()}")
        return merged_config

    def _init_modular_handlers(
        self,
        position_tracker: LivePositionTracker | None,
        execution_engine: LiveExecutionEngine | None,
        entry_handler: LiveEntryHandler | None,
        exit_handler: LiveExitHandler | None,
        market_data_handler: MarketDataHandler | None,
        event_logger: LiveEventLogger | None,
        health_monitor: HealthMonitor | None,
    ) -> None:
        """Initialize modular handlers with dependency injection or defaults.

        Args:
            position_tracker: Position tracking handler.
            execution_engine: Order execution handler.
            entry_handler: Entry signal processing handler.
            exit_handler: Exit condition checking handler.
            market_data_handler: Market data fetching handler.
            event_logger: Event logging handler.
            health_monitor: Health monitoring handler.
        """
        # Health monitor (no dependencies)
        self.health_monitor = health_monitor or HealthMonitor(
            max_consecutive_errors=self.max_consecutive_errors,
            base_check_interval=self.base_check_interval,
            min_check_interval=self.min_check_interval,
            max_check_interval=self.max_check_interval,
            error_cooldown=self.error_cooldown,
        )

        # Market data handler
        self.market_data_handler = market_data_handler or MarketDataHandler(
            data_provider=self.data_provider,
            sentiment_provider=self.sentiment_provider,
            data_freshness_threshold=self.data_freshness_threshold,
        )

        # Event logger
        self.event_logger = event_logger or LiveEventLogger(
            db_manager=self.db_manager,
            log_to_database=True,
            log_trades_to_file=self.log_trades,
            session_id=self.trading_session_id,
        )

        # Position tracker
        self.live_position_tracker = position_tracker or LivePositionTracker(
            db_manager=self.db_manager,
            fee_rate=self.fee_rate,
            slippage_rate=self.slippage_rate,
        )

        # Execution engine
        self.live_execution_engine = execution_engine or LiveExecutionEngine(
            fee_rate=self.fee_rate,
            slippage_rate=self.slippage_rate,
            enable_live_trading=self.enable_live_trading,
            exchange_interface=self.exchange_interface,
        )

        # Entry handler
        self.live_entry_handler = entry_handler or LiveEntryHandler(
            execution_engine=self.live_execution_engine,
            risk_manager=self.risk_manager,
            component_strategy=(
                self.strategy if isinstance(self.strategy, ComponentStrategy) else None
            ),
            dynamic_risk_manager=self.dynamic_risk_manager,
            max_position_size=self.max_position_size,
            default_take_profit_pct=self._resolve_take_profit_pct(),
        )

        # Wrap PartialExitPolicy in unified PartialOperationsManager
        partial_ops_manager = (
            PartialOperationsManager(policy=self.partial_manager)
            if self.partial_manager is not None
            else None
        )

        # Exit handler
        self.live_exit_handler = exit_handler or LiveExitHandler(
            execution_engine=self.live_execution_engine,
            position_tracker=self.live_position_tracker,
            risk_manager=self.risk_manager,
            trailing_stop_policy=self.trailing_stop_policy,
            partial_manager=partial_ops_manager,
            time_exit_policy=self.time_exit_policy,
            use_high_low_for_stops=self.use_high_low_for_stops,
            max_filled_price_deviation=self.max_filled_price_deviation,
        )

    def _apply_dynamic_risk_adjustment(
        self,
        original_size: float,
        current_time: datetime,
    ) -> float:
        """Apply dynamic risk adjustments to position size.

        Reduces position size during drawdown or adverse market conditions
        to preserve capital and prevent excessive losses.
        """
        if self.dynamic_risk_manager is None:
            return original_size

        try:
            perf_metrics = self.performance_tracker.get_metrics()

            # Guard against zero/None balances to prevent division by zero in drawdown calc
            balance = (
                float(self.current_balance)
                if self.current_balance and self.current_balance > 0
                else float(self.initial_balance)
            )
            peak = (
                float(perf_metrics.peak_balance)
                if perf_metrics.peak_balance and perf_metrics.peak_balance > 0
                else balance
            )
            # Peak should never be less than current balance
            peak_balance = max(peak, balance)

            adjusted_size = self._dynamic_risk_handler.apply_dynamic_risk(
                original_size=original_size,
                current_time=current_time,
                balance=balance,
                peak_balance=peak_balance,
                trading_session_id=self.trading_session_id,
            )
            self._log_dynamic_risk_adjustments()
            return adjusted_size

        except Exception as e:
            logger.warning("Failed to apply dynamic risk adjustment: %s", e)
            return original_size

    def _log_dynamic_risk_adjustments(self) -> None:
        """Log dynamic risk adjustments for observability and audit."""
        adjustments = self._dynamic_risk_handler.get_adjustment_objects(clear=True)
        for adjustment in adjustments:
            logger.info(
                "🎛️ Dynamic risk adjustment applied: size factor=%.2f, reason=%s",
                adjustment.position_size_factor,
                adjustment.primary_reason,
            )
            # Log both factor values (for analysis) and sizes (for debugging)
            log_risk_event(
                "dynamic_risk_adjustment",
                position_size_factor=adjustment.position_size_factor,
                reason=adjustment.primary_reason,
                original_value=1.0,
                adjusted_value=adjustment.position_size_factor,
                original_size=adjustment.original_size,
                adjusted_size=adjustment.adjusted_size,
                current_drawdown=adjustment.current_drawdown,
            )

            if self.db_manager and self.trading_session_id:
                try:
                    # Extract adjustment type from primary_reason (e.g., "drawdown_reduction" -> "drawdown")
                    # Use safe extraction to handle edge cases where reason doesn't contain "_"
                    reason = adjustment.primary_reason or "unknown"
                    adjustment_type = reason.split("_")[0] if "_" in reason else reason

                    # Log factor values (not position sizes) for backward compatibility
                    self.db_manager.log_risk_adjustment(
                        session_id=self.trading_session_id,
                        adjustment_type=adjustment_type,
                        trigger_reason=adjustment.primary_reason,
                        parameter_name="position_size_factor",
                        original_value=1.0,
                        adjusted_value=adjustment.position_size_factor,
                        adjustment_factor=adjustment.position_size_factor,
                        current_drawdown=adjustment.current_drawdown,
                        performance_score=None,
                        volatility_level=None,
                    )
                except Exception as log_e:
                    logger.warning("Failed to log risk adjustment to database: %s", log_e)

    def _get_dynamic_risk_adjusted_params(self) -> RiskParameters:
        """Get risk parameters with dynamic adjustments applied"""
        if not self.dynamic_risk_manager:
            return self.risk_manager.params

        try:
            # Calculate dynamic risk adjustments
            perf_metrics = self.performance_tracker.get_metrics()
            adjustments = self.dynamic_risk_manager.calculate_dynamic_risk_adjustments(
                current_balance=self.current_balance,
                peak_balance=perf_metrics.peak_balance or self.current_balance,
                session_id=self.trading_session_id,
            )

            # Apply adjustments to risk parameters
            adjusted_params = self.dynamic_risk_manager.apply_risk_adjustments(
                self.risk_manager.params, adjustments
            )

            return adjusted_params

        except Exception as e:
            logger.warning(f"Failed to get dynamic risk adjusted parameters: {e}")
            return self.risk_manager.params

    def _extract_component_risk_parameters(
        self, component_risk_manager: object
    ) -> RiskParameters | None:
        """Clone risk parameters from a component adapter, if available."""

        if component_risk_manager is None:
            return None

        core_manager = getattr(component_risk_manager, "_core_manager", None)
        if core_manager is None:
            return None

        params = getattr(core_manager, "params", None)
        if not isinstance(params, RiskParameters):
            return None

        return self._clone_risk_parameters(params)

    def _merge_risk_parameters(
        self,
        engine_params: RiskParameters | None,
        component_params: RiskParameters | None,
    ) -> RiskParameters | None:
        """Merges engine-provided and component-provided risk parameters.

        Component parameters take precedence over engine parameters when both
        are provided. Non-None component values override engine values.

        Args:
            engine_params: Risk parameters from the trading engine
            component_params: Risk parameters from the strategy component

        Returns:
            Merged risk parameters, or None if both inputs are None
        """

        if engine_params is None and component_params is None:
            return None

        if component_params is None:
            return self._clone_risk_parameters(engine_params)

        if engine_params is None:
            return component_params

        component_dict = asdict(component_params)
        engine_dict = asdict(engine_params)
        default_dict = asdict(RiskParameters())

        merged = dict(component_dict)
        for key, value in engine_dict.items():
            default_value = default_dict.get(key)

            # Preserve component overrides when the engine sticks with defaults.
            if value == default_value:
                continue

            merged[key] = value

        return RiskParameters(**merged)

    @staticmethod
    def _clone_risk_parameters(params: RiskParameters | None) -> RiskParameters | None:
        """Creates a deep-cloned copy of risk parameters for safe reuse.

        Args:
            params: Risk parameters to clone

        Returns:
            Deep copy of the risk parameters, or None if input is None
        """

        if params is None:
            return None

        return RiskParameters(**asdict(params))

    def _configure_strategy(self, strategy: ComponentStrategy | StrategyRuntime) -> None:
        """Normalizes strategy inputs and configures runtime bookkeeping.

        Handles both raw ComponentStrategy instances and wrapped StrategyRuntime
        instances, extracting the underlying strategy and setting up engine state.

        Args:
            strategy: Strategy instance to configure (raw or wrapped)
        """

        runtime = strategy if isinstance(strategy, StrategyRuntime) else None
        base_strategy = runtime.strategy if runtime is not None else strategy

        previous_component = getattr(self, "_component_strategy", None)

        self.strategy = base_strategy
        self._component_strategy = (
            base_strategy if isinstance(base_strategy, ComponentStrategy) else None
        )

        if (
            previous_component is not None
            and previous_component is not self._component_strategy
            and hasattr(previous_component, "set_additional_risk_context_provider")
        ):
            try:
                previous_component.set_additional_risk_context_provider(None)
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.debug("Failed to clear risk context provider on previous strategy: %s", exc)

        if runtime is not None:
            self._runtime = runtime
        elif self._component_strategy is not None:
            self._runtime = StrategyRuntime(self._component_strategy)
        else:
            self._runtime = None

        self._register_component_context_provider()

        if hasattr(self, "live_entry_handler"):
            self.live_entry_handler.set_component_strategy(self._component_strategy)

    def _register_component_context_provider(self) -> None:
        """Attaches the engine-provided risk context hook to component strategies.

        Registers a callback function that allows component strategies to request
        additional risk context (correlation data, etc.) during decision-making.
        """

        strategy = getattr(self, "_component_strategy", None)
        if strategy is None:
            return

        setter = getattr(strategy, "set_additional_risk_context_provider", None)
        if not callable(setter):
            return

        def provider(df: pd.DataFrame, index: int, signal) -> dict[str, Any] | None:
            return self._component_risk_context(df, index, signal)

        try:
            setter(provider)
        except (TypeError, AttributeError) as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to attach risk context provider to component strategy: %s", exc)

    def _component_risk_context(self, df: pd.DataFrame, index: int, signal) -> dict[str, Any]:
        """Build supplemental risk context (e.g., correlation data) for components."""

        strategy = getattr(self, "_component_strategy", None)
        if strategy is None:
            return {}

        if getattr(self, "correlation_engine", None) is None:
            return {}

        symbol = self._active_symbol or getattr(strategy, "trading_pair", None)
        if not symbol:
            self._component_risk_context_cache_key = None
            self._component_risk_context_cache = None
            return {}

        cache_key = (str(symbol), int(index))
        cached_key = getattr(self, "_component_risk_context_cache_key", None)
        if cached_key != cache_key:
            self._component_risk_context_cache_key = None
            self._component_risk_context_cache = None

        overrides = None
        if hasattr(strategy, "get_risk_overrides"):
            try:
                overrides = strategy.get_risk_overrides()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug(
                    "Failed to fetch component risk overrides for correlation context: %s",
                    exc,
                )

        # Only build correlation context when sizing a potential entry to avoid repeated
        # historical price lookups on every candle. Reuse the cached value if the same bar
        # has already triggered sizing.
        try:
            direction = getattr(signal, "direction", None)
        except Exception:
            direction = None

        if direction == SignalDirection.HOLD:
            return {}

        correlation_ctx = self._get_correlation_context(
            str(symbol),
            df,
            overrides,
            index=index,
        )
        if not correlation_ctx:
            return {}

        return {"correlation_ctx": correlation_ctx}

    def _get_correlation_context(
        self,
        symbol: str,
        df: pd.DataFrame,
        overrides: dict | None,
        *,
        index: int | None = None,
    ) -> dict | None:
        """Return cached correlation context for the given bar or build it on demand."""

        cache_key = (symbol, index) if index is not None else None
        cached_key = getattr(self, "_component_risk_context_cache_key", None)
        cached_ctx = getattr(self, "_component_risk_context_cache", None)
        if cache_key is not None and cache_key == cached_key and cached_ctx is not None:
            return cached_ctx

        context = self._build_correlation_context(symbol, df, overrides)
        if cache_key is not None:
            if context:
                self._component_risk_context_cache_key = cache_key
                self._component_risk_context_cache = context
            else:
                if cache_key == getattr(self, "_component_risk_context_cache_key", None):
                    self._component_risk_context_cache_key = None
                    self._component_risk_context_cache = None
        return context

    def _apply_policies_from_decision(self, decision) -> None:
        """Hydrate engine-level policies from component strategy output.

        Uses shared policy hydration logic for consistency with backtest engine.
        """
        # Use shared policy hydration logic
        apply_policies_to_engine(decision, self, self.db_manager)

        # Cache dynamic risk config for live engine state tracking
        if decision is not None:
            bundle = getattr(decision, "policies", None)
            if bundle:
                try:
                    dynamic_descriptor = getattr(bundle, "dynamic_risk", None)
                    if dynamic_descriptor is not None:
                        self._component_dynamic_risk_config = dynamic_descriptor.to_config()
                except Exception:
                    pass  # Ignore - shared function handles the main logic

    # Runtime integration helpers -------------------------------------------------

    def _is_runtime_strategy(self) -> bool:
        return self._runtime is not None

    def _strategy_name(self) -> str:
        """Returns the configured strategy name for logging and reporting.

        Returns:
            Strategy name, or "UnknownStrategy" if no strategy is configured
        """
        strategy = getattr(self, "strategy", None)
        if strategy is None:
            return "UnknownStrategy"
        return getattr(strategy, "name", strategy.__class__.__name__)

    def _prepare_strategy_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares dataframe for strategy processing.

        Component-based strategies compute indicators on-demand in process_candle(),
        so the dataframe is returned as-is. Legacy strategies would need upfront
        indicator calculation (not currently used).

        Args:
            df: Raw market data dataframe

        Returns:
            Prepared dataframe ready for strategy processing
        """
        if not self._is_runtime_strategy():
            # Component-based strategies don't need upfront indicator calculation
            # They compute indicators on-demand in process_candle()
            return df

        dataset = self._runtime.prepare_data(df)
        self._runtime_dataset = dataset
        self._runtime_warmup = max(0, int(dataset.warmup_period or 0))
        return dataset.data

    def _build_runtime_context(
        self,
        balance: float,
        current_price: float,
        current_time: datetime,
    ) -> RuntimeContext:
        positions: list[ComponentPosition] = []
        for position in self.live_position_tracker.positions.values():
            try:
                quantity = self._compute_component_quantity(position)
                component_position = ComponentPosition(
                    symbol=position.symbol,
                    side=position.side.value,
                    size=quantity,
                    entry_price=float(position.entry_price),
                    current_price=float(current_price),
                    entry_time=position.entry_time,
                )
                positions.append(component_position)
            except Exception as exc:
                logger.debug("Failed to translate live position for runtime: %s", exc)

        return RuntimeContext(balance=float(balance), current_positions=positions or None)

    def _compute_component_quantity(
        self, position: Position, balance_basis: float | None = None
    ) -> float:
        """Translate a position's fractional size into asset quantity for component strategies."""
        entry_price = float(position.entry_price)
        if entry_price <= 0:
            return 0.0

        basis = (
            balance_basis if balance_basis is not None else getattr(position, "entry_balance", None)
        )
        if basis is None or basis <= 0:
            basis = self.current_balance

        size_fraction = float(
            position.current_size if position.current_size is not None else position.size
        )
        return (size_fraction * float(basis)) / entry_price

    def _runtime_process_decision(
        self,
        df: pd.DataFrame,
        index: int,
        balance: float,
        current_price: float,
        current_time: datetime,
    ):
        if not self._is_runtime_strategy():
            return None
        if self._runtime_dataset is None:
            return None
        if index < self._runtime_warmup:
            return None

        context = self._build_runtime_context(balance, current_price, current_time)
        try:
            decision = self._runtime.process(index, context)
            self._apply_policies_from_decision(decision)
            return decision
        except (ValueError, KeyError, IndexError, AttributeError) as exc:
            logger.warning("Runtime decision failed in live engine at index %s: %s", index, exc)
            return None

    def _finalize_runtime(self) -> None:
        if self._is_runtime_strategy():
            try:
                self._runtime.finalize()
            finally:
                self._runtime_dataset = None
                self._runtime_warmup = 0

    def start(self, symbol: str, timeframe: str = "1h", max_steps: int | None = None) -> None:
        """Start the live trading engine"""
        if self.is_running:
            logger.warning("Trading engine is already running")
            return

        self.is_running = True
        self._active_symbol = symbol
        self.timeframe = timeframe  # Store the trading timeframe
        # Set base logging context for this engine run
        set_context(
            component="live_engine",
            strategy=getattr(self.strategy, "__class__", type("_", (), {})).__name__,
            symbol=symbol,
            timeframe=timeframe,
        )
        log_engine_event(
            "engine_start",
            initial_balance=self.current_balance,
            max_position_size=self.max_position_size,
            check_interval=self.check_interval,
            mode="live" if self.enable_live_trading else "paper",
        )
        logger.info(f"🚀 Starting live trading for {symbol} on {timeframe} timeframe")
        logger.info(f"Initial balance: ${self.current_balance:,.2f}")
        logger.info(f"Max position size: {self.max_position_size * 100:.1f}% of balance")
        logger.info(f"Check interval: {self.check_interval}s")

        if not self.enable_live_trading:
            logger.warning("⚠️  PAPER TRADING MODE - No real orders will be executed")

        # Try to recover from existing session first
        if self.resume_from_last_balance:
            recovered_balance = self._recover_existing_session()
            if recovered_balance is not None:
                self.current_balance = recovered_balance
                logger.info(
                    f"💾 Recovered balance from previous session: ${recovered_balance:,.2f}"
                )
                # Also recover active positions
                self._recover_active_positions()
            else:
                logger.info("🆕 No existing session found, starting fresh")

        # Create new trading session in database if none exists
        if self.trading_session_id is None:
            mode = TradeSource.LIVE if self.enable_live_trading else TradeSource.PAPER
            # Prepare time-exit session config for persistence
            tx_cfg = None
            if self.time_exit_policy:
                tx_cfg = {
                    "max_holding_hours": self.time_exit_policy.max_holding_hours,
                    "end_of_day_flat": self.time_exit_policy.end_of_day_flat,
                    "weekend_flat": self.time_exit_policy.weekend_flat,
                    "time_restrictions": {
                        "no_overnight": self.time_exit_policy.time_restrictions.no_overnight,
                        "no_weekend": self.time_exit_policy.time_restrictions.no_weekend,
                        "trading_hours_only": self.time_exit_policy.time_restrictions.trading_hours_only,
                    },
                }

            self.trading_session_id = self.db_manager.create_trading_session(
                strategy_name=self._strategy_name(),
                symbol=symbol,
                timeframe=timeframe,
                mode=mode,
                initial_balance=self.current_balance,  # Use current balance (might be recovered)
                strategy_config=getattr(self.strategy, "config", {}),
                time_exit_config=tx_cfg,
                market_timezone=(
                    self.time_exit_policy.market_timezone if self.time_exit_policy else None
                ),
            )

            # Update context with session id
            update_context(session_id=self.trading_session_id)

            # Initialize balance tracking
            self.db_manager.update_balance(
                self.current_balance, "session_start", "system", self.trading_session_id
            )

            # Set session ID on strategy for logging
            if hasattr(self.strategy, "session_id"):
                self.strategy.session_id = self.trading_session_id

        # Perform account synchronization if available
        self._pending_balance_correction = False
        self._pending_corrected_balance = None
        if self.account_synchronizer and self.enable_live_trading:
            try:
                logger.info("🔄 Performing initial account synchronization...")
                sync_result = self.account_synchronizer.sync_account_data(force=True)
                if sync_result.success:
                    logger.info("✅ Account synchronization completed")
                    # Update session ID for synchronizer
                    if self.trading_session_id:
                        self.account_synchronizer.session_id = self.trading_session_id
                    # Check if balance was corrected
                    balance_sync = sync_result.data.get("balance_sync", {})
                    if balance_sync.get("corrected", False):
                        corrected_balance = balance_sync.get("new_balance", self.current_balance)
                        # Atomic balance update with lock to prevent race conditions
                        with self._balance_lock:
                            self.current_balance = corrected_balance
                            self._pending_balance_correction = True
                            self._pending_corrected_balance = corrected_balance
                        logger.info(
                            f"💰 Balance corrected from exchange: ${corrected_balance:,.2f}"
                        )
                else:
                    logger.warning(f"⚠️ Account synchronization failed: {sync_result.message}")

                # Reconcile positions with exchange (detect offline stop-loss triggers)
                self._reconcile_positions_with_exchange()

            except Exception as e:
                logger.error(f"❌ Account synchronization error: {e}", exc_info=True)

        # If a balance correction was pending, log it now (outside session creation conditional)
        # Use lock to ensure atomic check and update
        with self._balance_lock:
            if (
                getattr(self, "_pending_balance_correction", False)
                and self.trading_session_id is not None
            ):
                corrected_balance = self._pending_corrected_balance
                self.db_manager.update_balance(
                    corrected_balance, "account_sync", "system", self.trading_session_id
                )
                self._pending_balance_correction = False
                self._pending_corrected_balance = None
                logger.info(f"💰 Balance corrected in database: ${corrected_balance:,.2f}")
            elif getattr(self, "_pending_balance_correction", False):
                # Balance correction was pending but no session ID available
                logger.warning(
                    "⚠️ Balance correction pending but no trading session ID available - skipping database update"
                )
                self._pending_balance_correction = False
                self._pending_corrected_balance = None

        # Set session ID on strategy for logging
        if hasattr(self.strategy, "session_id"):
            self.strategy.session_id = self.trading_session_id

        # Start order tracker for monitoring order fills (live trading only)
        if self.order_tracker and self.enable_live_trading:
            self.order_tracker.start()
            logger.info("📡 Order tracker started")

        # Start main trading loop in separate thread
        self.main_thread = threading.Thread(
            target=self._trading_loop, args=(symbol, timeframe, max_steps)
        )
        self.main_thread.daemon = True
        self.main_thread.start()

        try:
            # Keep main thread alive
            while self.is_running and self.main_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the trading engine gracefully"""
        if not self.is_running:
            return

        logger.info("🛑 Stopping trading engine...")
        self.is_running = False
        self.stop_event.set()

        # Stop order tracker first
        if self.order_tracker:
            self.order_tracker.stop()
            logger.info("📡 Order tracker stopped")

        # Close all open positions
        positions_snapshot = self.live_position_tracker.positions
        if positions_snapshot:
            logger.info("Closing %s open positions...", len(positions_snapshot))
            for position in list(positions_snapshot.values()):
                try:
                    # Get current price for position closure - MUST be valid
                    current_price = self.data_provider.get_current_price(position.symbol)
                    if current_price is None or current_price <= 0:
                        logger.critical(
                            "Cannot close position %s during shutdown - invalid price %s. "
                            "Position will remain open! Manual intervention required.",
                            position.symbol,
                            current_price,
                        )
                        continue

                    self._execute_exit(
                        position,
                        "Engine shutdown",
                        None,
                        float(current_price),
                        None,
                        None,
                        None,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to close position %s: %s", position.order_id, e, exc_info=True
                    )
                    self.live_position_tracker.remove_position(position.order_id)

        # Wait for main thread to finish (avoid joining current thread)
        if (
            self.main_thread
            and self.main_thread.is_alive()
            and self.main_thread != threading.current_thread()
        ):
            self.main_thread.join(timeout=30)

        # Print final statistics
        self._print_final_stats()

        # End the trading session in database
        if self.trading_session_id:
            self.db_manager.end_trading_session(
                session_id=self.trading_session_id, final_balance=self.current_balance
            )

        logger.info("Trading engine stopped")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)

    def _trading_loop(self, symbol: str, timeframe: str, max_steps: int | None = None) -> None:
        """Main trading loop"""
        logger.info("Trading loop started")
        steps = 0
        cfg = get_config()
        self._active_symbol = symbol
        try:
            heartbeat_every = int(cfg.get("ENGINE_HEARTBEAT_STEPS", "60"))
        except Exception:
            heartbeat_every = 60
        while self.is_running and not self.stop_event.is_set():
            if max_steps is not None and steps >= max_steps:
                logger.info(f"Reached max_steps={max_steps}, stopping engine for test.")
                self.stop()
                break
            steps += 1
            try:
                # For mock and real providers, update live data if supported
                if hasattr(self.data_provider, "update_live_data"):
                    try:
                        self.data_provider.update_live_data(symbol, timeframe)
                    except Exception as e:
                        logger.debug(f"update_live_data failed: {e}")
                # Fetch latest market data
                df = self._get_latest_data(symbol, timeframe)
                if df is None or df.empty:
                    log_data_event("no_data", reason="empty_frame")
                    logger.warning("No market data received")
                    self.check_interval = self._calculate_adaptive_interval()
                    self._sleep_with_interrupt(self.check_interval)
                    continue

                # Check data freshness to avoid redundant processing
                if not self._is_data_fresh(df):
                    logger.debug("Data is not fresh enough, using longer interval")
                    self.check_interval = self._calculate_adaptive_interval()
                    self._sleep_with_interrupt(self.check_interval)
                    continue
                # Add sentiment data if available
                if self.sentiment_provider:
                    df = self._add_sentiment_data(df, symbol)
                # Check for pending strategy/model updates (wrap in try-except to prevent loop crash)
                try:
                    if self.strategy_manager and self.strategy_manager.has_pending_update():
                        logger.info("🔄 Applying pending strategy/model update...")
                        success = self.strategy_manager.apply_pending_update()
                        if success:
                            self._finalize_runtime()
                            updated_strategy = self.strategy_manager.current_strategy
                            self._configure_strategy(updated_strategy)
                            self._runtime_dataset = None
                            self._runtime_warmup = 0
                            logger.info("✅ Strategy/model update applied successfully")
                            self._send_alert("Strategy/Model updated in live trading")
                        else:
                            logger.error("❌ Failed to apply strategy/model update")
                except Exception as e:
                    logger.error(
                        "❌ Exception during strategy update check/application: %s",
                        e,
                        exc_info=True,
                    )
                # Proceed to indicator calculation

                # Calculate indicators or prepare runtime dataset
                df = self._prepare_strategy_dataframe(df)
                # Remove warmup period and ensure we have enough data
                try:
                    essential_columns = ["open", "high", "low", "close", "volume"]
                    df = df.dropna(subset=essential_columns)
                except Exception:
                    # Fallback to conservative behavior if subset fails for any reason
                    df = df.dropna()
                # Context readiness gating
                ready, reason = self._is_context_ready(df)
                safety_mode = not ready
                if safety_mode:
                    logger.info("Safety mode active: %s", reason)
                if len(df) < 2:
                    try:
                        tail_nan_counts = df.tail(5).isna().sum().to_dict()
                    except Exception:
                        tail_nan_counts = {}
                    logger.warning(
                        "Insufficient data for analysis | rows=%s | tail_nan_counts=%s",
                        len(df),
                        tail_nan_counts,
                    )
                    self.check_interval = self._calculate_adaptive_interval()
                    self._sleep_with_interrupt(self.check_interval)
                    continue

                # Validate DataFrame is not empty before iloc access
                current_index = len(df) - 1
                if current_index < 0:
                    logger.error(
                        "DataFrame became empty after readiness check - skipping iteration"
                    )
                    self._sleep_with_interrupt(self.check_interval)
                    continue

                current_candle = df.iloc[current_index]
                current_price = current_candle["close"]
                current_time = current_candle.name if hasattr(current_candle, "name") else None
                if hasattr(current_time, "to_pydatetime"):
                    current_time = current_time.to_pydatetime()
                if not isinstance(current_time, datetime):
                    current_time = datetime.now(UTC)
                elif current_time.tzinfo is None:
                    current_time = current_time.replace(tzinfo=UTC)

                runtime_decision = self._runtime_process_decision(
                    df,
                    current_index,
                    self.current_balance,
                    float(current_price),
                    current_time,
                )
                if steps % heartbeat_every == 0:
                    log_engine_event(
                        "heartbeat",
                        step=steps,
                        open_positions=self.live_position_tracker.position_count,
                        balance=self.current_balance,
                        last_candle_time=str(df.index[-1]),
                    )
                logger.info(
                    f"Trading loop: current_index={current_index}, last_candle_time={df.index[-1]}"
                )
                # Update position PnL
                self.live_position_tracker.update_pnl(
                    float(current_price), fallback_balance=self.current_balance
                )
                # Apply trailing stop adjustments and update MFE/MAE before exit checks
                try:
                    self.live_exit_handler.update_trailing_stops(
                        df, current_index, float(current_price)
                    )
                except Exception as e:
                    logger.debug(f"Trailing stop update failed: {e}")
                # Update rolling MFE/MAE per position and persist lightweight updates
                self.live_position_tracker.update_mfe_mae(float(current_price))
                # Check exit conditions for existing positions
                self._check_exit_conditions(
                    df,
                    current_index,
                    current_price,
                    runtime_decision=None if safety_mode else runtime_decision,
                    candle=current_candle,
                    safety_mode=safety_mode,
                )
                # Evaluate partial exits and scale-ins for open positions
                if not safety_mode:
                    self.live_exit_handler.check_partial_operations(
                        df, current_index, float(current_price), self.current_balance
                    )
                # Check entry conditions if not at maximum positions
                if (not safety_mode) and (
                    self.live_position_tracker.position_count
                    < self.risk_manager.get_max_concurrent_positions()
                ):
                    self._check_entry_conditions(
                        df,
                        current_index,
                        symbol,
                        current_price,
                        current_time,
                        runtime_decision=runtime_decision,
                    )
                    # Check for short entry via legacy hook when available
                    if (not self._is_runtime_strategy()) and callable(
                        getattr(self.strategy, "check_short_entry_conditions", None)
                    ):
                        short_entry_signal = self.strategy.check_short_entry_conditions(
                            df, current_index
                        )
                        if short_entry_signal:
                            try:
                                overrides = (
                                    self.strategy.get_risk_overrides()
                                    if hasattr(self.strategy, "get_risk_overrides")
                                    else None
                                )
                            except Exception:
                                overrides = None
                            indicators = self._extract_indicators(df, current_index)
                            # Correlation context for short entries
                            short_correlation_ctx = self._get_correlation_context(
                                symbol,
                                df,
                                overrides,
                                index=current_index,
                            )
                            if overrides and overrides.get("position_sizer"):
                                short_fraction = self.risk_manager.calculate_position_fraction(
                                    df=df,
                                    index=current_index,
                                    balance=self.current_balance,
                                    price=current_price,
                                    indicators=indicators,
                                    strategy_overrides=overrides,
                                    correlation_ctx=short_correlation_ctx,
                                )
                                short_fraction = min(short_fraction, self.max_position_size)
                                short_position_size = short_fraction
                            else:
                                # All strategies should be component-based
                                self.logger.error(
                                    f"Strategy {self.strategy.name} does not support component-based position sizing"
                                )
                                short_position_size = 0.0

                            # Apply dynamic risk adjustments
                            short_position_size = self._apply_dynamic_risk_adjustment(
                                short_position_size,
                                current_time,
                            )
                            if short_position_size > 0:
                                if overrides and (
                                    ("stop_loss_pct" in overrides)
                                    or ("take_profit_pct" in overrides)
                                ):
                                    short_stop_loss, short_take_profit = (
                                        self.risk_manager.compute_sl_tp(
                                            df=df,
                                            index=current_index,
                                            entry_price=current_price,
                                            side="short",
                                            strategy_overrides=overrides,
                                        )
                                    )
                                    if short_take_profit is None:
                                        short_take_profit = current_price * (
                                            1 - getattr(self.strategy, "take_profit_pct", 0.04)
                                        )
                                else:
                                    # All strategies should be component-based
                                    self.logger.error(
                                        f"Strategy {self.strategy.name} does not support component-based stop loss calculation"
                                    )
                                    short_stop_loss = (
                                        current_price * 1.05
                                    )  # Default 5% stop for short
                                    short_take_profit = current_price * (
                                        1 - getattr(self.strategy, "take_profit_pct", 0.04)
                                    )
                                self._execute_entry(
                                    symbol=symbol,
                                    side=PositionSide.SHORT,
                                    size=short_position_size,
                                    price=float(current_price),
                                    stop_loss=short_stop_loss,
                                    take_profit=short_take_profit,
                                    signal_strength=0.0,
                                    signal_confidence=0.0,
                                )
                # Update performance metrics
                self._update_performance_metrics()
                # Log account snapshot to database periodically (configurable interval)
                now = datetime.now(UTC)
                if self.account_snapshot_interval > 0 and (
                    self.last_account_snapshot is None
                    or (now - self.last_account_snapshot).seconds >= self.account_snapshot_interval
                ):
                    self._log_account_snapshot()
                    self.last_account_snapshot = now

                    # Perform periodic account synchronization
                    if self.account_synchronizer and self.enable_live_trading:
                        try:
                            sync_result = self.account_synchronizer.sync_account_data()
                            if sync_result.success:
                                logger.debug("Periodic account sync completed")
                            else:
                                logger.warning(
                                    f"Periodic account sync failed: {sync_result.message}"
                                )
                        except Exception as e:
                            logger.error(f"Periodic account sync error: {e}")
                # Log status periodically
                if (
                    self.performance_tracker.get_metrics().total_trades % 10 == 0
                    or self.live_position_tracker.position_count > 0
                ):
                    self._log_status(symbol, current_price)
                # Reset error counter on successful iteration
                self.consecutive_errors = 0

                # Calculate and use adaptive interval for next iteration
                current_price = df.iloc[-1]["close"] if df is not None and not df.empty else None
                self.check_interval = self._calculate_adaptive_interval(current_price)

            except Exception as e:
                self.consecutive_errors += 1
                logger.error(
                    f"Error in trading loop (#{self.consecutive_errors}): {e}", exc_info=True
                )
                if self.consecutive_errors >= self.max_consecutive_errors:
                    logger.critical(
                        f"Too many consecutive errors ({self.consecutive_errors}). Stopping engine.",
                        exc_info=True,
                    )
                    self.stop()
                    break
                # Exponential backoff with adaptive intervals
                sleep_time = min(self.error_cooldown, self.check_interval * self.consecutive_errors)
                self._sleep_with_interrupt(sleep_time)
                continue

            # Sleep with current interval
            self._sleep_with_interrupt(self.check_interval)

        logger.info("Trading loop ended")
        self._finalize_runtime()

    def _is_context_ready(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Check if the current frame has enough context for strategy-driven decisions.

        Returns (ready, reason_if_not_ready).
        """
        try:
            rows = len(df)
            # Required rows from ML sequence length (for ML strategies only)
            try:
                seq_len = int(getattr(self.strategy, "sequence_length", 0) or 0)
            except Exception:
                seq_len = 0
            # Do not assume a large indicator window by default; strategies can opt-in via attribute
            try:
                max_window_attr = getattr(self.strategy, "max_indicator_window", 0)
                max_window = int(max_window_attr or 0)
            except Exception:
                max_window = 0
            min_needed_base = max(seq_len, max_window)
            min_needed = (min_needed_base + 1) if min_needed_base > 0 else 2

            if rows < min_needed:
                return False, f"insufficient_rows:{rows}<min_needed:{min_needed}"

            # Current index must have valid essentials
            idx = rows - 1
            essentials = ["open", "high", "low", "close", "volume"]
            for col in essentials:
                try:
                    if pd.isna(df.iloc[idx][col]):
                        return False, f"nan_in_essentials:{col}"
                except Exception:
                    return False, f"missing_essential:{col}"

            # Strategy-specific readiness: prediction availability for ML strategies
            if seq_len > 0:
                if "onnx_pred" in df.columns:
                    try:
                        if pd.isna(df["onnx_pred"].iloc[idx]):
                            return False, "prediction_unavailable_at_current_index"
                    except Exception:
                        return False, "prediction_column_access_error"

            # Data freshness check
            if not self._is_data_fresh(df):
                return False, "stale_data"

            return True, ""
        except Exception as e:
            logger.debug(f"Context readiness check failed: {e}")
            return False, "readiness_check_error"

    def _get_latest_data(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        """Fetch latest market data with error handling"""
        try:
            # Fetch with a generous limit to satisfy indicator and ML warmups
            df = self.data_provider.get_live_data(symbol, timeframe, limit=500)
            self.last_data_update = datetime.now(UTC)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}", exc_info=True)
            return None

    def _add_sentiment_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add sentiment data to price data"""
        try:
            if hasattr(self.sentiment_provider, "get_live_sentiment"):
                # Get live sentiment for recent data
                live_sentiment = self.sentiment_provider.get_live_sentiment()

                # Apply to recent candles (last 4 hours)
                recent_mask = df.index >= (df.index.max() - pd.Timedelta(hours=4))
                for feature, value in live_sentiment.items():
                    if feature not in df.columns:
                        df[feature] = 0.0
                    df.loc[recent_mask, feature] = value

                # Mark sentiment freshness
                df["sentiment_freshness"] = 0
                df.loc[recent_mask, "sentiment_freshness"] = 1

                logger.debug(f"Applied live sentiment to {recent_mask.sum()} recent candles")
            else:
                # Fallback to historical sentiment
                logger.debug("Using historical sentiment data")

        except Exception as e:
            logger.error(f"Failed to add sentiment data: {e}", exc_info=True)

        return df

    def _build_correlation_context(
        self, symbol: str, df: pd.DataFrame, overrides: dict | None
    ) -> dict | None:
        """
        Build correlation context dict for risk manager sizing, including corr matrix and optional exposure override.
        Returns None if correlation engine is unavailable or an error occurs.
        """
        try:
            if self.correlation_engine is None:
                return None
            # Build price series for candidate + currently open symbols
            symbols_to_check = set([symbol]) | set(
                p.symbol for p in self.live_position_tracker.positions.values()
            )
            price_series: dict[str, pd.Series] = {}
            end_ts = df.index[-1] if len(df) > 0 else None
            start_ts = (
                end_ts - pd.Timedelta(days=self.risk_manager.params.correlation_window_days)
                if end_ts is not None
                else None
            )
            if symbol:
                try:
                    price_series[str(symbol)] = df["close"].copy()
                except Exception:
                    pass
            for sym in symbols_to_check:
                s = str(sym)
                if s in price_series:
                    continue
                try:
                    if start_ts is not None and end_ts is not None:
                        # Use the strategy's actual trading timeframe instead of hardcoding "1h"
                        trading_timeframe = self.timeframe or "1h"  # Fallback to "1h" if not set
                        hist = self.data_provider.get_historical_data(
                            s,
                            timeframe=trading_timeframe,
                            start=start_ts.to_pydatetime(),
                            end=end_ts.to_pydatetime(),
                        )
                        if not hist.empty and "close" in hist:
                            price_series[s] = hist["close"]
                except Exception:
                    continue
            corr_matrix = self.correlation_engine.calculate_position_correlations(price_series)
            return {
                "engine": self.correlation_engine,
                "candidate_symbol": symbol,
                "corr_matrix": corr_matrix,
                "max_exposure_override": (
                    overrides.get("correlation_control", {}).get("max_correlated_exposure")
                    if overrides
                    else None
                ),
            }
        except Exception:
            return None

    def _check_exit_conditions(
        self,
        df: pd.DataFrame,
        current_index: int,
        current_price: float,
        runtime_decision=None,
        candle=None,
        safety_mode: bool = False,
    ):
        """Check if any positions should be closed."""
        positions_snapshot = self.live_position_tracker.positions
        if not positions_snapshot:
            return

        # Extract candle high/low for more realistic SL/TP detection (parity with backtest)
        candle_high = None
        candle_low = None
        if df is not None and current_index < len(df):
            row = df.iloc[current_index]
            if "high" in df.columns:
                candle_high = float(row["high"])
            if "low" in df.columns:
                candle_low = float(row["low"])

        # Extract context for logging
        indicators = self._extract_indicators(df, current_index)
        sentiment_data = self._extract_sentiment_data(df, current_index)
        ml_predictions = self._extract_ml_predictions(df, current_index)

        component_strategy = None if safety_mode else self._component_strategy
        decision_for_exit = None if safety_mode else runtime_decision

        for position in positions_snapshot.values():
            exit_check = self.live_exit_handler.check_exit_conditions(
                position=position,
                current_price=float(current_price),
                candle_high=candle_high,
                candle_low=candle_low,
                runtime_decision=decision_for_exit,
                component_strategy=component_strategy,
            )

            should_exit = exit_check.should_exit
            exit_reason = exit_check.exit_reason
            limit_price = exit_check.limit_price

            # Log exit decision for each position
            if self.db_manager:
                # Calculate current P&L for context
                # Validate entry_price to prevent division by zero from corrupted position data
                if position.entry_price <= 0:
                    logger.error(
                        f"Invalid entry_price {position.entry_price} for position {position.symbol} - "
                        "skipping P&L calculation for logging"
                    )
                    current_pnl = 0.0  # Fallback value for logging
                elif position.side == PositionSide.LONG:
                    current_pnl = (float(current_price) - float(position.entry_price)) / float(
                        position.entry_price
                    )
                else:
                    current_pnl = (float(position.entry_price) - float(current_price)) / float(
                        position.entry_price
                    )

                # Prepare logging reasons with TradingDecision data if available
                log_reasons = [
                    exit_reason if should_exit else "holding_position",
                    f"current_pnl_{current_pnl:.4f}",
                    f"position_age_{(datetime.now(UTC) - position.entry_time).total_seconds():.0f}s",
                    f"entry_price_{position.entry_price:.2f}",
                ]

                # Add regime context if available from TradingDecision
                if (
                    decision_for_exit
                    and hasattr(decision_for_exit, "regime")
                    and decision_for_exit.regime
                ):
                    regime = decision_for_exit.regime
                    log_reasons.append(
                        f"regime_trend_{regime.trend.value if hasattr(regime.trend, 'value') else regime.trend}"
                    )
                    log_reasons.append(
                        f"regime_volatility_{regime.volatility.value if hasattr(regime.volatility, 'value') else regime.volatility}"
                    )
                    log_reasons.append(f"regime_confidence_{regime.confidence:.2f}")

                # Add risk metrics if available from TradingDecision
                if (
                    decision_for_exit
                    and hasattr(decision_for_exit, "risk_metrics")
                    and decision_for_exit.risk_metrics
                ):
                    for key, value in decision_for_exit.risk_metrics.items():
                        if isinstance(value, int | float):
                            log_reasons.append(f"risk_{key}_{value:.4f}")

                # Extract signal confidence from TradingDecision if available
                confidence_score = indicators.get("prediction_confidence", 0.5)
                if (
                    decision_for_exit
                    and hasattr(decision_for_exit, "signal")
                    and decision_for_exit.signal
                ):
                    confidence_score = decision_for_exit.signal.confidence

                self.db_manager.log_strategy_execution(
                    strategy_name=self._strategy_name(),
                    symbol=position.symbol,
                    signal_type="exit",
                    action_taken="closed_position" if should_exit else "hold_position",
                    price=current_price,
                    timeframe="1m",
                    signal_strength=1.0 if should_exit else 0.0,
                    confidence_score=confidence_score,
                    indicators=indicators,
                    sentiment_data=sentiment_data if sentiment_data else None,
                    ml_predictions=ml_predictions if ml_predictions else None,
                    position_size=position.size,
                    reasons=log_reasons,
                    volume=indicators.get("volume"),
                    volatility=indicators.get("volatility"),
                    session_id=self.trading_session_id,
                )

            if should_exit:
                self._execute_exit(
                    position,
                    exit_reason,
                    limit_price,
                    float(current_price),
                    candle_high,
                    candle_low,
                )

    def _check_entry_conditions(
        self,
        df: pd.DataFrame,
        current_index: int,
        symbol: str,
        current_price: float,
        current_time: datetime,
        runtime_decision=None,
    ):
        """Check if new positions should be opened"""

        use_runtime = self._is_runtime_strategy()
        entry_signal = False
        position_size = 0.0
        entry_side = PositionSide.LONG
        runtime_strength = 0.0
        runtime_confidence = 0.0
        stop_loss = None
        take_profit = None
        overrides = None

        indicators = self._extract_indicators(df, current_index)
        sentiment_data = self._extract_sentiment_data(df, current_index)
        ml_predictions = self._extract_ml_predictions(df, current_index)

        if use_runtime:
            perf_metrics = self.performance_tracker.get_metrics()
            entry_signal_result = self.live_entry_handler.process_runtime_decision(
                runtime_decision=runtime_decision,
                balance=self.current_balance,
                current_price=float(current_price),
                current_time=datetime.now(UTC),
                peak_balance=perf_metrics.peak_balance or self.current_balance,
                trading_session_id=self.trading_session_id,
            )
            if entry_signal_result.should_enter and entry_signal_result.side is not None:
                entry_signal = True
                entry_side = entry_signal_result.side
                position_size = entry_signal_result.size_fraction
                stop_loss = entry_signal_result.stop_loss
                take_profit = entry_signal_result.take_profit
                runtime_strength = entry_signal_result.signal_strength
                runtime_confidence = entry_signal_result.signal_confidence
        elif isinstance(self.strategy, ComponentStrategy):
            # Component-based strategy: use process_candle() for decision
            # Note: runtime_decision should already be populated if this is a component strategy
            # This branch handles direct ComponentStrategy usage without StrategyRuntime wrapper
            try:
                decision = self.strategy.process_candle(
                    df, current_index, self.current_balance, None
                )
                self._apply_policies_from_decision(decision)

                notional_size = float(decision.position_size or 0.0)
                balance = float(self.current_balance or 0.0)
                size_fraction = 0.0 if balance <= 0 else max(0.0, notional_size / balance)
                bounded_fraction = min(size_fraction, self.max_position_size)

                if decision.signal.direction == SignalDirection.BUY and bounded_fraction > 0:
                    entry_signal = True
                    entry_side = PositionSide.LONG
                    position_size = bounded_fraction
                    runtime_strength = decision.signal.strength
                    runtime_confidence = decision.signal.confidence
                elif decision.signal.direction == SignalDirection.SELL and bounded_fraction > 0:
                    entry_signal = True
                    entry_side = PositionSide.SHORT
                    position_size = bounded_fraction
                    runtime_strength = decision.signal.strength
                    runtime_confidence = decision.signal.confidence
            except Exception as e:
                self.logger.warning(f"Component strategy decision failed: {e}")
                entry_signal = False
        else:
            # All strategies should be component-based
            self.logger.error(f"Strategy {self.strategy.name} is not a component-based strategy")
            entry_signal = False

        if entry_signal and not use_runtime:
            # Component strategies supply their own sizing. Retain correlation context computation
            # for downstream consumers that expect it to be populated as part of the entry check.
            self._get_correlation_context(symbol, df, None, index=current_index)

        if position_size > 0 and not use_runtime:
            position_size = self._apply_dynamic_risk_adjustment(position_size, current_time)

        if self.db_manager:
            # Prepare logging data - include TradingDecision data if available
            log_reasons = [
                (
                    "runtime_entry"
                    if use_runtime
                    else "entry_conditions_met" if entry_signal else "entry_conditions_not_met"
                ),
                (f"position_size_{position_size:.4f}" if position_size > 0 else "no_position_size"),
                f"max_positions_check_{self.live_position_tracker.position_count}_of_{self.risk_manager.get_max_concurrent_positions() if self.risk_manager else 1}",
                (
                    f"enter_short_{bool(getattr(runtime_decision, 'metadata', {}).get('enter_short'))}"
                    if use_runtime and runtime_decision is not None
                    else "enter_short_n/a"
                ),
            ]

            # Add regime context if available from TradingDecision
            if runtime_decision and hasattr(runtime_decision, "regime") and runtime_decision.regime:
                regime = runtime_decision.regime
                log_reasons.append(
                    f"regime_trend_{regime.trend.value if hasattr(regime.trend, 'value') else regime.trend}"
                )
                log_reasons.append(
                    f"regime_volatility_{regime.volatility.value if hasattr(regime.volatility, 'value') else regime.volatility}"
                )
                log_reasons.append(f"regime_confidence_{regime.confidence:.2f}")

            # Add risk metrics if available from TradingDecision
            if (
                runtime_decision
                and hasattr(runtime_decision, "risk_metrics")
                and runtime_decision.risk_metrics
            ):
                for key, value in runtime_decision.risk_metrics.items():
                    if isinstance(value, int | float):
                        log_reasons.append(f"risk_{key}_{value:.4f}")

            self.db_manager.log_strategy_execution(
                strategy_name=self._strategy_name(),
                symbol=symbol,
                signal_type="entry",
                action_taken=(
                    "opened_long"
                    if entry_signal and position_size > 0 and entry_side == PositionSide.LONG
                    else (
                        "opened_short"
                        if entry_signal and position_size > 0 and entry_side == PositionSide.SHORT
                        else "no_action"
                    )
                ),
                price=current_price,
                timeframe="1m",
                signal_strength=runtime_strength if use_runtime else (1.0 if entry_signal else 0.0),
                confidence_score=(
                    runtime_confidence
                    if use_runtime
                    else indicators.get("prediction_confidence", 0.5)
                ),
                indicators=indicators,
                sentiment_data=sentiment_data if sentiment_data else None,
                ml_predictions=ml_predictions if ml_predictions else None,
                position_size=position_size if position_size > 0 else None,
                reasons=log_reasons,
                volume=indicators.get("volume"),
                volatility=indicators.get("volatility"),
                session_id=self.trading_session_id,
            )

        if not entry_signal or position_size <= 0:
            return

        if use_runtime and self._component_strategy is not None:
            try:
                if stop_loss is None:
                    stop_loss = self._component_strategy.get_stop_loss_price(
                        float(current_price),
                        runtime_decision.signal if runtime_decision else None,
                        runtime_decision.regime if runtime_decision else None,
                    )
            except Exception:
                if stop_loss is None:
                    stop_loss = float(current_price) * (
                        0.95 if entry_side == PositionSide.LONG else 1.05
                    )
            if take_profit is None:
                tp_pct = self._resolve_take_profit_pct()
                take_profit = (
                    float(current_price) * (1 + tp_pct)
                    if entry_side == PositionSide.LONG
                    else float(current_price) * (1 - tp_pct)
                )
        elif isinstance(self.strategy, ComponentStrategy):
            # Component-based strategy: use get_stop_loss_price()
            try:
                # Create a signal from the decision
                signal = Signal(
                    direction=(
                        SignalDirection.BUY
                        if entry_side == PositionSide.LONG
                        else SignalDirection.SELL
                    ),
                    strength=runtime_strength,
                    confidence=runtime_confidence,
                    metadata={},
                )
                stop_loss = self.strategy.get_stop_loss_price(
                    float(current_price), signal, None  # regime context
                )
            except Exception as e:
                self.logger.debug(f"Component stop loss calculation failed: {e}")
                stop_loss = float(current_price) * (
                    0.95 if entry_side == PositionSide.LONG else 1.05
                )
            tp_pct = self._resolve_take_profit_pct()
            take_profit = (
                float(current_price) * (1 + tp_pct)
                if entry_side == PositionSide.LONG
                else float(current_price) * (1 - tp_pct)
            )
        else:
            try:
                overrides = (
                    self.strategy.get_risk_overrides()
                    if hasattr(self.strategy, "get_risk_overrides")
                    else None
                )
            except Exception:
                overrides = None

            if overrides and ("stop_loss_pct" in overrides or "take_profit_pct" in overrides):
                stop_loss, take_profit = self.risk_manager.compute_sl_tp(
                    df=df,
                    index=current_index,
                    entry_price=current_price,
                    side="long",
                    strategy_overrides=overrides,
                )
                if take_profit is None:
                    take_profit = current_price * (1 + overrides.get("take_profit_pct", 0.04))
            else:
                # All strategies should be component-based
                self.logger.error(
                    f"Strategy {self.strategy.name} does not support component-based stop loss calculation"
                )
                stop_loss = current_price * 0.95  # Default 5% stop for long
                take_profit = current_price * (1 + getattr(self.strategy, "take_profit_pct", 0.04))
            entry_side = PositionSide.LONG

        self._execute_entry(
            symbol=symbol,
            side=entry_side,
            size=position_size,
            price=float(current_price),
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_strength=runtime_strength,
            signal_confidence=runtime_confidence,
        )

    def _resolve_take_profit_pct(self) -> float:
        """Resolve the default take-profit percentage from risk parameters or strategy."""
        try:
            params = self.risk_manager.params if self.risk_manager else None
            if params and params.default_take_profit_pct is not None:
                try:
                    return float(params.default_take_profit_pct)
                except (TypeError, ValueError):
                    return 0.04
        except Exception:
            return 0.04

        value = getattr(self.strategy, "take_profit_pct", 0.04)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.04

    def _execute_entry(
        self,
        symbol: str,
        side: PositionSide,
        size: float,
        price: float,
        stop_loss: float | None,
        take_profit: float | None,
        signal_strength: float,
        signal_confidence: float,
    ) -> None:
        """Execute a new trading position using shared execution modules."""
        try:
            if size > self.max_position_size:
                logger.warning(
                    "Position size %.2f%% exceeds maximum %.2f%%. Capping at maximum.",
                    size * 100,
                    self.max_position_size * 100,
                )
                size = self.max_position_size

            # Build entry reasons for logging and analysis
            entry_reasons = [
                f"side_{side.value}",
                f"size_{size:.4f}",
                f"strength_{signal_strength:.2f}",
                f"confidence_{signal_confidence:.2f}",
            ]
            if stop_loss:
                entry_reasons.append(f"sl_{stop_loss:.2f}")
            if take_profit:
                entry_reasons.append(f"tp_{take_profit:.2f}")

            entry_signal = LiveEntrySignal(
                should_enter=True,
                side=side,
                size_fraction=size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasons=entry_reasons,
                signal_strength=signal_strength,
                signal_confidence=signal_confidence,
            )
            result = self.live_entry_handler.execute_entry(
                signal=entry_signal,
                symbol=symbol,
                current_price=price,
                balance=self.current_balance,
            )

            if not result.executed or result.position is None:
                logger.error("Failed to execute entry for %s: %s", symbol, result.error)
                return

            position = result.position
            entry_fee = result.entry_fee
            entry_slippage_cost = result.slippage_cost

            # Atomic balance update with full audit trail when trading session exists
            if self.trading_session_id is not None:
                try:
                    with self.db_manager.atomic_balance_update(
                        balance_change=-entry_fee,
                        reason=f"entry_fee_{symbol}",
                        updated_by="live_engine",
                        correlation_id=position.order_id,
                    ) as balance_result:
                        self.current_balance = balance_result["new_balance"]
                except (ValueError, Exception) as balance_err:
                    logger.error(
                        "Failed to update balance for entry fee %s: %s. Aborting entry.",
                        symbol,
                        balance_err,
                    )
                    # Critical: Entry executed but balance update failed
                    # Attempt emergency close to maintain consistency
                    if self.enable_live_trading and self.exchange_interface:
                        try:
                            close_side = (
                                OrderSide.SELL if side == PositionSide.LONG else OrderSide.BUY
                            )
                            # Validate entry_price to prevent division by zero
                            if position.entry_price <= 0:
                                logger.error(
                                    f"Cannot calculate emergency close quantity - invalid entry_price "
                                    f"{position.entry_price} for {symbol}"
                                )
                            else:
                                self.exchange_interface.place_market_order(
                                    symbol=symbol,
                                    side=close_side,
                                    quantity=position.size
                                    * result.position_value
                                    / position.entry_price,
                                )
                            logger.warning(
                                "Emergency close placed for %s due to balance update failure",
                                symbol,
                            )
                        except Exception as close_err:
                            logger.critical(
                                "CRITICAL: Emergency close FAILED after balance update failure for %s. "
                                "MANUAL INTERVENTION REQUIRED. Error: %s",
                                symbol,
                                close_err,
                            )
                    return
            else:
                # No trading session - update balance directly (testing/paper trading mode)
                self.current_balance -= entry_fee

            position.metadata["entry_fee"] = entry_fee
            position.metadata["entry_slippage_cost"] = entry_slippage_cost

            # CRITICAL: Register position with tracker IMMEDIATELY after execution
            # to minimize race window with OrderTracker callbacks.
            # If this fails after order execution, we have an orphaned position.
            try:
                self.live_position_tracker.open_position(
                    position=position,
                    session_id=self.trading_session_id,
                    strategy_name=self._strategy_name(),
                )
            except Exception as tracker_err:
                # Position executed on exchange but failed to track locally.
                # This is critical - attempt emergency close to avoid orphaned position.
                logger.critical(
                    "CRITICAL: Position tracking failed after order execution for %s. "
                    "Attempting emergency close. Error: %s",
                    symbol,
                    tracker_err,
                )
                if self.enable_live_trading and self.exchange_interface:
                    try:
                        close_side = OrderSide.SELL if side == PositionSide.LONG else OrderSide.BUY

                        # Validate entry_price before division to prevent crashes
                        if position.entry_price <= 0:
                            logger.critical(
                                "CRITICAL: Cannot calculate emergency close quantity for %s - "
                                "invalid entry_price %.8f. MANUAL INTERVENTION REQUIRED.",
                                symbol,
                                position.entry_price,
                            )
                        else:
                            self.exchange_interface.place_market_order(
                                symbol=symbol,
                                side=close_side,
                                quantity=position.size
                                * self.current_balance
                                / position.entry_price,
                            )
                            logger.info(
                                "Emergency close order placed for orphaned position %s", symbol
                            )
                    except Exception as close_err:
                        logger.critical(
                            "CRITICAL: Emergency close FAILED for %s. "
                            "MANUAL INTERVENTION REQUIRED. Error: %s",
                            symbol,
                            close_err,
                        )
                # Restore balance since position tracking failed (atomic refund)
                if self.trading_session_id is not None:
                    try:
                        with self.db_manager.atomic_balance_update(
                            balance_change=entry_fee,
                            reason=f"refund_entry_fee_{symbol}_tracking_failed",
                            updated_by="live_engine",
                            correlation_id=position.order_id,
                        ) as balance_result:
                            self.current_balance = balance_result["new_balance"]
                    except Exception as refund_err:
                        logger.critical(
                            "CRITICAL: Failed to refund entry fee after position tracking failure for %s. "
                            "Balance state inconsistent. Error: %s",
                            symbol,
                            refund_err,
                        )
                else:
                    # No trading session - update balance directly
                    self.current_balance += entry_fee
                return

            # Update risk manager tracking for new position.
            # If this fails, close the position to maintain state consistency.
            if self.risk_manager:
                try:
                    self.risk_manager.update_position(
                        symbol=symbol,
                        side=side.value,
                        size=size,
                        entry_price=position.entry_price,
                    )
                except (AttributeError, ValueError, KeyError, TypeError) as e:
                    # Risk manager update failed - state is now inconsistent.
                    # Close position to prevent exceeding risk limits.
                    logger.error(
                        "Risk manager update failed for %s position %s. "
                        "Closing position to maintain risk consistency. Error: %s",
                        side.value,
                        symbol,
                        e,
                    )
                    self._execute_exit(
                        position,
                        "Risk manager sync failure",
                        None,
                        price,
                        None,
                        None,
                        None,
                        skip_live_close=False,
                    )
                    return

            logger.info(
                "🚀 Opened %s position: %s @ $%.2f (Size: %.2f%%)",
                side.value,
                symbol,
                position.entry_price,
                size * 100,
            )
            log_order_event(
                "open_position",
                order_id=position.order_id,
                symbol=symbol,
                side=side.value,
                entry_price=position.entry_price,
                size=size,
            )

            # Register with order tracker AFTER position is fully tracked.
            # This ensures callbacks can find the position in the tracker.
            if position.order_id and self.order_tracker:
                try:
                    self.order_tracker.track_order(position.order_id, symbol)
                except Exception as e:
                    # Order tracking failure is non-critical - position exists and is tracked.
                    # Stop-loss monitoring may be affected but position is safe.
                    logger.warning(
                        "Failed to track order %s for %s (position still valid): %s",
                        position.order_id,
                        symbol,
                        e,
                    )

            # Send alert if configured
            self._send_alert(
                f"Position Opened: {symbol} {side.value} @ ${position.entry_price:.2f}"
            )

            # Place server-side stop-loss order for protection with retry logic
            if self.enable_live_trading and stop_loss and self.exchange_interface:
                sl_side = OrderSide.SELL if side == PositionSide.LONG else OrderSide.BUY
                sl_order_id = None
                max_retries = 3
                retry_delay = 1.0
                # Use stored quantity directly to ensure stop-loss covers exact position size
                if position.quantity is not None and position.quantity > 0:
                    quantity = position.quantity
                else:
                    # Fallback for legacy positions without quantity field
                    entry_balance = (
                        float(position.entry_balance)
                        if position.entry_balance is not None and position.entry_balance > 0
                        else float(self.current_balance)
                    )
                    position_value = size * entry_balance
                    quantity = (
                        position_value / float(position.entry_price)
                        if position.entry_price
                        else 0.0
                    )

                for attempt in range(max_retries):
                    try:
                        sl_order_id = self.exchange_interface.place_stop_loss_order(
                            symbol=symbol,
                            side=sl_side,
                            quantity=quantity,
                            stop_price=stop_loss,
                        )
                        if sl_order_id:
                            break
                    except Exception as sl_err:
                        logger.warning(
                            "Stop-loss placement attempt %s/%s failed: %s",
                            attempt + 1,
                            max_retries,
                            sl_err,
                        )

                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2

                if sl_order_id:
                    logger.info(
                        "Server-side stop-loss placed: %s @ $%.2f order_id=%s",
                        symbol,
                        stop_loss,
                        sl_order_id,
                    )
                    self.live_position_tracker.set_stop_loss_order_id(
                        position.order_id, sl_order_id
                    )
                    if self.order_tracker:
                        self.order_tracker.track_order(sl_order_id, symbol)
                else:
                    logger.critical(
                        "CRITICAL: Failed to place stop-loss after %s attempts for %s - closing position",
                        max_retries,
                        symbol,
                    )
                    self._send_alert(
                        f"⚠️ EMERGENCY: Closing {symbol} position - stop-loss placement failed"
                    )
                    self._execute_exit(
                        position,
                        "Stop-loss placement failed",
                        None,
                        price,
                        None,
                        None,
                        None,
                        skip_live_close=True,
                    )

        except Exception as e:
            logger.error("Failed to open position: %s", e, exc_info=True)
            if self.trading_session_id is not None:
                self.db_manager.log_event(
                    event_type="ERROR",
                    message=f"Failed to open position: {str(e)}",
                    severity="error",
                    component="LiveTradingEngine",
                    details={"stack_trace": str(e)},
                    session_id=self.trading_session_id,
                )
            else:
                logger.warning("⚠️ Cannot log error to database - no trading session ID available")

    def _handle_order_fill(
        self, order_id: str, symbol: str, filled_qty: float, avg_price: float
    ) -> None:
        """
        Handle a fully filled order notification from OrderTracker.

        This handles both entry order fills and stop-loss order fills.
        For stop-loss fills, it closes the associated position.

        Args:
            order_id: The filled order ID
            symbol: Trading symbol
            filled_qty: Total quantity filled
            avg_price: Average fill price
        """
        logger.info(
            f"Order fill confirmed: {order_id} {symbol} qty={filled_qty} @ ${avg_price:.2f}"
        )
        log_order_event(
            "order_filled",
            order_id=order_id,
            symbol=symbol,
            filled_quantity=filled_qty,
            average_price=avg_price,
        )

        # Check if this is a stop-loss order fill - need to close the position
        # Find matching position under lock, then close outside lock to avoid deadlock
        position_to_close: Position | None = None
        for pos_order_id, position in self.live_position_tracker.positions.items():
            if position.stop_loss_order_id == order_id:
                position_to_close = position
                logger.warning(
                    "Stop-loss order %s filled for position %s at $%.2f - closing position",
                    order_id,
                    pos_order_id,
                    avg_price,
                )
                break
        if position_to_close:
            if self.live_position_tracker.has_position(position_to_close.order_id):
                # Close position using the actual SL fill price (outside lock)
                self._execute_exit(
                    position_to_close,
                    reason="stop_loss",
                    limit_price=avg_price,
                    current_price=float(avg_price),
                    candle_high=None,
                    candle_low=None,
                    candle=None,
                    skip_live_close=True,
                )
            else:
                logger.info(
                    "Stop-loss fill received for already closed position %s",
                    position_to_close.order_id,
                )

    def _handle_partial_fill(
        self, order_id: str, symbol: str, new_filled_qty: float, avg_price: float
    ) -> None:
        """
        Handle a partial fill notification from OrderTracker.

        For stop-loss partial fills, logs a critical warning since position
        remains exposed. Full handling of partial SL fills would require
        placing a new SL order for the remaining quantity.

        Args:
            order_id: The partially filled order ID
            symbol: Trading symbol
            new_filled_qty: Additional quantity filled since last check
            avg_price: Average fill price
        """
        logger.info(f"Partial fill: {order_id} {symbol} +{new_filled_qty} @ ${avg_price:.2f}")
        log_order_event(
            "partial_fill",
            order_id=order_id,
            symbol=symbol,
            new_filled_quantity=new_filled_qty,
            average_price=avg_price,
        )

        # Check if this is a stop-loss order partial fill - log critical warning
        # Partial SL fills leave the position partially exposed without protection
        for pos_order_id, position in self.live_position_tracker.positions.items():
            if position.stop_loss_order_id == order_id:
                logger.critical(
                    "PARTIAL STOP-LOSS FILL: Position %s SL order %s partially filled (%.4f @ $%.2f). "
                    "Remaining position may be unprotected - manual intervention recommended.",
                    pos_order_id,
                    order_id,
                    new_filled_qty,
                    avg_price,
                )
                log_order_event(
                    "partial_sl_fill_warning",
                    order_id=order_id,
                    position_order_id=pos_order_id,
                    symbol=symbol,
                    filled_quantity=new_filled_qty,
                    average_price=avg_price,
                )
                return

    def _handle_order_cancel(self, order_id: str, symbol: str) -> None:
        """
        Handle an order cancellation/rejection notification from OrderTracker.

        Args:
            order_id: The cancelled/rejected order ID
            symbol: Trading symbol
        """
        logger.warning(f"Order cancelled/rejected: {order_id} {symbol}")
        log_order_event(
            "order_cancelled",
            order_id=order_id,
            symbol=symbol,
        )
        # Check if this was an entry order for a position we thought we had
        # Use atomic pop() for thread safety (called from OrderTracker background thread)
        removed_position = self.live_position_tracker.get_position(order_id)
        if removed_position:
            self.live_position_tracker.remove_position(order_id)
        if removed_position is not None:
            logger.error(f"Entry order {order_id} was cancelled - removing phantom position")

    def _execute_exit(
        self,
        position: Position,
        reason: str,
        limit_price: float | None,
        current_price: float,
        candle_high: float | None,
        candle_low: float | None,
        candle,
        skip_live_close: bool = False,
    ) -> None:
        """Close a position using shared execution modules."""
        try:
            if position.entry_price <= 0:
                logger.error(
                    "Invalid entry_price %s for position %s - cannot close position safely",
                    position.entry_price,
                    position.symbol,
                )
                return

            metrics = self.live_position_tracker.mfe_mae_tracker.get_position_metrics(
                position.order_id
            )

            sl_already_filled = False
            sl_fill_price: float | None = None
            if not skip_live_close:
                sl_already_filled, sl_fill_price = self._check_stop_loss_filled(position)
            else:
                sl_already_filled = True

            base_price = None
            if sl_already_filled and sl_fill_price is not None:
                base_price = float(sl_fill_price)
            elif limit_price is not None:
                base_price = float(limit_price)
            else:
                base_price = float(current_price)

            if sl_already_filled or skip_live_close:
                exit_result = self.live_exit_handler.execute_filled_exit(
                    position=position,
                    exit_reason=reason,
                    filled_price=base_price,
                    current_balance=self.current_balance,
                )
            else:
                exit_result = self.live_exit_handler.execute_exit(
                    position=position,
                    exit_reason=reason,
                    current_price=float(current_price),
                    limit_price=limit_price,
                    current_balance=self.current_balance,
                    candle_high=candle_high,
                    candle_low=candle_low,
                    data_provider=self.data_provider,
                )
            if not exit_result.success:
                logger.error(
                    "Failed to close position %s: %s",
                    position.order_id,
                    exit_result.error,
                )
                return

            realized_pnl = exit_result.realized_pnl - exit_result.exit_fee

            # Atomic balance update with full audit trail for realized P&L
            if self.trading_session_id is not None:
                try:
                    with self.db_manager.atomic_balance_update(
                        balance_change=realized_pnl,
                        reason=f"realized_pnl_{position.symbol}_{reason}",
                        updated_by="live_engine",
                        correlation_id=position.order_id,
                    ) as balance_result:
                        self.current_balance = balance_result["new_balance"]
                except Exception as balance_err:
                    logger.error(
                        "Failed to update balance for realized P&L %s: %s. Trade will be logged but balance inconsistent.",
                        position.symbol,
                        balance_err,
                    )
                    # Continue processing to log the trade even if balance update fails
                    # This allows for manual reconciliation
            else:
                # No trading session - update balance directly (testing/paper trading mode)
                self.current_balance += realized_pnl

            exit_price = float(exit_result.exit_price)
            exit_fee = exit_result.exit_fee
            exit_slippage_cost = exit_result.slippage_cost
            pnl_percent = exit_result.realized_pnl_percent

            entry_fee = float(position.metadata.get("entry_fee", 0.0))
            entry_slippage_cost = float(position.metadata.get("entry_slippage_cost", 0.0))
            total_fee = entry_fee + exit_fee
            total_slippage = entry_slippage_cost + exit_slippage_cost
            net_trade_pnl = realized_pnl - entry_fee

            trade = Trade(
                symbol=position.symbol,
                side=position.side,
                size=float(
                    position.current_size if position.current_size is not None else position.size
                ),
                entry_price=position.entry_price,
                exit_price=exit_price,
                entry_time=position.entry_time,
                exit_time=datetime.now(UTC),
                pnl=net_trade_pnl,
                pnl_percent=pnl_percent,
                exit_reason=reason,
            )

            self.performance_tracker.record_trade(
                trade=trade, fee=total_fee, slippage=total_slippage
            )

            self.completed_trades.append(trade)
            if self.log_trades:
                self._log_trade(trade)

            if self.trading_session_id is not None:
                self.db_manager.log_trade(
                    symbol=position.symbol,
                    side=position.side.value,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    size=float(
                        position.current_size
                        if position.current_size is not None
                        else position.size
                    ),
                    pnl=net_trade_pnl,
                    strategy_name=self._strategy_name(),
                    exit_reason=reason,
                    entry_time=position.entry_time,
                    exit_time=datetime.now(UTC),
                    session_id=self.trading_session_id,
                    mfe=(metrics.mfe if metrics else None),
                    mae=(metrics.mae if metrics else None),
                    mfe_price=(metrics.mfe_price if metrics else None),
                    mae_price=(metrics.mae_price if metrics else None),
                    mfe_time=(metrics.mfe_time if metrics else None),
                    mae_time=(metrics.mae_time if metrics else None),
                )

            if (
                self.enable_live_trading
                and self.exchange_interface
                and position.stop_loss_order_id
                and not sl_already_filled
            ):
                try:
                    cancelled = self.exchange_interface.cancel_order(
                        position.stop_loss_order_id, position.symbol
                    )
                    if cancelled:
                        logger.info(
                            "Cancelled stop-loss order %s for %s",
                            position.stop_loss_order_id,
                            position.symbol,
                        )
                except Exception as e:
                    logger.warning("Error cancelling stop-loss order: %s", e)

                if self.order_tracker:
                    self.order_tracker.stop_tracking(position.stop_loss_order_id)

            logger.info(
                "📈 Closed %s position for %s: PnL=$%.2f, Reason=%s, Balance=$%.2f",
                position.side.value,
                position.symbol,
                net_trade_pnl,
                reason,
                self.current_balance,
            )
            log_order_event(
                "close_position",
                order_id=position.order_id,
                symbol=position.symbol,
                side=position.side.value,
                exit_price=exit_price,
                pnl=net_trade_pnl,
                pnl_percent=trade.pnl_percent,
                reason=reason,
            )
        except Exception as e:
            logger.error("Failed to close position %s: %s", position.order_id, e, exc_info=True)

    def _check_stop_loss_filled(self, position: Position) -> tuple[bool, float | None]:
        """Check if a stop-loss order already filled on the exchange."""
        if (
            not self.enable_live_trading
            or not self.exchange_interface
            or not position.stop_loss_order_id
        ):
            return False, None

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                sl_order = self.exchange_interface.get_order(
                    position.stop_loss_order_id, position.symbol
                )
                if sl_order and sl_order.status == ExchangeOrderStatus.FILLED:
                    logger.info(
                        "Stop-loss order %s already filled at $%.2f - using actual fill price",
                        position.stop_loss_order_id,
                        sl_order.average_price,
                    )
                    return True, sl_order.average_price
                return False, None
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(
                    "Transient error checking stop-loss order %s (attempt %s/%s): %s",
                    position.stop_loss_order_id,
                    attempt + 1,
                    max_attempts,
                    e,
                )
                if attempt < max_attempts - 1:
                    time.sleep(2**attempt)
            except Exception as e:
                logger.error(
                    "Unexpected error checking stop-loss order %s: %s",
                    position.stop_loss_order_id,
                    e,
                    exc_info=True,
                )
                return False, None

        logger.error(
            "Failed to check stop-loss order %s after %s attempts; assuming not filled",
            position.stop_loss_order_id,
            max_attempts,
        )
        log_order_event(
            "sl_check_failed",
            order_id=position.stop_loss_order_id,
            symbol=position.symbol,
        )
        return False, None

    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        # Update performance tracker on every metric update cycle
        # Note: Less frequent than backtest (every candle vs every update cycle)
        # This trade-off reduces overhead while maintaining statistical validity for risk metrics
        self.performance_tracker.update_balance(self.current_balance, timestamp=datetime.now(UTC))

    def _extract_indicators(self, df: pd.DataFrame, index: int) -> dict:
        """Extract indicator values from dataframe for logging"""
        if index >= len(df):
            return {}

        indicators = {}
        current_row = df.iloc[index]

        # Common indicators to extract
        indicator_columns = [
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "atr",
            "volatility",
            "trend_ma",
            "short_ma",
            "long_ma",
            "volume_ma",
            "trend_strength",
            "regime",
            "body_size",
            "upper_wick",
            "lower_wick",
        ]

        for col in indicator_columns:
            if col in df.columns and not pd.isna(current_row[col]):
                indicators[col] = float(current_row[col])

        # Add basic OHLCV data
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                indicators[col] = float(current_row[col])

        return indicators

    def _extract_sentiment_data(self, df: pd.DataFrame, index: int) -> dict:
        """Extract sentiment data from dataframe for logging"""
        if index >= len(df):
            return {}

        sentiment_data = {}
        current_row = df.iloc[index]

        # Sentiment columns to extract
        sentiment_columns = [
            "sentiment_primary",
            "sentiment_momentum",
            "sentiment_volatility",
            "sentiment_extreme_positive",
            "sentiment_extreme_negative",
            "sentiment_ma_3",
            "sentiment_ma_7",
            "sentiment_ma_14",
            "sentiment_confidence",
            "sentiment_freshness",
        ]

        for col in sentiment_columns:
            if col in df.columns and not pd.isna(current_row[col]):
                sentiment_data[col] = float(current_row[col])

        return sentiment_data

    def _extract_ml_predictions(self, df: pd.DataFrame, index: int) -> dict:
        """Extract ML prediction data from dataframe for logging"""
        if index >= len(df):
            return {}

        ml_data = {}
        current_row = df.iloc[index]

        # ML prediction columns to extract
        ml_columns = ["ml_prediction", "prediction_confidence", "onnx_pred"]

        for col in ml_columns:
            if col in df.columns and not pd.isna(current_row[col]):
                ml_data[col] = float(current_row[col])

        return ml_data

    def _log_account_snapshot(self):
        """Log current account state to database"""
        try:
            # Calculate total exposure using the active fraction per position
            positions_snapshot = self.live_position_tracker.positions
            total_exposure = sum(
                float(pos.current_size if pos.current_size is not None else pos.size)
                * (
                    float(pos.entry_balance)
                    if pos.entry_balance is not None and pos.entry_balance > 0
                    else float(self.current_balance)
                )
                for pos in positions_snapshot.values()
            )

            # Calculate equity (balance + unrealized P&L)
            unrealized_pnl = sum(float(pos.unrealized_pnl) for pos in positions_snapshot.values())
            equity = float(self.current_balance) + unrealized_pnl

            # Calculate current drawdown percentage
            current_drawdown = 0
            perf_metrics = self.performance_tracker.get_metrics()
            if perf_metrics.peak_balance > 0:
                current_drawdown = (
                    (perf_metrics.peak_balance - self.current_balance)
                    / perf_metrics.peak_balance
                    * 100
                )

            # TODO: Calculate daily P&L (requires tracking of day start balance)
            daily_pnl = 0  # Placeholder

            # Log snapshot to database
            if self.trading_session_id is not None:
                self.db_manager.log_account_snapshot(
                    balance=self.current_balance,
                    equity=equity,
                    total_pnl=perf_metrics.total_pnl,
                    open_positions=self.live_position_tracker.position_count,
                    total_exposure=total_exposure,
                    drawdown=current_drawdown,
                    daily_pnl=daily_pnl,
                    session_id=self.trading_session_id,
                )
            else:
                logger.warning(
                    "⚠️ Cannot log account snapshot to database - no trading session ID available"
                )

        except Exception as e:
            logger.error(f"Failed to log account snapshot: {e}")

    def _log_status(self, symbol: str, current_price: float):
        """Log current trading status"""
        total_unrealized = sum(
            float(pos.unrealized_pnl) for pos in self.live_position_tracker.positions.values()
        )
        perf_metrics = self.performance_tracker.get_metrics()
        win_rate = perf_metrics.win_rate * 100

        logger.info(
            f"📊 Status: {symbol} @ ${current_price:.2f} | "
            f"Balance: ${self.current_balance:.2f} | "
            f"Positions: {self.live_position_tracker.position_count} | "
            f"Unrealized: ${total_unrealized:.2f} | "
            f"Trades: {perf_metrics.total_trades} ({win_rate:.1f}% win)"
        )

    def _log_trade(self, trade: Trade):
        """Log trade to file"""
        try:
            # Create logs/trades directory if it doesn't exist
            os.makedirs("logs/trades", exist_ok=True)

            log_file = f"logs/trades/trades_{datetime.now(UTC).strftime('%Y%m')}.json"
            trade_data = {
                "timestamp": trade.exit_time.isoformat(),
                "symbol": trade.symbol,
                "side": trade.side.value,
                "size": trade.size,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "pnl": trade.pnl,
                "pnl_percent": trade.pnl_percent,
                "exit_reason": trade.exit_reason,
                "duration_minutes": (trade.exit_time - trade.entry_time).total_seconds() / 60,
            }

            # Append to log file
            with open(log_file, "a") as f:
                f.write(json.dumps(trade_data) + "\n")

        except Exception as e:
            logger.error(f"Failed to log trade: {e}", exc_info=True)

    def _send_alert(self, message: str):
        """Send trading alert (webhook, email, etc.)"""
        if not self.alert_webhook_url:
            return

        try:
            import requests

            payload = {
                "text": f"🤖 Trading Bot: {message}",
                "timestamp": datetime.now(UTC).isoformat(),
            }
            requests.post(self.alert_webhook_url, json=payload, timeout=10)
        except Exception as e:
            logger.error(f"Failed to send alert: {e}", exc_info=True)

    def _sleep_with_interrupt(self, seconds: float):
        """Sleep in small increments to allow for interrupt and float seconds"""
        end_time = time.time() + seconds
        poll_interval = DEFAULT_SLEEP_POLL_INTERVAL  # Use configurable interval instead of 0.1
        while time.time() < end_time:
            if self.stop_event.is_set():
                break
            time.sleep(min(poll_interval, end_time - time.time()))

    def _calculate_adaptive_interval(self, current_price: float | None = None) -> int:
        """Calculate adaptive check interval based on recent trading activity and market conditions"""
        # Base interval from configuration
        interval = self.base_check_interval

        # Factor in recent trading activity
        recent_trades = len(
            [
                p
                for p in self.live_position_tracker.positions.values()
                if p.entry_time > datetime.now(UTC) - timedelta(hours=1)
            ]
        )
        if recent_trades > 0:
            # More frequent checks if we have recent activity
            interval = max(self.min_check_interval, interval // 2)
        elif self.live_position_tracker.position_count == 0:
            # Less frequent checks if no active positions
            interval = min(self.max_check_interval, interval * 2)

        # Consider time of day (basic market hours awareness)
        current_hour = datetime.now(UTC).hour
        if current_hour < 6 or current_hour > 22:  # Off-hours (UTC)
            interval = min(self.max_check_interval, interval * 1.5)

        return int(interval)

    def _is_data_fresh(self, df: pd.DataFrame) -> bool:
        """Check if the data is fresh enough to warrant processing"""
        if df is None or df.empty:
            return False

        latest_timestamp = df.index[-1] if hasattr(df.index[-1], "timestamp") else datetime.now(UTC)
        if isinstance(latest_timestamp, str):
            try:
                latest_timestamp = pd.to_datetime(latest_timestamp)
            except (ValueError, TypeError):
                return True  # Assume fresh if we can't parse timestamp

        # Normalizes to UTC to avoid naive/aware datetime comparisons.
        if isinstance(latest_timestamp, pd.Timestamp):
            if latest_timestamp.tz is None:
                latest_timestamp = latest_timestamp.tz_localize(UTC)
            else:
                latest_timestamp = latest_timestamp.tz_convert(UTC)
        elif isinstance(latest_timestamp, datetime):
            if latest_timestamp.tzinfo is None:
                latest_timestamp = latest_timestamp.replace(tzinfo=UTC)
            else:
                latest_timestamp = latest_timestamp.astimezone(UTC)

        age_seconds = (datetime.now(UTC) - latest_timestamp).total_seconds()
        return age_seconds <= self.data_freshness_threshold

    def _print_final_stats(self):
        """Print final trading statistics"""
        # Validate initial_balance before division to prevent crashes
        if self.initial_balance <= 0:
            logger.error(
                "Cannot calculate total return - invalid initial_balance: %.8f. "
                "Skipping final statistics.",
                self.initial_balance,
            )
            return

        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        perf_metrics = self.performance_tracker.get_metrics()
        win_rate = perf_metrics.win_rate * 100

        print("\n" + "=" * 60)
        print("🏁 FINAL TRADING STATISTICS")
        print("=" * 60)
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${self.current_balance:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Total PnL: ${perf_metrics.total_pnl:+,.2f}")
        print(f"Max Drawdown: {perf_metrics.max_drawdown * 100:.2f}%")
        print(f"Total Trades: {perf_metrics.total_trades}")
        print(f"Winning Trades: {perf_metrics.winning_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Active Positions: {self.live_position_tracker.position_count}")

        if self.completed_trades:
            avg_trade = sum(trade.pnl for trade in self.completed_trades) / len(
                self.completed_trades
            )
            print(f"Average Trade: ${avg_trade:.2f}")

        print("=" * 60)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get current performance summary"""
        # Get comprehensive metrics from performance tracker
        perf_metrics = self.performance_tracker.get_metrics()

        # Convert to percentages for backward compatibility
        win_rate = perf_metrics.win_rate * 100
        current_drawdown = perf_metrics.current_drawdown * 100
        max_drawdown_pct = perf_metrics.max_drawdown * 100

        return {
            # Core metrics from tracker
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "total_return": perf_metrics.total_return_pct,
            "total_return_pct": perf_metrics.total_return_pct,
            "total_pnl": perf_metrics.total_pnl,
            "current_drawdown": current_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "total_trades": perf_metrics.total_trades,
            "winning_trades": perf_metrics.winning_trades,
            "win_rate": win_rate,
            "win_rate_pct": win_rate,
            # New metrics from tracker
            "sharpe_ratio": perf_metrics.sharpe_ratio,
            "sortino_ratio": perf_metrics.sortino_ratio,
            "calmar_ratio": perf_metrics.calmar_ratio,
            "var_95": perf_metrics.var_95,
            "expectancy": perf_metrics.expectancy,
            "profit_factor": perf_metrics.profit_factor,
            "avg_win": perf_metrics.avg_win,
            "avg_loss": perf_metrics.avg_loss,
            "largest_win": perf_metrics.largest_win,
            "largest_loss": perf_metrics.largest_loss,
            "avg_trade_duration_hours": perf_metrics.avg_trade_duration_hours,
            "consecutive_wins": perf_metrics.consecutive_wins,
            "consecutive_losses": perf_metrics.consecutive_losses,
            "total_fees_paid": perf_metrics.total_fees_paid,
            "total_slippage_cost": perf_metrics.total_slippage_cost,
            # Live-specific metrics
            "active_positions": self.live_position_tracker.position_count,
            "last_update": self.last_data_update.isoformat() if self.last_data_update else None,
            "is_running": self.is_running,
        }

    def _recover_existing_session(self) -> float | None:
        """Try to recover from an existing active session"""
        try:
            # Check if there's an active session
            active_session_id = self.db_manager.get_active_session_id()
            if active_session_id:
                logger.info(f"🔍 Found active session #{active_session_id}")

                # Try to recover balance
                recovered_balance = self.db_manager.recover_last_balance(active_session_id)
                if recovered_balance and recovered_balance > 0:
                    self.trading_session_id = active_session_id
                    logger.info(
                        f"🎯 Recovered session #{active_session_id} with balance ${recovered_balance:,.2f}"
                    )
                    return recovered_balance
                else:
                    logger.warning("⚠️  Active session found but no balance to recover")
            else:
                logger.info("🆕 No active session found")

            return None
        except Exception as e:
            logger.error(f"❌ Error recovering session: {e}", exc_info=True)
            return None

    def _recover_active_positions(self) -> None:
        """Recover active positions from database"""
        try:
            if not self.trading_session_id:
                return

            # Get active positions from database
            db_positions = self.db_manager.get_active_positions(self.trading_session_id)

            if not db_positions:
                logger.info("📊 No active positions to recover")
                return

            logger.info(f"🔄 Recovering {len(db_positions)} active positions...")

            for pos_data in db_positions:
                # Convert database position to Position object
                # Handle both uppercase and lowercase side values from database
                side_value = pos_data["side"]
                if isinstance(side_value, str):
                    side_value = side_value.lower()

                stored_entry_balance = pos_data.get("entry_balance")
                try:
                    entry_balance = (
                        float(stored_entry_balance)
                        if stored_entry_balance is not None
                        else float(self.current_balance)
                    )
                except (TypeError, ValueError):
                    logger.warning(
                        "Recovered position %s has invalid entry balance %s; falling back to current balance",
                        pos_data.get("symbol"),
                        stored_entry_balance,
                    )
                    entry_balance = float(self.current_balance)

                position = Position(
                    symbol=pos_data["symbol"],
                    side=PositionSide(side_value),
                    size=pos_data["size"],
                    entry_price=pos_data["entry_price"],
                    entry_time=pos_data["entry_time"],
                    entry_balance=entry_balance,
                    stop_loss=pos_data.get("stop_loss"),
                    take_profit=pos_data.get("take_profit"),
                    unrealized_pnl=float(pos_data.get("unrealized_pnl", 0.0) or 0.0),
                    unrealized_pnl_percent=float(
                        pos_data.get("unrealized_pnl_percent", 0.0) or 0.0
                    ),
                    order_id=str(pos_data["id"]),  # Use database ID as order_id
                    stop_loss_order_id=pos_data.get("stop_loss_order_id"),
                )

                if position.order_id:
                    self.live_position_tracker.track_recovered_position(
                        position, db_id=pos_data.get("id")
                    )

                # Register recovered stop-loss order with OrderTracker for monitoring
                if position.stop_loss_order_id and self.order_tracker:
                    self.order_tracker.track_order(position.stop_loss_order_id, position.symbol)
                    logger.info(
                        f"📡 Recovered and tracking stop-loss order {position.stop_loss_order_id} "
                        f"for position {position.symbol}"
                    )

                # Update risk manager tracking for recovered positions
                if self.risk_manager:
                    try:
                        self.risk_manager.update_position(
                            symbol=position.symbol,
                            side=position.side.value,
                            size=position.size,
                            entry_price=position.entry_price,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to update risk manager for recovered position {position.symbol}: {e}"
                        )

                logger.info(
                    f"✅ Recovered position: {pos_data['symbol']} {pos_data['side']} @ ${pos_data['entry_price']:.2f}"
                )

            logger.info(f"🎯 Successfully recovered {len(db_positions)} positions")

        except Exception as e:
            logger.error(f"❌ Error recovering positions: {e}", exc_info=True)

    def _reconcile_positions_with_exchange(self) -> None:
        """
        Reconcile local positions with exchange state on startup.

        This detects positions that were closed while the bot was offline
        (e.g., by stop-loss orders triggering) and updates local state accordingly.
        """
        if not self.exchange_interface or not self.enable_live_trading:
            return

        positions_snapshot = self.live_position_tracker.positions
        if not positions_snapshot:
            logger.info("📊 No local positions to reconcile")
            return

        logger.info("🔄 Reconciling %s positions with exchange...", len(positions_snapshot))

        try:
            # Get open orders from exchange
            exchange_orders = self.exchange_interface.get_open_orders()
            exchange_order_ids = {order.order_id for order in exchange_orders}

            # Check each local position
            positions_to_close = []
            for _order_id, position in positions_snapshot.items():
                # Check if the position's entry order is still open (shouldn't be for filled orders)
                # More importantly, check if stop-loss order is still active
                if position.stop_loss_order_id:
                    if position.stop_loss_order_id not in exchange_order_ids:
                        # Stop-loss order is gone - may have triggered
                        logger.warning(
                            f"⚠️ Stop-loss order {position.stop_loss_order_id} not found "
                            f"on exchange for {position.symbol} - position may have closed"
                        )
                        # Check if we can verify the order status
                        try:
                            sl_order = self.exchange_interface.get_order(
                                position.stop_loss_order_id, position.symbol
                            )
                            if sl_order and sl_order.status == ExchangeOrderStatus.FILLED:
                                logger.info(
                                    f"✅ Confirmed: Stop-loss triggered for {position.symbol} "
                                    f"@ ${sl_order.average_price or 'unknown'}"
                                )
                                positions_to_close.append((position, sl_order.average_price))
                        except Exception as e:
                            logger.warning(f"Could not verify stop-loss order status: {e}")

            # Close positions that were stopped out
            for position, exit_price in positions_to_close:
                logger.info(
                    f"🔄 Marking position {position.symbol} as closed (stop-loss triggered offline)"
                )
                # Update balance based on stop-loss exit
                if exit_price:
                    fraction = (
                        position.current_size
                        if position.current_size is not None
                        else position.size
                    )
                    # Guard against division by zero
                    if position.entry_price <= 0:
                        logger.error(
                            f"Invalid entry_price {position.entry_price} for position "
                            f"{position.symbol} - skipping reconciliation"
                        )
                        continue
                    if position.side == PositionSide.LONG:
                        pnl_pct = (exit_price - position.entry_price) / position.entry_price
                    else:
                        pnl_pct = (position.entry_price - exit_price) / position.entry_price

                    # Use entry_balance for PnL calculation to maintain backtest-live parity
                    basis_balance = (
                        float(position.entry_balance)
                        if position.entry_balance is not None and position.entry_balance > 0
                        else self.current_balance
                    )
                    # Calculate exit fee and slippage (matching normal exit flow)
                    exit_position_notional = (
                        basis_balance * fraction * (exit_price / position.entry_price)
                    )
                    exit_fee = self.live_execution_engine.calculate_exit_fee(exit_position_notional)
                    exit_slippage_cost = self.live_execution_engine.calculate_slippage_cost(
                        exit_position_notional
                    )
                    realized_pnl = pnl_pct * fraction * basis_balance - exit_fee

                    # Atomic balance update for offline stop-loss reconciliation
                    if self.trading_session_id is not None:
                        try:
                            with self.db_manager.atomic_balance_update(
                                balance_change=realized_pnl,
                                reason=f"offline_stop_loss_{position.symbol}",
                                updated_by="live_engine_reconciliation",
                                correlation_id=position.order_id,
                            ) as balance_result:
                                self.current_balance = balance_result["new_balance"]
                                logger.info(
                                    f"💰 Adjusted balance for offline stop-loss: ${realized_pnl:+,.2f} "
                                    f"(fee: ${exit_fee:.2f}) -> ${self.current_balance:,.2f}"
                                )
                        except Exception as balance_err:
                            logger.error(
                                "Failed to update balance for offline stop-loss %s: %s. Skipping reconciliation.",
                                position.symbol,
                                balance_err,
                            )
                            continue
                    else:
                        # No trading session - update balance directly
                        self.current_balance += realized_pnl
                        logger.info(
                            f"💰 Adjusted balance for offline stop-loss: ${realized_pnl:+,.2f} "
                            f"(fee: ${exit_fee:.2f}) -> ${self.current_balance:,.2f}"
                        )
                    trade = Trade(
                        symbol=position.symbol,
                        side=position.side,
                        size=fraction,
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        entry_time=position.entry_time,
                        exit_time=datetime.now(UTC),
                        pnl=realized_pnl,
                        pnl_percent=pnl_pct,
                        exit_reason="stop_loss_offline",
                    )
                    self.performance_tracker.record_trade(
                        trade=trade, fee=exit_fee, slippage=exit_slippage_cost
                    )
                    self.completed_trades.append(trade)
                    if self.log_trades:
                        self._log_trade(trade)

                # Stop tracking the SL order to prevent memory leak
                if position.stop_loss_order_id and self.order_tracker:
                    self.order_tracker.stop_tracking(position.stop_loss_order_id)

                # Remove from local positions
                if position.order_id:
                    self.live_position_tracker.remove_position(position.order_id)
                if self.risk_manager:
                    try:
                        self.risk_manager.close_position(position.symbol)
                    except Exception as e:
                        logger.warning(
                            "Failed to update risk manager for reconciled position %s: %s",
                            position.symbol,
                            e,
                        )

                # Close in database
                db_ids = self.live_position_tracker.position_db_ids
                position_db_id = db_ids.get(position.order_id)
                if position_db_id:
                    self.db_manager.close_position(position_id=position_db_id)

            if positions_to_close:
                logger.info(
                    f"🔄 Reconciliation complete: {len(positions_to_close)} positions "
                    "closed (stopped out while offline)"
                )
            else:
                logger.info("✅ All positions verified - no offline closures detected")

        except Exception as e:
            logger.error(f"❌ Error reconciling positions with exchange: {e}", exc_info=True)

    def _handle_strategy_change(self, swap_data: dict[str, Any]):
        """Handle strategy change callback"""
        logger.info(f"🔄 Strategy change requested: {swap_data}")

        # If requested to close positions, close them now
        if swap_data.get("close_positions", False):
            logger.info("🚪 Closing all positions before strategy swap")
            for position in list(self.live_position_tracker.positions.values()):
                # Validate price before closing to prevent data corruption
                try:
                    current_price = self.data_provider.get_current_price(position.symbol)
                except Exception as exc:
                    logger.error(
                        "Cannot close position %s during strategy change - price fetch failed: %s. "
                        "Position will remain open.",
                        position.symbol,
                        exc,
                    )
                    continue
                if current_price is None or current_price <= 0:
                    logger.error(
                        "Cannot close position %s during strategy change - invalid price %s. "
                        "Position will remain open.",
                        position.symbol,
                        current_price,
                    )
                    continue

                self._execute_exit(
                    position,
                    "Strategy change - close requested",
                    None,
                    float(current_price),
                    None,
                    None,
                    None,
                )
        else:
            logger.info("📊 Keeping existing positions during strategy swap")

    def _handle_model_update(self, update_data: dict[str, Any]):
        """Handle model update callback"""
        logger.info(f"🤖 Model update requested: {update_data}")
        # Model update logic is handled in strategy_manager.apply_pending_update()

    def hot_swap_strategy(
        self, new_strategy_name: str, close_positions: bool = False, new_config: dict | None = None
    ) -> bool:
        """
        Hot-swap to a new strategy during live trading

        Args:
            new_strategy_name: Name of new strategy
            close_positions: Whether to close existing positions
            new_config: Configuration for new strategy

        Returns:
            True if swap was initiated successfully
        """

        if not self.strategy_manager:
            logger.error("Strategy manager not initialized - hot swapping disabled")
            return False

        logger.info(f"🔄 Initiating hot-swap to strategy: {new_strategy_name}")

        success = self.strategy_manager.hot_swap_strategy(
            new_strategy_name=new_strategy_name,
            new_config=new_config,
            close_existing_positions=close_positions,
        )

        if success:
            logger.info("✅ Hot-swap initiated successfully - will apply on next cycle")
            strategy_name = self._strategy_name()
            self._send_alert(f"Strategy hot-swap initiated: {strategy_name} → {new_strategy_name}")
        else:
            logger.error("❌ Hot-swap initiation failed")

        return success

    def update_model(self, new_model_path: str) -> bool:
        """
        Update ML models during live trading

        Args:
            new_model_path: Path to new model file

        Returns:
            True if update was initiated successfully
        """

        if not self.strategy_manager:
            logger.error("Strategy manager not initialized - model updates disabled")
            return False

        strategy_name = self._strategy_name().lower()

        logger.info(f"🤖 Initiating model update for strategy: {strategy_name}")

        success = self.strategy_manager.update_model(
            strategy_name=strategy_name, new_model_path=new_model_path, validate_model=True
        )

        if success:
            logger.info("✅ Model update initiated successfully - will apply on next cycle")
            self._send_alert(f"Model update initiated for {strategy_name}")
        else:
            logger.error("❌ Model update initiation failed")

        return success

    def _build_trailing_policy(self) -> TrailingStopPolicy | None:
        """Construct trailing policy from risk parameters and strategy overrides.

        Uses shared risk configuration logic for consistency with backtest engine.
        """
        return build_trailing_stop_policy(self.strategy, self.risk_manager)
