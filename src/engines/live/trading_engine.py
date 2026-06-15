from __future__ import annotations

import json
import logging
import os
import queue
import signal
import sys
import threading
import time
import traceback
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

from src.config import get_config
from src.config.constants import (
    DEFAULT_ACCOUNT_SNAPSHOT_INTERVAL,
    DEFAULT_CHECK_INTERVAL,
    DEFAULT_DATA_FRESHNESS_THRESHOLD,
    DEFAULT_DB_OUTAGE_CLOSE_ONLY_SECONDS,
    DEFAULT_DYNAMIC_RISK_ENABLED,
    DEFAULT_END_OF_DAY_FLAT,
    DEFAULT_ERROR_COOLDOWN,
    DEFAULT_FEE_RATE,
    DEFAULT_INITIAL_BALANCE,
    DEFAULT_MARKET_TIMEZONE,
    DEFAULT_MAX_CHECK_INTERVAL,
    DEFAULT_MAX_FILLED_PRICE_DEVIATION,
    DEFAULT_MAX_HOLDING_HOURS,
    DEFAULT_MAX_POSITION_SIZE,
    DEFAULT_MIN_CHECK_INTERVAL,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_TAKE_PROFIT_PCT,
    DEFAULT_TIME_RESTRICTIONS,
    DEFAULT_WEEKEND_FLAT,
)
from src.config.feature_flags import is_enabled
from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.coinbase_provider import CoinbaseProvider
from src.data_providers.data_provider import DataProvider
from src.data_providers.sentiment_provider import SentimentDataProvider
from src.database.manager import DatabaseManager
from src.database.models import EventType, TradeSource
from src.engines.live.config import LiveEngineSettings

# Modular handlers (optional injection for testability)
from src.engines.live.data.market_data_handler import MarketDataHandler
from src.engines.live.dynamic_risk_coordinator import LiveDynamicRiskCoordinator
from src.engines.live.execution.entry_coordinator import LiveEntryCoordinator
from src.engines.live.execution.entry_handler import LiveEntryHandler
from src.engines.live.execution.execution_engine import LiveExecutionEngine
from src.engines.live.execution.exit_coordinator import LiveExitCoordinator
from src.engines.live.execution.exit_handler import LiveExitHandler
from src.engines.live.execution.market_data_coordinator import LiveMarketDataCoordinator
from src.engines.live.execution.order_fill_coordinator import LiveOrderFillCoordinator
from src.engines.live.execution.position_tracker import (
    LivePosition,
    LivePositionTracker,
)
from src.engines.live.execution.stop_loss_manager import LiveStopLossManager
from src.engines.live.health.health_monitor import HealthMonitor
from src.engines.live.logging.event_logger import LiveEventLogger
from src.engines.live.loop_timing import LiveLoopTimingCoordinator
from src.engines.live.monitoring import (
    LiveAccountMonitor,
    extract_indicators,
    extract_ml_predictions,
    extract_sentiment_data,
)
from src.engines.live.recovery import LiveSessionRecoverer
from src.engines.live.strategy_hot_swap import StrategyHotSwapCoordinator
from src.engines.live.strategy_manager import StrategyManager
from src.engines.live.strategy_runtime import StrategyRuntimeCoordinator

# Re-exported close-accounting helpers: the exit path (now LiveExitCoordinator)
# uses them directly, and tests import them from this module, so keep the
# re-export here even though this module no longer references them.
from src.engines.live.trade_close_accounting import (  # noqa: F401
    _close_entry_fee_usd,
    _close_position_portion,
    _closed_base_quantity,
)
from src.engines.live.ws_health import WebSocketHealthMonitor
from src.engines.shared.correlation_handler import CorrelationHandler
from src.engines.shared.dynamic_risk_handler import DynamicRiskHandler
from src.engines.shared.execution.execution_model import ExecutionModel
from src.engines.shared.models import (
    BaseTrade,
    PositionSide,
)
from src.engines.shared.partial_operations_manager import PartialOperationsManager
from src.engines.shared.risk_configuration import (
    build_trailing_stop_policy,
    merge_dynamic_risk_config,
)
from src.infrastructure.logging.context import set_context, update_context
from src.infrastructure.logging.events import (
    log_data_event,
    log_engine_event,
)
from src.position_management.correlation_engine import CorrelationConfig, CorrelationEngine
from src.position_management.dynamic_risk import DynamicRiskConfig, DynamicRiskManager
from src.position_management.partial_manager import PartialExitPolicy
from src.position_management.time_exits import TimeExitPolicy, TimeRestrictions
from src.position_management.trailing_stops import TrailingStopPolicy
from src.regime.detector import RegimeDetector
from src.risk.risk_manager import RiskManager, RiskParameters
from src.strategies.components import Position as ComponentPosition
from src.strategies.components import RuntimeContext, StrategyRuntime
from src.strategies.components import Strategy as ComponentStrategy

from .account_sync import AccountSynchronizer
from .order_tracker import OrderTracker

if TYPE_CHECKING:
    from src.config.config_manager import ConfigManager
    from src.engines.live.kline_buffer import KlineBuffer
    from src.engines.live.reconciliation import PeriodicReconciler
    from src.engines.live.user_data_processor import UserDataProcessor
    from src.strategies.components.runtime import SupportsRuntimeHooks
    from src.strategies.components.strategy import TradingDecision

logger = logging.getLogger(__name__)

# Type aliases for backward compatibility - use shared models
# Position uses LivePosition which has stop_loss_order_id for server-side stop tracking
Position = LivePosition
# Trade uses BaseTrade which has all required fields plus MFE/MAE tracking
Trade = BaseTrade


def _create_exchange_provider(provider: str, config: ConfigManager, testnet: bool = False):
    """Factory to create exchange provider and return (provider_instance, provider_name).

    Args:
        provider: Exchange provider name ('binance' or 'coinbase')
        config: Configuration manager containing API credentials
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
        max_position_size: float = DEFAULT_MAX_POSITION_SIZE,
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
        enable_partial_operations: bool = True,  # Enable by default for better profit capture
        # Execution realism parameters (parity with backtest engine)
        fee_rate: float = DEFAULT_FEE_RATE,
        slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
        use_high_low_for_stops: bool = True,  # Check candle high/low for SL/TP detection
        max_filled_price_deviation: float = DEFAULT_MAX_FILLED_PRICE_DEVIATION,
        # Handler injection (all optional - defaults created if not provided)
        position_tracker: LivePositionTracker | None = None,
        execution_engine: LiveExecutionEngine | None = None,
        entry_handler: LiveEntryHandler | None = None,
        exit_handler: LiveExitHandler | None = None,
        market_data_handler: MarketDataHandler | None = None,
        event_logger: LiveEventLogger | None = None,
        health_monitor: HealthMonitor | None = None,
        settings: LiveEngineSettings | None = None,
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
        settings : LiveEngineSettings, optional
            Pre-resolved construction-time settings (feature flags / env /
            app config). The runner builds these explicitly; when omitted the
            engine resolves them itself (#486).
        """

        self._validate_inputs(
            initial_balance=initial_balance,
            max_position_size=max_position_size,
            check_interval=check_interval,
            account_snapshot_interval=account_snapshot_interval,
        )
        self._resolve_settings(settings)
        self._init_coordinators()
        self._configure_strategy(strategy)
        self.data_provider = data_provider
        self.sentiment_provider = sentiment_provider
        self._init_risk_manager(risk_parameters)
        self._init_risk_policies(
            trailing_stop_policy=trailing_stop_policy,
            enable_dynamic_risk=enable_dynamic_risk,
            dynamic_risk_config=dynamic_risk_config,
        )

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
        self._init_partial_operations(
            enable_partial_operations=enable_partial_operations,
            partial_manager=partial_manager,
        )
        self._init_correlation(data_provider)
        self._init_database(database_url)
        self._init_dynamic_risk_manager()

        self._init_exchange_interface(
            provider=provider,
            testnet=testnet,
            enable_live_trading=enable_live_trading,
        )
        self._resume_balance_from_snapshot()
        self._init_strategy_manager(strategy, enable_hot_swapping=enable_hot_swapping)
        self._seed_trading_state()

        # Performance tracker (unified with backtest engine)
        from src.performance.tracker import PerformanceTracker

        self.performance_tracker = PerformanceTracker(initial_balance)

        # Error handling
        self.max_consecutive_errors = max_consecutive_errors
        self.consecutive_errors = 0
        # Monotonic timestamp marking when the database first became unreachable
        # inside the trading loop (None while reachable). Lets the engine ride
        # out transient DB outages instead of counting them toward
        # max_consecutive_errors (incident 2026-05-19: a Railway internal-DNS
        # outage made Postgres unresolvable and shut the live bot down).
        self.db_unreachable_since: float | None = None
        self.error_cooldown = DEFAULT_ERROR_COOLDOWN

        self._init_time_exit_policy(time_exit_policy)

        # Threading
        self.main_thread: threading.Thread | None = None
        # Set when the trading loop dies abnormally (unhandled crash or error
        # exhaustion) so start() can exit non-zero for an orchestrator restart (#630).
        self._loop_crashed = False
        self.stop_event = threading.Event()

        # Optional regime detector (feature-gated)
        self.regime_detector = None
        try:
            if self.settings.regime_detection_enabled:
                self.regime_detector = RegimeDetector()
        except Exception:
            self.regime_detector = None

        self._install_signal_handlers()

        # Execution modeling
        self.execution_fill_policy = self.settings.execution_fill_policy
        self.execution_model = ExecutionModel(self.execution_fill_policy)

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

    def _validate_inputs(
        self,
        *,
        initial_balance: float,
        max_position_size: float,
        check_interval: int,
        account_snapshot_interval: int,
    ) -> None:
        """Validate constructor inputs, raising ``ValueError`` on bad values."""
        if initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        if max_position_size <= 0 or max_position_size > 1:
            raise ValueError("Max position size must be between 0 and 1")
        if check_interval <= 0:
            raise ValueError("Check interval must be positive")
        if account_snapshot_interval < 0:
            raise ValueError("Account snapshot interval must be non-negative")

    def _resolve_settings(self, settings: LiveEngineSettings | None) -> None:
        """Resolve construction-time settings (feature flags / env / config)."""
        # Construction-time settings. Pass this module's lookups so test
        # patches on trading_engine.is_enabled / trading_engine.get_config
        # keep intercepting resolution.
        self.settings = settings or LiveEngineSettings.resolve(
            flag_lookup=is_enabled,
            env_lookup=os.getenv,
            config_lookup=get_config,
        )

    def _init_coordinators(self) -> None:
        """Construct the coordinator family that owns extracted engine behaviors."""
        self._runtime_dataset = None
        self._runtime_warmup = 0
        # Strategy-runtime state, owned by StrategyRuntimeCoordinator and assigned
        # via configure_strategy below. Declared here so the type-checker tracks
        # the attributes now that the coordinator — not an engine method — writes
        # them.
        self.strategy: SupportsRuntimeHooks | StrategyRuntime
        self._component_strategy: ComponentStrategy | None
        self._runtime: StrategyRuntime | None
        # Strategy-runtime coordinator owns strategy normalization and the
        # per-candle runtime decision pipeline; built before _configure_strategy
        # (its first caller) reads/writes engine strategy state at call time (#486).
        self.strategy_coordinator = StrategyRuntimeCoordinator(engine_state=self)
        # Hot-swap / model-update lifecycle. Reads/writes engine strategy state
        # at call time; all mutation runs on the trading-loop thread (#486).
        self.hot_swap_coordinator = StrategyHotSwapCoordinator(engine_state=self)
        # WebSocket stream health + reconnect subsystem. Owns no state of its own
        # (thread handle, counters, queue all live on the engine); reads/writes
        # engine attrs at call time, preserving the lock-free single-writer
        # threading model (#486).
        self.ws_health_monitor = WebSocketHealthMonitor(engine_state=self)
        # Entry decision + execution pipeline. Holds no state of its own; reads/
        # writes engine state (balance, trackers, risk manager, session) through
        # the engine backref at call time, preserving the base-asset locking and
        # ordering of the real-money entry path (#486).
        self.entry_coordinator = LiveEntryCoordinator(engine_state=self)
        self.exit_coordinator = LiveExitCoordinator(engine_state=self)
        self.dynamic_risk_coordinator = LiveDynamicRiskCoordinator(engine_state=self)
        self.loop_timing_coordinator = LiveLoopTimingCoordinator(engine_state=self)
        self.market_data_coordinator = LiveMarketDataCoordinator(engine_state=self)
        self.order_fill_coordinator = LiveOrderFillCoordinator(engine_state=self)

    def _init_risk_manager(self, risk_parameters: RiskParameters | None) -> None:
        """Build the canonical RiskManager and bind it to component strategies."""
        # Duck-typed component risk adapter; attribute access is hasattr-guarded.
        component_risk: Any = None
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

    def _init_risk_policies(
        self,
        *,
        trailing_stop_policy: TrailingStopPolicy | None,
        enable_dynamic_risk: bool,
        dynamic_risk_config: DynamicRiskConfig | None,
    ) -> None:
        """Resolve trailing-stop + dynamic-risk config and seed correlation cache."""
        # Trailing stop policy
        self.trailing_stop_policy = trailing_stop_policy or self._build_trailing_policy()
        self._trailing_stop_opt_in = self.trailing_stop_policy is not None

        # Dynamic risk management
        self.enable_dynamic_risk = enable_dynamic_risk
        self.dynamic_risk_manager: DynamicRiskManager | None = None
        self._component_dynamic_risk_config: DynamicRiskConfig | None = None
        if enable_dynamic_risk:
            config = dynamic_risk_config or DynamicRiskConfig()
            # Will be initialized after db_manager is available
            self._dynamic_risk_config = config

        # Cache component-provided correlation context to avoid repeated lookups per bar.
        self._component_risk_context_cache_key: tuple[str, int] | None = None
        self._component_risk_context_cache: dict[str, Any] | None = None

    def _init_partial_operations(
        self,
        *,
        enable_partial_operations: bool,
        partial_manager: PartialExitPolicy | None,
    ) -> None:
        """Resolve the partial exit/scale-in policy (gated by #734 feature flag)."""
        # Partial operations policy.
        #
        # DISABLED by default behind the `live_partial_operations` feature flag
        # (#734): the live engine currently applies partial exits / scale-ins as
        # BOOKKEEPING ONLY — no exchange order is placed — and with mismatched
        # units (fraction-of-original-position applied to fraction-of-balance
        # state). On a real account this desyncs tracked size from actual
        # holdings (stranded inventory / un-repaid margin borrows / -2010 close
        # failures), books phantom realized PnL, and frees risk budget that is
        # still deployed. The flag exists only for development of the proper
        # fix; do NOT enable it for live capital until #734 is resolved.
        if (
            enable_partial_operations or partial_manager is not None
        ) and not self.settings.partial_operations_allowed:
            logger.warning(
                "Partial exits/scale-ins are DISABLED (#734): the live engine "
                "executes them as bookkeeping only (no exchange order, mismatched "
                "units), which desyncs tracked size from real holdings and books "
                "phantom PnL. Set feature flag live_partial_operations=true only "
                "for development of the fix."
            )
            enable_partial_operations = False
            partial_manager = None
        self.enable_partial_operations = bool(enable_partial_operations)
        if partial_manager is not None:
            self.partial_manager: PartialExitPolicy | None = partial_manager
        elif enable_partial_operations:
            # Check strategy overrides first, then fall back to risk parameters
            strategy_overrides = (
                self.strategy.get_risk_overrides()
                if hasattr(self.strategy, "get_risk_overrides")
                else None
            )
            if isinstance(strategy_overrides, dict) and "partial_operations" in strategy_overrides:
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

    def _init_correlation(self, data_provider: DataProvider) -> None:
        """Initialize the correlation engine and entry-time correlation handler."""
        # Correlation engine setup
        try:
            corr_cfg = CorrelationConfig(
                correlation_window_days=self.risk_manager.params.correlation_window_days,
                correlation_threshold=self.risk_manager.params.correlation_threshold,
                max_correlated_exposure=self.risk_manager.params.max_correlated_exposure,
                correlation_update_frequency_hours=self.risk_manager.params.correlation_update_frequency_hours,
            )
            self.correlation_engine: CorrelationEngine | None = CorrelationEngine(config=corr_cfg)
        except Exception:
            self.correlation_engine = None

        # Correlation handler — applies correlation-based size reduction at
        # entry. Mirrors backtest engine wiring (src/engines/backtest/engine.py:343-350)
        # so live entries reduce size for correlated exposure the same way
        # backtest does. Without this, live silently over-concentrates in
        # correlated pairs that backtest would have de-risked.
        self.correlation_handler: CorrelationHandler | None = None
        if self.correlation_engine is not None:
            try:
                self.correlation_handler = CorrelationHandler(
                    correlation_engine=self.correlation_engine,
                    risk_manager=self.risk_manager,
                    data_provider=data_provider,
                    strategy=self.strategy,
                )
            except Exception as e:
                logger.warning(
                    "Failed to initialize live correlation handler: %s — "
                    "live entries will run without correlation controls.",
                    e,
                )

    def _init_database(self, database_url: str | None) -> None:
        """Connect the (required) database manager and seed session state."""
        # Initialize database manager
        try:
            self.db_manager = DatabaseManager(database_url)
        except (ConnectionError, OSError, ValueError) as e:
            print(
                f"❌ Could not connect to the PostgreSQL database: {e}\nThe trading engine cannot start without a database connection. Exiting..."
            )
            raise RuntimeError("Database connection required. Service stopped.") from e
        self.trading_session_id: int | None = None
        # On a clean restart the engine recovers balance from the most recent
        # inactive session but creates a NEW session. This holds that old
        # session id so start() can carry its OPEN positions forward into the
        # new session (#668); None when recovery took the active/crash path or
        # found no recent session.
        self._recovered_inactive_session_id: int | None = None

    def _init_dynamic_risk_manager(self) -> None:
        """Build the dynamic-risk manager now that the database is available."""
        # Initialize dynamic risk manager after database is available
        if self.enable_dynamic_risk:
            try:
                # Merge strategy risk overrides with engine config
                final_config = self._merge_dynamic_risk_config(self._dynamic_risk_config)
                self.dynamic_risk_manager = DynamicRiskManager(
                    config=final_config,
                    db_manager=self.db_manager,
                    risk_parameters=self.risk_manager.params,
                    positions_provider=self.risk_manager.get_positions_snapshot,
                )
                logger.info("Dynamic risk management enabled")
            except Exception as e:
                logger.warning("Failed to initialize dynamic risk manager: %s", e)
                self.dynamic_risk_manager = None
        self._dynamic_risk_handler = DynamicRiskHandler(self.dynamic_risk_manager)

    def _init_exchange_interface(
        self,
        *,
        provider: str,
        testnet: bool,
        enable_live_trading: bool,
    ) -> None:
        """Initialize exchange interface, account synchronizer, and order tracker."""
        # Initialize exchange interface, account synchronizer, and order tracker.
        # Typed Any: the provider factory is untyped and concrete providers expose
        # duck-typed margin/WS extensions beyond the base interface.
        self.exchange_interface: Any = None
        self.account_synchronizer = None
        self.order_tracker: OrderTracker | None = None
        if enable_live_trading:
            try:
                app_config = get_config()
                self.exchange_interface, provider_name = _create_exchange_provider(
                    provider, app_config, testnet
                )
                if self.exchange_interface:
                    use_margin = getattr(self.exchange_interface, "is_margin_mode", False)
                    self.account_synchronizer = AccountSynchronizer(
                        self.exchange_interface,
                        self.db_manager,
                        self.trading_session_id,
                        use_margin=use_margin,
                    )
                    # Initialize order tracker for monitoring order fills
                    self.order_tracker = OrderTracker(
                        exchange=self.exchange_interface,
                        poll_interval=5,
                        on_fill=self._handle_order_fill,
                        on_partial_fill=self._handle_partial_fill,
                        on_cancel=self._handle_order_cancel,
                        on_tracking_lost=self._handle_order_tracking_lost,
                    )
                    logger.info(
                        f"{provider_name} exchange interface and account synchronizer initialized"
                    )
                else:
                    logger.warning(
                        f"{provider_name} API credentials not found - account sync disabled"
                    )
            except Exception as e:
                logger.warning("Failed to initialize exchange interface: %s", e)

            # Fail fast if live trading requested but exchange interface unavailable
            if self.exchange_interface is None:
                raise ValueError(
                    "Cannot enable live trading without exchange interface. "
                    "Ensure valid API credentials are configured for the selected provider."
                )

    def _resume_balance_from_snapshot(self) -> None:
        """Optionally resume balance from the last account snapshot (live only)."""
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
                logger.warning("Could not resume from last balance: %s", e)

    def _init_strategy_manager(
        self, strategy: ComponentStrategy | StrategyRuntime, *, enable_hot_swapping: bool
    ) -> None:
        """Wire the hot-swap StrategyManager and strategy DB logging."""
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
                logger.info("Hot swapping enabled for %s", managed_strategy.__class__.__name__)
            else:
                logger.info("Hot swapping disabled: provided strategy does not implement Strategy")

        # Set up strategy logging if database is available
        if self.db_manager:
            if hasattr(self.strategy, "set_database_manager"):
                self.strategy.set_database_manager(self.db_manager)

    def _seed_trading_state(self) -> None:
        """Seed trading-loop, reconciliation, and WebSocket runtime state."""
        # Trading state
        self.is_running = False
        self._close_only_mode = False  # No new entries when True; exits still run
        # Set during start() for live trading
        self._periodic_reconciler: PeriodicReconciler | None = None
        # Shared per-base-asset cooldown for the orphaned-borrow sweep, so the
        # startup sweep and the periodic reconciler don't both act in one window.
        self._orphan_sweep_cooldown: dict[str, float] = {}
        # Shared per-base-asset exchange-mutation lock: entry/exit and the
        # orphaned-borrow sweep serialise on it, so an active repay can never race a
        # just-placed borrow that isn't tracked yet (#703).
        from src.engines.live.reconciliation import BaseAssetLockRegistry

        self._base_asset_locks = BaseAssetLockRegistry()
        self.completed_trades: list[Trade] = []
        self.last_data_update: datetime | None = None
        # Track when we last logged account state
        self.last_account_snapshot: datetime | None = None
        self.timeframe: str | None = None  # Will be set when trading starts
        self._active_symbol: str | None = None

        # WebSocket stream state (populated during start() if provider supports it)
        self._kline_buffer: KlineBuffer | None = None
        self._user_data_processor: UserDataProcessor | None = None
        self._ws_kline_active = False
        # Duck-typed: the unwrapped provider exposing WS kline extensions.
        self._ws_kline_provider: Any = None
        self._ws_health_thread: threading.Thread | None = None
        # Consecutive unproductive user-stream reconnects; trips a circuit breaker
        # that stops the futile reconnect loop and runs REST-only (#616).
        self._user_reconnect_failures = 0
        # Consecutive unproductive kline reconnects; drives the *recovering* REST
        # fallback (#662) — momentary REST while reconnecting, auto-return to
        # WS-primary the instant a real kline event resumes (never REST-until-restart).
        self._kline_reconnect_failures = 0
        # Stop-loss fills are detected on the OrderTracker poll thread but their
        # bookkeeping exit is deferred to the trading loop so a slow/failing close
        # can't block order polling or force-remove a filled order (#631).
        self._pending_fill_exits: queue.SimpleQueue = queue.SimpleQueue()

    def _init_time_exit_policy(self, time_exit_policy: TimeExitPolicy | None) -> None:
        """Construct the time-exit policy from overrides when not injected."""
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

    def _install_signal_handlers(self) -> None:
        """Register SIGINT/SIGTERM handlers for graceful shutdown (main thread)."""
        # Setup graceful shutdown (main thread only)
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        else:
            logger.debug("Skipping signal handler registration outside main thread")

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
            logger.info("Merged strategy dynamic risk overrides from %s", self._strategy_name())
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

        # Position tracker (explicit annotation: the positions property reads
        # this attribute before mypy can infer it through the init cycle)
        self.live_position_tracker: LivePositionTracker = position_tracker or LivePositionTracker(
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
        # Wire db_manager for order journaling (session_id set during start())
        self.live_execution_engine.db_manager = self.db_manager

        # Entry handler. correlation_handler matches backtest engine wiring
        # so live and backtest both reduce position size for correlated
        # exposure at entry.
        self.live_entry_handler = entry_handler or LiveEntryHandler(
            execution_engine=self.live_execution_engine,
            execution_model=self.execution_model,
            risk_manager=self.risk_manager,
            component_strategy=(
                self.strategy if isinstance(self.strategy, ComponentStrategy) else None
            ),
            dynamic_risk_manager=self.dynamic_risk_manager,
            correlation_handler=self.correlation_handler,
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
            execution_model=self.execution_model,
            risk_manager=self.risk_manager,
            trailing_stop_policy=self.trailing_stop_policy,
            partial_manager=partial_ops_manager,
            time_exit_policy=self.time_exit_policy,
            use_high_low_for_stops=self.use_high_low_for_stops,
            max_filled_price_deviation=self.max_filled_price_deviation,
        )

        # Stop-loss lifecycle handler — owns every exchange-facing stop-loss
        # call (place/cancel/query/re-protect) so the engine orchestrates
        # without touching the exchange interface directly (#486).
        self.stop_loss_manager = LiveStopLossManager(
            engine_state=self,
            send_alert=self._send_alert,
        )

        # Account monitor — snapshots, status lines, performance summaries.
        self.account_monitor = LiveAccountMonitor(engine_state=self)

        # Startup recovery — session balance, persisted positions, exchange
        # reconciliation. Reads/writes engine state at call time (#486).
        self.session_recoverer = LiveSessionRecoverer(engine_state=self)

    def _apply_dynamic_risk_adjustment(
        self,
        original_size: float,
        current_time: datetime,
    ) -> float:
        """Apply dynamic risk adjustments to position size (delegated to LiveDynamicRiskCoordinator)."""
        return self.dynamic_risk_coordinator.apply_dynamic_risk_adjustment(
            original_size, current_time
        )

    def _log_dynamic_risk_adjustments(self) -> None:
        """Log dynamic risk adjustments for observability/audit (delegated to LiveDynamicRiskCoordinator)."""
        return self.dynamic_risk_coordinator.log_dynamic_risk_adjustments()

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
            logger.warning("Failed to get dynamic risk adjusted parameters: %s", e)
            return self.risk_manager.params

    def _extract_component_risk_parameters(
        self, component_risk_manager: object
    ) -> RiskParameters | None:
        """Clone risk parameters from a component adapter, if available."""
        return self.strategy_coordinator.extract_component_risk_parameters(component_risk_manager)

    def _merge_risk_parameters(
        self,
        engine_params: RiskParameters | None,
        component_params: RiskParameters | None,
    ) -> RiskParameters | None:
        """Merge engine-provided and component-provided risk parameters."""
        return self.strategy_coordinator.merge_risk_parameters(engine_params, component_params)

    @staticmethod
    def _clone_risk_parameters(params: RiskParameters | None) -> RiskParameters | None:
        """Create a deep-cloned copy of risk parameters for safe reuse."""
        return StrategyRuntimeCoordinator.clone_risk_parameters(params)

    def _configure_strategy(self, strategy: ComponentStrategy | StrategyRuntime) -> None:
        """Normalize strategy inputs and configure runtime bookkeeping."""
        self.strategy_coordinator.configure_strategy(strategy)

    def _register_component_context_provider(self) -> None:
        """Attach the engine-provided risk context hook to component strategies."""
        self.strategy_coordinator.register_component_context_provider()

    def _component_risk_context(self, df: pd.DataFrame, index: int, signal) -> dict[str, Any]:
        """Build supplemental risk context (e.g., correlation data) for components."""
        return self.strategy_coordinator.component_risk_context(df, index, signal)

    def _get_correlation_context(
        self,
        symbol: str,
        df: pd.DataFrame,
        overrides: dict | None,
        *,
        index: int | None = None,
    ) -> dict | None:
        """Return cached correlation context for the given bar or build it on demand."""
        return self.strategy_coordinator.get_correlation_context(symbol, df, overrides, index=index)

    def _apply_policies_from_decision(self, decision) -> None:
        """Hydrate engine-level policies from component strategy output."""
        self.strategy_coordinator.apply_policies_from_decision(decision)

    # Runtime integration helpers -------------------------------------------------

    def _is_runtime_strategy(self) -> bool:
        return self.strategy_coordinator.is_runtime_strategy()

    def _strategy_name(self) -> str:
        """Returns the configured strategy name for logging and reporting."""
        return self.strategy_coordinator.strategy_name()

    def _prepare_strategy_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe for strategy processing."""
        return self.strategy_coordinator.prepare_strategy_dataframe(df)

    def _build_component_positions(
        self,
        current_price: float,
    ) -> list[ComponentPosition]:
        """Translate live positions into the strategy-side ComponentPosition list."""
        return self.strategy_coordinator.build_component_positions(current_price)

    def _build_runtime_context(
        self,
        balance: float,
        current_price: float,
        current_time: datetime,
    ) -> RuntimeContext:
        """Build the StrategyRuntime context with current balance and live positions."""
        return self.strategy_coordinator.build_runtime_context(balance, current_price, current_time)

    def _compute_component_quantity(
        self, position: Position, balance_basis: float | None = None
    ) -> float:
        """Translate a position's fractional size into asset quantity for component strategies."""
        return self.strategy_coordinator.compute_component_quantity(position, balance_basis)

    def _runtime_process_decision(
        self,
        df: pd.DataFrame,
        index: int,
        balance: float,
        current_price: float,
        current_time: datetime,
    ) -> TradingDecision | None:
        return self.strategy_coordinator.runtime_process_decision(
            df, index, balance, current_price, current_time
        )

    def _finalize_runtime(self) -> None:
        self.strategy_coordinator.finalize_runtime()

    def start(
        self,
        symbol: str,
        timeframe: str = "1h",
        max_steps: int | None = None,
        exit_on_crash: bool = False,
    ) -> None:
        """Start the live trading engine.

        ``exit_on_crash`` makes an abnormal loop death exit the process non-zero
        so an orchestrator restarts it (#630). It defaults to False so start()
        stays a well-behaved library call for callers that read results after it
        returns (e.g. the migration baseline tool); the production runner opts in.
        """
        if self.is_running:
            logger.warning("Trading engine is already running")
            return

        self._begin_session_runtime(symbol, timeframe)
        self._bootstrap_trading_session(symbol, timeframe)
        self._carry_forward_open_positions()
        self._self_heal_terminal_positions()
        self._synchronize_account_on_start()

        # Set session ID on strategy for logging
        if hasattr(self.strategy, "session_id"):
            self.strategy.session_id = self.trading_session_id

        self._start_runtime_services(symbol, timeframe)
        self._run_main_loop_until_stopped(symbol, timeframe, max_steps)

        # After a clean stop this is a no-op; after an abnormal loop death it
        # exits the process non-zero (when opted in) so the orchestrator
        # restarts it (#630).
        self._exit_if_loop_crashed(exit_on_crash)

    def _begin_session_runtime(self, symbol: str, timeframe: str) -> None:
        """Set runtime flags + logging context and emit the startup banner."""
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
        logger.info("🚀 Starting live trading for %s on %s timeframe", symbol, timeframe)
        logger.info("Initial balance: $%.2f", self.current_balance)
        logger.info("Max position size: %.1f%% of balance", self.max_position_size * 100)
        logger.info("Check interval: %ss", self.check_interval)

        if not self.enable_live_trading:
            logger.warning("⚠️  PAPER TRADING MODE - No real orders will be executed")

    def _bootstrap_trading_session(self, symbol: str, timeframe: str) -> None:
        """Recover prior balance/positions and create + wire the trading session."""
        # Try to recover from existing session first
        if self.resume_from_last_balance:
            recovered_balance = self._recover_existing_session()
            if recovered_balance is not None:
                # _recover_existing_session() already coerced this to a finite float
                # and rejected corrupt (non-finite) state. The float() here is a
                # cheap defensive invariant so current_balance never becomes a
                # Decimal — which would break downstream float arithmetic such as
                # _print_final_stats (CODE.md "Arithmetic & Financial Calculations").
                self.current_balance = float(recovered_balance)
                logger.info(
                    "💾 Recovered balance from previous session: $%.2f",
                    recovered_balance,
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

            # Wire session_id and strategy_name to execution engine for order journaling
            self.live_execution_engine.session_id = self.trading_session_id
            self.live_execution_engine.strategy_name = self._strategy_name()

            # Wire the event logger so snapshot/daily-P&L logging is session
            # scoped; on a clean restart also point day-start recovery at the
            # prior session, where today's earlier snapshots live (#766).
            self.event_logger.set_session_id(self.trading_session_id)
            self.event_logger.set_recovery_session_id(self._recovered_inactive_session_id)

    def _carry_forward_open_positions(self) -> None:
        """Carry OPEN positions forward from a recovered inactive session (#668)."""
        # Carry OPEN positions forward on a clean restart (#668). The inactive
        # session recovered above (balance only) still owns any OPEN position via
        # Position.session_id, so _recover_active_positions() at line ~1281 saw a
        # None session id and loaded nothing — the position would be orphaned.
        # Re-point those positions onto the new session, then reload them into the
        # live tracker. Ordering: reassign → recover-into-tracker (which self-heals
        # first) → the heal + exchange reconciliation below re-verify the position
        # and its server-side stop-loss against the exchange.
        if self._recovered_inactive_session_id is not None and self.trading_session_id is not None:
            try:
                moved_ids = self.db_manager.reassign_open_positions_to_session(
                    old_session_id=self._recovered_inactive_session_id,
                    new_session_id=self.trading_session_id,
                    symbol=self._active_symbol,
                    strategy_name=self._strategy_name(),
                )
                if moved_ids:
                    logger.info(
                        "🔁 Carried %d OPEN position(s) forward from inactive session "
                        "#%s into new session #%s; reloading into tracker",
                        len(moved_ids),
                        self._recovered_inactive_session_id,
                        self.trading_session_id,
                    )
                    # Reload now that the rows belong to the new session. Safe and
                    # idempotent if it already ran (empty session ⇒ no-op).
                    self._recover_active_positions()
            except Exception as reassign_err:
                # A failure here must not abort startup, but it is capital-critical
                # (an OPEN position stays orphaned), so log loudly for alerting.
                logger.critical(
                    "Failed to carry OPEN positions forward from inactive session "
                    "#%s into session #%s (positions may be orphaned — MANUAL "
                    "RECONCILIATION REQUIRED): %s",
                    self._recovered_inactive_session_id,
                    self.trading_session_id,
                    reassign_err,
                    exc_info=True,
                )
            finally:
                # Clear so a later start()/stop()/start() re-entry cannot re-trigger a
                # stale-session reassign (#668, P3).
                self._recovered_inactive_session_id = None

    def _self_heal_terminal_positions(self) -> None:
        """Close any OPEN position in this session that already has a terminal Trade (#657)."""
        # Startup self-heal (#657): close any OPEN position in this session that
        # already has a terminal Trade. Deliberately NOT gated behind
        # enable_live_trading/exchange — the whole bug was that closing was
        # paper-blind, so this must run in paper mode too. Pure DB reconciliation
        # (no exchange calls), idempotent, and complements the atomic status flip
        # now performed inside log_trade. Placed before account sync so the books
        # are consistent before any exchange reconciliation reads them.
        if self.trading_session_id is not None:
            try:
                healed = self.db_manager.heal_positions_with_terminal_trades(
                    self.trading_session_id
                )
                if healed:
                    logger.info(
                        "🩹 Startup self-heal closed %d stale-OPEN position(s) with "
                        "terminal trades (session #%s)",
                        healed,
                        self.trading_session_id,
                    )
            except Exception as heal_err:
                logger.warning("Startup position self-heal failed (continuing): %s", heal_err)

    def _synchronize_account_on_start(self) -> None:
        """Sync balance/positions with the exchange and persist any balance correction."""
        # Perform account synchronization if available
        self._pending_balance_correction = False
        self._pending_corrected_balance = None
        if self.account_synchronizer and self.enable_live_trading:
            try:
                logger.info("🔄 Performing initial account synchronization...")
                sync_result = self.account_synchronizer.sync_account_data(
                    force=True, symbol=self._active_symbol
                )
                if sync_result.success:
                    logger.info("✅ Account synchronization completed")
                    # Update session ID for synchronizer
                    if self.trading_session_id:
                        self.account_synchronizer.session_id = self.trading_session_id
                    # Check if balance was corrected
                    balance_sync = sync_result.data.get("balance_sync", {})
                    if balance_sync.get("corrected", False):
                        previous_balance = balance_sync.get("old_balance", self.current_balance)
                        corrected_balance = balance_sync.get("new_balance", self.current_balance)
                        # Atomic balance update with lock to prevent race conditions
                        with self._balance_lock:
                            self.current_balance = corrected_balance
                            self._pending_balance_correction = True
                            self._pending_corrected_balance = corrected_balance
                        logger.info(
                            "💰 Balance corrected from exchange: $%.2f",
                            corrected_balance,
                        )
                        # A silent balance overwrite masked a real capital-erosion
                        # incident before — make every correction auditable.
                        self._record_event(
                            EventType.WARNING,
                            (
                                "Balance overwritten from exchange: "
                                f"{previous_balance} -> {corrected_balance}"
                            ),
                            severity="warning",
                            component="balance",
                            error_code="BALANCE_OVERWRITE",
                        )
                else:
                    logger.warning("⚠️ Account synchronization failed: %s", sync_result.message)

                # Reconcile positions with exchange (detect offline stop-loss triggers)
                self._reconcile_positions_with_exchange()

                # Reconciliation paths (e.g. PositionReconciler._reconcile_filled_entry)
                # may create LivePositions via track_recovered_position without
                # registering them with risk_manager. The DB-recovery path in
                # _recover_active_positions does register; the reconciler path
                # currently does not. Sweep the tracker after reconciliation so
                # every tracked position is known to the risk manager — this
                # restores the parity invariant (also enforced on every
                # backtest entry) that risk_manager has visibility into all
                # active positions for per-symbol caps and correlation gating.
                self._ensure_positions_registered_with_risk_manager()

            except Exception as e:
                logger.error("❌ Account synchronization error: %s", e, exc_info=True)

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
                logger.info("💰 Balance corrected in database: $%.2f", corrected_balance)
            elif getattr(self, "_pending_balance_correction", False):
                # Balance correction was pending but no session ID available
                logger.warning(
                    "⚠️ Balance correction pending but no trading session ID available - skipping database update"
                )
                self._pending_balance_correction = False
                self._pending_corrected_balance = None

    def _start_runtime_services(self, symbol: str, timeframe: str) -> None:
        """Start the order tracker, periodic reconciler, and WebSocket streams."""
        # Start order tracker for monitoring order fills (live trading only)
        if self.order_tracker and self.enable_live_trading:
            self.order_tracker.start()
            logger.info("📡 Order tracker started")

        # Start periodic reconciler (live trading only, not paper mode)
        if self.enable_live_trading and self.exchange_interface and self.trading_session_id:
            try:
                from src.engines.live.reconciliation import PeriodicReconciler

                use_margin = getattr(self.exchange_interface, "is_margin_mode", False)
                self._periodic_reconciler = PeriodicReconciler(
                    exchange_interface=self.exchange_interface,
                    position_tracker=self.live_position_tracker,
                    db_manager=self.db_manager,
                    session_id=self.trading_session_id,
                    on_critical=self._enter_close_only_mode,
                    use_margin=use_margin,
                    symbols=[self._active_symbol] if self._active_symbol else [],
                    sweep_cooldown=self._orphan_sweep_cooldown,
                    lock_registry=self._base_asset_locks,
                    data_provider=self.data_provider,
                )
                self._periodic_reconciler.start()
                logger.info("🔄 Periodic reconciler started")
            except Exception as e:
                logger.warning("Failed to start periodic reconciler: %s", e)
                # A silently-disabled reconciler is exactly the kind of failure
                # that ran invisible for months — surface it in system_events.
                self._record_event(
                    EventType.ERROR,
                    f"Periodic reconciler failed to start: {e}",
                    severity="error",
                    component="reconciler",
                    error_code="RECONCILER_START_FAILED",
                    exc=e,
                )

        # Try to start WebSocket streams for reduced API weight
        self._start_websocket_streams(symbol, timeframe)

    def _run_main_loop_until_stopped(
        self, symbol: str, timeframe: str, max_steps: int | None
    ) -> None:
        """Launch the trading-loop thread and block until it stops, then tear down."""
        # Start main trading loop in separate thread
        self.main_thread = threading.Thread(
            target=self._run_trading_loop, args=(symbol, timeframe, max_steps)
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

    def _enter_close_only_mode(self) -> None:
        """Enter close-only mode: no new entries, exits/stops/trailing still active."""
        if not self._close_only_mode:
            self._close_only_mode = True
            logger.critical("🚨 CLOSE-ONLY MODE ACTIVATED — no new entries until manual review")
            # Emit once on transition (guarded above) so the kill-switch is
            # visible in system_events and pages an operator.
            self._record_event(
                EventType.ALERT,
                "Close-only mode activated — no new entries until manual review",
                severity="critical",
                component="risk",
                error_code="CLOSE_ONLY",
                alert=True,
            )

    def resume_trading(self) -> None:
        """Resume normal trading after close-only mode review."""
        if self._close_only_mode:
            self._close_only_mode = False
            logger.info("✅ Close-only mode deactivated — normal trading resumed")

    def _start_websocket_streams(self, symbol: str, timeframe: str) -> None:
        """Initialize WebSocket streams for reduced API weight."""
        return self.ws_health_monitor.start_websocket_streams(symbol, timeframe)

    def _start_ws_health_monitor(self) -> None:
        """Start daemon thread to monitor WebSocket stream health."""
        return self.ws_health_monitor.start_ws_health_monitor()

    def _ensure_ws_health_monitor_alive(self) -> None:
        """Watchdog-on-watchdog: restart the WS health monitor if its thread died."""
        return self.ws_health_monitor.ensure_ws_health_monitor_alive()

    def _ws_health_loop(self) -> None:
        """Monitor WebSocket streams and trigger reconnection on failure."""
        return self.ws_health_monitor.ws_health_loop()

    def _drain_pending_fill_exits(self) -> None:
        """Execute stop-loss-fill exits deferred from the OrderTracker poll thread."""
        return self.ws_health_monitor.drain_pending_fill_exits()

    def _check_kline_health(self) -> None:
        """Kline stream health with a *recovering* REST fallback (#662)."""
        return self.ws_health_monitor.check_kline_health()

    def _should_probe_kline_reconnect(self, failures: int) -> bool:
        """Whether to attempt a kline WS reconnect on this health cycle (#662)."""
        return self.ws_health_monitor.should_probe_kline_reconnect(failures)

    def _check_user_stream_health(self) -> None:
        """User-stream health with a *recovering* REST fallback (#717)."""
        return self.ws_health_monitor.check_user_stream_health()

    def _should_probe_user_reconnect(self, failures: int) -> bool:
        """Whether to probe a user-stream WS reconnect on this degraded cycle (#717/#723)."""
        return self.ws_health_monitor.should_probe_user_reconnect(failures)

    @staticmethod
    def _user_probe_boundary_reached(failures: int) -> bool:
        """True iff ``failures`` is an exponential-backoff probe boundary (#723)."""
        return WebSocketHealthMonitor.user_probe_boundary_reached(failures)

    @staticmethod
    def _user_next_probe_eta_minutes(failures: int) -> float:
        """Approx minutes until the next user degraded-probe after ``failures`` (#723)."""
        return WebSocketHealthMonitor.user_next_probe_eta_minutes(failures)

    def _should_hard_reconnect_user(self) -> bool:
        """Whether the degraded probe should use a HARD reconnect this cycle (#723)."""
        return self.ws_health_monitor.should_hard_reconnect_user()

    def _restore_user_ws_primary(self) -> None:
        """Return the user stream to WS-primary once a real event confirms delivery."""
        return self.ws_health_monitor.restore_user_ws_primary()

    def _handle_kline_disconnect(self) -> None:
        """Resync kline history from REST and attempt one WS reconnect."""
        return self.ws_health_monitor.handle_kline_disconnect()

    def _handle_user_stream_disconnect(self, *, hard: bool = False) -> None:
        """Handle user data stream failure. Resync orders and attempt reconnect."""
        return self.ws_health_monitor.handle_user_stream_disconnect(hard=hard)

    def stop(self) -> None:
        """Stop the trading engine gracefully."""
        if not self.is_running:
            return

        logger.info("🛑 Stopping trading engine...")
        self.is_running = False
        self.stop_event.set()

        # Stop inbound WS streams FIRST (no new events arrive)
        if self._ws_kline_provider and hasattr(self._ws_kline_provider, "stop_streams"):
            self._ws_kline_provider.stop_streams()
        if self.exchange_interface and hasattr(self.exchange_interface, "stop_streams"):
            self.exchange_interface.stop_streams()
        # Stop/drain UserDataProcessor (process remaining queued events)
        if self._user_data_processor:
            self._user_data_processor.stop()

        # Stop periodic reconciler
        if self._periodic_reconciler:
            self._periodic_reconciler.stop()
            logger.info("🔄 Periodic reconciler stopped")

        # Stop order tracker
        if self.order_tracker:
            self.order_tracker.stop()
            logger.info("📡 Order tracker stopped")

        # Close or preserve open positions depending on trading mode
        positions_snapshot = self.live_position_tracker.positions
        if positions_snapshot:
            if self.enable_live_trading:
                # LIVE: close all positions on exchange before shutdown.
                logger.info("Closing %s open live positions...", len(positions_snapshot))
                for position in list(positions_snapshot.values()):
                    try:
                        current_price = self.data_provider.get_current_price(position.symbol)
                        if current_price is None or current_price <= 0:
                            logger.critical(
                                "Cannot close live position %s during shutdown — invalid price %s. "
                                "Manual intervention required.",
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
                        # Tracked positions always carry a non-None order_id.
                        self.live_position_tracker.remove_position(cast(str, position.order_id))
            else:
                # PAPER: preserve open positions in DB so they survive restart.
                # _recover_active_positions() will reload them on next start().
                # The first candle evaluation after recovery will check SL/TP.
                logger.info(
                    "Paper mode: preserving %s open positions for restart recovery",
                    len(positions_snapshot),
                )

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
        logger.info("Received signal %s", signum)
        self.stop()
        sys.exit(0)

    def _run_trading_loop(self, symbol: str, timeframe: str, max_steps: int | None = None) -> None:
        """Thread target so an unhandled exception can't kill the loop *silently*.

        A bare daemon thread that raises just vanishes, leaving the process alive
        but brain-dead (HTTP server up, loop gone) — the zombie that hid the
        2026-05-19 outage for 12 days. Catch any unhandled exception, record it,
        and let start() turn it into a non-zero exit for an orchestrator restart (#630).
        """
        try:
            self._trading_loop(symbol, timeframe, max_steps)
        except Exception as e:
            self._loop_crashed = True
            logger.critical("Trading loop terminated unexpectedly: %s", e, exc_info=True)

    def _exit_if_loop_crashed(self, exit_on_crash: bool) -> None:
        """Exit the process non-zero if the trading loop died abnormally (#630).

        A clean stop (signal, explicit stop, or max_steps) leaves this a no-op.
        On an abnormal death (unhandled crash or consecutive-error exhaustion):
        when *exit_on_crash* (the production runner), exit 1 so the orchestrator
        restarts the process instead of leaving the bot silently dead; otherwise
        return so library callers that read results after start() keep working.
        """
        if not (self._loop_crashed and exit_on_crash):
            return
        # The loop thread may still be unwinding its own shutdown (e.g. closing
        # positions); wait so a daemon thread isn't killed mid-cleanup. We exit
        # regardless of the join result (the daemon dies with the process), but
        # surface a wedged cleanup so it's visible rather than silent.
        if self.main_thread is not None and self.main_thread != threading.current_thread():
            self.main_thread.join(timeout=30)
            if self.main_thread.is_alive():
                logger.error(
                    "Loop thread still alive after 30s join; exiting anyway — "
                    "shutdown cleanup may be incomplete."
                )
        logger.critical(
            "Trading loop ended abnormally; exiting with code 1 to trigger an "
            "orchestrator restart instead of running dead."
        )
        sys.exit(1)

    def _trading_loop(self, symbol: str, timeframe: str, max_steps: int | None = None) -> None:
        """Main trading loop"""
        from src.infrastructure import liveness  # shared loop-liveness for /health (#627)

        logger.info("Trading loop started")
        steps = 0
        cfg = get_config()
        self._active_symbol = symbol
        try:
            # get() with a non-None default always returns str.
            heartbeat_every = int(cast(str, cfg.get("ENGINE_HEARTBEAT_STEPS", "60")))
        except Exception:
            heartbeat_every = 60
        while self.is_running and not self.stop_event.is_set():
            if max_steps is not None and steps >= max_steps:
                logger.info("Reached max_steps=%s, stopping engine for test.", max_steps)
                self.stop()
                break
            steps += 1
            liveness.beat()  # record loop liveness for the /health endpoint (#627)
            try:
                # Supervise the WS health monitor and drain deferred stop-loss
                # exits first, so both run every iteration even when a data outage
                # would `continue` below before reaching them (#631).
                self._ensure_ws_health_monitor_alive()
                self._drain_pending_fill_exits()
                # For mock and real providers, update live data if supported.
                # Skip when WS kline cache is active (no REST needed).
                if not self._ws_kline_active and hasattr(self.data_provider, "update_live_data"):
                    try:
                        self.data_provider.update_live_data(symbol, timeframe)
                    except Exception as e:
                        logger.debug("update_live_data failed: %s", e)
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
                        if self._apply_pending_strategy_update():
                            self._send_alert("Strategy/Model updated in live trading")
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
                    logger.debug("Trailing stop update failed: %s", e)
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
                # Evaluate partial exits and scale-ins for open positions.
                # Pass current candle time so positions entered on this bar
                # are skipped, matching backtest same-bar protection.
                if not safety_mode:
                    self.live_exit_handler.check_partial_operations(
                        df,
                        current_index,
                        float(current_price),
                        self.current_balance,
                        candle_time=current_time,
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
                    self._process_legacy_short_entry(
                        df, current_index, symbol, current_price, current_time
                    )
                # Update performance metrics
                self._update_performance_metrics()
                self._log_periodic_account_state()
                # Log status periodically
                if (
                    self.performance_tracker.get_metrics().total_trades % 10 == 0
                    or self.live_position_tracker.position_count > 0
                ):
                    self._log_status(symbol, current_price)
                # Reset error counter on successful iteration
                self.consecutive_errors = 0
                self.db_unreachable_since = None

                # Calculate and use adaptive interval for next iteration
                current_price = df.iloc[-1]["close"] if df is not None and not df.empty else None
                self.check_interval = self._calculate_adaptive_interval(current_price)

            except Exception as e:
                # Transient database-connectivity errors (a brief Postgres
                # outage or DNS hiccup) must NOT shut the engine down: ride them
                # out with a bounded backoff and keep the process alive so it
                # reconnects when the database returns. Counting them toward
                # max_consecutive_errors killed the live bot during a multi-hour
                # Railway internal-DNS outage on 2026-05-19.
                if self._is_transient_db_error(e):
                    if self.db_unreachable_since is None:
                        self.db_unreachable_since = time.monotonic()
                    unreachable_for = time.monotonic() - self.db_unreachable_since
                    # Prolonged outage: stop opening new positions (exits and
                    # server-side stop-losses still run) to avoid order churn
                    # while DB writes keep failing. Stays close-only until a
                    # manual resume_trading() after review.
                    if (
                        unreachable_for >= DEFAULT_DB_OUTAGE_CLOSE_ONLY_SECONDS
                        and not self._close_only_mode
                    ):
                        logger.critical(
                            "Database unreachable for %.0fs (>= %ds) — entering "
                            "close-only mode (new entries suspended).",
                            unreachable_for,
                            DEFAULT_DB_OUTAGE_CLOSE_ONLY_SECONDS,
                        )
                        self._enter_close_only_mode()
                    logger.warning(
                        "Database temporarily unreachable in trading loop (%s); "
                        "backing off %.0fs and retrying — not counted toward "
                        "shutdown (unreachable for %.0fs).",
                        type(e).__name__,
                        self.error_cooldown,
                        unreachable_for,
                    )
                    self._sleep_with_interrupt(self.error_cooldown)
                    continue

                self.consecutive_errors += 1
                logger.error(
                    f"Error in trading loop (#{self.consecutive_errors}): {e}", exc_info=True
                )
                if self.consecutive_errors >= self.max_consecutive_errors:
                    logger.critical(
                        f"Too many consecutive errors ({self.consecutive_errors}). Stopping engine.",
                        exc_info=True,
                    )
                    # Abnormal stop: signal start() to exit non-zero for a restart (#630).
                    self._loop_crashed = True
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

    def _process_legacy_short_entry(
        self,
        df: pd.DataFrame,
        current_index: int,
        symbol: str,
        current_price: float,
        current_time: datetime,
    ) -> None:
        """Evaluate + execute a legacy duck-typed short entry (delegated to LiveEntryCoordinator)."""
        self.entry_coordinator.process_legacy_short_entry(
            df, current_index, symbol, current_price, current_time
        )

    def _log_periodic_account_state(self) -> None:
        """Log the periodic account snapshot and run periodic exchange account sync."""
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
                    sync_result = self.account_synchronizer.sync_account_data(
                        symbol=self._active_symbol
                    )
                    if sync_result.success:
                        logger.debug("Periodic account sync completed")
                        # Apply any balance correction to the in-memory balance
                        # that drives sizing (the DB is already updated inside the
                        # sync). Without this a mid-session margin-equity
                        # correction would not reach live sizing until restart.
                        balance_sync = sync_result.data.get("balance_sync", {})
                        if balance_sync.get("corrected", False):
                            corrected = balance_sync.get("new_balance")
                            if corrected is not None:
                                with self._balance_lock:
                                    self.current_balance = corrected
                                logger.info(
                                    "💰 Balance corrected mid-session from " "exchange: $%.2f",
                                    corrected,
                                )
                    else:
                        logger.warning(
                            "Periodic account sync failed: %s",
                            sync_result.message,
                        )
                except Exception as e:
                    logger.error("Periodic account sync error: %s", e)

    @staticmethod
    def _is_transient_db_error(exc: BaseException) -> bool:
        """Return True if *exc* is a transient database-connectivity error.

        Brief Postgres unavailability, dropped connections, or DNS hiccups
        should be ridden out with a backoff rather than counted toward
        ``max_consecutive_errors``. Otherwise a short infrastructure blip
        shuts the live engine down — which is exactly what happened on
        2026-05-19 when Railway's internal DNS failed to resolve
        ``postgres.railway.internal`` for several hours. ``pool_pre_ping``
        reconnects automatically once the database returns.

        Permanent faults (bad credentials, missing role/database, permission
        denied) are deliberately NOT treated as transient: retrying them
        forever would keep the bot alive but brain-dead, so they fall through
        to the normal consecutive-error path and fail fast.
        """
        # Permanent misconfiguration — never retry. These are psycopg2
        # OperationalError subclasses too, so they must be matched BEFORE the
        # broad isinstance() net below.
        permanent_markers = (
            "password authentication failed",
            "authentication failed",
            "no password supplied",
            "permission denied",
            "does not exist",  # role/database missing — will not self-heal
        )
        transient_markers = (
            "could not translate host name",
            "temporary failure in name resolution",
            "name or service not known",
            "could not connect",
            "connection refused",
            "server closed the connection",
            "connection already closed",
            "ssl connection has been closed",
            "terminating connection",
            "the database system is starting up",
            "connection timed out",
            "could not receive data from server",
            "no route to host",
        )
        seen: set[int] = set()
        cur: BaseException | None = exc
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            text = str(cur).lower()
            if any(marker in text for marker in permanent_markers):
                return False
            if any(marker in text for marker in transient_markers):
                return True
            if isinstance(cur, OperationalError | InterfaceError):
                return True
            if isinstance(cur, DBAPIError) and getattr(cur, "connection_invalidated", False):
                return True
            cur = getattr(cur, "orig", None) or cur.__cause__ or cur.__context__
        return False

    def _is_context_ready(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Check the frame has enough context for decisions (delegated to LiveMarketDataCoordinator)."""
        return self.market_data_coordinator.is_context_ready(df)

    def _get_latest_data(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        """Fetch latest market data from WS cache or REST (delegated to LiveMarketDataCoordinator)."""
        return self.market_data_coordinator.get_latest_data(symbol, timeframe)

    def _add_sentiment_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add sentiment data to price data (delegated to LiveMarketDataCoordinator)."""
        return self.market_data_coordinator.add_sentiment_data(df, symbol)

    def _build_correlation_context(
        self, symbol: str, df: pd.DataFrame, overrides: dict | None
    ) -> dict | None:
        """Build correlation-sizing context (delegated to LiveMarketDataCoordinator)."""
        return self.market_data_coordinator.build_correlation_context(symbol, df, overrides)

    def _check_exit_conditions(
        self,
        df: pd.DataFrame,
        current_index: int,
        current_price: float,
        runtime_decision=None,
        candle=None,
        safety_mode: bool = False,
    ):
        """Check if any positions should be closed (delegated to LiveExitCoordinator)."""
        return self.exit_coordinator.check_exit_conditions(
            df,
            current_index,
            current_price,
            runtime_decision=runtime_decision,
            candle=candle,
            safety_mode=safety_mode,
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
        """Check if new positions should be opened (delegated to LiveEntryCoordinator)."""
        return self.entry_coordinator.check_entry_conditions(
            df,
            current_index,
            symbol,
            current_price,
            current_time,
            runtime_decision,
        )

    def _resolve_take_profit_pct(self) -> float:
        """Resolve the default take-profit percentage from risk parameters or strategy."""
        try:
            params = self.risk_manager.params if self.risk_manager else None
            if params and params.default_take_profit_pct is not None:
                try:
                    return float(params.default_take_profit_pct)
                except (TypeError, ValueError):
                    return DEFAULT_TAKE_PROFIT_PCT
        except Exception:
            return DEFAULT_TAKE_PROFIT_PCT

        value = getattr(self.strategy, "take_profit_pct", DEFAULT_TAKE_PROFIT_PCT)
        try:
            return float(value)
        except (TypeError, ValueError):
            return DEFAULT_TAKE_PROFIT_PCT

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
        """Serialise the entry on the symbol's base-asset lock, then execute it.

        Delegated to LiveEntryCoordinator; the lock is held across order submit
        -> position tracking (and any emergency-close fallback, which re-acquires
        it re-entrantly) so the orphaned-borrow sweep can't repay a borrow this
        entry just created (#703).
        """
        return self.entry_coordinator.execute_entry(
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_strength=signal_strength,
            signal_confidence=signal_confidence,
        )

    def _execute_entry_locked(
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
        """Execute a new trading position (delegated to LiveEntryCoordinator)."""
        return self.entry_coordinator.execute_entry_locked(
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_strength=signal_strength,
            signal_confidence=signal_confidence,
        )

    def _handle_order_fill(
        self, order_id: str, symbol: str, filled_qty: float, avg_price: float
    ) -> None:
        """Handle a fully filled order from OrderTracker (delegated to LiveOrderFillCoordinator)."""
        return self.order_fill_coordinator.handle_order_fill(
            order_id, symbol, filled_qty, avg_price
        )

    def _handle_partial_fill(
        self, order_id: str, symbol: str, new_filled_qty: float, avg_price: float
    ) -> None:
        """Handle a partial fill from OrderTracker (delegated to LiveOrderFillCoordinator)."""
        return self.order_fill_coordinator.handle_partial_fill(
            order_id, symbol, new_filled_qty, avg_price
        )

    def _handle_stop_loss_cancelled(self, order_id: str, symbol: str) -> bool:
        """Escalate a terminated stop-loss order (delegated to LiveOrderFillCoordinator)."""
        return self.order_fill_coordinator.handle_stop_loss_cancelled(order_id, symbol)

    def _handle_order_cancel(self, order_id: str, symbol: str, filled_qty: float = 0.0) -> None:
        """Handle an order cancel/reject from OrderTracker (delegated to LiveOrderFillCoordinator)."""
        return self.order_fill_coordinator.handle_order_cancel(order_id, symbol, filled_qty)

    def _handle_order_tracking_lost(self, order_id: str, symbol: str, failures: int) -> None:
        """Handle OrderTracker abandoning an UNKNOWN-state order (delegated to LiveOrderFillCoordinator)."""
        return self.order_fill_coordinator.handle_order_tracking_lost(order_id, symbol, failures)

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
        """Serialise the close on the position\'s base-asset lock, then execute it (#703).

        Delegated to LiveExitCoordinator. Re-entrant: an entry that already holds
        the lock (its SL-failed emergency close routes here) re-acquires it on the
        same thread without deadlock.
        """
        return self.exit_coordinator.execute_exit(
            position,
            reason,
            limit_price,
            current_price,
            candle_high,
            candle_low,
            candle,
            skip_live_close=skip_live_close,
        )

    def _execute_exit_locked(
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
        """Close a position using shared execution modules (delegated to LiveExitCoordinator)."""
        return self.exit_coordinator.execute_exit_locked(
            position,
            reason,
            limit_price,
            current_price,
            candle_high,
            candle_low,
            candle,
            skip_live_close=skip_live_close,
        )

    def _cancel_stop_loss_order(self, position: Position) -> bool:
        """Cancel a position's resting stop-loss order and stop tracking it.

        Returns True only when the exchange confirms the cancel. The close path uses
        this before a market exit so the stop no longer reserves the base asset
        (otherwise the close is rejected -2010 on margin, #710). A False result means
        the order may still rest, or may have just filled, so the caller must NOT
        submit a close (it would -2010, or over-sell an already-closed position).
        """
        return self.stop_loss_manager.cancel(position)

    def _stop_loss_filled_quantity(self, position: Position) -> float | None:
        """Return the filled (executed) base quantity of a position's stop-loss order.

        ``0.0`` for an unfilled stop, the filled base quantity for a partial/full fill,
        or ``None`` if the order cannot be read (missing / API error). The close path
        treats ``None`` and any non-zero fill as "unsafe to inline-close" and defers to
        the reconciler — a partially-filled stop means held base != tracked size, so a
        full-size close would over-sell (long) / over-buy (short). (#710)
        """
        return self.stop_loss_manager.filled_quantity(position)

    def _position_still_held(self, position: Position) -> bool:
        """Whether the position's inventory is still actually held on the exchange.

        Checked before an inline re-protect so a stop is not re-placed on a position an
        ambiguous / already-executed close has actually closed (which would orphan a
        stop). Conservative: any unreadable/uncertain state returns ``False`` (do not
        re-place; the reconciler reconciles exchange truth). (#710)
        """
        return self.stop_loss_manager.position_still_held(position)

    def _held_protection_quantity(self, position: Position) -> float:
        """Base quantity to protect, scaled for any prior partial exits.

        Mirrors the reconciler's re-placement sizing ``quantity * current/original`` so
        a re-protected stop covers the *remaining* held size, not the full entry size.
        """
        return self.stop_loss_manager.held_protection_quantity(position)

    def _reprotect_position(self, position: Position) -> None:
        """Re-place a stop-loss after a failed close left a position momentarily naked.

        Reached only when a market close failed *after* its clean (zero-fill) resting
        stop was cancelled to free the base balance (#710). Re-establish protection
        immediately rather than waiting for the ~120s reconciler — but first verify the
        position is still actually held (the close may be ambiguous / already executed)
        to avoid orphaning a stop, and size for any prior partial exits. The reconciler
        is the ultimate backstop if this attempt cannot run or also fails.
        """
        self.stop_loss_manager.reprotect(position)

    def _check_stop_loss_filled(self, position: Position) -> tuple[bool, float | None]:
        """Check if a stop-loss order already filled on the exchange."""
        return self.stop_loss_manager.check_filled(position)

    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        self.account_monitor.update_performance_metrics()

    def _extract_indicators(self, df: pd.DataFrame, index: int) -> dict:
        """Extract indicator values from dataframe for logging"""
        return extract_indicators(df, index)

    def _extract_sentiment_data(self, df: pd.DataFrame, index: int) -> dict:
        """Extract sentiment data from dataframe for logging"""
        return extract_sentiment_data(df, index)

    def _extract_ml_predictions(self, df: pd.DataFrame, index: int) -> dict:
        """Extract ML prediction data from dataframe for logging"""
        return extract_ml_predictions(df, index)

    def _log_account_snapshot(self):
        """Log current account state to database via the event logger.

        ``LiveAccountMonitor`` routes through
        ``LiveEventLogger.log_account_snapshot`` so daily P&L tracking — and
        its day-start recovery across restarts (#766) — stays on the live
        path.
        """
        self.account_monitor.log_account_snapshot()

    def _log_status(self, symbol: str, current_price: float):
        """Log current trading status"""
        self.account_monitor.log_status(symbol, current_price)

    def _log_trade(self, trade: Trade):
        """Log trade to file"""
        try:
            # Create logs/trades directory if it doesn't exist
            os.makedirs("logs/trades", exist_ok=True)

            log_file = f"logs/trades/trades_{datetime.now(UTC).strftime('%Y%m')}.json"
            trade_data = {
                "timestamp": trade.exit_time.isoformat(),
                "symbol": trade.symbol,
                # BaseTrade.__post_init__ normalizes str sides to PositionSide.
                "side": cast(PositionSide, trade.side).value,
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
            logger.error("Failed to log trade: %s", e, exc_info=True)

    def _record_event(
        self,
        event_type: EventType,
        message: str,
        *,
        severity: str = "error",
        component: str | None = None,
        error_code: str | None = None,
        exc: BaseException | None = None,
        alert: bool = False,
    ) -> None:
        """Emit a structured ``system_events`` row (and optionally an alert).

        Populates the long-dormant observability columns (``component``,
        ``error_code``, ``stack_trace``, ``alert_sent``, ``alert_method``) so
        operators can triage incidents from ``system_events`` instead of grepping
        application logs. A stack trace is captured only when ``exc`` is supplied
        and an exception is currently being handled.

        Entirely fault-isolated: any logging, DB, or alert failure is swallowed so
        observability can never break the trading loop.
        """
        try:
            # Capture the active traceback only when an exception is in flight;
            # format_exc() returns the literal string "NoneType: None\n" outside
            # an except block, which would be noise in the audit trail.
            stack_trace = traceback.format_exc() if exc is not None else None
            if stack_trace is not None and stack_trace.startswith("NoneType: None"):
                stack_trace = None

            alert_sent = False
            alert_method: str | None = None
            if alert:
                # Record the real OUTCOME, not just intent: _send_alert returns
                # False when no webhook is configured or the POST fails, so
                # alert_sent reflects whether an operator was actually paged.
                alert_sent = bool(self._send_alert(message))
                alert_method = "webhook" if alert_sent else None

            self.db_manager.log_event(
                event_type=event_type,
                message=message,
                severity=severity,
                component=component,
                error_code=error_code,
                stack_trace=stack_trace,
                session_id=self.trading_session_id,
                alert_sent=alert_sent,
                alert_method=alert_method,
            )
        except Exception as e:
            # Observability must never propagate into the trading loop.
            logger.warning("observability event failed: %s", e)

    def _send_alert(self, message: str) -> bool:
        """Send a trading alert (webhook).

        Returns True only if an alert was actually delivered (a webhook POST
        returned a 2xx status); False when no webhook is configured, the POST
        raised, or the endpoint returned a non-2xx status (4xx/5xx). Callers
        persist this as the real ``alert_sent`` so operators aren't misled into
        thinking a critical event paged them when it didn't.
        """
        if not self.alert_webhook_url:
            return False

        try:
            import requests  # type: ignore[import-untyped]  # types-requests not installed

            payload = {
                "text": f"🤖 Trading Bot: {message}",
                "timestamp": datetime.now(UTC).isoformat(),
            }
            resp = requests.post(self.alert_webhook_url, json=payload, timeout=10)
            # A 4xx/5xx response means the alert was NOT delivered — requests
            # does not raise on HTTP error status by default, so check it
            # explicitly or alert_sent would falsely read True (#688 review).
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error("Failed to send alert: %s", e, exc_info=True)
            return False

    def _sleep_with_interrupt(self, seconds: float) -> None:
        """Sleep in small increments to allow for interrupt (delegated to LiveLoopTimingCoordinator)."""
        return self.loop_timing_coordinator.sleep_with_interrupt(seconds)

    def _calculate_adaptive_interval(self, current_price: float | None = None) -> int:
        """Adaptive check interval from activity + market conditions (delegated to LiveLoopTimingCoordinator)."""
        return self.loop_timing_coordinator.calculate_adaptive_interval(current_price)

    def _is_data_fresh(self, df: pd.DataFrame) -> bool:
        """Check the data is fresh enough to process (delegated to LiveLoopTimingCoordinator)."""
        return self.loop_timing_coordinator.is_data_fresh(df)

    def _print_final_stats(self):
        """Print final trading statistics"""
        self.account_monitor.print_final_stats()

    def get_performance_summary(self) -> dict[str, Any]:
        """Get current performance summary"""
        return self.account_monitor.performance_summary()

    def _recover_existing_session(self) -> float | None:
        """Try to recover balance from an existing session.

        See ``LiveSessionRecoverer.recover_existing_session`` for the full
        crash-recovery / clean-restart semantics (#668).
        """
        return self.session_recoverer.recover_existing_session()

    def _ensure_positions_registered_with_risk_manager(self) -> None:
        """Register every tracked position with the risk manager (idempotent).

        See ``LiveSessionRecoverer.ensure_positions_registered_with_risk_manager``
        for the parity rationale.
        """
        self.session_recoverer.ensure_positions_registered_with_risk_manager()

    def _recover_active_positions(self) -> None:
        """Recover active positions from database"""
        self.session_recoverer.recover_active_positions()

    def _reconcile_positions_with_exchange(self) -> None:
        """Reconcile local positions with exchange state on startup.

        Delegates to PositionReconciler for comprehensive order-based verification
        (via ``LiveSessionRecoverer``), falling back to legacy SL-based
        reconciliation if the reconciler is unavailable.
        """
        self.session_recoverer.reconcile_positions_with_exchange()

    def _handle_strategy_change(self, swap_data: dict[str, Any]):
        """Handle strategy change callback"""
        self.hot_swap_coordinator.handle_strategy_change(swap_data)

    def _handle_model_update(self, update_data: dict[str, Any]):
        """Handle model update callback"""
        self.hot_swap_coordinator.handle_model_update(update_data)

    def hot_swap_strategy(
        self, new_strategy_name: str, close_positions: bool = False, new_config: dict | None = None
    ) -> bool:
        """Hot-swap to a new strategy during live trading."""
        return self.hot_swap_coordinator.hot_swap_strategy(
            new_strategy_name, close_positions=close_positions, new_config=new_config
        )

    def update_model(self, new_model_path: str) -> bool:
        """Update ML models during live trading."""
        return self.hot_swap_coordinator.update_model(new_model_path)

    def _build_trailing_policy(self) -> TrailingStopPolicy | None:
        """Construct trailing policy from risk parameters and strategy overrides.

        Uses shared risk configuration logic for consistency with backtest engine.
        """
        return build_trailing_stop_policy(self.strategy, self.risk_manager)

    def _apply_pending_strategy_update(self) -> bool:
        """Apply a queued strategy/model update from the StrategyManager.

        See ``StrategyHotSwapCoordinator.apply_pending_strategy_update`` for the
        full pipeline; kept as a method so the run loop and unit tests drive the
        same code path.
        """
        return self.hot_swap_coordinator.apply_pending_strategy_update()

    def _refresh_strategy_dependencies(self) -> None:
        """Refresh engine state derived from the active strategy (post hot-swap)."""
        self.hot_swap_coordinator.refresh_strategy_dependencies()

    def _refresh_partial_manager_after_swap(
        self,
        new_overrides: dict[str, Any],
    ) -> None:
        """Rebuild engine-level partial_manager from new strategy overrides."""
        self.hot_swap_coordinator.refresh_partial_manager_after_swap(new_overrides)

    def _refresh_time_exit_policy_after_swap(
        self,
        new_overrides: dict[str, Any],
    ) -> None:
        """Rebuild engine-level time_exit_policy from new strategy overrides."""
        self.hot_swap_coordinator.refresh_time_exit_policy_after_swap(new_overrides)
