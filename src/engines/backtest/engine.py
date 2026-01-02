"""Backtesting engine for strategy evaluation.

This module provides the main Backtester class that orchestrates backtest
execution by delegating to specialized handlers for entry, exit, risk,
regime switching, and logging.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
from pandas import DataFrame
from sqlalchemy.exc import SQLAlchemyError

from src.config.config_manager import get_config
from src.config.constants import (
    DEFAULT_CONFIDENCE_SCORE,
    DEFAULT_DYNAMIC_RISK_ENABLED,
    DEFAULT_END_OF_DAY_FLAT,
    DEFAULT_FEE_RATE,
    DEFAULT_INITIAL_BALANCE,
    DEFAULT_MARKET_TIMEZONE,
    DEFAULT_MAX_HOLDING_HOURS,
    DEFAULT_MAX_POSITION_SIZE,
    DEFAULT_MFE_MAE_PRECISION_DECIMALS,
    DEFAULT_REGIME_LOOKBACK_BUFFER,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_TIME_RESTRICTIONS,
    DEFAULT_WEEKEND_FLAT,
)
from src.database.models import TradeSource
from src.engines.backtest.execution import (
    EntryHandler,
    ExecutionEngine,
    ExitHandler,
    PositionTracker,
)
from src.engines.backtest.logging import EventLogger
from src.engines.backtest.models import ActiveTrade, Trade
from src.engines.backtest.regime import RegimeHandler
from src.engines.backtest.risk import CorrelationHandler
from src.engines.backtest.utils import extract_indicators as util_extract_indicators
from src.engines.backtest.utils import extract_ml_predictions as util_extract_ml
from src.engines.backtest.utils import extract_sentiment_data as util_extract_sentiment
from src.engines.shared.partial_operations_manager import PartialOperationsManager
from src.engines.shared.policy_hydration import apply_policies_to_engine
from src.engines.shared.risk_configuration import (
    build_trailing_stop_policy,
    merge_dynamic_risk_config,
)
from src.infrastructure.logging.context import set_context, update_context
from src.infrastructure.logging.events import log_engine_event
from src.position_management.correlation_engine import CorrelationConfig, CorrelationEngine
from src.position_management.dynamic_risk import DynamicRiskConfig, DynamicRiskManager
from src.position_management.time_exits import TimeExitPolicy, TimeRestrictions
from src.position_management.trailing_stops import TrailingStopPolicy
from src.regime.detector import RegimeDetector
from src.risk.risk_manager import RiskManager
from src.strategies.components import Position as ComponentPosition
from src.strategies.components import (
    RuntimeContext,
    StrategyRuntime,
)
from src.strategies.components import Strategy as ComponentStrategy

if TYPE_CHECKING:
    from src.data_providers.data_provider import DataProvider
    from src.data_providers.sentiment_provider import SentimentDataProvider
    from src.database.manager import DatabaseManager

logger = logging.getLogger(__name__)

# Use centralized constant for regime lookback buffer
REGIME_LOOKBACK_BUFFER = DEFAULT_REGIME_LOOKBACK_BUFFER


def _compute_regime_lookback(regime_switcher: Any) -> int:
    """Determine how many candles are needed for regime analysis.

    Args:
        regime_switcher: Regime switcher instance.

    Returns:
        Number of candles needed for regime lookback.
    """
    if not regime_switcher:
        return 0

    configs: list[Any] = []

    detector = getattr(regime_switcher, "regime_detector", None)
    if detector is not None:
        cfg = getattr(detector, "config", None)
        if cfg is not None:
            configs.append(cfg)

    timeframe_detectors = getattr(regime_switcher, "timeframe_detectors", {}) or {}
    for detector in timeframe_detectors.values():
        cfg = getattr(detector, "config", None)
        if cfg is not None:
            configs.append(cfg)

    if not configs:
        return 0

    slope_window = max((getattr(cfg, "slope_window", 0) or 0) for cfg in configs)
    atr_lookback = max((getattr(cfg, "atr_percentile_lookback", 0) or 0) for cfg in configs)

    base_lookback = max(slope_window, atr_lookback)
    return int(base_lookback + REGIME_LOOKBACK_BUFFER)


class Backtester:
    """Backtesting engine for trading strategies.

    Orchestrates backtest execution by delegating to specialized handlers:
    - ExecutionEngine: Handles fees, slippage, next-bar execution
    - PositionTracker: Manages active trade state and MFE/MAE
    - EntryHandler: Processes entry signals and execution
    - ExitHandler: Processes exit signals and execution
    - CorrelationHandler: Applies correlation-based sizing
    - RegimeHandler: Manages regime-based strategy switching
    - EventLogger: Coordinates database logging
    """

    def __init__(
        self,
        strategy: ComponentStrategy | StrategyRuntime,
        data_provider: DataProvider,
        sentiment_provider: SentimentDataProvider | None = None,
        risk_parameters: Any | None = None,
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
        database_url: str | None = None,
        log_to_database: bool | None = None,
        default_take_profit_pct: float | None = None,
        legacy_stop_loss_indexing: bool = True,
        enable_engine_risk_exits: bool = True,
        time_exit_policy: TimeExitPolicy | None = None,
        enable_dynamic_risk: bool = DEFAULT_DYNAMIC_RISK_ENABLED,
        dynamic_risk_config: DynamicRiskConfig | None = None,
        trailing_stop_policy: TrailingStopPolicy | None = None,
        partial_manager: Any | None = None,
        enable_regime_switching: bool = False,
        regime_config: Any | None = None,
        strategy_mapping: Any | None = None,
        switching_config: Any | None = None,
        fee_rate: float = DEFAULT_FEE_RATE,
        slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
        use_next_bar_execution: bool = False,
        use_high_low_for_stops: bool = True,
        max_position_size: float | None = None,
    ) -> None:
        """Initialize backtester with strategy and configuration.

        Args:
            strategy: Trading strategy to backtest.
            data_provider: Provider for historical market data.
            sentiment_provider: Optional provider for sentiment data.
            risk_parameters: Risk management parameters.
            initial_balance: Starting account balance.
            database_url: Database URL for logging.
            log_to_database: Whether to log to database.
            default_take_profit_pct: Default take profit percentage.
            legacy_stop_loss_indexing: Use legacy SL calculation behavior.
            enable_engine_risk_exits: Enable SL/TP checks in engine.
            time_exit_policy: Policy for time-based exits.
            enable_dynamic_risk: Enable dynamic risk management.
            dynamic_risk_config: Configuration for dynamic risk.
            trailing_stop_policy: Policy for trailing stops.
            partial_manager: Manager for partial exits/scale-ins.
            enable_regime_switching: Enable regime-based strategy switching.
            regime_config: Configuration for regime detection.
            strategy_mapping: Mapping of regimes to strategies.
            switching_config: Configuration for strategy switching.
            fee_rate: Fee percentage per trade.
            slippage_rate: Slippage percentage per trade.
            use_next_bar_execution: Execute entries on next bar's open.
            use_high_low_for_stops: Use high/low for SL/TP detection.
            max_position_size: Maximum position size as fraction of balance (backward compatibility).
        """
        if initial_balance <= 0:
            raise ValueError("Initial balance must be positive")

        # Validate max_position_size if provided
        if max_position_size is not None:
            if max_position_size <= 0 or max_position_size > 1:
                raise ValueError("Max position size must be between 0 and 1")

        # Core state
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.trades: list[Trade] = []
        self.dynamic_risk_adjustments: list[dict] = []

        # Performance tracking
        from src.performance.tracker import PerformanceTracker

        self.performance_tracker = PerformanceTracker(initial_balance)

        # Configure strategy
        self._runtime_dataset = None
        self._runtime_warmup = 0
        self._configure_strategy(strategy)

        name_source = strategy if isinstance(strategy, StrategyRuntime) else self.strategy
        self.initial_strategy_name = getattr(name_source, "name", name_source.__class__.__name__)

        # Providers
        self.data_provider = data_provider
        self.sentiment_provider = sentiment_provider

        # Risk management - handle backward compatibility for max_position_size
        if max_position_size is not None:
            from src.risk.risk_manager import RiskParameters

            if risk_parameters is None:
                risk_parameters = RiskParameters(max_position_size=max_position_size)
            else:
                # Update existing risk_parameters with max_position_size
                if hasattr(risk_parameters, "max_position_size"):
                    risk_parameters.max_position_size = max_position_size

        self.risk_parameters = risk_parameters
        self.risk_manager = RiskManager(risk_parameters)
        self.enable_dynamic_risk = enable_dynamic_risk
        self.dynamic_risk_manager = None

        if enable_dynamic_risk:
            config = dynamic_risk_config or DynamicRiskConfig()
            final_config = self._merge_dynamic_risk_config(config, strategy)
            self.dynamic_risk_manager = DynamicRiskManager(final_config, db_manager=None)

        # Build policies
        self._custom_trailing_stop_policy = trailing_stop_policy is not None
        self.trailing_stop_policy = trailing_stop_policy or self._build_trailing_policy()
        self._trailing_stop_opt_in = self.trailing_stop_policy is not None

        self._custom_time_exit_policy = time_exit_policy is not None
        self.time_exit_policy = time_exit_policy or self._build_time_exit_policy()

        self.partial_manager = partial_manager
        self._partial_operations_opt_in = partial_manager is not None

        # Feature flags
        self.default_take_profit_pct = default_take_profit_pct
        self.legacy_stop_loss_indexing = legacy_stop_loss_indexing
        self.enable_engine_risk_exits = enable_engine_risk_exits

        # Early stop tracking
        self.early_stop_reason: str | None = None
        self.early_stop_date: datetime | None = None
        self.early_stop_candle_index: int | None = None
        self._early_stop_max_drawdown = (
            self.risk_manager.params.max_drawdown if risk_parameters is not None else 0.5
        )

        # Initialize handlers
        self.execution_engine = ExecutionEngine(
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            use_next_bar_execution=use_next_bar_execution,
        )

        self.position_tracker = PositionTracker(
            mfe_mae_precision=DEFAULT_MFE_MAE_PRECISION_DECIMALS
        )

        # Correlation engine
        self.correlation_engine: CorrelationEngine | None = None
        try:
            corr_cfg = CorrelationConfig(
                correlation_window_days=self.risk_manager.params.correlation_window_days,
                correlation_threshold=self.risk_manager.params.correlation_threshold,
                max_correlated_exposure=self.risk_manager.params.max_correlated_exposure,
                correlation_update_frequency_hours=self.risk_manager.params.correlation_update_frequency_hours,
            )
            self.correlation_engine = CorrelationEngine(config=corr_cfg)
        except Exception:
            pass

        self.correlation_handler: CorrelationHandler | None = None
        if self.correlation_engine is not None:
            self.correlation_handler = CorrelationHandler(
                correlation_engine=self.correlation_engine,
                risk_manager=self.risk_manager,
                data_provider=data_provider,
                strategy=self.strategy,
            )

        # Regime detector
        self.regime_detector: RegimeDetector | None = None
        try:
            self.regime_detector = RegimeDetector()
        except Exception:
            pass

        # Regime switching
        self.enable_regime_switching = enable_regime_switching
        self.regime_handler: RegimeHandler | None = None
        self._init_regime_switching(regime_config, strategy_mapping, switching_config)

        # Database logging
        self._init_database_logging(log_to_database, database_url)

        # Initialize entry/exit handlers
        self.entry_handler = EntryHandler(
            execution_engine=self.execution_engine,
            position_tracker=self.position_tracker,
            risk_manager=self.risk_manager,
            component_strategy=self._component_strategy,
            dynamic_risk_manager=self.dynamic_risk_manager,
            correlation_handler=self.correlation_handler,
            default_take_profit_pct=default_take_profit_pct,
            max_position_size=self.risk_manager.params.max_position_size,
        )

        # Wrap PartialExitPolicy in unified PartialOperationsManager
        partial_ops_manager = (
            PartialOperationsManager(policy=partial_manager)
            if partial_manager is not None
            else None
        )

        self.exit_handler = ExitHandler(
            execution_engine=self.execution_engine,
            position_tracker=self.position_tracker,
            risk_manager=self.risk_manager,
            trailing_stop_policy=self.trailing_stop_policy,
            time_exit_policy=self.time_exit_policy,
            partial_manager=partial_ops_manager,
            enable_engine_risk_exits=enable_engine_risk_exits,
            use_high_low_for_stops=use_high_low_for_stops,
        )

        # For backward compatibility - expose current_trade through position_tracker
        # Tests may access backtester.current_trade directly

    @property
    def current_trade(self) -> ActiveTrade | None:
        """Get the current active trade (backward compatibility)."""
        return self.position_tracker.current_trade

    @current_trade.setter
    def current_trade(self, value: ActiveTrade | None) -> None:
        """Set the current active trade (backward compatibility)."""
        self.position_tracker.current_trade = value

    @property
    def regime_switcher(self):
        """Get regime switcher (backward compatibility)."""
        if self.regime_handler:
            return self.regime_handler.regime_switcher
        return None

    @property
    def total_fees_paid(self) -> float:
        """Get total fees paid (backward compatibility)."""
        return self.execution_engine.total_fees_paid

    @property
    def total_slippage_cost(self) -> float:
        """Get total slippage cost (backward compatibility)."""
        return self.execution_engine.total_slippage_cost

    @property
    def fee_rate(self) -> float:
        """Get fee rate (backward compatibility)."""
        return self.execution_engine.fee_rate

    @property
    def slippage_rate(self) -> float:
        """Get slippage rate (backward compatibility)."""
        return self.execution_engine.slippage_rate

    @property
    def use_next_bar_execution(self) -> bool:
        """Get next bar execution setting (backward compatibility)."""
        return self.execution_engine.use_next_bar_execution

    @property
    def use_high_low_for_stops(self) -> bool:
        """Get high/low for stops setting (backward compatibility)."""
        return self.exit_handler.use_high_low_for_stops

    @property
    def max_position_size(self) -> float:
        """Get max position size (backward compatibility)."""
        if self.risk_parameters is None:
            return DEFAULT_MAX_POSITION_SIZE  # Default for backward compatibility
        return self.risk_manager.params.max_position_size

    def _init_regime_switching(
        self,
        regime_config: Any,
        strategy_mapping: Any,
        switching_config: Any,
    ) -> None:
        """Initialize regime-based strategy switching if enabled."""
        if not self.enable_regime_switching:
            return

        try:
            from src.engines.live.regime_strategy_switcher import RegimeStrategySwitcher
            from src.engines.live.strategy_manager import StrategyManager

            strategy_manager = StrategyManager()
            strategy_key = self._normalize_strategy_key(self.strategy.name)
            strategy_manager.load_strategy(strategy_key)

            regime_switcher = RegimeStrategySwitcher(
                strategy_manager=strategy_manager,
                regime_config=regime_config,
                strategy_mapping=strategy_mapping,
                switching_config=switching_config,
            )

            self.regime_handler = RegimeHandler(
                regime_switcher=regime_switcher,
                strategy_manager=strategy_manager,
                initial_strategy_name=self.initial_strategy_name,
            )

            logger.info("Regime-aware strategy switching enabled for backtesting")
        except Exception as e:
            logger.warning(
                "Failed to initialize regime switching: %s. Continuing without regime switching.",
                e,
            )
            self.enable_regime_switching = False

    def _normalize_strategy_key(self, name: str) -> str:
        """Normalize strategy name to registry format."""
        key = name.lower().replace(" ", "_").replace("strategy", "")
        if key == "mlbasic":
            return "ml_basic"
        elif key == "ensembleweighted":
            return "ensemble_weighted"
        elif key == "momentumleverage":
            return "momentum_leverage"
        return key

    def _init_database_logging(
        self,
        log_to_database: bool | None,
        database_url: str | None,
    ) -> None:
        """Initialize database logging configuration."""
        # Auto-detect test environment
        if log_to_database is None:
            database_url_env = os.getenv("DATABASE_URL", "")
            is_pytest = os.environ.get("PYTEST_CURRENT_TEST") is not None
            log_to_database = not (
                database_url_env.startswith("sqlite://") or database_url_env == "" or is_pytest
            )

        self.log_to_database = log_to_database
        self.db_manager: DatabaseManager | None = None
        self.trading_session_id: int | None = None

        if log_to_database:
            self._connect_database(database_url)

        self.event_logger = EventLogger(
            db_manager=self.db_manager,
            log_to_database=self.log_to_database,
        )

    def _connect_database(self, database_url: str | None) -> None:
        """Connect to database for logging."""
        from src.database.manager import DatabaseManager

        try:
            selected_db_url = database_url
            if selected_db_url is None:
                try:
                    cfg = get_config()
                    selected_db_url = cfg.get("PRODUCTION_DATABASE_URL")
                except Exception:
                    pass

            self.db_manager = DatabaseManager(selected_db_url)
            if self.db_manager and hasattr(self.strategy, "set_database_manager"):
                self.strategy.set_database_manager(self.db_manager)

        except (SQLAlchemyError, ValueError) as db_err:
            logger.warning(
                "Database connection failed (%s). Falling back to in-memory SQLite.",
                db_err,
            )
            try:
                self.db_manager = DatabaseManager("sqlite:///:memory:")
                if self.db_manager and hasattr(self.strategy, "set_database_manager"):
                    self.strategy.set_database_manager(self.db_manager)
            except Exception as sqlite_err:
                logger.warning("Fallback SQLite also failed: %s", sqlite_err)
                self.db_manager = None

    def _configure_strategy(self, strategy: ComponentStrategy | StrategyRuntime) -> None:
        """Normalize strategy inputs and prepare runtime state."""
        runtime = strategy if isinstance(strategy, StrategyRuntime) else None
        base_strategy = runtime.strategy if runtime is not None else strategy

        self.strategy = base_strategy
        self._component_strategy = (
            base_strategy if isinstance(base_strategy, ComponentStrategy) else None
        )

        if runtime is not None:
            self._runtime = runtime
        elif self._component_strategy is not None:
            self._runtime = StrategyRuntime(self._component_strategy)
        else:
            self._runtime = None

    def _is_runtime_strategy(self) -> bool:
        """Check if using runtime-based strategy."""
        return self._runtime is not None

    def _merge_dynamic_risk_config(
        self, base_config: DynamicRiskConfig, strategy: Any
    ) -> DynamicRiskConfig:
        """Merge strategy risk overrides with base dynamic risk configuration.

        Uses shared risk configuration logic for consistency.
        """
        return merge_dynamic_risk_config(base_config, strategy)

    def _build_trailing_policy(self) -> TrailingStopPolicy | None:
        """Build trailing stop policy from strategy/risk overrides.

        Uses shared risk configuration logic for consistency.
        """
        return build_trailing_stop_policy(self.strategy, self.risk_manager)

    def _build_time_exit_policy(self) -> TimeExitPolicy | None:
        """Build time exit policy from strategy/risk overrides."""
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

        if not time_cfg and self.risk_manager:
            params = getattr(self.risk_manager, "params", None)
            if params:
                time_cfg = getattr(params, "time_exits", None)

        if not time_cfg:
            return None

        try:
            restrictions_cfg = (
                time_cfg.get("time_restrictions") if isinstance(time_cfg, dict) else None
            )
            if restrictions_cfg is None:
                restrictions_cfg = DEFAULT_TIME_RESTRICTIONS

            restrictions = TimeRestrictions(
                no_overnight=bool(restrictions_cfg.get("no_overnight", False)),
                no_weekend=bool(restrictions_cfg.get("no_weekend", False)),
                trading_hours_only=bool(restrictions_cfg.get("trading_hours_only", False)),
            )

            return TimeExitPolicy(
                max_holding_hours=time_cfg.get("max_holding_hours", DEFAULT_MAX_HOLDING_HOURS),
                end_of_day_flat=time_cfg.get("end_of_day_flat", DEFAULT_END_OF_DAY_FLAT),
                weekend_flat=time_cfg.get("weekend_flat", DEFAULT_WEEKEND_FLAT),
                market_timezone=time_cfg.get("market_timezone", DEFAULT_MARKET_TIMEZONE),
                time_restrictions=restrictions,
            )
        except Exception:
            return None

    def _prepare_strategy_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare strategy data using the runtime."""
        if not self._is_runtime_strategy():
            return df

        dataset = self._runtime.prepare_data(df)
        self._runtime_dataset = dataset
        self._runtime_warmup = max(0, int(dataset.warmup_period or 0))
        return dataset.data

    def _build_runtime_context(
        self, balance: float, current_price: float, current_time: datetime
    ) -> RuntimeContext:
        """Build runtime context from current engine state."""
        positions: list[ComponentPosition] = []
        trade = self.position_tracker.current_trade

        if trade is not None:
            notional = getattr(trade, "component_notional", None)
            if notional is None:
                notional = float(trade.current_size) * float(balance)
            try:
                positions.append(
                    ComponentPosition(
                        symbol=trade.symbol,
                        side=trade.side,
                        size=float(notional),
                        entry_price=float(trade.entry_price),
                        current_price=float(current_price),
                        entry_time=trade.entry_time,
                    )
                )
            except Exception:
                pass

        return RuntimeContext(balance=float(balance), current_positions=positions or None)

    def _get_runtime_decision(
        self, df: pd.DataFrame, index: int, current_price: float, current_time: datetime
    ) -> Any:
        """Get decision from runtime strategy."""
        if not self._is_runtime_strategy() or self._runtime_dataset is None:
            return None
        if index < self._runtime_warmup:
            return None

        context = self._build_runtime_context(self.balance, current_price, current_time)
        try:
            decision = self._runtime.process(index, context)
            self._apply_policies_from_decision(decision)
            return decision
        except Exception as e:
            logger.warning("Runtime decision failed at index %s: %s", index, e)
            return None

    def _apply_policies_from_decision(self, decision: Any) -> None:
        """Hydrate engine-level policies from runtime decisions.

        Uses shared policy hydration logic for consistency with live engine.
        """
        apply_policies_to_engine(decision, self, self.db_manager)

    def _finalize_runtime(self) -> None:
        """Clean up runtime state after backtest."""
        if self._is_runtime_strategy():
            try:
                self._runtime.finalize()
            finally:
                self._runtime_dataset = None
                self._runtime_warmup = 0

    def _switch_strategy(self, new_strategy: ComponentStrategy, df: pd.DataFrame) -> None:
        """Switch to a new strategy during regime switching."""
        self._configure_strategy(new_strategy)

        # Update handlers
        self.entry_handler.set_component_strategy(self._component_strategy)
        if self.correlation_handler:
            self.correlation_handler.set_strategy(self.strategy)

        # Prepare runtime dataset for new strategy
        try:
            df = self._prepare_strategy_dataframe(df)
            logger.debug("Prepared runtime dataset for switched strategy")
        except Exception as e:
            logger.warning("Failed to prepare runtime dataset: %s", e)
            self._runtime_dataset = None
            self._runtime_warmup = 0

    def run(
        self, symbol: str, timeframe: str, start: datetime, end: datetime | None = None
    ) -> dict:
        """Run backtest for the specified period.

        Args:
            symbol: Trading symbol.
            timeframe: Candle timeframe.
            start: Start datetime.
            end: End datetime (optional).

        Returns:
            Dictionary with backtest results including metrics and trades.
        """
        try:
            # Set logging context
            set_context(
                component="backtester",
                strategy=getattr(self.strategy, "__class__", type("_", (), {})).__name__,
                symbol=symbol,
                timeframe=timeframe,
            )
            log_engine_event(
                "backtest_start",
                initial_balance=self.initial_balance,
                start=start.isoformat(),
                end=end.isoformat() if end else None,
            )

            # Create trading session
            self._create_trading_session(symbol, timeframe, start)

            # Fetch and prepare data
            df = self._fetch_and_prepare_data(symbol, timeframe, start, end)
            if df.empty:
                return self._build_empty_results()

            # Calculate hold return for comparison
            hold_return = self._calculate_hold_return(df)

            logger.info("Starting backtest with %d candles", len(df))

            # Run main backtest loop
            balance_history, yearly_balance = self._run_main_loop(df, symbol, timeframe)

            # Compute final metrics
            return self._build_final_results(
                df, start, end, balance_history, yearly_balance, hold_return
            )

        except Exception as e:
            logger.error("Error running backtest: %s", e)
            raise
        finally:
            self._finalize_runtime()

    def _create_trading_session(self, symbol: str, timeframe: str, start: datetime) -> None:
        """Create trading session in database if enabled."""
        if not self.log_to_database or not self.db_manager:
            return

        self.trading_session_id = self.event_logger.create_trading_session(
            strategy_name=self.strategy.__class__.__name__,
            symbol=symbol,
            timeframe=timeframe,
            source=TradeSource.BACKTEST,
            initial_balance=self.initial_balance,
            strategy_config=getattr(self.strategy, "config", {}),
            start_time=start,
        )

        if hasattr(self.strategy, "session_id"):
            self.strategy.session_id = self.trading_session_id

        update_context(session_id=self.trading_session_id)

        if self.enable_dynamic_risk and self.dynamic_risk_manager and self.db_manager:
            self.dynamic_risk_manager.db_manager = self.db_manager

    def _fetch_and_prepare_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime | None,
    ) -> pd.DataFrame:
        """Fetch historical data and prepare for backtesting."""
        df: DataFrame = self.data_provider.get_historical_data(symbol, timeframe, start, end)

        if df.empty:
            return df

        # Validate required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except (ValueError, TypeError):
                df.index = pd.date_range(start=start, periods=len(df), freq="h")

        # Merge sentiment data
        if self.sentiment_provider:
            df = self._merge_sentiment_data(df, symbol, timeframe, start, end)

        # Prepare strategy data
        df = self._prepare_strategy_dataframe(df)

        # Drop rows with missing essential data
        essential_columns = ["open", "high", "low", "close", "volume"]
        df = df.dropna(subset=essential_columns)

        return df

    def _merge_sentiment_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime | None,
    ) -> pd.DataFrame:
        """Merge sentiment data into price DataFrame."""
        if not self.sentiment_provider:
            return df

        sentiment_df = self.sentiment_provider.get_historical_sentiment(symbol, start, end)
        if not sentiment_df.empty:
            sentiment_df = self.sentiment_provider.aggregate_sentiment(
                sentiment_df, window=timeframe
            )
            df = df.join(sentiment_df, how="left")
            if "sentiment_score" in df.columns:
                df["sentiment_score"] = df["sentiment_score"].ffill().fillna(0)

        return df

    def _calculate_hold_return(self, df: pd.DataFrame) -> float:
        """Calculate buy-and-hold return for comparison with robust validation."""
        if df.empty or len(df) < 2:
            logger.warning("Insufficient data for hold return calculation")
            return 0.0

        start_price = float(df["close"].iloc[0])
        end_price = float(df["close"].iloc[-1])

        # Guard against zero or invalid prices to prevent division by zero
        if start_price <= 0 or end_price <= 0:
            logger.warning(
                "Invalid prices for hold return: start=%.8f, end=%.8f",
                start_price,
                end_price,
            )
            return 0.0

        return ((end_price / start_price) - 1) * 100

    def _run_main_loop(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> tuple[list[tuple], dict[int, dict[str, float]]]:
        """Run the main backtest iteration loop.

        Returns:
            Tuple of (balance_history, yearly_balance).
        """
        balance_history: list[tuple] = []
        yearly_balance: dict[int, dict[str, float]] = {}
        total_trades = 0
        winning_trades = 0
        max_drawdown_running = 0.0

        for i in range(len(df)):
            candle = df.iloc[i]
            current_time = candle.name
            current_price = float(candle["close"])
            open_price = float(candle["open"])

            # Process pending entry from previous candle
            if self.execution_engine.has_pending_entry and not self.position_tracker.has_position:
                entry_result = self.entry_handler.process_pending_entry(
                    symbol=symbol,
                    open_price=open_price,
                    current_time=current_time,
                    balance=self.balance,
                )
                if entry_result.executed:
                    self._deduct_entry_fee(entry_result.entry_fee)

                    # Final validation: ensure balance is still positive
                    if self.balance < 0:
                        logger.critical(
                            "Balance went negative: %.2f after entry fee %.2f",
                            self.balance,
                            entry_result.entry_fee,
                        )
                        raise RuntimeError(f"Balance corruption: balance={self.balance:.2f}")

            # Get runtime decision
            runtime_decision = self._get_runtime_decision(df, i, current_price, current_time)

            # Track balance
            balance_history.append((current_time, self.balance))

            # Update performance tracker every candle for accurate intraday tracking
            # Note: This differs from live engine which updates less frequently (on metric update cycles)
            # This higher sampling rate provides more granular volatility metrics in backtests
            self.performance_tracker.update_balance(self.balance, timestamp=current_time)

            # Track yearly balance
            yr = current_time.year
            if yr not in yearly_balance:
                yearly_balance[yr] = {"start": self.balance, "end": self.balance}
            else:
                yearly_balance[yr]["end"] = self.balance

            # Sync peak_balance from tracker (single source of truth)
            self.peak_balance = self.performance_tracker.peak_balance
            current_drawdown = (
                (self.peak_balance - self.balance) / self.peak_balance
                if self.peak_balance > 0
                else 0.0
            )
            max_drawdown_running = max(max_drawdown_running, current_drawdown)

            # Regime switching check
            if self.regime_handler and self.regime_handler.should_analyze_regime(i):
                new_strategy, switched, _ = self.regime_handler.analyze_and_switch_if_needed(
                    df=df,
                    candle_index=i,
                    current_time=current_time,
                    timeframe=timeframe,
                    balance=self.balance,
                    current_strategy=self.strategy,
                )
                if switched and new_strategy:
                    self._switch_strategy(new_strategy, df)

            # Position exit path
            if self.position_tracker.has_position:
                exit_occurred, trade_result = self._process_position_exit(
                    runtime_decision=runtime_decision,
                    candle=candle,
                    current_price=current_price,
                    current_time=current_time,
                    df=df,
                    index=i,
                    symbol=symbol,
                    timeframe=timeframe,
                )

                if exit_occurred and trade_result:
                    total_trades += 1
                    if trade_result.pnl > 0:
                        winning_trades += 1
                    yearly_balance[current_time.year]["end"] = self.balance

                    # Check max drawdown
                    if current_drawdown > self._early_stop_max_drawdown:
                        self.early_stop_reason = (
                            f"Maximum drawdown exceeded ({current_drawdown:.1%})"
                        )
                        self.early_stop_date = current_time
                        self.early_stop_candle_index = i
                        logger.warning("Maximum drawdown exceeded. Stopping backtest.")
                        break

            # Entry path
            elif self._is_runtime_strategy():
                self._process_entry_signal(
                    runtime_decision=runtime_decision,
                    df=df,
                    index=i,
                    current_price=current_price,
                    current_time=current_time,
                    symbol=symbol,
                    timeframe=timeframe,
                )

        return balance_history, yearly_balance

    def _process_position_exit(
        self,
        runtime_decision: Any,
        candle: pd.Series,
        current_price: float,
        current_time: datetime,
        df: pd.DataFrame,
        index: int,
        symbol: str,
        timeframe: str,
    ) -> tuple[bool, Trade | None]:
        """Process position exit logic.

        Returns:
            Tuple of (exit_occurred, completed_trade).
        """
        # Update trailing stop
        trailing_updated, log_msg = self.exit_handler.update_trailing_stop(
            current_price=current_price,
            df=df,
            index=index,
        )
        if trailing_updated and log_msg:
            self.event_logger.log_risk_adjustment(
                strategy_name=self.strategy.__class__.__name__,
                symbol=symbol,
                adjustment_type="trailing_stop_update",
                current_price=current_price,
                timeframe=timeframe,
                details=[log_msg],
            )

        # Process partial operations
        indicators = util_extract_indicators(df, index)
        partial_result = self.exit_handler.check_partial_operations(
            current_price=current_price,
            df=df,
            index=index,
            indicators=indicators,
        )
        self.balance += partial_result.realized_pnl

        # Update MFE/MAE
        self.position_tracker.update_metrics(current_price, current_time)

        # Check exit conditions
        exit_check = self.exit_handler.check_exit_conditions(
            runtime_decision=runtime_decision,
            candle=candle,
            current_price=current_price,
            symbol=symbol,
            component_strategy=self._component_strategy,
        )

        # Log exit decision
        if self.event_logger.enabled:
            sentiment_data = util_extract_sentiment(df, index)
            ml_predictions = util_extract_ml(df, index)
            current_pnl_pct = self.exit_handler.calculate_current_pnl_pct(current_price)

            self.event_logger.log_exit_decision(
                strategy_name=self.strategy.__class__.__name__,
                symbol=symbol,
                current_price=current_price,
                timeframe=timeframe,
                action_taken="closed_position" if exit_check.should_exit else "hold_position",
                signal_strength=1.0 if exit_check.should_exit else 0.0,
                confidence_score=indicators.get("prediction_confidence", DEFAULT_CONFIDENCE_SCORE),
                position_size=(
                    self.position_tracker.current_trade.size
                    if self.position_tracker.current_trade
                    else None
                ),
                indicators=indicators,
                sentiment_data=sentiment_data,
                ml_predictions=ml_predictions,
                reasons=[
                    exit_check.exit_reason if exit_check.should_exit else "holding_position",
                    f"current_pnl_{current_pnl_pct:.4f}",
                ],
            )

        # Execute exit if needed
        if exit_check.should_exit:
            completed_trade, net_pnl, exit_fee, slippage = self.exit_handler.execute_exit(
                exit_price=exit_check.exit_price,
                exit_reason=exit_check.exit_reason,
                current_time=current_time,
                balance=self.balance,
                symbol=symbol,
            )

            self.balance += net_pnl
            self.trades.append(completed_trade)

            # Update performance tracking
            self.performance_tracker.record_trade(
                trade=completed_trade, fee=exit_fee, slippage=slippage
            )
            self.performance_tracker.update_balance(self.balance, timestamp=current_time)

            # Sync peak_balance from tracker (single source of truth)
            self.peak_balance = self.performance_tracker.peak_balance

            logger.info(
                "Exited %s at %.2f, Balance: %.2f",
                completed_trade.side,
                exit_check.exit_price,
                self.balance,
            )

            self.event_logger.log_completed_trade(
                trade=completed_trade,
                symbol=symbol,
                strategy_name=self.strategy.__class__.__name__,
                source=TradeSource.BACKTEST,
            )

            return True, completed_trade

        return False, None

    def _process_entry_signal(
        self,
        runtime_decision: Any,
        df: pd.DataFrame,
        index: int,
        current_price: float,
        current_time: datetime,
        symbol: str,
        timeframe: str,
    ) -> None:
        """Process entry signal logic."""
        # Get entry signal
        signal = self.entry_handler.process_runtime_decision(
            runtime_decision=runtime_decision,
            balance=self.balance,
            current_price=current_price,
            current_time=current_time,
            df=df,
            index=index,
            symbol=symbol,
            timeframe=timeframe,
            peak_balance=self.peak_balance,
            trading_session_id=self.trading_session_id,
        )

        if not signal.should_enter:
            # Log no-action periodically
            if self.event_logger.should_log_candle(index):
                indicators = util_extract_indicators(df, index)
                sentiment_data = util_extract_sentiment(df, index)
                ml_predictions = util_extract_ml(df, index)

                self.event_logger.log_entry_decision(
                    strategy_name=self.strategy.__class__.__name__,
                    symbol=symbol,
                    current_price=current_price,
                    timeframe=timeframe,
                    action_taken="no_action",
                    indicators=indicators,
                    sentiment_data=sentiment_data,
                    ml_predictions=ml_predictions,
                    reasons=signal.reasons,
                )
            return

        # Execute entry
        entry_result = self.entry_handler.execute_entry(
            signal=signal,
            symbol=symbol,
            current_price=current_price,
            current_time=current_time,
            balance=self.balance,
        )

        if entry_result.executed:
            self._deduct_entry_fee(entry_result.entry_fee)

            # Final validation: ensure balance is still positive
            if self.balance < 0:
                logger.critical(
                    "Balance went negative: %.2f after entry fee %.2f",
                    self.balance,
                    entry_result.entry_fee,
                )
                raise RuntimeError(f"Balance corruption: balance={self.balance:.2f}")

            # Log entry
            indicators = util_extract_indicators(df, index)
            sentiment_data = util_extract_sentiment(df, index)
            ml_predictions = util_extract_ml(df, index)

            self.event_logger.log_entry_decision(
                strategy_name=self.strategy.__class__.__name__,
                symbol=symbol,
                current_price=current_price,
                timeframe=timeframe,
                action_taken=f"opened_{signal.side}",
                signal_strength=(runtime_decision.signal.strength if runtime_decision else 0.0),
                confidence_score=(runtime_decision.signal.confidence if runtime_decision else 0.0),
                position_size=signal.size_fraction,
                indicators=indicators,
                sentiment_data=sentiment_data,
                ml_predictions=ml_predictions,
                reasons=signal.reasons,
            )

            logger.info(
                "Entered %s position at %.2f",
                signal.side,
                current_price,
            )

        # Collect dynamic risk adjustments from handler
        if self.enable_dynamic_risk:
            adjustments = self.entry_handler.get_dynamic_risk_adjustments()
            self.dynamic_risk_adjustments.extend(adjustments)

    def _build_empty_results(self) -> dict:
        """Build results for empty data case."""
        results = {
            "total_trades": 0,
            "final_balance": self.initial_balance,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "avg_trade_duration": 0.0,
            "total_fees": 0.0,
            "total_slippage_cost": 0.0,
            "trades": [],
            "hold_return": 0.0,
            "trading_vs_hold_difference": 0.0,
            "annualized_return": 0.0,
            "yearly_returns": {},
            "prediction_metrics": {
                "prediction_accuracy": 0.0,
                "prediction_mae": 0.0,
            },
            "session_id": None,
            "early_stop_reason": None,
            "early_stop_date": None,
            "early_stop_candle_index": None,
            "dynamic_risk_adjustments": [],
            "dynamic_risk_summary": None,
            "execution_settings": (
                self.execution_engine.get_execution_settings()
                if hasattr(self, "execution_engine")
                else {}
            ),
        }

        # Add regime switching results
        if self.regime_handler:
            results.update(self.regime_handler.get_results())
            results["final_strategy"] = self.strategy.name
        else:
            results.update(RegimeHandler.get_disabled_results())

        return results

    def _build_final_results(
        self,
        df: pd.DataFrame,
        start: datetime,
        end: datetime | None,
        balance_history: list[tuple],
        yearly_balance: dict[int, dict[str, float]],
        hold_return: float,
    ) -> dict:
        """Build final backtest results dictionary."""
        # Calculate prediction metrics
        pred_metrics = self._calculate_prediction_metrics(df)

        # Get performance metrics from tracker
        perf_metrics = self.performance_tracker.get_metrics()

        # Calculate yearly returns
        yearly_returns = {}
        for yr, bal in yearly_balance.items():
            if bal["start"] > 0:
                yearly_returns[str(yr)] = (bal["end"] / bal["start"] - 1) * 100

        # End trading session
        self.event_logger.end_trading_session(self.balance)

        # Warn about pending entry
        pending = self.execution_engine.clear_pending_entry()
        if pending:
            logger.warning(
                "Backtest ended with pending %s entry that was never executed.",
                pending.get("side", "unknown"),
            )

        # Build results using performance tracker metrics
        results = {
            # Core metrics from tracker
            "total_trades": perf_metrics.total_trades,
            "win_rate": perf_metrics.win_rate * 100,  # Convert to percentage for backward compat
            "total_return": perf_metrics.total_return_pct,
            "max_drawdown": perf_metrics.max_drawdown * 100,  # Convert to percentage
            "sharpe_ratio": perf_metrics.sharpe_ratio,
            "final_balance": self.balance,
            "annualized_return": perf_metrics.annualized_return,
            # New metrics from tracker
            "sortino_ratio": perf_metrics.sortino_ratio,
            "calmar_ratio": perf_metrics.calmar_ratio,
            "var_95": perf_metrics.var_95,
            "var_95_note": (
                "Requires 30+ days of data for statistical validity"
                if perf_metrics.var_95 == 0.0
                else None
            ),
            "expectancy": perf_metrics.expectancy,
            "profit_factor": perf_metrics.profit_factor,
            "avg_win": perf_metrics.avg_win,
            "avg_loss": perf_metrics.avg_loss,
            "largest_win": perf_metrics.largest_win,
            "largest_loss": perf_metrics.largest_loss,
            "avg_trade_duration_hours": perf_metrics.avg_trade_duration_hours,
            "consecutive_wins": perf_metrics.consecutive_wins,
            "consecutive_losses": perf_metrics.consecutive_losses,
            # Backtest-specific metrics
            "yearly_returns": yearly_returns,
            "hold_return": hold_return,
            "trading_vs_hold_difference": perf_metrics.total_return_pct - hold_return,
            "session_id": self.trading_session_id if self.log_to_database else None,
            "early_stop_reason": self.early_stop_reason,
            "early_stop_date": self.early_stop_date,
            "early_stop_candle_index": self.early_stop_candle_index,
            "prediction_metrics": pred_metrics,
            "dynamic_risk_adjustments": (
                self.dynamic_risk_adjustments if self.enable_dynamic_risk else []
            ),
            "dynamic_risk_summary": (
                self._summarize_dynamic_risk_adjustments() if self.enable_dynamic_risk else None
            ),
            "total_fees": perf_metrics.total_fees_paid,
            "total_slippage_cost": perf_metrics.total_slippage_cost,
            "execution_settings": self.execution_engine.get_execution_settings(),
        }

        # Add regime switching results
        if self.regime_handler:
            results.update(self.regime_handler.get_results())
            results["final_strategy"] = self.strategy.name
        else:
            results.update(RegimeHandler.get_disabled_results())

        return results

    def _calculate_prediction_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate prediction accuracy metrics if predictions are present."""
        pred_acc = 0.0
        mae = 0.0
        mape = 0.0
        brier = 0.0

        try:
            if "onnx_pred" in df.columns:
                pred_series = df["onnx_pred"].dropna()
                actual_series = df["close"].reindex(pred_series.index)

                from src.performance.metrics import (
                    brier_score_direction,
                    directional_accuracy,
                    mean_absolute_error,
                    mean_absolute_percentage_error,
                )

                mae = mean_absolute_error(pred_series, actual_series)
                mape = mean_absolute_percentage_error(pred_series, actual_series)
                pred_acc = directional_accuracy(pred_series, actual_series)

                if "prediction_confidence" in df.columns:
                    p_up = (pred_series.shift(1) < pred_series).astype(float) * df[
                        "prediction_confidence"
                    ].reindex(pred_series.index).fillna(0.5) + (
                        pred_series.shift(1) >= pred_series
                    ).astype(
                        float
                    ) * (
                        1.0
                        - df["prediction_confidence"]
                        .reindex(pred_series.index)
                        .fillna(DEFAULT_CONFIDENCE_SCORE)
                    )
                    actual_up = (actual_series.diff() > 0).astype(float)
                    brier = brier_score_direction(p_up.fillna(0.5), actual_up.fillna(0.0))
        except Exception as e:
            logger.warning("Failed to calculate prediction metrics: %s", e)

        return {
            "directional_accuracy_pct": pred_acc,
            "mae": mae,
            "mape_pct": mape,
            "brier_score_direction": brier,
        }

    def _summarize_dynamic_risk_adjustments(self) -> dict:
        """Summarize dynamic risk adjustments for backtest results."""
        if not self.dynamic_risk_adjustments:
            return {
                "total_adjustments": 0,
                "adjustment_frequency": 0.0,
                "average_factor": 1.0,
                "most_common_reason": None,
                "max_reduction": 0.0,
                "time_under_adjustment": 0.0,
            }

        factors = [adj["position_size_factor"] for adj in self.dynamic_risk_adjustments]
        reasons = [adj["primary_reason"] for adj in self.dynamic_risk_adjustments]
        reason_counts = Counter(reasons)

        return {
            "total_adjustments": len(self.dynamic_risk_adjustments),
            "adjustment_frequency": len(self.dynamic_risk_adjustments) / 100.0,
            "average_factor": sum(factors) / len(factors) if factors else 1.0,
            "most_common_reason": reason_counts.most_common(1)[0][0] if reason_counts else None,
            "max_reduction": 1.0 - min(factors) if factors else 0.0,
            "time_under_adjustment": len(self.dynamic_risk_adjustments) / 100.0,
            "reason_breakdown": dict(reason_counts),
        }

    # Backward compatibility helper methods
    def _extract_indicators(self, df: pd.DataFrame, index: int) -> dict:
        """Extract indicators from DataFrame row."""
        return util_extract_indicators(df, index)

    def _extract_sentiment_data(self, df: pd.DataFrame, index: int) -> dict:
        """Extract sentiment data from DataFrame row."""
        return util_extract_sentiment(df, index)

    def _extract_ml_predictions(self, df: pd.DataFrame, index: int) -> dict:
        """Extract ML predictions from DataFrame row."""
        return util_extract_ml(df, index)

    def _get_dynamic_risk_adjusted_size(
        self, original_size: float, current_time: datetime
    ) -> float:
        """Apply dynamic risk adjustments to position size (backward compatibility).

        This method is deprecated. The functionality has been moved to
        EntryHandler._apply_dynamic_risk().

        Args:
            original_size: Original position size fraction.
            current_time: Current timestamp.

        Returns:
            Adjusted position size fraction.
        """
        if not self.dynamic_risk_manager:
            return original_size

        return self.entry_handler._apply_dynamic_risk(
            original_size=original_size,
            current_time=current_time,
            balance=self.balance,
            peak_balance=self.peak_balance,
            trading_session_id=self.trading_session_id,
        )

    def _update_peak_balance(self) -> None:
        """Update peak balance for drawdown tracking (backward compatibility).

        This method is deprecated. Peak balance is now automatically updated
        in the main backtest loop.
        """
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

    def _deduct_entry_fee(self, entry_fee: float) -> None:
        """Deduct entry fee from balance with validation.

        Validates that entry fee is non-negative, does not exceed balance,
        and that balance remains non-negative after deduction. Raises
        RuntimeError on validation failure to prevent balance corruption.

        Args:
            entry_fee: The fee to deduct from balance (must be >= 0).

        Raises:
            RuntimeError: If entry fee is negative, exceeds balance,
                or balance goes negative after deduction.
        """
        if entry_fee < 0:
            logger.critical(
                "Entry fee %.8f is negative - fee calculation error!",
                entry_fee,
            )
            raise RuntimeError(f"Invalid entry fee {entry_fee:.8f}: fees cannot be negative")

        if entry_fee > self.balance:
            logger.critical(
                "Entry fee %.2f exceeds balance %.2f - position sizing error!",
                entry_fee,
                self.balance,
            )
            raise RuntimeError(
                f"Critical error: Entry fee {entry_fee:.2f} "
                f"exceeds balance {self.balance:.2f}. Aborting to prevent corruption."
            )

        self.balance -= entry_fee

        if self.balance < 0:
            logger.critical(
                "Balance went negative: %.2f after entry fee %.2f",
                self.balance,
                entry_fee,
            )
            raise RuntimeError(f"Balance corruption: balance={self.balance:.2f}")

    def _apply_correlation_control(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        index: int,
        candidate_fraction: float,
    ) -> float:
        """Apply correlation control to candidate position size (backward compatibility).

        This method is deprecated. The functionality has been moved to
        CorrelationHandler.apply_correlation_control().

        Args:
            symbol: Trading symbol.
            timeframe: Candle timeframe.
            df: DataFrame with market data.
            index: Current candle index.
            candidate_fraction: Proposed position size fraction.

        Returns:
            Adjusted position size fraction.
        """
        if self.correlation_handler is None:
            return candidate_fraction

        return self.correlation_handler.apply_correlation_control(
            symbol=symbol,
            timeframe=timeframe,
            df=df,
            index=index,
            candidate_fraction=candidate_fraction,
        )

    def _load_strategy_by_name(self, strategy_name: str):
        """Load strategy by name (backward compatibility).

        This method is deprecated. The functionality has been moved to
        RegimeHandler._load_strategy().

        Args:
            strategy_name: Name of strategy to load.

        Returns:
            Strategy instance, or None on failure.
        """
        if self.regime_handler is None:
            return None

        return self.regime_handler._load_strategy(strategy_name)
