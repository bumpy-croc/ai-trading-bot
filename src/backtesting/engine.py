"""
Backtesting engine for strategy evaluation.

This module provides a comprehensive backtesting framework.
"""

import logging
from datetime import datetime
from typing import Any, Optional

import pandas as pd  # type: ignore
from pandas import DataFrame  # type: ignore
from performance.metrics import cash_pnl
from regime.detector import RegimeDetector
from sqlalchemy.exc import SQLAlchemyError

from backtesting.models import Trade as CompletedTrade
from backtesting.utils import (
    compute_performance_metrics,
)
from backtesting.utils import (
    extract_indicators as util_extract_indicators,
)
from backtesting.utils import (
    extract_ml_predictions as util_extract_ml,
)
from backtesting.utils import (
    extract_sentiment_data as util_extract_sentiment,
)
from config.config_manager import get_config
from config.constants import DEFAULT_INITIAL_BALANCE
from data_providers.data_provider import DataProvider
from data_providers.sentiment_provider import SentimentDataProvider
from database.manager import DatabaseManager
from database.models import TradeSource
from performance.metrics import cash_pnl
from position_management.dynamic_risk import DynamicRiskManager, DynamicRiskConfig
from regime.detector import RegimeDetector
from risk.risk_manager import RiskManager
from strategies.base import BaseStrategy
from position_management.time_exits import TimeExitPolicy
from position_management.partial_manager import PartialExitPolicy, PositionState

logger = logging.getLogger(__name__)


class ActiveTrade:
    """Represents an active trade during backtest iteration"""

    def __init__(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        entry_time: datetime,
        size: float,
        stop_loss: float,
        take_profit: Optional[float] = None,
    ):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.size = min(size, 1.0)  # Limit position size to 100% of balance (fraction)
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.exit_reason: Optional[str] = None
        # Partial operations runtime state
        self.original_size: float = self.size
        self.current_size: float = self.size
        self.partial_exits_taken: int = 0
        self.scale_ins_taken: int = 0


class Backtester:
    """Backtesting engine for trading strategies"""

    def __init__(
        self,
        strategy: BaseStrategy,
        data_provider: DataProvider,
        sentiment_provider: Optional[SentimentDataProvider] = None,
        risk_parameters: Optional[Any] = None,
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
        enable_short_trading: bool = False,
        database_url: Optional[str] = None,
        log_to_database: Optional[bool] = None,
        enable_time_limit_exit: bool = False,
        default_take_profit_pct: Optional[float] = None,
        legacy_stop_loss_indexing: bool = True,  # Preserve historical behavior by default
        enable_engine_risk_exits: bool = False,  # Enforce engine-level SL/TP exits (off to preserve baseline)
        time_exit_policy: TimeExitPolicy | None = None,
        # Dynamic risk management
        enable_dynamic_risk: bool = False,  # Disabled by default for backtesting to preserve historical results
        dynamic_risk_config: Optional[DynamicRiskConfig] = None,
        partial_manager: Optional[PartialExitPolicy] = None,
        enable_partial_operations: bool = False,
    ):
        self.strategy = strategy
        self.data_provider = data_provider
        self.sentiment_provider = sentiment_provider
        self.risk_parameters = risk_parameters
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.trades: list[dict] = []
        self.current_trade: Optional[ActiveTrade] = None
        self.dynamic_risk_adjustments: list[dict] = []  # Track dynamic risk adjustments

        # Dynamic risk management
        self.enable_dynamic_risk = enable_dynamic_risk
        self.dynamic_risk_manager = None
        self.session_id = None  # Will be set when database is available
        if enable_dynamic_risk:
            config = dynamic_risk_config or DynamicRiskConfig()
            # Merge strategy overrides with base config
            final_config = self._merge_dynamic_risk_config(config, strategy)
            # Initialize without db_manager for now - will be set later if database logging is enabled
            self.dynamic_risk_manager = DynamicRiskManager(final_config, db_manager=None)

        # Feature flags for parity tuning
        self.enable_short_trading = enable_short_trading
        self.enable_time_limit_exit = enable_time_limit_exit
        self.default_take_profit_pct = default_take_profit_pct
        self.legacy_stop_loss_indexing = legacy_stop_loss_indexing
        self.enable_engine_risk_exits = enable_engine_risk_exits

        # Risk manager (parity with live engine)
        self.risk_manager = RiskManager(risk_parameters)
        # Partial operations policy
        if partial_manager is not None:
            self.partial_manager = partial_manager
        elif enable_partial_operations:
            rp = self.risk_manager.params
            self.partial_manager = PartialExitPolicy(
                exit_targets=rp.partial_exit_targets or [],
                exit_sizes=rp.partial_exit_sizes or [],
                scale_in_thresholds=rp.scale_in_thresholds or [],
                scale_in_sizes=rp.scale_in_sizes or [],
                max_scale_ins=rp.max_scale_ins,
            )
        else:
            self.partial_manager = None

        # Regime detector (always available for analytics/tests)
        try:
            self.regime_detector = RegimeDetector()
        except Exception:
            self.regime_detector = None

        # Early stop tracking
        self.early_stop_reason: Optional[str] = None
        self.early_stop_date: Optional[datetime] = None
        self.early_stop_candle_index: Optional[int] = None
        # Use legacy 50% drawdown threshold unless explicit risk params provided, to preserve historical parity
        self._early_stop_max_drawdown = (
            self.risk_manager.params.max_drawdown if risk_parameters is not None else 0.5
        )

        # Database logging
        # Auto-detect test environment and default log_to_database accordingly
        if log_to_database is None:
            # Default to False in test environments (when DATABASE_URL is SQLite or not set)
            import os

            database_url_env = os.getenv("DATABASE_URL", "")
            # More reliable pytest detection using PYTEST_CURRENT_TEST
            is_pytest = os.environ.get("PYTEST_CURRENT_TEST") is not None
            log_to_database = not (
                database_url_env.startswith("sqlite://") or database_url_env == "" or is_pytest
            )

        self.log_to_database = log_to_database
        self.db_manager = None
        self.trading_session_id = None
        if log_to_database:
            try:
                # Prefer production DB for backtest persistence by default
                selected_db_url = database_url
                if selected_db_url is None:
                    try:
                        cfg = get_config()
                        selected_db_url = cfg.get("PRODUCTION_DATABASE_URL")
                    except Exception:
                        selected_db_url = None

                self.db_manager = DatabaseManager(selected_db_url)
                # Set up strategy logging
                if self.db_manager:
                    self.strategy.set_database_manager(self.db_manager)
            except (SQLAlchemyError, ValueError) as db_err:
                # Fallback to in-memory SQLite to satisfy tests that expect db_manager presence
                logger.warning(
                    f"Database connection failed ({db_err}). Falling back to in-memory SQLite database for logging."
                )
                try:
                    self.db_manager = DatabaseManager("sqlite:///:memory:")
                    if self.db_manager:
                        self.strategy.set_database_manager(self.db_manager)
                except (SQLAlchemyError, ValueError) as sqlite_err:
                    logger.warning(
                        f"Fallback SQLite initialization failed ({sqlite_err}). Disabling database logging."
                    )
                    self.log_to_database = False

                    class DummyDBManager:
                        def __getattr__(self, _):
                            def _noop(*args, **kwargs):
                                return None

                            return _noop

                    self.db_manager = DummyDBManager()

    def _merge_dynamic_risk_config(self, base_config: DynamicRiskConfig, strategy) -> DynamicRiskConfig:
        """Merge strategy risk overrides with base dynamic risk configuration"""
        try:
            # Get strategy risk overrides
            strategy_overrides = strategy.get_risk_overrides() if strategy else None
            if not strategy_overrides or 'dynamic_risk' not in strategy_overrides:
                return base_config
                
            dynamic_overrides = strategy_overrides['dynamic_risk']
            
            # Create a new config with merged values
            merged_config = DynamicRiskConfig(
                enabled=dynamic_overrides.get('enabled', base_config.enabled),
                performance_window_days=dynamic_overrides.get('performance_window_days', base_config.performance_window_days),
                drawdown_thresholds=dynamic_overrides.get('drawdown_thresholds', base_config.drawdown_thresholds),
                risk_reduction_factors=dynamic_overrides.get('risk_reduction_factors', base_config.risk_reduction_factors),
                recovery_thresholds=dynamic_overrides.get('recovery_thresholds', base_config.recovery_thresholds),
                volatility_adjustment_enabled=dynamic_overrides.get('volatility_adjustment_enabled', base_config.volatility_adjustment_enabled),
                volatility_window_days=dynamic_overrides.get('volatility_window_days', base_config.volatility_window_days),
                high_volatility_threshold=dynamic_overrides.get('high_volatility_threshold', base_config.high_volatility_threshold),
                low_volatility_threshold=dynamic_overrides.get('low_volatility_threshold', base_config.low_volatility_threshold),
                volatility_risk_multipliers=dynamic_overrides.get('volatility_risk_multipliers', base_config.volatility_risk_multipliers)
            )
            
            # Note: Using print since logger might not be available yet during initialization
            print(f"Merged strategy dynamic risk overrides from {strategy.__class__.__name__}")
            return merged_config
            
        except Exception as e:
            print(f"Failed to merge strategy dynamic risk overrides: {e}")
            return base_config

    def _get_dynamic_risk_adjusted_size(self, original_size: float, current_time: datetime) -> float:
        """Apply dynamic risk adjustments to position size in backtesting"""
        if not self.dynamic_risk_manager:
            return original_size
            
        try:
            # Calculate dynamic risk adjustments
            adjustments = self.dynamic_risk_manager.calculate_dynamic_risk_adjustments(
                current_balance=self.balance,
                peak_balance=self.peak_balance,
                session_id=self.trading_session_id
            )
            
            # Apply position size adjustment
            adjusted_size = original_size * adjustments.position_size_factor
            
            # Log significant adjustments for analysis
            if abs(adjustments.position_size_factor - 1.0) > 0.1:  # >10% change
                logger.debug(
                    f"Dynamic risk adjustment at {current_time}: "
                    f"size factor={adjustments.position_size_factor:.2f}, "
                    f"reason={adjustments.primary_reason}"
                )
                
                # Track adjustment for backtest results
                self.dynamic_risk_adjustments.append({
                    'timestamp': current_time,
                    'position_size_factor': adjustments.position_size_factor,
                    'stop_loss_tightening': adjustments.stop_loss_tightening,
                    'daily_risk_factor': adjustments.daily_risk_factor,
                    'primary_reason': adjustments.primary_reason,
                    'current_drawdown': adjustments.adjustment_details.get('current_drawdown'),
                    'balance': self.balance,
                    'peak_balance': self.peak_balance,
                    'original_size': original_size,
                    'adjusted_size': adjusted_size
                })
            
            return adjusted_size
            
        except Exception as e:
            logger.warning(f"Failed to apply dynamic risk adjustment: {e}")
            return original_size

    def _update_peak_balance(self):
        """Update peak balance for drawdown tracking"""
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

    def run(
        self, symbol: str, timeframe: str, start: datetime, end: Optional[datetime] = None
    ) -> dict:
        """Run backtest with sentiment data if available"""
        try:
            # Create trading session in database if enabled
            if self.log_to_database and self.db_manager:
                self.trading_session_id = self.db_manager.create_trading_session(
                    strategy_name=self.strategy.__class__.__name__,
                    symbol=symbol,
                    timeframe=timeframe,
                    mode=TradeSource.BACKTEST,
                    initial_balance=self.initial_balance,
                    strategy_config=getattr(self.strategy, "config", {}),
                    session_name=f"Backtest_{symbol}_{start.strftime('%Y%m%d')}",
                )

                # Set session ID on strategy for logging
                if hasattr(self.strategy, "session_id"):
                    self.strategy.session_id = self.trading_session_id

                # Update dynamic risk manager with database connection
                if self.enable_dynamic_risk and self.dynamic_risk_manager and self.db_manager:
                    self.dynamic_risk_manager.db_manager = self.db_manager

            # Fetch price data
            df: DataFrame = self.data_provider.get_historical_data(symbol, timeframe, start, end)
            if df.empty:
                # Return empty results for empty data
                return {
                    "total_trades": 0,
                    "final_balance": self.initial_balance,
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "win_rate": 0.0,
                    "avg_trade_duration": 0.0,
                    "total_fees": 0.0,
                    "trades": [],
                }

            # Validate required columns
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Validate index type - must be datetime-like for time-series analysis
            if not isinstance(df.index, pd.DatetimeIndex):
                # Try to convert to datetime index if possible
                try:
                    df.index = pd.to_datetime(df.index)
                except (ValueError, TypeError):
                    # If conversion fails, create a dummy datetime index
                    df.index = pd.date_range(start=start, periods=len(df), freq="h")

            # Fetch/merge sentiment data if provider is available
            if self.sentiment_provider:
                df = self._merge_sentiment_data(df, symbol, timeframe, start, end)

            # Calculate indicators
            df = self.strategy.calculate_indicators(df)

            # Remove warmup period - only drop rows where essential price data is missing
            # Don't drop rows just because ML predictions or sentiment data is missing
            essential_columns = ["open", "high", "low", "close", "volume"]
            df = df.dropna(subset=essential_columns)

            logger.info(f"Starting backtest with {len(df)} candles")

            # Preserve legacy behavior: enforce long-only unless explicit flag is set
            if not self.enable_short_trading:
                if hasattr(self.strategy, "check_short_entry_conditions"):
                    # No change to strategy; engine will simply skip short entries via flag
                    pass

            # -----------------------------
            # Metrics & tracking variables
            # -----------------------------
            total_trades = 0
            winning_trades = 0
            max_drawdown_running = 0  # interim tracker (still used for intra-loop stopping)

            # Track balance over time to enable robust performance stats
            balance_history: list[tuple] = []  # (timestamp, balance)

            # Helper dict to track first/last balance of each calendar year
            yearly_balance: dict[int, dict[str, float]] = {}

            # Iterate through candles
            for i in range(len(df)):
                candle = df.iloc[i]
                current_time = candle.name
                current_price = float(candle["close"])

                # Record current balance for time-series analytics
                balance_history.append((current_time, self.balance))

                # Track yearly start/end balances for return calc
                yr = current_time.year
                if yr not in yearly_balance:
                    yearly_balance[yr] = {"start": self.balance, "end": self.balance}
                else:
                    yearly_balance[yr]["end"] = self.balance

                # Update max drawdown
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance
                current_drawdown = (
                    (self.peak_balance - self.balance) / self.peak_balance
                    if self.peak_balance > 0
                    else 0.0
                )
                max_drawdown_running = max(max_drawdown_running, current_drawdown)

                # Check for exit if in position
                if self.current_trade is not None:
                    exit_signal = self.strategy.check_exit_conditions(
                        df, i, self.current_trade.entry_price
                    )

                    # Evaluate partial exits and scale-ins before full exit
                    try:
                        if self.partial_manager is None:
                            raise RuntimeError("partial_ops_disabled")
                        state = PositionState(
                            entry_price=self.current_trade.entry_price,
                            side=self.current_trade.side,
                            original_size=self.current_trade.original_size,
                            current_size=self.current_trade.current_size,
                            partial_exits_taken=self.current_trade.partial_exits_taken,
                            scale_ins_taken=self.current_trade.scale_ins_taken,
                        )
                        # Partial exits cascade
                        actions = self.partial_manager.check_partial_exits(state, current_price)
                        for act in actions:
                            # Translate to fraction-of-balance using original size
                            exec_of_original = float(act["size"])  # fraction of ORIGINAL
                            delta = exec_of_original * state.original_size
                            exec_frac = min(delta, state.current_size)
                            if exec_frac <= 0:
                                continue
                            # Realize PnL for the exited fraction
                            move = (
                                (current_price - self.current_trade.entry_price)
                                / self.current_trade.entry_price
                                if self.current_trade.side == "long"
                                else (self.current_trade.entry_price - current_price)
                                / self.current_trade.entry_price
                            )
                            pnl_pct = move * exec_frac
                            pnl_cash = cash_pnl(pnl_pct, self.balance)
                            self.balance += pnl_cash
                            # Update state
                            state.current_size = max(0.0, state.current_size - exec_frac)
                            state.partial_exits_taken += 1
                            self.current_trade.current_size = state.current_size
                            self.current_trade.partial_exits_taken = state.partial_exits_taken
                            # Risk manager update
                            try:
                                self.risk_manager.adjust_position_after_partial_exit(symbol, exec_frac)
                            except Exception:
                                pass
                            # Optional DB logging of partial operations could be added later for backtests

                        # Scale-in opportunity
                        scale = self.partial_manager.check_scale_in_opportunity(state, current_price, self._extract_indicators(df, i))
                        if scale is not None:
                            add_of_original = float(scale["size"])  # fraction of ORIGINAL
                            if add_of_original > 0:
                                # Enforce caps
                                delta_add = add_of_original * state.original_size
                                remaining_daily = max(0.0, self.risk_manager.params.max_daily_risk - self.risk_manager.daily_risk_used)
                                add_effective = min(delta_add, remaining_daily)
                                if add_effective > 0:
                                    state.current_size = min(1.0, state.current_size + add_effective)
                                    self.current_trade.current_size = state.current_size
                                    self.current_trade.size = min(1.0, self.current_trade.size + add_effective)
                                    state.scale_ins_taken += 1
                                    self.current_trade.scale_ins_taken = state.scale_ins_taken
                                    try:
                                        self.risk_manager.adjust_position_after_scale_in(symbol, add_effective)
                                    except Exception:
                                        pass
                    except Exception as e:
                        if str(e) != "partial_ops_disabled":
                            logger.debug(f"Partial ops evaluation skipped/failed: {e}")

                    # Additional parity checks: stop loss, take profit, and time-limit
                    hit_stop_loss = False
                    hit_take_profit = False
                    if self.enable_engine_risk_exits and self.current_trade.stop_loss is not None:
                        if self.current_trade.side == "long":
                            hit_stop_loss = current_price <= float(self.current_trade.stop_loss)
                        else:
                            hit_stop_loss = current_price >= float(self.current_trade.stop_loss)
                    if self.enable_engine_risk_exits and self.current_trade.take_profit is not None:
                        if self.current_trade.side == "long":
                            hit_take_profit = current_price >= float(self.current_trade.take_profit)
                        else:
                            hit_take_profit = current_price <= float(self.current_trade.take_profit)
                    hit_time_limit = False
                    if self.enable_time_limit_exit:
                        if self.time_exit_policy is not None:
                            try:
                                should_exit, _ = self.time_exit_policy.check_time_exit_conditions(
                                    self.current_trade.entry_time, current_time
                                )
                                hit_time_limit = should_exit
                            except Exception:
                                hit_time_limit = False
                        else:
                            hit_time_limit = (
                                (current_time - self.current_trade.entry_time).total_seconds() > 86400
                            )

                    should_exit = exit_signal or hit_stop_loss or hit_take_profit or hit_time_limit
                    exit_reason = (
                        "Strategy signal"
                        if exit_signal
                        else (
                            "Stop loss"
                            if hit_stop_loss
                            else (
                                "Take profit"
                                if hit_take_profit
                                else "Time limit" if hit_time_limit else "Hold"
                            )
                        )
                    )

                    # Log exit decision
                    if self.log_to_database and self.db_manager:
                        indicators = self._extract_indicators(df, i)
                        sentiment_data = self._extract_sentiment_data(df, i)
                        ml_predictions = self._extract_ml_predictions(df, i)

                        # Calculate current P&L for context (percentage vs entry)
                        current_pnl_pct = (
                            current_price - self.current_trade.entry_price
                        ) / self.current_trade.entry_price
                        if self.current_trade.side != "long":
                            current_pnl_pct = -current_pnl_pct

                        self.db_manager.log_strategy_execution(
                            strategy_name=self.strategy.__class__.__name__,
                            symbol=symbol,
                            signal_type="exit",
                            action_taken="closed_position" if should_exit else "hold_position",
                            price=current_price,
                            timeframe=timeframe,
                            signal_strength=1.0 if should_exit else 0.0,
                            confidence_score=indicators.get("prediction_confidence", 0.5),
                            indicators=indicators,
                            sentiment_data=sentiment_data if sentiment_data else None,
                            ml_predictions=ml_predictions if ml_predictions else None,
                            position_size=self.current_trade.size,
                            reasons=[
                                exit_reason if should_exit else "holding_position",
                                f"current_pnl_{current_pnl_pct:.4f}",
                                f"position_age_{(current_time - self.current_trade.entry_time).total_seconds():.0f}s",
                                f"entry_price_{self.current_trade.entry_price:.2f}",
                            ],
                            volume=indicators.get("volume"),
                            volatility=indicators.get("volatility"),
                            session_id=self.trading_session_id,
                        )

                    if should_exit:
                        # Close the trade
                        exit_price = current_price
                        exit_time = current_time

                        # Calculate PnL percent (sized) and then convert to cash
                        fraction = float(getattr(self.current_trade, "current_size", self.current_trade.size))
                        if self.current_trade.side == "long":
                            trade_pnl_pct = (
                                (exit_price - self.current_trade.entry_price)
                                / self.current_trade.entry_price
                            ) * fraction
                        else:
                            trade_pnl_pct = (
                                (self.current_trade.entry_price - exit_price)
                                / self.current_trade.entry_price
                            ) * fraction
                        trade_pnl_cash = cash_pnl(trade_pnl_pct, self.balance)

                        # Update balance
                        self.balance += trade_pnl_cash

                        # Update peak balance for drawdown tracking
                        self._update_peak_balance()

                        # Update metrics
                        total_trades += 1
                        if trade_pnl_cash > 0:
                            winning_trades += 1

                        # Log trade
                        logger.info(
                            f"Exited {self.current_trade.side} at {current_price}, Balance: {self.balance:.2f}"
                        )

                        # After updating self.balance, update yearly_balance for the exit year
                        exit_year = exit_time.year
                        if exit_year in yearly_balance:
                            yearly_balance[exit_year]["end"] = self.balance

                        # Log to database if enabled
                        if self.log_to_database and self.db_manager:
                            self.db_manager.log_trade(
                                symbol=symbol,
                                side=self.current_trade.side,
                                entry_price=self.current_trade.entry_price,
                                exit_price=exit_price,
                                size=fraction,
                                entry_time=self.current_trade.entry_time,
                                exit_time=exit_time,
                                pnl=trade_pnl_cash,
                                exit_reason=exit_reason,
                                strategy_name=self.strategy.__class__.__name__,
                                source=TradeSource.BACKTEST,
                                stop_loss=self.current_trade.stop_loss,
                                take_profit=self.current_trade.take_profit,
                                session_id=self.trading_session_id,
                            )

                        # Store completed trade
                        self.trades.append(
                            CompletedTrade(
                                symbol=symbol,
                                side=self.current_trade.side,
                                entry_price=self.current_trade.entry_price,
                                exit_price=exit_price,
                                entry_time=self.current_trade.entry_time,
                                exit_time=exit_time,
                                size=fraction,
                                pnl=trade_pnl_cash,
                                exit_reason=exit_reason,
                                stop_loss=self.current_trade.stop_loss,
                                take_profit=self.current_trade.take_profit,
                            )
                        )

                        # Notify risk manager to close tracked position
                        try:
                            self.risk_manager.close_position(symbol)
                        except Exception as e:
                            logger.warning(
                                f"Failed to update risk manager on close for {symbol}: {e}"
                            )

                        self.current_trade = None

                        # Check if maximum drawdown exceeded (use risk params if present)
                        max_dd_threshold = self._early_stop_max_drawdown
                        if current_drawdown > max_dd_threshold:
                            self.early_stop_reason = (
                                f"Maximum drawdown exceeded ({current_drawdown:.1%})"
                            )
                            self.early_stop_date = current_time
                            self.early_stop_candle_index = i
                            logger.warning(
                                f"Maximum drawdown exceeded ({current_drawdown:.1%}). Stopping backtest."
                            )
                            break

                # Check for entry if not in position
                elif self.strategy.check_entry_conditions(df, i):
                    # Calculate position size (as fraction of balance)
                    try:
                        overrides = (
                            self.strategy.get_risk_overrides()
                            if hasattr(self.strategy, "get_risk_overrides")
                            else None
                        )
                    except Exception:
                        overrides = None
                    if overrides and overrides.get("position_sizer"):
                        size_fraction = self.risk_manager.calculate_position_fraction(
                            df=df,
                            index=i,
                            balance=self.balance,
                            price=current_price,
                            indicators=self._extract_indicators(df, i),
                            strategy_overrides=overrides,
                        )
                    else:
                        size_fraction = self.strategy.calculate_position_size(df, i, self.balance)

                    # Apply dynamic risk adjustments
                    if size_fraction > 0 and self.enable_dynamic_risk:
                        size_fraction = self._get_dynamic_risk_adjusted_size(
                            size_fraction, current_time
                        )

                    # Log entry decision
                    if self.log_to_database and self.db_manager:
                        indicators = self._extract_indicators(df, i)
                        sentiment_data = self._extract_sentiment_data(df, i)
                        ml_predictions = self._extract_ml_predictions(df, i)

                        self.db_manager.log_strategy_execution(
                            strategy_name=self.strategy.__class__.__name__,
                            symbol=symbol,
                            signal_type="entry",
                            action_taken="opened_long" if size_fraction > 0 else "no_action",
                            price=current_price,
                            timeframe=timeframe,
                            signal_strength=1.0 if size_fraction > 0 else 0.0,
                            confidence_score=indicators.get("prediction_confidence", 0.5),
                            indicators=indicators,
                            sentiment_data=sentiment_data if sentiment_data else None,
                            ml_predictions=ml_predictions if ml_predictions else None,
                            position_size=size_fraction if size_fraction > 0 else None,
                            reasons=[
                                "entry_conditions_met",
                                (
                                    f"position_size_{size_fraction:.4f}"
                                    if size_fraction > 0
                                    else "no_position_size"
                                ),
                                f"balance_{self.balance:.2f}",
                            ],
                            volume=indicators.get("volume"),
                            volatility=indicators.get("volatility"),
                            session_id=self.trading_session_id,
                        )

                    if size_fraction > 0:
                        # Enter new trade
                        # Optionally use legacy indexing behavior for stop-loss calculation to preserve parity
                        sl_index = (len(df) - 1) if self.legacy_stop_loss_indexing else i
                        try:
                            overrides = (
                                self.strategy.get_risk_overrides()
                                if hasattr(self.strategy, "get_risk_overrides")
                                else None
                            )
                        except Exception:
                            overrides = None
                        if overrides and (
                            ("stop_loss_pct" in overrides) or ("take_profit_pct" in overrides)
                        ):
                            stop_loss, take_profit = self.risk_manager.compute_sl_tp(
                                df=df,
                                index=sl_index,
                                entry_price=current_price,
                                side="long",
                                strategy_overrides=overrides,
                            )
                            if take_profit is None:
                                tp_pct = (
                                    self.default_take_profit_pct
                                    if self.default_take_profit_pct is not None
                                    else getattr(self.strategy, "take_profit_pct", 0.04)
                                )
                                take_profit = current_price * (1 + tp_pct)
                        else:
                            stop_loss = self.strategy.calculate_stop_loss(
                                df, sl_index, current_price, "long"
                            )
                            tp_pct = (
                                self.default_take_profit_pct
                                if self.default_take_profit_pct is not None
                                else getattr(self.strategy, "take_profit_pct", 0.04)
                            )
                            take_profit = current_price * (1 + tp_pct)
                        self.current_trade = ActiveTrade(
                            symbol=symbol,
                            side="long",
                            entry_price=current_price,
                            entry_time=current_time,
                            size=size_fraction,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                        )
                        logger.info(f"Entered long position at {current_price}")

                        # Update risk manager with opened position to track daily risk usage
                        try:
                            self.risk_manager.update_position(
                                symbol=symbol,
                                side="long",
                                size=size_fraction,
                                entry_price=current_price,
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to update risk manager for opened long on {symbol}: {e}"
                            )

                # Optional short entry if supported by strategy
                elif (
                    self.enable_short_trading
                    and hasattr(self.strategy, "check_short_entry_conditions")
                    and self.strategy.check_short_entry_conditions(df, i)
                ):
                    try:
                        overrides = (
                            self.strategy.get_risk_overrides()
                            if hasattr(self.strategy, "get_risk_overrides")
                            else None
                        )
                    except Exception:
                        overrides = None
                    if overrides and overrides.get("position_sizer"):
                        size_fraction = self.risk_manager.calculate_position_fraction(
                            df=df,
                            index=i,
                            balance=self.balance,
                            price=current_price,
                            indicators=self._extract_indicators(df, i),
                            strategy_overrides=overrides,
                        )
                    else:
                        size_fraction = self.strategy.calculate_position_size(df, i, self.balance)

                    # Apply dynamic risk adjustments
                    if size_fraction > 0 and self.enable_dynamic_risk:
                        size_fraction = self._get_dynamic_risk_adjusted_size(
                            size_fraction, current_time
                        )

                    if self.log_to_database and self.db_manager:
                        indicators = self._extract_indicators(df, i)
                        sentiment_data = self._extract_sentiment_data(df, i)
                        ml_predictions = self._extract_ml_predictions(df, i)
                        self.db_manager.log_strategy_execution(
                            strategy_name=self.strategy.__class__.__name__,
                            symbol=symbol,
                            signal_type="entry",
                            action_taken="opened_short" if size_fraction > 0 else "no_action",
                            price=current_price,
                            timeframe=timeframe,
                            signal_strength=1.0 if size_fraction > 0 else 0.0,
                            confidence_score=indicators.get("prediction_confidence", 0.5),
                            indicators=indicators,
                            sentiment_data=sentiment_data if sentiment_data else None,
                            ml_predictions=ml_predictions if ml_predictions else None,
                            position_size=size_fraction if size_fraction > 0 else None,
                            reasons=[
                                "short_entry_conditions_met",
                                (
                                    f"position_size_{size_fraction:.4f}"
                                    if size_fraction > 0
                                    else "no_position_size"
                                ),
                                f"balance_{self.balance:.2f}",
                            ],
                            volume=indicators.get("volume"),
                            volatility=indicators.get("volatility"),
                            session_id=self.trading_session_id,
                        )

                    if size_fraction > 0:
                        sl_index = (len(df) - 1) if self.legacy_stop_loss_indexing else i
                        if overrides and (
                            ("stop_loss_pct" in overrides) or ("take_profit_pct" in overrides)
                        ):
                            stop_loss, take_profit = self.risk_manager.compute_sl_tp(
                                df=df,
                                index=sl_index,
                                entry_price=current_price,
                                side="short",
                                strategy_overrides=overrides,
                            )
                            if take_profit is None:
                                tp_pct = (
                                    self.default_take_profit_pct
                                    if self.default_take_profit_pct is not None
                                    else getattr(self.strategy, "take_profit_pct", 0.04)
                                )
                                take_profit = current_price * (1 - tp_pct)
                        else:
                            stop_loss = self.strategy.calculate_stop_loss(
                                df, sl_index, current_price, "short"
                            )
                            tp_pct = (
                                self.default_take_profit_pct
                                if self.default_take_profit_pct is not None
                                else getattr(self.strategy, "take_profit_pct", 0.04)
                            )
                            take_profit = current_price * (1 - tp_pct)
                        self.current_trade = ActiveTrade(
                            symbol=symbol,
                            side="short",
                            entry_price=current_price,
                            entry_time=current_time,
                            size=size_fraction,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                        )
                        logger.info(f"Entered short position at {current_price}")

                        # Update risk manager with opened position to track daily risk usage
                        try:
                            self.risk_manager.update_position(
                                symbol=symbol,
                                side="short",
                                size=size_fraction,
                                entry_price=current_price,
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to update risk manager for opened short on {symbol}: {e}"
                            )

                # Log no-action cases (when no position and no entry signal)
                else:
                    # Only log every 10th candle to avoid spam, but capture key decision points
                    if i % 10 == 0 and self.log_to_database and self.db_manager:
                        indicators = self._extract_indicators(df, i)
                        sentiment_data = self._extract_sentiment_data(df, i)
                        ml_predictions = self._extract_ml_predictions(df, i)

                        self.db_manager.log_strategy_execution(
                            strategy_name=self.strategy.__class__.__name__,
                            symbol=symbol,
                            signal_type="entry",
                            action_taken="no_action",
                            price=current_price,
                            timeframe=timeframe,
                            signal_strength=0.0,
                            confidence_score=indicators.get("prediction_confidence", 0.5),
                            indicators=indicators,
                            sentiment_data=sentiment_data if sentiment_data else None,
                            ml_predictions=ml_predictions if ml_predictions else None,
                            reasons=[
                                "no_entry_conditions",
                                f"balance_{self.balance:.2f}",
                                f"candle_{i}_of_{len(df)}",
                            ],
                            volume=indicators.get("volume"),
                            volatility=indicators.get("volatility"),
                            session_id=self.trading_session_id,
                        )

            # Calculate final metrics
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            # ----------------------------------------------
            # Prediction accuracy metrics (if predictions present)
            # ----------------------------------------------
            pred_acc = 0.0
            mae = 0.0
            mape = 0.0
            brier = 0.0
            try:
                if "onnx_pred" in df.columns:
                    # Align predicted price at t with actual close at t
                    pred_series = df["onnx_pred"].dropna()
                    actual_series = df["close"].reindex(pred_series.index)
                    from performance.metrics import (
                        brier_score_direction,
                        directional_accuracy,
                        mean_absolute_error,
                        mean_absolute_percentage_error,
                    )

                    mae = mean_absolute_error(pred_series, actual_series)
                    mape = mean_absolute_percentage_error(pred_series, actual_series)
                    pred_acc = directional_accuracy(pred_series, actual_series)
                    # Proxy prob_up from confidence if available
                    if "prediction_confidence" in df.columns:
                        p_up = (pred_series.shift(1) < pred_series).astype(float) * df[
                            "prediction_confidence"
                        ].reindex(pred_series.index).fillna(0.5) + (
                            pred_series.shift(1) >= pred_series
                        ).astype(
                            float
                        ) * (
                            1.0 - df["prediction_confidence"].reindex(pred_series.index).fillna(0.5)
                        )
                        actual_up = (actual_series.diff() > 0).astype(float)
                        brier = brier_score_direction(p_up.fillna(0.5), actual_up.fillna(0.0))
            except Exception as e:
                # Keep zeros if any issue
                logger.warning(f"Failed to calculate brier score: {e}")
                brier = 0.0

            # Build balance history DataFrame for metrics
            bh_df = (
                pd.DataFrame(balance_history, columns=["timestamp", "balance"]).set_index(
                    "timestamp"
                )
                if balance_history
                else pd.DataFrame()
            )
            total_return, max_drawdown_pct, sharpe_ratio, annualized_return = (
                compute_performance_metrics(
                    self.initial_balance,
                    self.balance,
                    pd.Timestamp(start),
                    pd.Timestamp(end) if end else None,
                    bh_df,
                )
            )

            # ---------------------------------------------
            # Yearly returns based on account balance
            # ---------------------------------------------
            yearly_returns: dict[str, float] = {}
            for yr, bal in yearly_balance.items():
                start_bal = bal["start"]
                end_bal = bal["end"]
                if start_bal > 0:
                    yearly_returns[str(yr)] = (end_bal / start_bal - 1) * 100

            # End trading session in database if enabled
            if self.log_to_database and self.db_manager and self.trading_session_id:
                self.db_manager.end_trading_session(
                    session_id=self.trading_session_id, final_balance=self.balance
                )

            return {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "total_return": total_return,
                "max_drawdown": max_drawdown_pct,
                "sharpe_ratio": sharpe_ratio,
                "final_balance": self.balance,
                "annualized_return": annualized_return,
                "yearly_returns": yearly_returns,
                "session_id": self.trading_session_id if self.log_to_database else None,
                "early_stop_reason": self.early_stop_reason,
                "early_stop_date": self.early_stop_date,
                "early_stop_candle_index": self.early_stop_candle_index,
                "prediction_metrics": {
                    "directional_accuracy_pct": pred_acc,
                    "mae": mae,
                    "mape_pct": mape,
                    "brier_score_direction": brier,
                },
                "dynamic_risk_adjustments": self.dynamic_risk_adjustments if self.enable_dynamic_risk else [],
                "dynamic_risk_summary": self._summarize_dynamic_risk_adjustments() if self.enable_dynamic_risk else None,
            }

        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise

    def _summarize_dynamic_risk_adjustments(self) -> dict:
        """Summarize dynamic risk adjustments for backtest results"""
        if not self.dynamic_risk_adjustments:
            return {
                "total_adjustments": 0,
                "adjustment_frequency": 0.0,
                "average_factor": 1.0,
                "most_common_reason": None,
                "max_reduction": 0.0,
                "time_under_adjustment": 0.0
            }
        
        total_adjustments = len(self.dynamic_risk_adjustments)
        factors = [adj['position_size_factor'] for adj in self.dynamic_risk_adjustments]
        reasons = [adj['primary_reason'] for adj in self.dynamic_risk_adjustments]
        
        # Count reason frequencies
        from collections import Counter
        reason_counts = Counter(reasons)
        most_common_reason = reason_counts.most_common(1)[0][0] if reason_counts else None
        
        # Calculate statistics
        average_factor = sum(factors) / len(factors) if factors else 1.0
        max_reduction = 1.0 - min(factors) if factors else 0.0
        
        # Estimate time under adjustment (simplified)
        if len(self.dynamic_risk_adjustments) > 1:
            time_under_adjustment = len(self.dynamic_risk_adjustments) / 100.0  # Rough estimate
        else:
            time_under_adjustment = 0.0
        
        return {
            "total_adjustments": total_adjustments,
            "adjustment_frequency": total_adjustments / 100.0,  # Rough frequency
            "average_factor": average_factor,
            "most_common_reason": most_common_reason,
            "max_reduction": max_reduction,
            "time_under_adjustment": time_under_adjustment,
            "reason_breakdown": dict(reason_counts)
        }

    # --------------------
    # Modularized helpers
    # --------------------
    def _merge_sentiment_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime],
    ) -> pd.DataFrame:
        if not self.sentiment_provider:
            return df
        sentiment_df = self.sentiment_provider.get_historical_sentiment(symbol, start, end)
        if not sentiment_df.empty:
            sentiment_df = self.sentiment_provider.aggregate_sentiment(
                sentiment_df, window=timeframe
            )
            df = df.join(sentiment_df, how="left")
            # Forward fill sentiment scores and freshness flag for parity
            if "sentiment_score" in df.columns:
                df["sentiment_score"] = df["sentiment_score"].ffill()
                df["sentiment_score"] = df["sentiment_score"].fillna(0)
        return df

    def _extract_indicators(self, df: pd.DataFrame, index: int) -> dict:
        return util_extract_indicators(df, index)

    def _extract_sentiment_data(self, df: pd.DataFrame, index: int) -> dict:
        return util_extract_sentiment(df, index)

    def _extract_ml_predictions(self, df: pd.DataFrame, index: int) -> dict:
        return util_extract_ml(df, index)
