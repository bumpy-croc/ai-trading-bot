import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd  # type: ignore
from pandas import DataFrame  # type: ignore

# Shared performance metrics
from performance.metrics import (
    cash_pnl,
)
from regime.detector import RegimeDetector
from sqlalchemy.exc import SQLAlchemyError

from backtesting.models import Trade as CompletedTrade
from backtesting.utils import (
    compute_performance_metrics,
)

# New modular utilities and models
from backtesting.utils import (
    extract_indicators as util_extract_indicators,
)
from backtesting.utils import (
    extract_ml_predictions as util_extract_ml,
)
from backtesting.utils import (
    extract_sentiment_data as util_extract_sentiment,
)
from config.constants import DEFAULT_INITIAL_BALANCE
from data_providers.data_provider import DataProvider
from data_providers.sentiment_provider import SentimentDataProvider
from database.manager import DatabaseManager
from database.models import TradeSource
from risk.risk_manager import RiskManager
from strategies.base import BaseStrategy
from config.config_manager import get_config

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
    ):
        self.strategy = strategy
        self.data_provider = data_provider
        self.sentiment_provider = sentiment_provider
        self.risk_parameters = risk_parameters
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.trades: List[CompletedTrade] = []
        self.current_trade: Optional[ActiveTrade] = None

        # Feature flags for parity tuning
        self.enable_short_trading = enable_short_trading
        self.enable_time_limit_exit = enable_time_limit_exit
        self.default_take_profit_pct = default_take_profit_pct
        self.legacy_stop_loss_indexing = legacy_stop_loss_indexing
        self.enable_engine_risk_exits = enable_engine_risk_exits

        # Risk manager (parity with live engine)
        self.risk_manager = RiskManager(risk_parameters)
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

    def run(
        self, symbol: str, timeframe: str, start: datetime, end: Optional[datetime] = None
    ) -> Dict:
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
            balance_history: List[tuple] = []  # (timestamp, balance)

            # Helper dict to track first/last balance of each calendar year
            yearly_balance: Dict[int, Dict[str, float]] = {}

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
                    hit_time_limit = (
                        self.enable_time_limit_exit
                        and (current_time - self.current_trade.entry_time).total_seconds() > 86400
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
                        if self.current_trade.side == "long":
                            trade_pnl_pct = (
                                (exit_price - self.current_trade.entry_price)
                                / self.current_trade.entry_price
                            ) * self.current_trade.size
                        else:
                            trade_pnl_pct = (
                                (self.current_trade.entry_price - exit_price)
                                / self.current_trade.entry_price
                            ) * self.current_trade.size
                        trade_pnl_cash = cash_pnl(trade_pnl_pct, self.balance)

                        # Update balance
                        self.balance += trade_pnl_cash

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
                                size=self.current_trade.size,
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
                                size=self.current_trade.size,
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
            except Exception:
                # Keep zeros if any issue
                pass

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
            yearly_returns: Dict[str, float] = {}
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
            }

        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise

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

    def _extract_indicators(self, df: pd.DataFrame, index: int) -> Dict:
        return util_extract_indicators(df, index)

    def _extract_sentiment_data(self, df: pd.DataFrame, index: int) -> Dict:
        return util_extract_sentiment(df, index)

    def _extract_ml_predictions(self, df: pd.DataFrame, index: int) -> Dict:
        return util_extract_ml(df, index)
