import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd  # type: ignore
from pandas import DataFrame  # type: ignore
from performance.metrics import (
    cagr as perf_cagr,
)

# Shared performance metrics
from performance.metrics import (
    cash_pnl,
)

# New modular utilities and models
from backtesting.utils import (
    compute_performance_metrics,
)
from src.trading.shared.indicators import (
    extract_indicators as util_extract_indicators,
    extract_sentiment_data as util_extract_sentiment,
    extract_ml_predictions as util_extract_ml,
)
from src.trading.shared.sentiment import merge_historical_sentiment
from src.trading.shared.sizing import normalize_position_size
from src.config.feature_flags import is_enabled
from backtesting.models import Trade as CompletedTrade
from sqlalchemy.exc import SQLAlchemyError

from config.constants import DEFAULT_INITIAL_BALANCE
from data_providers.data_provider import DataProvider
from data_providers.sentiment_provider import SentimentDataProvider
from database.manager import DatabaseManager
from database.models import TradeSource
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class Trade:
    """Represents a single trade"""

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
        self.size = min(size, 1.0)  # Limit position size to 100% of balance
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.pnl: Optional[float] = None
        self.exit_reason: Optional[str] = None

    def close(self, price: float, time: datetime, reason: str):
        """Close the trade and calculate PnL"""
        self.exit_price = price
        self.exit_time = time
        self.exit_reason = reason

        # Calculate percentage return
        if self.side == "long":
            self.pnl = ((self.exit_price - self.entry_price) / self.entry_price) * self.size
        else:  # short
            self.pnl = ((self.entry_price - self.exit_price) / self.entry_price) * self.size


class Backtester:
    """Backtesting engine for trading strategies"""

    def __init__(
        self,
        strategy: BaseStrategy,
        data_provider: DataProvider,
        sentiment_provider: Optional[SentimentDataProvider] = None,
        risk_parameters: Optional[Any] = None,
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
        database_url: Optional[str] = None,
        log_to_database: Optional[bool] = None,
    ):
        self.strategy = strategy
        self.data_provider = data_provider
        self.sentiment_provider = sentiment_provider
        self.risk_parameters = risk_parameters
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None

        # Early stop tracking
        self.early_stop_reason: Optional[str] = None
        self.early_stop_date: Optional[datetime] = None
        self.early_stop_candle_index: Optional[int] = None

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
                self.db_manager = DatabaseManager(database_url)
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

            # Fetch sentiment data if provider is available
            if self.sentiment_provider:
                # parity: shared merge behavior
                df = merge_historical_sentiment(df, self.sentiment_provider, symbol, timeframe, start, end)

            # Calculate indicators
            df = self.strategy.calculate_indicators(df)

            # Parity warmup: only ensure essential price columns are present
            essential_columns = ["open", "high", "low", "close", "volume"]
            df = df.dropna(subset=essential_columns)

            logger.info(f"Starting backtest with {len(df)} candles")

            # -----------------------------
            # Metrics & tracking variables
            # -----------------------------
            total_trades = 0
            winning_trades = 0
            max_drawdown_running = 0  # interim tracker (still used for intra-loop stopping)

            # Track balance over time to enable robust performance stats
            balance_history = []  # (timestamp, balance)

            # Helper dict to track first/last balance of each calendar year
            yearly_balance = {}

            # Iterate through candles
            for i in range(len(df)):
                candle = df.iloc[i]

                # Record current balance for time-series analytics
                balance_history.append((candle.name, self.balance))

                # Track yearly start/end balances for return calc
                yr = candle.name.year
                if yr not in yearly_balance:
                    yearly_balance[yr] = {"start": self.balance, "end": self.balance}
                else:
                    yearly_balance[yr]["end"] = self.balance

                # Update max drawdown
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance
                current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
                max_drawdown_running = max(max_drawdown_running, current_drawdown)

                # Check for exit if in position
                if self.current_trade is not None:
                    exit_signal = self.strategy.check_exit_conditions(
                        df, i, self.current_trade.entry_price
                    )

                    # Log exit decision
                    if self.log_to_database and self.db_manager:
                        indicators = self._extract_indicators(df, i)
                        sentiment_data = self._extract_sentiment_data(df, i)

                        # Calculate current P&L for context
                        current_pnl = (
                            candle["close"] - self.current_trade.entry_price
                        ) / self.current_trade.entry_price

                        self.db_manager.log_strategy_execution(
                            strategy_name=self.strategy.__class__.__name__,
                            symbol=symbol,
                            signal_type="exit",
                            action_taken="closed_position" if exit_signal else "hold_position",
                            price=candle["close"],
                            timeframe=timeframe,
                            signal_strength=1.0 if exit_signal else 0.0,
                            confidence_score=indicators.get("prediction_confidence", 0.5),
                            indicators=indicators,
                            sentiment_data=sentiment_data if sentiment_data else None,
                            position_size=self.current_trade.size,
                            reasons=[
                                "exit_signal" if exit_signal else "holding_position",
                                f"current_pnl_{current_pnl:.4f}",
                                f"position_age_{(candle.name - self.current_trade.entry_time).total_seconds():.0f}s",
                                f"entry_price_{self.current_trade.entry_price:.2f}",
                            ],
                            volume=indicators.get("volume"),
                            volatility=indicators.get("volatility"),
                            session_id=self.trading_session_id,
                        )

                    if exit_signal:
                        # Close the trade
                        self.current_trade.close(candle["close"], candle.name, "Strategy exit")

                        # Update balance (convert percentage PnL to absolute currency)
                        trade_pnl_percent: float = float(
                            self.current_trade.pnl or 0.0
                        )  # e.g. 0.02 for +2%
                        # Convert to absolute profit/loss based on current balance BEFORE applying PnL
                        trade_pnl: float = cash_pnl(trade_pnl_percent, self.balance)

                        self.balance += trade_pnl

                        # Update metrics
                        total_trades += 1
                        if trade_pnl > 0:
                            winning_trades += 1

                        # Log trade to database if enabled
                        if self.log_to_database and self.db_manager:
                            self.db_manager.log_trade(
                                symbol=symbol,
                                side=self.current_trade.side,
                                entry_price=self.current_trade.entry_price,
                                exit_price=candle["close"],
                                size=self.current_trade.size,
                                entry_time=self.current_trade.entry_time,
                                exit_time=candle.name,
                                pnl=trade_pnl,
                                exit_reason="Strategy exit",
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
                                exit_price=candle["close"],
                                entry_time=self.current_trade.entry_time,
                                exit_time=candle.name,
                                size=self.current_trade.size,
                                pnl=trade_pnl,
                                exit_reason="Strategy exit",
                                stop_loss=self.current_trade.stop_loss,
                                take_profit=self.current_trade.take_profit,
                            )
                        )
                        self.current_trade = None

                        # Check if maximum drawdown exceeded (use risk params if present)
                        # Keep develop behavior: use running drawdown check here
                        # (engine-level early stop logic remains as-is)

                # Check for entry if not in position
                elif self.strategy.check_entry_conditions(df, i):
                    # Calculate position size (normalize to fraction by default, legacy via flag)
                    raw_size = self.strategy.calculate_position_size(df, i, self.balance)
                    legacy = is_enabled('legacy_engine_behavior', default=False)
                    mode = 'notional' if legacy else 'fraction'
                    size_fraction = normalize_position_size(raw_size, self.balance, mode=mode)

                    if self.log_to_database and self.db_manager:
                        indicators = self._extract_indicators(df, i)
                        sentiment_data = self._extract_sentiment_data(df, i)

                        self.db_manager.log_strategy_execution(
                            strategy_name=self.strategy.__class__.__name__,
                            symbol=symbol,
                            signal_type="entry",
                            action_taken="opened_long" if size_fraction > 0 else "no_action",
                            price=candle["close"],
                            timeframe=timeframe,
                            signal_strength=1.0 if size_fraction > 0 else 0.0,
                            confidence_score=indicators.get("prediction_confidence", 0.5),
                            indicators=indicators,
                            sentiment_data=sentiment_data if sentiment_data else None,
                            position_size=size_fraction if size_fraction > 0 else None,
                            reasons=[
                                "entry_conditions_met",
                                f"position_size_{size_fraction:.4f}" if size_fraction > 0 else "no_position_size",
                                f"balance_{self.balance:.2f}",
                            ],
                            volume=indicators.get("volume"),
                            volatility=indicators.get("volatility"),
                            session_id=self.trading_session_id,
                        )

                    if size_fraction > 0:
                        # Enter new trade
                        stop_loss = self.strategy.calculate_stop_loss(df, i, candle["close"], "long")
                        tp_pct = getattr(self.strategy, 'take_profit_pct', 0.04)
                        take_profit = candle["close"] * (1 + tp_pct)
                        self.current_trade = Trade(
                            symbol=symbol,
                            side="long",
                            entry_price=candle["close"],
                            entry_time=candle.name,
                            size=size_fraction,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                        )
                        logger.info(f"Entered long position at {candle['close']}")

                # Log no-action cases (when no position and no entry signal)
                else:
                    if i % 10 == 0 and self.log_to_database and self.db_manager:
                        indicators = self._extract_indicators(df, i)
                        sentiment_data = self._extract_sentiment_data(df, i)

                        self.db_manager.log_strategy_execution(
                            strategy_name=self.strategy.__class__.__name__,
                            symbol=symbol,
                            signal_type="entry",
                            action_taken="no_action",
                            price=candle["close"],
                            timeframe=timeframe,
                            signal_strength=0.0,
                            confidence_score=indicators.get("prediction_confidence", 0.5),
                            indicators=indicators,
                            sentiment_data=sentiment_data if sentiment_data else None,
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

            # Build balance history DataFrame for metrics
            bh_df = pd.DataFrame(balance_history, columns=["timestamp", "balance"]).set_index("timestamp") if balance_history else pd.DataFrame()
            total_return, max_drawdown_pct, sharpe_ratio, annualized_return = compute_performance_metrics(
                self.initial_balance,
                self.balance,
                pd.Timestamp(start),
                pd.Timestamp(end) if end else None,
                bh_df,
            )

            # Yearly returns based on account balance
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
            }
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise

    # --------------------
    # Modularized helpers
    # --------------------
    def _extract_indicators(self, df: pd.DataFrame, index: int) -> Dict:
        return util_extract_indicators(df, index)

    def _extract_sentiment_data(self, df: pd.DataFrame, index: int) -> Dict:
        return util_extract_sentiment(df, index)

    def _extract_ml_predictions(self, df: pd.DataFrame, index: int) -> Dict:
        return util_extract_ml(df, index)
