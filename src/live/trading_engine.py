from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd
from performance.metrics import (
    Side,
    pnl_percent,
)
from regime.detector import RegimeDetector

from config.constants import DEFAULT_INITIAL_BALANCE
from data_providers.binance_provider import BinanceProvider
from data_providers.coinbase_provider import CoinbaseProvider
from data_providers.data_provider import DataProvider
from data_providers.sentiment_provider import SentimentDataProvider
from database.manager import DatabaseManager
from database.models import TradeSource
from live.strategy_manager import StrategyManager
from risk.risk_manager import RiskManager, RiskParameters
from strategies.base import BaseStrategy

from .account_sync import AccountSynchronizer

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class Position:
    """Represents an active trading position"""

    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: float | None = None
    take_profit: float | None = None
    unrealized_pnl: float = 0.0
    order_id: str | None = None


@dataclass
class Trade:
    """Represents a completed trade"""

    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    entry_time: datetime
    exit_price: float | None = None
    exit_time: datetime | None = None
    pnl: float | None = None
    exit_reason: str | None = None


def _create_exchange_provider(provider: str, config: dict):
    """Factory to create exchange provider and return (provider_instance, provider_name)."""
    if provider == "coinbase":
        api_key = config.get("COINBASE_API_KEY")
        api_secret = config.get("COINBASE_API_SECRET")
        if api_key and api_secret:
            return CoinbaseProvider(api_key, api_secret, testnet=False), "Coinbase"
        else:
            return None, "Coinbase (no credentials)"
    else:
        api_key = config.get("BINANCE_API_KEY")
        api_secret = config.get("BINANCE_API_SECRET")
        if api_key and api_secret:
            return BinanceProvider(api_key, api_secret, testnet=False), "Binance"
        else:
            return None, "Binance (no credentials)"


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
        strategy: BaseStrategy,
        data_provider: DataProvider,
        sentiment_provider: SentimentDataProvider | None = None,
        risk_parameters: RiskParameters | None = None,
        check_interval: int = 60,  # seconds
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
        max_position_size: float = 0.1,  # 10% of balance per position
        enable_live_trading: bool = False,  # Safety flag - must be explicitly enabled
        log_trades: bool = True,
        alert_webhook_url: str | None = None,
        enable_hot_swapping: bool = True,  # Enable strategy hot-swapping
        resume_from_last_balance: bool = True,  # Resume balance from last account snapshot
        database_url: str | None = None,  # Database connection URL
        max_consecutive_errors: int = 10,  # Maximum consecutive errors before shutdown
        account_snapshot_interval: int = 1800,  # Account snapshot interval in seconds (30 minutes)
        provider: str = "binance",  # 'binance' (default) or 'coinbase'
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

        self.strategy = strategy
        self.data_provider = data_provider
        self.sentiment_provider = sentiment_provider
        self.risk_manager = RiskManager(risk_parameters)
        self.check_interval = check_interval
        self.initial_balance = initial_balance
        self.current_balance = initial_balance  # Will be updated during startup
        self.max_position_size = max_position_size
        self.enable_live_trading = enable_live_trading
        self.log_trades = log_trades
        self.alert_webhook_url = alert_webhook_url
        self.enable_hot_swapping = enable_hot_swapping
        self.resume_from_last_balance = resume_from_last_balance
        self.account_snapshot_interval = account_snapshot_interval

        # Initialize database manager
        try:
            self.db_manager = DatabaseManager(database_url)
        except Exception as e:
            logger.critical(
                f"❌ Could not connect to the PostgreSQL database: {e}\nThe trading engine cannot start without a database connection. Exiting."
            )
            raise RuntimeError("Database connection required. Service stopped.")
        self.trading_session_id: int | None = None

        # Initialize exchange interface and account synchronizer
        self.exchange_interface = None
        self.account_synchronizer = None
        if enable_live_trading:
            try:
                from src.config import get_config

                config = get_config()
                self.exchange_interface, provider_name = _create_exchange_provider(provider, config)
                if self.exchange_interface:
                    self.account_synchronizer = AccountSynchronizer(
                        self.exchange_interface, self.db_manager, self.trading_session_id
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
            self.strategy_manager = StrategyManager()
            self.strategy_manager.current_strategy = strategy
            self.strategy_manager.on_strategy_change = self._handle_strategy_change
            self.strategy_manager.on_model_update = self._handle_model_update

        # Set up strategy logging if database is available
        if self.db_manager:
            self.strategy.set_database_manager(self.db_manager)

        # Trading state
        self.is_running = False
        self.positions: dict[str, Position] = {}
        self.position_db_ids: dict[str, int] = {}  # Map order_id to database position ID
        self.completed_trades: list[Trade] = []
        self.last_data_update = None
        self.last_account_snapshot = None  # Track when we last logged account state

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0

        # Error handling
        self.consecutive_errors = 0
        self.max_consecutive_errors = max_consecutive_errors
        self.error_cooldown = 300  # 5 minutes

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

        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(
            f"LiveTradingEngine initialized - Live Trading: {'ENABLED' if enable_live_trading else 'DISABLED'}"
        )

    def start(self, symbol: str, timeframe: str = "1h", max_steps: int = None):
        """Start the live trading engine"""
        if self.is_running:
            logger.warning("Trading engine is already running")
            return

        self.is_running = True
        logger.info(f"🚀 Starting live trading for {symbol} on {timeframe} timeframe")
        logger.info(f"Initial balance: ${self.current_balance:,.2f}")
        logger.info(f"Max position size: {self.max_position_size*100:.1f}% of balance")
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
            self.trading_session_id = self.db_manager.create_trading_session(
                strategy_name=self.strategy.__class__.__name__,
                symbol=symbol,
                timeframe=timeframe,
                mode=mode,
                initial_balance=self.current_balance,  # Use current balance (might be recovered)
                strategy_config=getattr(self.strategy, "config", {}),
            )

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
                        self.current_balance = corrected_balance
                        logger.info(
                            f"💰 Balance corrected from exchange: ${corrected_balance:,.2f}"
                        )
                        # Defer DB update until session is created
                        self._pending_balance_correction = True
                        self._pending_corrected_balance = corrected_balance
                else:
                    logger.warning(f"⚠️ Account synchronization failed: {sync_result.message}")
            except Exception as e:
                logger.error(f"❌ Account synchronization error: {e}", exc_info=True)

        # If a balance correction was pending, log it now (outside session creation conditional)
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

    def stop(self):
        """Stop the trading engine gracefully"""
        if not self.is_running:
            return

        logger.info("🛑 Stopping trading engine...")
        self.is_running = False
        self.stop_event.set()

        # Close all open positions
        if self.positions:
            logger.info(f"Closing {len(self.positions)} open positions...")
            for position in list(self.positions.values()):
                try:
                    self._close_position(position, "Engine shutdown")
                except Exception as e:
                    logger.error(
                        f"Failed to close position {position.order_id}: {e}", exc_info=True
                    )
                    # Force remove from positions dict if close fails
                    if position.order_id in self.positions:
                        del self.positions[position.order_id]

        # Wait for main thread to finish
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=30)

        # Print final statistics
        self._print_final_stats()

        # End the trading session in database
        if self.trading_session_id:
            self.db_manager.end_trading_session(
                session_id=self.trading_session_id, final_balance=self.current_balance
            )

        logger.info("Trading engine stopped")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)

    def _trading_loop(self, symbol: str, timeframe: str, max_steps: int = None):
        """Main trading loop"""
        logger.info("Trading loop started")
        steps = 0
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
                    logger.warning("No market data received")
                    self._sleep_with_interrupt(self.check_interval)
                    continue
                # Add sentiment data if available
                if self.sentiment_provider:
                    df = self._add_sentiment_data(df, symbol)
                # Check for pending strategy/model updates
                if self.strategy_manager and self.strategy_manager.has_pending_update():
                    logger.info("🔄 Applying pending strategy/model update...")
                    success = self.strategy_manager.apply_pending_update()
                    if success:
                        self.strategy = self.strategy_manager.current_strategy
                        logger.info("✅ Strategy/model update applied successfully")
                        self._send_alert("Strategy/Model updated in live trading")
                    else:
                        logger.error("❌ Failed to apply strategy/model update")
                # Calculate indicators
                df = self.strategy.calculate_indicators(df)
                # Remove warmup period and ensure we have enough data
                df = df.dropna()
                if len(df) < 2:
                    logger.warning("Insufficient data for analysis")
                    self._sleep_with_interrupt(self.check_interval)
                    continue
                current_index = len(df) - 1
                current_candle = df.iloc[current_index]
                current_price = current_candle["close"]
                logger.info(
                    f"Trading loop: current_index={current_index}, last_candle_time={df.index[-1]}"
                )
                # Update position PnL
                self._update_position_pnl(current_price)
                # Check exit conditions for existing positions
                self._check_exit_conditions(df, current_index, current_price)
                # Check entry conditions if not at maximum positions
                if len(self.positions) < self.risk_manager.get_max_concurrent_positions():
                    self._check_entry_conditions(df, current_index, symbol, current_price)
                    # Check for short entry if strategy supports it
                    if hasattr(self.strategy, "check_short_entry_conditions"):
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
                            if overrides and overrides.get("position_sizer"):
                                short_fraction = self.risk_manager.calculate_position_fraction(
                                    df=df,
                                    index=current_index,
                                    balance=self.current_balance,
                                    price=current_price,
                                    indicators=indicators,
                                    strategy_overrides=overrides,
                                )
                                short_fraction = min(short_fraction, self.max_position_size)
                                short_position_size = short_fraction
                            else:
                                short_position_size = self.strategy.calculate_position_size(
                                    df, current_index, self.current_balance
                                )
                                short_position_size = min(
                                    short_position_size, self.max_position_size
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
                                    short_stop_loss = self.strategy.calculate_stop_loss(
                                        df, current_index, current_price, PositionSide.SHORT
                                    )
                                    short_take_profit = current_price * (
                                        1 - getattr(self.strategy, "take_profit_pct", 0.04)
                                    )
                                self._open_position(
                                    symbol,
                                    PositionSide.SHORT,
                                    short_position_size,
                                    current_price,
                                    short_stop_loss,
                                    short_take_profit,
                                )
                # Update performance metrics
                self._update_performance_metrics()
                # Log account snapshot to database periodically (configurable interval)
                now = datetime.now()
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
                if self.total_trades % 10 == 0 or len(self.positions) > 0:
                    self._log_status(symbol, current_price)
                # Reset error counter on successful iteration
                self.consecutive_errors = 0
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
                # Sleep longer after errors
                sleep_time = min(self.error_cooldown, self.check_interval * self.consecutive_errors)
                self._sleep_with_interrupt(sleep_time)
                continue

            # Normal sleep between checks
            self._sleep_with_interrupt(self.check_interval)

        logger.info("Trading loop ended")

    def _get_latest_data(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        """Fetch latest market data with error handling"""
        try:
            df = self.data_provider.get_live_data(symbol, timeframe, limit=200)
            self.last_data_update = datetime.now()
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

    def _update_position_pnl(self, current_price: float):
        """Update unrealized PnL for all positions"""
        for position in self.positions.values():
            if position.side == PositionSide.LONG:
                position.unrealized_pnl = pnl_percent(
                    position.entry_price, current_price, Side.LONG, position.size
                )
            else:  # SHORT
                position.unrealized_pnl = pnl_percent(
                    position.entry_price, current_price, Side.SHORT, position.size
                )

    def _check_exit_conditions(self, df: pd.DataFrame, current_index: int, current_price: float):
        """Check if any positions should be closed"""
        positions_to_close = []

        # Extract context for logging
        indicators = self._extract_indicators(df, current_index)
        sentiment_data = self._extract_sentiment_data(df, current_index)
        ml_predictions = self._extract_ml_predictions(df, current_index)

        for position in self.positions.values():
            should_exit = False
            exit_reason = ""

            # Check strategy exit conditions
            if self.strategy.check_exit_conditions(df, current_index, position.entry_price):
                should_exit = True
                exit_reason = "Strategy signal"

            # Check stop loss
            elif position.stop_loss and self._check_stop_loss(position, current_price):
                should_exit = True
                exit_reason = "Stop loss"

            # Check take profit
            elif position.take_profit and self._check_take_profit(position, current_price):
                should_exit = True
                exit_reason = "Take profit"

            # Check maximum position time (24 hours)
            elif (datetime.now() - position.entry_time).total_seconds() > 86400:
                should_exit = True
                exit_reason = "Time limit"

            # Log exit decision for each position
            if self.db_manager:
                # Calculate current P&L for context
                if position.side == PositionSide.LONG:
                    current_pnl = (float(current_price) - float(position.entry_price)) / float(
                        position.entry_price
                    )
                else:
                    current_pnl = (float(position.entry_price) - float(current_price)) / float(
                        position.entry_price
                    )

                self.db_manager.log_strategy_execution(
                    strategy_name=self.strategy.__class__.__name__,
                    symbol=position.symbol,
                    signal_type="exit",
                    action_taken="closed_position" if should_exit else "hold_position",
                    price=current_price,
                    timeframe="1m",
                    signal_strength=1.0 if should_exit else 0.0,
                    confidence_score=indicators.get("prediction_confidence", 0.5),
                    indicators=indicators,
                    sentiment_data=sentiment_data if sentiment_data else None,
                    ml_predictions=ml_predictions if ml_predictions else None,
                    position_size=position.size,
                    reasons=[
                        exit_reason if should_exit else "holding_position",
                        f"current_pnl_{current_pnl:.4f}",
                        f"position_age_{(datetime.now() - position.entry_time).total_seconds():.0f}s",
                        f"entry_price_{position.entry_price:.2f}",
                    ],
                    volume=indicators.get("volume"),
                    volatility=indicators.get("volatility"),
                    session_id=self.trading_session_id,
                )

            if should_exit:
                positions_to_close.append((position, exit_reason))

        # Close positions
        for position, reason in positions_to_close:
            self._close_position(position, reason)

    def _check_entry_conditions(
        self, df: pd.DataFrame, current_index: int, symbol: str, current_price: float
    ):
        """Check if new positions should be opened"""
        # Check strategy entry conditions
        entry_signal = self.strategy.check_entry_conditions(df, current_index)

        # Extract context for logging
        indicators = self._extract_indicators(df, current_index)
        sentiment_data = self._extract_sentiment_data(df, current_index)
        ml_predictions = self._extract_ml_predictions(df, current_index)

        # Calculate position size if entry signal is present
        position_size = 0.0
        if entry_signal:
            # Prefer strategy sizing by default; use risk manager only if overrides specify a sizer
            try:
                overrides = (
                    self.strategy.get_risk_overrides()
                    if hasattr(self.strategy, "get_risk_overrides")
                    else None
                )
            except Exception:
                overrides = None
            if overrides and overrides.get("position_sizer"):
                fraction = self.risk_manager.calculate_position_fraction(
                    df=df,
                    index=current_index,
                    balance=self.current_balance,
                    price=current_price,
                    indicators=indicators,
                    strategy_overrides=overrides,
                )
                # Enforce engine-level cap
                fraction = min(fraction, self.max_position_size)
                position_size = fraction
            else:
                position_size = self.strategy.calculate_position_size(
                    df, current_index, self.current_balance
                )
                position_size = min(position_size, self.max_position_size)

        # Log strategy execution decision
        if self.db_manager:
            self.db_manager.log_strategy_execution(
                strategy_name=self.strategy.__class__.__name__,
                symbol=symbol,
                signal_type="entry",
                action_taken="opened_long" if entry_signal and position_size > 0 else "no_action",
                price=current_price,
                timeframe="1m",  # Could be made configurable
                signal_strength=1.0 if entry_signal else 0.0,
                confidence_score=indicators.get("prediction_confidence", 0.5),
                indicators=indicators,
                sentiment_data=sentiment_data if sentiment_data else None,
                ml_predictions=ml_predictions if ml_predictions else None,
                position_size=position_size if position_size > 0 else None,
                reasons=[
                    "entry_conditions_met" if entry_signal else "entry_conditions_not_met",
                    (
                        f"position_size_{position_size:.4f}"
                        if position_size > 0
                        else "no_position_size"
                    ),
                    f"max_positions_check_{len(self.positions)}_of_{self.risk_manager.get_max_concurrent_positions() if self.risk_manager else 1}",
                ],
                volume=indicators.get("volume"),
                volatility=indicators.get("volatility"),
                session_id=self.trading_session_id,
            )

        # Only proceed if we have a valid entry signal and position size
        if not entry_signal or position_size <= 0:
            return

        # Calculate risk management levels
        try:
            overrides = (
                self.strategy.get_risk_overrides()
                if hasattr(self.strategy, "get_risk_overrides")
                else None
            )
        except Exception:
            overrides = None
        if overrides and (("stop_loss_pct" in overrides) or ("take_profit_pct" in overrides)):
            stop_loss, take_profit = self.risk_manager.compute_sl_tp(
                df=df,
                index=current_index,
                entry_price=current_price,
                side="long",
                strategy_overrides=overrides,
            )
            if take_profit is None:
                take_profit = current_price * 1.04
        else:
            stop_loss = self.strategy.calculate_stop_loss(df, current_index, current_price, "long")
            take_profit = current_price * 1.04

        # Open new position
        self._open_position(
            symbol, PositionSide.LONG, position_size, current_price, stop_loss, take_profit
        )

    def _open_position(
        self,
        symbol: str,
        side: PositionSide,
        size: float,
        price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ):
        """Open a new trading position"""
        try:
            # Enforce maximum position size limit
            if size > self.max_position_size:
                logger.warning(
                    f"Position size {size:.2%} exceeds maximum {self.max_position_size:.2%}. Capping at maximum."
                )
                size = self.max_position_size

            position_value = size * self.current_balance

            if self.enable_live_trading:
                # Execute real order
                order_id = self._execute_order(symbol, side, position_value, price)
                if not order_id:
                    logger.error("Failed to execute order")
                    return
            else:
                order_id = f"paper_{int(time.time())}"
                logger.info(f"📄 PAPER TRADE - Would open {side.value} position")

            # Create position
            position = Position(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=price,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                order_id=order_id,
            )

            self.positions[order_id] = position

            # Log position to database
            if self.trading_session_id is not None:
                position_db_id = self.db_manager.log_position(
                    symbol=symbol,
                    side=side.value,
                    entry_price=price,
                    size=size,
                    strategy_name=self.strategy.__class__.__name__,
                    order_id=order_id,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    quantity=position_value / price,  # Calculate actual quantity
                    session_id=self.trading_session_id,
                )
                self.position_db_ids[order_id] = position_db_id
            else:
                logger.warning(
                    "⚠️ Cannot log position to database - no trading session ID available"
                )
                self.position_db_ids[order_id] = None

            logger.info(
                f"🚀 Opened {side.value} position: {symbol} @ ${price:.2f} (Size: ${position_value:.2f})"
            )

            # Send alert if configured
            self._send_alert(f"Position Opened: {symbol} {side.value} @ ${price:.2f}")

            # Update risk manager with the newly opened position so daily risk is tracked
            if self.risk_manager:
                try:
                    self.risk_manager.update_position(
                        symbol=symbol, side=side.value, size=size, entry_price=price
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to update risk manager for opened position {symbol}: {e}"
                    )

        except Exception as e:
            logger.error(f"Failed to open position: {e}", exc_info=True)
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

    def _execute_order(
        self, symbol: str, side: PositionSide, value: float, price: float
    ) -> str | None:
        """Execute a real market order (implement based on your exchange)"""
        # This is a placeholder - implement actual order execution
        logger.warning("Real order execution not implemented - using paper trading")
        return f"real_{int(time.time())}"

    def _close_order(self, symbol: str, order_id: str) -> bool:
        """Close a real market order (implement based on your exchange)"""
        # This is a placeholder - implement actual order closing
        logger.warning("Real order closing not implemented - using paper trading")
        return True

    def _check_stop_loss(self, position: Position, current_price: float) -> bool:
        """Check if stop loss should be triggered"""
        if not position.stop_loss:
            return False

        if position.side == PositionSide.LONG:
            return current_price <= position.stop_loss
        else:
            return current_price >= position.stop_loss

    def _check_take_profit(self, position: Position, current_price: float) -> bool:
        """Check if take profit should be triggered"""
        if not position.take_profit:
            return False

        if position.side == PositionSide.LONG:
            return current_price >= position.take_profit
        else:
            return current_price <= position.take_profit

    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

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
            # Calculate total exposure
            total_exposure = sum(pos.size * self.current_balance for pos in self.positions.values())

            # Calculate equity (balance + unrealized P&L)
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            equity = self.current_balance + unrealized_pnl

            # Calculate current drawdown percentage
            current_drawdown = 0
            if self.peak_balance > 0:
                current_drawdown = (
                    (self.peak_balance - self.current_balance) / self.peak_balance * 100
                )

            # TODO: Calculate daily P&L (requires tracking of day start balance)
            daily_pnl = 0  # Placeholder

            # Log snapshot to database
            if self.trading_session_id is not None:
                self.db_manager.log_account_snapshot(
                    balance=self.current_balance,
                    equity=equity,
                    total_pnl=self.total_pnl,
                    open_positions=len(self.positions),
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
            pos.unrealized_pnl * self.current_balance for pos in self.positions.values()
        )
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        logger.info(
            f"📊 Status: {symbol} @ ${current_price:.2f} | "
            f"Balance: ${self.current_balance:.2f} | "
            f"Positions: {len(self.positions)} | "
            f"Unrealized: ${total_unrealized:.2f} | "
            f"Trades: {self.total_trades} ({win_rate:.1f}% win)"
        )

    def _log_trade(self, trade: Trade):
        """Log trade to file"""
        try:
            # Create logs/trades directory if it doesn't exist
            os.makedirs("logs/trades", exist_ok=True)

            log_file = f"logs/trades/trades_{datetime.now().strftime('%Y%m')}.json"
            trade_data = {
                "timestamp": trade.exit_time.isoformat(),
                "symbol": trade.symbol,
                "side": trade.side.value,
                "size": trade.size,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "pnl": trade.pnl,
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
                "timestamp": datetime.now().isoformat(),
            }
            requests.post(self.alert_webhook_url, json=payload, timeout=10)
        except Exception as e:
            logger.error(f"Failed to send alert: {e}", exc_info=True)

    def _sleep_with_interrupt(self, seconds: float):
        """Sleep in small increments to allow for interrupt and float seconds"""
        end_time = time.time() + seconds
        while time.time() < end_time:
            if self.stop_event.is_set():
                break
            time.sleep(min(0.1, end_time - time.time()))

    def _print_final_stats(self):
        """Print final trading statistics"""
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        print("\n" + "=" * 60)
        print("🏁 FINAL TRADING STATISTICS")
        print("=" * 60)
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${self.current_balance:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Total PnL: ${self.total_pnl:+,.2f}")
        print(f"Max Drawdown: {self.max_drawdown*100:.2f}%")
        print(f"Total Trades: {self.total_trades}")
        print(f"Winning Trades: {self.winning_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Active Positions: {len(self.positions)}")

        if self.completed_trades:
            avg_trade = sum(trade.pnl for trade in self.completed_trades) / len(
                self.completed_trades
            )
            print(f"Average Trade: ${avg_trade:.2f}")

        print("=" * 60)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get current performance summary"""
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        current_drawdown = (
            (self.peak_balance - self.current_balance) / self.peak_balance * 100
            if self.peak_balance > 0
            else 0
        )

        return {
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "total_return": total_return,  # Keep both for backward compatibility
            "total_return_pct": total_return,
            "total_pnl": self.total_pnl,
            "current_drawdown": current_drawdown,  # Add current drawdown
            "max_drawdown_pct": self.max_drawdown * 100,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": win_rate,  # Keep both for backward compatibility
            "win_rate_pct": win_rate,
            "active_positions": len(self.positions),
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
                position = Position(
                    symbol=pos_data["symbol"],
                    side=PositionSide(pos_data["side"]),
                    size=pos_data["size"],
                    entry_price=pos_data["entry_price"],
                    entry_time=pos_data["entry_time"],
                    stop_loss=pos_data.get("stop_loss"),
                    take_profit=pos_data.get("take_profit"),
                    unrealized_pnl=pos_data.get("unrealized_pnl", 0.0),
                    order_id=str(pos_data["id"]),  # Use database ID as order_id
                )

                # Add to active positions
                if position.order_id:
                    self.positions[position.order_id] = position
                    self.position_db_ids[position.order_id] = pos_data["id"]

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

    def _handle_strategy_change(self, swap_data: dict[str, Any]):
        """Handle strategy change callback"""
        logger.info(f"🔄 Strategy change requested: {swap_data}")

        # If requested to close positions, close them now
        if swap_data.get("close_positions", False):
            logger.info("🚪 Closing all positions before strategy swap")
            for position in list(self.positions.values()):
                self._close_position(position, "Strategy change - close requested")
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
            strategy_name = getattr(self.strategy, "name", self.strategy.__class__.__name__)
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

        strategy_name = getattr(self.strategy, "name", self.strategy.__class__.__name__).lower()

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
