from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import pandas as pd

from src.config.constants import (
    DEFAULT_ACCOUNT_SNAPSHOT_INTERVAL,
    DEFAULT_CHECK_INTERVAL,
    DEFAULT_DATA_FRESHNESS_THRESHOLD,
    DEFAULT_DYNAMIC_RISK_ENABLED,
    DEFAULT_END_OF_DAY_FLAT,
    DEFAULT_INITIAL_BALANCE,
    DEFAULT_MARKET_TIMEZONE,
    DEFAULT_MAX_CHECK_INTERVAL,
    DEFAULT_MAX_HOLDING_HOURS,
    DEFAULT_MFE_MAE_PRECISION_DECIMALS,
    DEFAULT_MFE_MAE_UPDATE_FREQUENCY_SECONDS,
    DEFAULT_MIN_CHECK_INTERVAL,
    DEFAULT_SLEEP_POLL_INTERVAL,
    DEFAULT_TIME_RESTRICTIONS,
    DEFAULT_WEEKEND_FLAT,
)
from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.coinbase_provider import CoinbaseProvider
from src.data_providers.data_provider import DataProvider
from src.data_providers.sentiment_provider import SentimentDataProvider
from src.database.manager import DatabaseManager
from src.database.models import TradeSource
from src.live.strategy_manager import StrategyManager
from src.performance.metrics import Side, pnl_percent
from src.position_management.dynamic_risk import DynamicRiskConfig, DynamicRiskManager
from src.position_management.mfe_mae_tracker import MFEMAETracker
from src.position_management.time_exits import TimeExitPolicy, TimeRestrictions
from src.regime.detector import RegimeDetector
from src.risk.risk_manager import RiskManager, RiskParameters
from src.strategies.base import BaseStrategy
from src.utils.logging_context import set_context, update_context
from src.utils.logging_events import (
    log_data_event,
    log_engine_event,
    log_order_event,
    log_risk_event,
)

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
        # Dynamic risk management
        enable_dynamic_risk: bool = DEFAULT_DYNAMIC_RISK_ENABLED,
        dynamic_risk_config: DynamicRiskConfig | None = None,
        time_exit_policy: TimeExitPolicy | None = None,
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
        
        # Dynamic risk management
        self.enable_dynamic_risk = enable_dynamic_risk
        self.dynamic_risk_manager = None
        if enable_dynamic_risk:
            config = dynamic_risk_config or DynamicRiskConfig()
            # Will be initialized after db_manager is available
            self._dynamic_risk_config = config
        
        # Timing configuration
        self.base_check_interval = check_interval
        self.check_interval = check_interval
        self.min_check_interval = DEFAULT_MIN_CHECK_INTERVAL
        self.max_check_interval = DEFAULT_MAX_CHECK_INTERVAL
        self.data_freshness_threshold = DEFAULT_DATA_FRESHNESS_THRESHOLD
        self.last_data_timestamp = None
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
                    config=final_config,
                    db_manager=self.db_manager
                )
                logger.info("Dynamic risk management enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize dynamic risk manager: {e}")
                self.dynamic_risk_manager = None
        
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

        # MFE/MAE tracker
        self.mfe_mae_tracker = MFEMAETracker(precision_decimals=DEFAULT_MFE_MAE_PRECISION_DECIMALS)
        self._last_mfe_mae_persist: datetime | None = None

        # Error handling
        self.max_consecutive_errors = max_consecutive_errors
        self.consecutive_errors = 0

        # Time exit policy (construct from overrides if not provided)
        self.time_exit_policy = time_exit_policy
        if self.time_exit_policy is None:
            overrides = None
            try:
                overrides = self.strategy.get_risk_overrides() if hasattr(self.strategy, "get_risk_overrides") else None
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
                        max_holding_hours=time_cfg.get("max_holding_hours", DEFAULT_MAX_HOLDING_HOURS),
                        end_of_day_flat=time_cfg.get("end_of_day_flat", DEFAULT_END_OF_DAY_FLAT),
                        weekend_flat=time_cfg.get("weekend_flat", DEFAULT_WEEKEND_FLAT),
                        market_timezone=time_cfg.get("market_timezone", DEFAULT_MARKET_TIMEZONE),
                        time_restrictions=restrictions,
                    )
            except Exception:
                pass

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

    def _merge_dynamic_risk_config(self, base_config: DynamicRiskConfig) -> DynamicRiskConfig:
        """Merge strategy risk overrides with base dynamic risk configuration"""
        try:
            # Get strategy risk overrides
            strategy_overrides = self.strategy.get_risk_overrides() if self.strategy else None
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
            
            logger.info(f"Merged strategy dynamic risk overrides from {self.strategy.__class__.__name__}")
            return merged_config
            
        except Exception as e:
            logger.warning(f"Failed to merge strategy dynamic risk overrides: {e}")
            return base_config

    def _get_dynamic_risk_adjusted_size(self, original_size: float) -> float:
        """Apply dynamic risk adjustments to position size"""
        if not self.dynamic_risk_manager:
            return original_size
            
        try:
            # Calculate dynamic risk adjustments
            adjustments = self.dynamic_risk_manager.calculate_dynamic_risk_adjustments(
                current_balance=self.current_balance,
                peak_balance=self.peak_balance,
                session_id=self.trading_session_id
            )
            
            # Apply position size adjustment
            adjusted_size = original_size * adjustments.position_size_factor
            
            # Log the adjustment if significant
            if abs(adjustments.position_size_factor - 1.0) > 0.1:  # >10% change
                logger.info(
                    f"🎛️ Dynamic risk adjustment applied: "
                    f"size factor={adjustments.position_size_factor:.2f}, "
                    f"reason={adjustments.primary_reason}"
                )
                log_risk_event(
                    "dynamic_risk_adjustment",
                    position_size_factor=adjustments.position_size_factor,
                    reason=adjustments.primary_reason,
                )
                
                # Log to database for tracking
                if self.db_manager and self.trading_session_id:
                    try:
                        self.db_manager.log_risk_adjustment(
                            session_id=self.trading_session_id,
                            adjustment_type=adjustments.primary_reason.split('_')[0],  # e.g., 'drawdown' from 'drawdown_15.0%'
                            trigger_reason=adjustments.primary_reason,
                            parameter_name='position_size_factor',
                            original_value=1.0,
                            adjusted_value=adjustments.position_size_factor,
                            adjustment_factor=adjustments.position_size_factor,
                            current_drawdown=adjustments.adjustment_details.get('current_drawdown'),
                            performance_score=None,  # Could be enhanced to include performance score
                            volatility_level=adjustments.adjustment_details.get('performance_metrics', {}).get('estimated_volatility')
                        )
                    except Exception as log_e:
                        logger.warning(f"Failed to log risk adjustment to database: {log_e}")
            
            return adjusted_size
            
        except Exception as e:
            logger.warning(f"Failed to apply dynamic risk adjustment: {e}")
            return original_size

    def _get_dynamic_risk_adjusted_params(self) -> RiskParameters:
        """Get risk parameters with dynamic adjustments applied"""
        if not self.dynamic_risk_manager:
            return self.risk_manager.params
            
        try:
            # Calculate dynamic risk adjustments
            adjustments = self.dynamic_risk_manager.calculate_dynamic_risk_adjustments(
                current_balance=self.current_balance,
                peak_balance=self.peak_balance,
                session_id=self.trading_session_id
            )
            
            # Apply adjustments to risk parameters
            adjusted_params = self.dynamic_risk_manager.apply_risk_adjustments(
                self.risk_manager.params,
                adjustments
            )
            
            return adjusted_params
            
        except Exception as e:
            logger.warning(f"Failed to get dynamic risk adjusted parameters: {e}")
            return self.risk_manager.params

    def start(self, symbol: str, timeframe: str = "1h", max_steps: int = None):
        """Start the live trading engine"""
        if self.is_running:
            logger.warning("Trading engine is already running")
            return

        self.is_running = True
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
                strategy_name=self.strategy.__class__.__name__,
                symbol=symbol,
                timeframe=timeframe,
                mode=mode,
                initial_balance=self.current_balance,  # Use current balance (might be recovered)
                strategy_config=getattr(self.strategy, "config", {}),
                time_exit_config=tx_cfg,
                market_timezone=(self.time_exit_policy.market_timezone if self.time_exit_policy else None),
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

        # Wait for main thread to finish (avoid joining current thread)
        if self.main_thread and self.main_thread.is_alive() and self.main_thread != threading.current_thread():
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
        heartbeat_every = int(os.getenv("ENGINE_HEARTBEAT_STEPS", "60")) if os.getenv("ENGINE_HEARTBEAT_STEPS") else 60
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
                    self.check_interval = self._calculate_adaptive_interval()
                    self._sleep_with_interrupt(self.check_interval)
                    continue
                current_index = len(df) - 1
                current_candle = df.iloc[current_index]
                current_price = current_candle["close"]
                if steps % heartbeat_every == 0:
                    log_engine_event(
                        "heartbeat",
                        step=steps,
                        open_positions=len(self.positions),
                        balance=self.current_balance,
                        last_candle_time=str(df.index[-1]),
                    )
                logger.info(
                    f"Trading loop: current_index={current_index}, last_candle_time={df.index[-1]}"
                )
                # Update position PnL
                self._update_position_pnl(current_price)

                # Update rolling MFE/MAE per position and persist lightweight updates
                self._update_positions_mfe_mae(current_price)

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
                            
                            # Apply dynamic risk adjustments
                            short_position_size = self._get_dynamic_risk_adjusted_size(short_position_size)
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

    def _update_positions_mfe_mae(self, current_price: float):
        """Compute and persist rolling MFE/MAE for active positions."""
        now = datetime.utcnow()
        for order_id, position in self.positions.items():
            # fraction is position.size (fraction of balance)
            self.mfe_mae_tracker.update_position_metrics(
                position_key=order_id,
                entry_price=float(position.entry_price),
                current_price=float(current_price),
                side=position.side.value,
                position_fraction=float(position.size),
                current_time=now,
            )
        # Throttle DB persistence to avoid overhead
        should_persist = False
        if self._last_mfe_mae_persist is None:
            should_persist = True
        else:
            delta = (now - self._last_mfe_mae_persist).total_seconds()
            should_persist = delta >= float(DEFAULT_MFE_MAE_UPDATE_FREQUENCY_SECONDS)
        if not should_persist:
            return
        self._last_mfe_mae_persist = now
        for order_id, _position in self.positions.items():
            db_id = self.position_db_ids.get(order_id)
            if db_id is not None:
                try:
                    m = self.mfe_mae_tracker.get_position_metrics(order_id)
                    if not m:
                        continue
                    self.db_manager.update_position(
                        position_id=db_id,
                        current_price=float(current_price),
                        mfe=float(m.mfe),
                        mae=float(m.mae),
                        mfe_price=float(m.mfe_price) if m.mfe_price is not None else None,
                        mae_price=float(m.mae_price) if m.mae_price is not None else None,
                        mfe_time=m.mfe_time,
                        mae_time=m.mae_time,
                    )
                except Exception as e:
                    logger.debug(f"MFE/MAE DB update failed for {order_id}: {e}")

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

            # Check time-based exits via policy (fallback to 24h if not provided)
            else:
                hit_time_exit = False
                reason = None
                if self.time_exit_policy is not None:
                    hit_time_exit, reason = self.time_exit_policy.check_time_exit_conditions(
                        position.entry_time, datetime.utcnow()
                    )
                else:
                    # Use UTC consistently to avoid timezone drift with naive datetimes
                    hit_time_exit = (datetime.utcnow() - position.entry_time).total_seconds() > 86400
                    reason = "Time limit"
                if hit_time_exit:
                    should_exit = True
                    exit_reason = reason or "Time exit"

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
                        f"position_age_{(datetime.utcnow() - position.entry_time).total_seconds():.0f}s",
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

        # Apply dynamic risk adjustments
        if position_size > 0:
            position_size = self._get_dynamic_risk_adjusted_size(position_size)

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
                entry_time=datetime.utcnow(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                order_id=order_id,
            )

            self.positions[order_id] = position

            # Initialize MFE/MAE cache for this position
            self.mfe_mae_tracker.clear(order_id)  # ensure fresh state

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
            log_order_event(
                "open_position",
                order_id=order_id,
                symbol=symbol,
                side=side.value,
                entry_price=price,
                size=size,
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

    def _close_position(self, position: Position, reason: str):
        """Close a position and update balance"""
        try:
            current_price_raw = self.data_provider.get_current_price(position.symbol)
            try:
                current_price = float(current_price_raw)
            except Exception:
                current_price = None
            if not current_price:
                # Fallback: try latest data frame
                try:
                    df = self._get_latest_data(position.symbol, "1m")
                    if df is not None and not df.empty and "close" in df.columns:
                        current_price = float(df["close"].iloc[-1])
                except Exception:
                    current_price = None
            if not current_price:
                # As a last resort use entry price to allow cleanup and logging
                logger.warning(
                    f"Falling back to entry price for {position.symbol} during close; live price unavailable"
                )
                current_price = float(position.entry_price)

            # Calculate P&L
            if position.side == PositionSide.LONG:
                pnl = pnl_percent(position.entry_price, current_price, Side.LONG, position.size)
            else:
                pnl = pnl_percent(position.entry_price, current_price, Side.SHORT, position.size)

            # Update balance
            position_value = position.size * self.current_balance
            realized_pnl = position_value * pnl
            self.current_balance += realized_pnl
            self.total_pnl += realized_pnl

            # Update peak balance for drawdown tracking
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance

            # Fetch final MFE/MAE metrics for this position
            metrics = self.mfe_mae_tracker.get_position_metrics(position.order_id)

            # Create trade record
            trade = Trade(
                symbol=position.symbol,
                side=position.side,
                size=position.size,
                entry_price=position.entry_price,
                exit_price=current_price,
                entry_time=position.entry_time,
                exit_time=datetime.now(),
                pnl=pnl,
                exit_reason=reason,
            )

            # Update statistics
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1

            # Log trade
            self.completed_trades.append(trade)
            if self.log_trades:
                self._log_trade(trade)

            # Log to database
            if self.trading_session_id is not None:
                self.db_manager.log_trade(
                    symbol=position.symbol,
                    side=position.side.value,
                    entry_price=position.entry_price,
                    exit_price=current_price,
                    size=position.size,
                    pnl=pnl,
                    strategy_name=self.strategy.__class__.__name__,
                    exit_reason=reason,
                    entry_time=position.entry_time,
                    exit_time=datetime.now(),
                    session_id=self.trading_session_id,
                    mfe=(metrics.mfe if metrics else None),
                    mae=(metrics.mae if metrics else None),
                    mfe_price=(metrics.mfe_price if metrics else None),
                    mae_price=(metrics.mae_price if metrics else None),
                    mfe_time=(metrics.mfe_time if metrics else None),
                    mae_time=(metrics.mae_time if metrics else None),
                )

                # Close position in database if it exists
                if position.order_id in self.position_db_ids:
                    position_db_id = self.position_db_ids[position.order_id]
                    
                    # Update position with final MFE/MAE metrics before closing
                    if metrics:
                        self.db_manager.update_position(
                            position_id=position_db_id,
                            mfe=metrics.mfe,
                            mae=metrics.mae,
                            mfe_price=metrics.mfe_price,
                            mae_price=metrics.mae_price,
                            mfe_time=metrics.mfe_time,
                            mae_time=metrics.mae_time,
                        )
                    
                    self.db_manager.close_position(
                        position_id=position_db_id,
                        exit_price=current_price,
                        exit_time=datetime.now(),
                        pnl=pnl,
                    )
                    del self.position_db_ids[position.order_id]

            # Close real order if needed
            if self.enable_live_trading:
                self._close_order(position.symbol, position.order_id)

            # Remove from active positions and tracker cache
            if position.order_id in self.positions:
                del self.positions[position.order_id]
            self.mfe_mae_tracker.clear(position.order_id)

            logger.info(
                f"📈 Closed {position.side.value} position for {position.symbol}: "
                f"P&L={pnl:.2%}, Reason={reason}, "
                f"Balance=${self.current_balance:,.2f}"
            )
            log_order_event(
                "close_position",
                order_id=position.order_id,
                symbol=position.symbol,
                side=position.side.value,
                exit_price=current_price,
                pnl=pnl,
                reason=reason,
            )

        except Exception as e:
            logger.error(f"Failed to close position {position.order_id}: {e}", exc_info=True)
            # Ensure local cleanup so engine/shutdown does not leave dangling positions
            try:
                if position.order_id in self.positions:
                    del self.positions[position.order_id]
                self.mfe_mae_tracker.clear(position.order_id)
            except Exception:
                # Best-effort cleanup; ignore secondary errors
                pass

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
        poll_interval = DEFAULT_SLEEP_POLL_INTERVAL  # Use configurable interval instead of 0.1
        while time.time() < end_time:
            if self.stop_event.is_set():
                break
            time.sleep(min(poll_interval, end_time - time.time()))

    def _calculate_adaptive_interval(self, current_price: float = None) -> int:
        """Calculate adaptive check interval based on recent trading activity and market conditions"""
        # Base interval from configuration
        interval = self.base_check_interval

        # Factor in recent trading activity
        recent_trades = len(
            [
                p
                for p in self.positions.values()
                if p.entry_time > datetime.now() - timedelta(hours=1)
            ]
        )
        if recent_trades > 0:
            # More frequent checks if we have recent activity
            interval = max(self.min_check_interval, interval // 2)
        elif len(self.positions) == 0:
            # Less frequent checks if no active positions
            interval = min(self.max_check_interval, interval * 2)

        # Consider time of day (basic market hours awareness)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Off-hours (UTC)
            interval = min(self.max_check_interval, interval * 1.5)

        return int(interval)

    def _is_data_fresh(self, df: pd.DataFrame) -> bool:
        """Check if the data is fresh enough to warrant processing"""
        if df is None or df.empty:
            return False

        latest_timestamp = df.index[-1] if hasattr(df.index[-1], "timestamp") else datetime.now()
        if isinstance(latest_timestamp, str):
            try:
                latest_timestamp = pd.to_datetime(latest_timestamp)
            except (ValueError, TypeError):
                return True  # Assume fresh if we can't parse timestamp

        age_seconds = (datetime.now() - latest_timestamp).total_seconds()
        return age_seconds <= self.data_freshness_threshold

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
        print(f"Max Drawdown: {self.max_drawdown * 100:.2f}%")
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
