"""
Backtesting engine for strategy evaluation.

This module provides a comprehensive backtesting framework.
"""

import hashlib
import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd  # type: ignore
import psutil
from pandas import DataFrame  # type: ignore
from sqlalchemy.exc import SQLAlchemyError

from src.backtesting.models import Trade as CompletedTrade
from src.backtesting.utils import (
    compute_performance_metrics,
)
from src.backtesting.utils import (
    extract_indicators as util_extract_indicators,
)
from src.backtesting.utils import (
    extract_ml_predictions as util_extract_ml,
)
from src.backtesting.utils import (
    extract_sentiment_data as util_extract_sentiment,
)
from src.config.config_manager import get_config
from src.config.constants import DEFAULT_INITIAL_BALANCE, DEFAULT_MFE_MAE_PRECISION_DECIMALS
from src.data_providers.data_provider import DataProvider
from src.data_providers.sentiment_provider import SentimentDataProvider
from src.database.manager import DatabaseManager
from src.database.models import TradeSource
from src.performance.metrics import cash_pnl
from src.position_management.correlation_engine import CorrelationConfig, CorrelationEngine
from src.position_management.dynamic_risk import DynamicRiskConfig, DynamicRiskManager
from src.position_management.mfe_mae_tracker import MFEMAETracker
from src.position_management.partial_manager import PositionState
from src.position_management.time_exits import TimeExitPolicy
from src.position_management.trailing_stops import TrailingStopPolicy
from src.regime.detector import RegimeDetector
from src.risk.risk_manager import RiskManager
from src.strategies.base import BaseStrategy
from src.utils.logging_context import set_context, update_context
from src.utils.logging_events import log_engine_event

logger = logging.getLogger(__name__)

# Cache configuration constants
MAX_CACHE_SIZE = 10000  # Maximum number of candles to cache
CHUNK_SIZE = 5000       # Process large datasets in chunks
MEMORY_THRESHOLD = 80   # Memory usage threshold percentage
CACHE_DIR = "cache/backtesting"  # Directory for persistent cache
CACHE_EXPIRY_DAYS = 7   # Cache expiration in days
PROGRESS_UPDATE_INTERVAL = 100  # Update progress every N candles


class PersistentCacheManager:
    """Manages persistent disk-based caching for backtesting predictions."""
    
    def __init__(self, cache_dir: str = CACHE_DIR, expiry_days: int = CACHE_EXPIRY_DAYS):
        self._cache_dir = Path(cache_dir)
        self.expiry_days = expiry_days
        self._cache_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def cache_dir(self) -> Path:
        """Get the cache directory as a Path object."""
        return self._cache_dir
    
    @cache_dir.setter
    def cache_dir(self, value: str | Path) -> None:
        """Set the cache directory, ensuring it's always a Path object."""
        self._cache_dir = Path(value)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return Path(self.cache_dir) / f"{cache_key}.pkl"
    
    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get the metadata file path for a cache key."""
        return Path(self.cache_dir) / f"{cache_key}.meta"
    
    def _is_cache_expired(self, cache_key: str) -> bool:
        """Check if a cache entry has expired."""
        metadata_path = self._get_metadata_path(cache_key)
        if not metadata_path.exists():
            return True
            
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            created_at = datetime.fromisoformat(metadata['created_at'])
            return datetime.now() - created_at > timedelta(days=self.expiry_days)
        except (json.JSONDecodeError, KeyError, ValueError):
            return True
    
    def get(self, cache_key: str) -> Optional[dict]:
        """Get cached data from disk."""
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)
        
        if not cache_path.exists() or not metadata_path.exists():
            return None
            
        if self._is_cache_expired(cache_key):
            self.delete(cache_key)
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, FileNotFoundError):
            return None
    
    def set(self, cache_key: str, data: dict) -> bool:
        """Save data to disk cache."""
        try:
            cache_path = self._get_cache_path(cache_key)
            metadata_path = self._get_metadata_path(cache_key)
            
            # Save data
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Save metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'size': len(str(data)),
                'cache_key': cache_key
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
                
            return True
        except (OSError, pickle.PickleError) as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
            return False
    
    def delete(self, cache_key: str) -> bool:
        """Delete cached data from disk."""
        try:
            cache_path = self._get_cache_path(cache_key)
            metadata_path = self._get_metadata_path(cache_key)
            
            if cache_path.exists():
                cache_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
                
            return True
        except OSError as e:
            logger.warning(f"Failed to delete cache {cache_key}: {e}")
            return False
    
    def cleanup_expired(self) -> int:
        """Clean up expired cache entries."""
        cleaned = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_key = cache_file.stem
            if self._is_cache_expired(cache_key):
                if self.delete(cache_key):
                    cleaned += 1
        return cleaned
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        total_files = 0
        total_size = 0
        expired_files = 0
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            total_files += 1
            total_size += cache_file.stat().st_size
            
            cache_key = cache_file.stem
            if self._is_cache_expired(cache_key):
                expired_files += 1
        
        return {
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'expired_files': expired_files,
            'cache_dir': str(self.cache_dir)
        }


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
        # Trailing state
        self.trailing_stop_activated: bool = False
        self.breakeven_triggered: bool = False
        self.trailing_stop_price: Optional[float] = None


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
        enable_time_limit_exit: bool = False,
        default_take_profit_pct: Optional[float] = None,
        legacy_stop_loss_indexing: bool = True,  # Preserve historical behavior by default
        enable_engine_risk_exits: bool = False,  # Enforce engine-level SL/TP exits (off to preserve baseline)
        time_exit_policy: TimeExitPolicy | None = None,
        # Dynamic risk management
        enable_dynamic_risk: bool = False,  # Disabled by default for backtesting to preserve historical results
        dynamic_risk_config: Optional[DynamicRiskConfig] = None,
        # Trailing stops
        trailing_stop_policy: Optional[TrailingStopPolicy] = None,
        # Partial operations
        partial_manager: Optional[Any] = None,
        # Internal caching control
        disable_results_cache: bool = False,  # Disable internal feature/strategy/ML caching
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
        self.trailing_stop_policy = trailing_stop_policy
        self.partial_manager = partial_manager
        self.disable_results_cache = disable_results_cache
        
        # Performance optimization: feature extraction caching
        self._feature_cache: dict[str, dict] = {}  # Cache for indicators, sentiment, ML data per candle
        self._feature_cache_size = 0  # Track current cache size
        
        # Performance optimization: strategy calculation caching
        self._strategy_cache: dict[str, dict] = {}  # Cache for strategy calculations per candle
        self._strategy_cache_size = 0  # Track current cache size
        
        # Performance optimization: ML prediction caching
        self._ml_predictions_cache: dict[str, float] = {}  # Cache for ML predictions per candle
        self._ml_predictions_cache_size = 0  # Track current cache size
        
        # Cache management
        self._model_version: Optional[str] = None  # Current model version
        self._use_original_method = False  # Fallback flag for when caching fails
        self._cache_hits = 0  # Track cache performance
        self._cache_misses = 0
        
        # Persistent caching
        self._persistent_cache = PersistentCacheManager()
        self._data_hash: Optional[str] = None  # Hash of current dataset
        self._enable_persistent_cache = True  # Enable disk caching
        self._current_df: Optional[pd.DataFrame] = None  # Current DataFrame for cache key generation
        
        # Progress tracking
        self._progress_callback: Optional[callable] = None  # Progress callback function
        self._last_progress_update = 0  # Last progress update time

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
        self.enable_time_limit_exit = enable_time_limit_exit
        self.default_take_profit_pct = default_take_profit_pct
        self.legacy_stop_loss_indexing = legacy_stop_loss_indexing
        self.enable_engine_risk_exits = enable_engine_risk_exits

        # MFE/MAE tracker for active trade
        self.mfe_mae_tracker = MFEMAETracker(precision_decimals=DEFAULT_MFE_MAE_PRECISION_DECIMALS)

        # Risk manager (parity with live engine)
        self.risk_manager = RiskManager(risk_parameters)
        # Correlation engine (for correlation-aware backtests)
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
                        f"Fallback SQLite database connection also failed: {sqlite_err}. Proceeding without DB logging."
                    )
                    self.db_manager = None

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
            # Get historical data first to check if we need to clear cache
            df = self.data_provider.get_historical_data(
                symbol=symbol, timeframe=timeframe, start=start, end=end
            )
            
            # Only clear cache if dataset has actually changed
            if self._should_clear_cache_for_new_dataset(df):
                logger.debug("Dataset changed, clearing cache")
                self._clear_feature_cache()
                # Reset data hash for new dataset
                self._data_hash = None
            else:
                logger.debug("Dataset unchanged, preserving cache for reuse")
            
            # Set base logging context for this backtest run
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
                # Update context with session id
                update_context(session_id=self.trading_session_id)

                # Update dynamic risk manager with database connection
                if self.enable_dynamic_risk and self.dynamic_risk_manager and self.db_manager:
                    self.dynamic_risk_manager.db_manager = self.db_manager

            # Check if data is empty (df was already fetched above)
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

            # Remove warmup period - only drop rows where essential price data is missing
            # Don't drop rows just because ML predictions or sentiment data is missing
            essential_columns = ["open", "high", "low", "close", "volume"]
            df = df.dropna(subset=essential_columns)

            # Pre-compute all feature extractions for performance optimization
            self._precompute_features(df)
            
            # Pre-compute strategy calculations for performance
            self._precompute_strategy_calculations(df)
            
            # Pre-compute ML predictions for performance (most expensive operation)
            self._precompute_ml_predictions(df)
            
            # Add cached predictions to the DataFrame for strategy use
            if self._ml_predictions_cache:
                # Create a Series with the correct index alignment
                # Map the cached predictions to the actual DataFrame indices
                predictions_series = pd.Series(index=df.index, dtype=float)
                for i in range(len(df)):
                    cache_key = self._get_cache_key(i)
                    if cache_key in self._ml_predictions_cache:
                        predictions_series.iloc[i] = self._ml_predictions_cache[cache_key]
                
                df['onnx_pred'] = predictions_series
                df['ml_prediction'] = df['onnx_pred']  # Alias for compatibility
                
                # Calculate prediction confidence if needed
                if 'close' in df.columns:
                    df['prediction_confidence'] = df.apply(
                        lambda row: min(1.0, abs(row['onnx_pred'] - row['close']) / row['close'] * 5.0) 
                        if pd.notna(row['onnx_pred']) and row['close'] > 0 else 0.0, 
                        axis=1
                    )
                    
                logger.info(f"Added {len(self._ml_predictions_cache)} ML predictions to DataFrame")
                logger.debug("Using pre-computed ML predictions, skipping strategy calculate_indicators")
            else:
                # Fallback to original method if no cached predictions
                logger.info("No cached ML predictions found, using original calculate_indicators method")
                df = self.strategy.calculate_indicators(df)

            logger.info(f"Starting backtest with {len(df)} candles")

            # Short trading is always enabled for strategies that define short entry conditions

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
                # Use cached data for performance
                cache_key = self._get_cache_key(i)
                if cache_key in self._strategy_cache:
                    cached_data = self._strategy_cache[cache_key]
                    current_time = cached_data['current_time']
                    current_price = cached_data['current_price']
                else:
                    # Fallback to direct DataFrame access
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
                    # Apply trailing stop before evaluating exits
                    if self.trailing_stop_policy is not None:
                        # Determine ATR if present
                        atr_val = None
                        try:
                            if "atr" in df.columns:
                                v = df["atr"].iloc[i]
                                atr_val = float(v) if v is not None and not pd.isna(v) else None
                        except Exception:
                            atr_val = None
                        new_sl, act, be = self.trailing_stop_policy.update_trailing_stop(
                            side=self.current_trade.side,
                            entry_price=float(self.current_trade.entry_price),
                            current_price=float(current_price),
                            existing_stop=float(self.current_trade.stop_loss) if self.current_trade.stop_loss is not None else None,
                            position_fraction=float(self.current_trade.size),
                            atr=atr_val,
                            trailing_activated=bool(self.current_trade.trailing_stop_activated),
                            breakeven_triggered=bool(self.current_trade.breakeven_triggered),
                        )
                        changed = False
                        if new_sl is not None and (
                            self.current_trade.stop_loss is None
                            or (self.current_trade.side == "long" and new_sl > float(self.current_trade.stop_loss))
                            or (self.current_trade.side != "long" and new_sl < float(self.current_trade.stop_loss))
                        ):
                            self.current_trade.stop_loss = new_sl
                            self.current_trade.trailing_stop_price = new_sl
                            changed = True
                        if act != self.current_trade.trailing_stop_activated or be != self.current_trade.breakeven_triggered:
                            self.current_trade.trailing_stop_activated = act
                            self.current_trade.breakeven_triggered = be
                            changed = True or changed
                        if changed and self.log_to_database and self.db_manager:
                            try:
                                self.db_manager.log_strategy_execution(
                                    strategy_name=self.strategy.__class__.__name__,
                                    symbol=symbol,
                                    signal_type="risk_adjustment",
                                    action_taken="trailing_stop_update",
                                    price=current_price,
                                    timeframe=timeframe,
                                    reasons=[
                                        f"sl_updated={self.current_trade.stop_loss}",
                                        f"activated={self.current_trade.trailing_stop_activated}",
                                        f"breakeven={self.current_trade.breakeven_triggered}",
                                    ],
                                    session_id=self.trading_session_id,
                                )
                            except Exception:
                                pass

                    # Use cached data for exit conditions check
                    exit_signal = self._check_exit_conditions_cached(
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

                    # Update MFE/MAE tracker for the active trade
                    try:
                        self.mfe_mae_tracker.update_position_metrics(
                            position_key="active",
                            entry_price=float(self.current_trade.entry_price),
                            current_price=float(current_price),
                            side=self.current_trade.side,
                            position_fraction=float(self.current_trade.size),
                            current_time=current_time,
                        )
                    except Exception:
                        pass

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

                        # Snapshot MFE/MAE
                        metrics = self.mfe_mae_tracker.get_position_metrics("active")

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
                                mfe=(metrics.mfe if metrics else None),
                                mae=(metrics.mae if metrics else None),
                                mfe_price=(metrics.mfe_price if metrics else None),
                                mae_price=(metrics.mae_price if metrics else None),
                                mfe_time=(metrics.mfe_time if metrics else None),
                                mae_time=(metrics.mae_time if metrics else None),
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
                                mfe=metrics.mfe if metrics else 0.0,
                                mae=metrics.mae if metrics else 0.0,
                                mfe_price=metrics.mfe_price if metrics else None,
                                mae_price=metrics.mae_price if metrics else None,
                                mfe_time=metrics.mfe_time if metrics else None,
                                mae_time=metrics.mae_time if metrics else None,
                            )
                        )

                        # Clear tracker for next trade
                        self.mfe_mae_tracker.clear("active")

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

                # Check for entry if not in position using cached method
                elif self._check_entry_conditions_cached(df, i):
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
                        # Build correlation context if available
                        correlation_ctx = None
                        try:
                            if self.correlation_engine is not None:
                                # Use available df as candidate series and fetch for other open symbols
                                # Use current open positions tracked by risk manager
                                open_symbols = set(self.risk_manager.positions.keys()) if getattr(self, "risk_manager", None) and self.risk_manager.positions else set()
                                symbols_to_check = set([symbol]) | open_symbols
                                price_series: dict[str, pd.Series] = {str(symbol): df["close"].copy()}
                                end_ts = df.index[-1] if len(df) > 0 else None
                                start_ts = end_ts - pd.Timedelta(days=self.risk_manager.params.correlation_window_days) if end_ts is not None else None
                                for sym in symbols_to_check:
                                    s = str(sym)
                                    if s in price_series:
                                        continue
                                    try:
                                        if start_ts is not None and end_ts is not None:
                                            hist = self.data_provider.get_historical_data(
                                                s,
                                                timeframe=timeframe,
                                                start=start_ts.to_pydatetime(),
                                                end=end_ts.to_pydatetime(),
                                            )
                                            if not hist.empty and "close" in hist:
                                                price_series[s] = hist["close"]
                                    except Exception:
                                        continue
                                corr_matrix = self.correlation_engine.calculate_position_correlations(price_series)
                                correlation_ctx = {
                                    "engine": self.correlation_engine,
                                    "candidate_symbol": symbol,
                                    "corr_matrix": corr_matrix,
                                    "max_exposure_override": overrides.get("correlation_control", {}).get("max_correlated_exposure") if overrides else None,
                                }
                        except Exception:
                            correlation_ctx = None
                        size_fraction = self.risk_manager.calculate_position_fraction(
                            df=df,
                            index=i,
                            balance=self.balance,
                            price=current_price,
                            indicators=self._extract_indicators(df, i),
                            strategy_overrides=overrides,
                            correlation_ctx=correlation_ctx,
                        )
                    else:
                        size_fraction = self._calculate_position_size_cached(df, i, self.balance)

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

                # Short entry if supported by strategy
                elif self._check_short_entry_conditions_cached(df, i):
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
                        size_fraction = self._calculate_position_size_cached(df, i, self.balance)

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
            
            # Log comprehensive cache performance statistics
            cache_stats = self.get_cache_stats()
            total_cache_requests = self._cache_hits + self._cache_misses
            if total_cache_requests > 0:
                logger.info(f"Cache performance: {self._cache_hits} hits, {self._cache_misses} misses ({cache_stats['hit_rate']:.1f}% hit rate)")
                logger.info(f"Memory cache sizes: features={self._feature_cache_size}, strategy={self._strategy_cache_size}, ml_predictions={self._ml_predictions_cache_size}")
                
                if self._enable_persistent_cache and 'total_files' in cache_stats:
                    logger.info(f"Disk cache: {cache_stats['total_files']} files, {cache_stats['total_size_mb']:.1f}MB")
                    if cache_stats['expired_files'] > 0:
                        logger.info(f"Expired cache files: {cache_stats['expired_files']}")
            
            # Note: Cache is preserved after backtest completion to allow inspection
            # and potential reuse in subsequent operations

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

    def _precompute_features(self, df: pd.DataFrame) -> None:
        """Pre-compute all feature extractions for the entire DataFrame to avoid redundant calculations."""
        # Skip pre-computation if caching is disabled
        if self.disable_results_cache:
            logger.debug("Skipping feature pre-computation - results cache disabled")
            return
            
        logger.debug(f"Pre-computing features for {len(df)} candles")
        
        # Generate data hash for this dataset
        self._get_data_hash(df)
        
        # Try to load from persistent cache first
        if self._enable_persistent_cache:
            persistent_key = self._get_persistent_cache_key("features")
            cached_features = self._persistent_cache.get(persistent_key)
            if cached_features:
                logger.info(f"Loaded features from persistent cache: {len(cached_features)} candles")
                self._feature_cache.update(cached_features)
                self._feature_cache_size = len(self._feature_cache)
                return
        
        # Pre-compute features with progress tracking
        for i in range(len(df)):
            # Extract all features once and cache them
            indicators = util_extract_indicators(df, i)
            sentiment_data = util_extract_sentiment(df, i)
            ml_predictions = util_extract_ml(df, i)
            
            cache_key = self._get_cache_key(i)
            self._feature_cache[cache_key] = {
                'indicators': indicators,
                'sentiment_data': sentiment_data,
                'ml_predictions': ml_predictions
            }
            self._feature_cache_size += 1
            
            # Update progress
            self._update_progress(i + 1, len(df), "Pre-computing features")
            
            # Check if cache is getting full
            if self._is_cache_full():
                self._cleanup_old_cache_entries()
        
        # Save to persistent cache
        if self._enable_persistent_cache and self._feature_cache:
            persistent_key = self._get_persistent_cache_key("features")
            if self._persistent_cache.set(persistent_key, self._feature_cache):
                logger.debug(f"Saved features to persistent cache: {len(self._feature_cache)} candles")
        
        logger.debug(f"Pre-computed features for {len(self._feature_cache)} candles")

    def _precompute_strategy_calculations(self, df: pd.DataFrame) -> None:
        """Pre-compute strategy calculations to avoid redundant operations during backtesting."""
        # Skip pre-computation if caching is disabled
        if self.disable_results_cache:
            logger.debug("Skipping strategy pre-computation - results cache disabled")
            return
            
        logger.debug(f"Pre-computing strategy calculations for {len(df)} candles")
        
        # Try to load from persistent cache first
        if self._enable_persistent_cache:
            persistent_key = self._get_persistent_cache_key("strategy")
            cached_strategy = self._persistent_cache.get(persistent_key)
            if cached_strategy:
                logger.info(f"Loaded strategy calculations from persistent cache: {len(cached_strategy)} candles")
                self._strategy_cache.update(cached_strategy)
                self._strategy_cache_size = len(self._strategy_cache)
                return
        
        # Pre-compute strategy calculations with progress tracking
        for i in range(len(df)):
            # Pre-compute common strategy calculations
            candle = df.iloc[i]
            current_price = float(candle["close"])
            
            # Cache basic calculations that are used multiple times
            cache_key = self._get_cache_key(i)
            self._strategy_cache[cache_key] = {
                'current_price': current_price,
                'current_time': candle.name,
                'candle_data': {
                    'open': float(candle.get('open', current_price)),
                    'high': float(candle.get('high', current_price)),
                    'low': float(candle.get('low', current_price)),
                    'close': current_price,
                    'volume': float(candle.get('volume', 0))
                }
            }
            self._strategy_cache_size += 1
            
            # Update progress
            self._update_progress(i + 1, len(df), "Pre-computing strategy calculations")
            
            # Check if cache is getting full
            if self._is_cache_full():
                self._cleanup_old_cache_entries()
        
        # Save to persistent cache
        if self._enable_persistent_cache and self._strategy_cache:
            persistent_key = self._get_persistent_cache_key("strategy")
            if self._persistent_cache.set(persistent_key, self._strategy_cache):
                logger.debug(f"Saved strategy calculations to persistent cache: {len(self._strategy_cache)} candles")
        
        logger.debug(f"Pre-computed strategy calculations for {len(self._strategy_cache)} candles")

    def _precompute_ml_predictions(self, df: pd.DataFrame) -> None:
        """Pre-compute ML predictions to avoid expensive inference during backtesting."""
        # Skip pre-computation if caching is disabled
        if self.disable_results_cache:
            logger.debug("Skipping ML predictions pre-computation - results cache disabled")
            return
            
        logger.debug(f"Pre-computing ML predictions for {len(df)} candles")
        
        # Only pre-compute if the strategy uses ML predictions
        if not hasattr(self.strategy, 'calculate_indicators'):
            return
            
        # Check if this is an ML strategy by looking for ONNX-related attributes
        if not (hasattr(self.strategy, 'ort_session') or hasattr(self.strategy, 'prediction_engine')):
            return
        
        # Try to load from persistent cache first
        if self._enable_persistent_cache:
            persistent_key = self._get_persistent_cache_key("ml_predictions")
            cached_predictions = self._persistent_cache.get(persistent_key)
            if cached_predictions:
                logger.info(f"Loaded ML predictions from persistent cache: {len(cached_predictions)} candles")
                self._ml_predictions_cache.update(cached_predictions)
                self._ml_predictions_cache_size = len(self._ml_predictions_cache)
                return
        
        # Check memory usage before starting
        if not self._check_memory_usage():
            logger.warning("High memory usage detected. Skipping ML prediction pre-computation.")
            self._use_original_method = True
            return
            
        try:
            # Use chunked processing for large datasets
            if len(df) > MAX_CACHE_SIZE:
                logger.info(f"Large dataset detected ({len(df)} candles). Using chunked processing.")
                self._precompute_ml_predictions_chunked(df)
            else:
                # Process entire dataset at once for smaller datasets
                self._precompute_ml_predictions_single(df)
            
            # Save to persistent cache
            if self._enable_persistent_cache and self._ml_predictions_cache:
                persistent_key = self._get_persistent_cache_key("ml_predictions")
                if self._persistent_cache.set(persistent_key, self._ml_predictions_cache):
                    logger.debug(f"Saved ML predictions to persistent cache: {len(self._ml_predictions_cache)} candles")
                
        except Exception as e:
            logger.warning(f"Failed to pre-compute ML predictions: {e}")
            logger.warning("Falling back to original method for ML predictions.")
            self._use_original_method = True
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")

    def _precompute_ml_predictions_single(self, df: pd.DataFrame) -> None:
        """Pre-compute ML predictions for smaller datasets."""
        df_with_predictions = self.strategy.calculate_indicators(df.copy())
        
        # Cache the predictions with versioned keys
        predictions_found = 0
        for i in range(len(df_with_predictions)):
            if 'onnx_pred' in df_with_predictions.columns:
                pred = df_with_predictions['onnx_pred'].iloc[i]
                if pd.notna(pred):
                    cache_key = self._get_cache_key(i)
                    self._ml_predictions_cache[cache_key] = float(pred)
                    self._ml_predictions_cache_size += 1
                    predictions_found += 1
                    
                    # Check if cache is getting full
                    if self._is_cache_full():
                        self._cleanup_old_cache_entries()
                        
        logger.info(f"Pre-computed ML predictions for {predictions_found} candles out of {len(df_with_predictions)} total")

    def _precompute_ml_predictions_chunked(self, df: pd.DataFrame) -> None:
        """Pre-compute ML predictions for large datasets using chunked processing."""
        total_predictions = 0
        total_chunks = (len(df) + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        for chunk_idx, start_idx in enumerate(range(0, len(df), CHUNK_SIZE)):
            end_idx = min(start_idx + CHUNK_SIZE, len(df))
            chunk_df = df.iloc[start_idx:end_idx]
            
            logger.debug(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({start_idx}-{end_idx}, {len(chunk_df)} candles)")
            
            try:
                # Process this chunk
                chunk_predictions = self._process_ml_chunk(chunk_df, start_idx)
                total_predictions += chunk_predictions
                
                # Update progress
                self._update_progress(chunk_idx + 1, total_chunks, "Processing ML prediction chunks")
                
                # Check memory usage after each chunk
                if not self._check_memory_usage():
                    logger.warning("High memory usage detected during chunked processing. Stopping early.")
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to process chunk {start_idx}-{end_idx}: {e}")
                continue
                
        logger.info(f"Pre-computed ML predictions for {total_predictions} candles using chunked processing")

    def _process_ml_chunk(self, chunk_df: pd.DataFrame, start_idx: int) -> int:
        """Process a single chunk of ML predictions."""
        df_with_predictions = self.strategy.calculate_indicators(chunk_df.copy())
        
        predictions_found = 0
        for i in range(len(df_with_predictions)):
            if 'onnx_pred' in df_with_predictions.columns:
                pred = df_with_predictions['onnx_pred'].iloc[i]
                if pd.notna(pred):
                    cache_key = self._get_cache_key(start_idx + i)
                    self._ml_predictions_cache[cache_key] = float(pred)
                    self._ml_predictions_cache_size += 1
                    predictions_found += 1
                    
                    # Check if cache is getting full
                    if self._is_cache_full():
                        self._cleanup_old_cache_entries()
                        
        return predictions_found

    def _check_exit_conditions_cached(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Optimized exit conditions check using cached data."""
        # If caching is disabled, call the strategy method directly
        if self.disable_results_cache:
            if hasattr(self.strategy, 'check_exit_conditions'):
                try:
                    return self.strategy.check_exit_conditions(df, index, entry_price)
                except Exception:
                    return False
            return False
        
        cache_key = self._get_cache_key(index)
        
        # Try to use cached data first
        if cache_key in self._strategy_cache:
            self._cache_hits += 1
            cached_data = self._strategy_cache[cache_key]
            
            # Use cached data to avoid expensive DataFrame operations
            if 'exit_conditions_result' in cached_data:
                return cached_data['exit_conditions_result']
            
            # If not cached, compute and cache the result
            if hasattr(self.strategy, 'check_exit_conditions'):
                try:
                    result = self.strategy.check_exit_conditions(df, index, entry_price)
                    cached_data['exit_conditions_result'] = result
                    return result
                except Exception:
                    cached_data['exit_conditions_result'] = False
                    return False
            return False
        else:
            self._cache_misses += 1
            # Fallback to original method if not cached
            if hasattr(self.strategy, 'check_exit_conditions'):
                try:
                    result = self.strategy.check_exit_conditions(df, index, entry_price)
                    # Cache the result for future use
                    self._strategy_cache[cache_key] = {
                        'exit_conditions_result': result,
                        'current_price': df.iloc[index]['close'],
                        'current_time': df.index[index]
                    }
                    self._strategy_cache_size = len(self._strategy_cache)
                    return result
                except Exception:
                    return False
            return False

    def _check_entry_conditions_cached(self, df: pd.DataFrame, index: int) -> bool:
        """Optimized entry conditions check using cached data."""
        # If caching is disabled, call the strategy method directly
        if self.disable_results_cache:
            if hasattr(self.strategy, 'check_entry_conditions'):
                try:
                    return self.strategy.check_entry_conditions(df, index)
                except Exception:
                    return False
            return False
        
        cache_key = self._get_cache_key(index)
        
        # Try to use cached data first
        if cache_key in self._strategy_cache:
            self._cache_hits += 1
            cached_data = self._strategy_cache[cache_key]
            
            # Use cached data to avoid expensive DataFrame operations
            if 'entry_conditions_result' in cached_data:
                return cached_data['entry_conditions_result']
            
            # If not cached, compute and cache the result
            if hasattr(self.strategy, 'check_entry_conditions'):
                try:
                    result = self.strategy.check_entry_conditions(df, index)
                    cached_data['entry_conditions_result'] = result
                    return result
                except Exception:
                    cached_data['entry_conditions_result'] = False
                    return False
            return False
        else:
            self._cache_misses += 1
            # Fallback to original method if not cached
            if hasattr(self.strategy, 'check_entry_conditions'):
                try:
                    result = self.strategy.check_entry_conditions(df, index)
                    # Cache the result for future use
                    self._strategy_cache[cache_key] = {
                        'entry_conditions_result': result,
                        'current_price': df.iloc[index]['close'],
                        'current_time': df.index[index]
                    }
                    self._strategy_cache_size = len(self._strategy_cache)
                    return result
                except Exception:
                    return False
            return False

    def _check_short_entry_conditions_cached(self, df: pd.DataFrame, index: int) -> bool:
        """Optimized short entry conditions check using cached data."""
        # If caching is disabled, call the strategy method directly
        if self.disable_results_cache:
            if hasattr(self.strategy, 'check_short_entry_conditions'):
                try:
                    return self.strategy.check_short_entry_conditions(df, index)
                except Exception:
                    return False
            return False
        
        cache_key = self._get_cache_key(index)
        
        # Try to use cached data first
        if cache_key in self._strategy_cache:
            self._cache_hits += 1
            cached_data = self._strategy_cache[cache_key]
            
            # Use cached data to avoid expensive DataFrame operations
            if 'short_entry_conditions_result' in cached_data:
                return cached_data['short_entry_conditions_result']
            
            # If not cached, compute and cache the result
            if hasattr(self.strategy, 'check_short_entry_conditions'):
                try:
                    result = self.strategy.check_short_entry_conditions(df, index)
                    cached_data['short_entry_conditions_result'] = result
                    return result
                except Exception:
                    cached_data['short_entry_conditions_result'] = False
                    return False
            return False
        else:
            self._cache_misses += 1
            # Fallback to original method if not cached
            if hasattr(self.strategy, 'check_short_entry_conditions'):
                try:
                    result = self.strategy.check_short_entry_conditions(df, index)
                    # Cache the result for future use
                    if cache_key not in self._strategy_cache:
                        self._strategy_cache[cache_key] = {
                            'current_price': df.iloc[index]['close'],
                            'current_time': df.index[index]
                        }
                    self._strategy_cache[cache_key]['short_entry_conditions_result'] = result
                    self._strategy_cache_size = len(self._strategy_cache)
                    return result
                except Exception:
                    return False
            return False

    def _calculate_position_size_cached(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Optimized position size calculation using cached data."""
        # If caching is disabled, call the strategy method directly
        if self.disable_results_cache:
            if hasattr(self.strategy, 'calculate_position_size'):
                try:
                    return self.strategy.calculate_position_size(df, index, balance)
                except Exception:
                    return 0.0
            return 0.0
        
        cache_key = self._get_cache_key(index)
        
        # Try to use cached data first
        if cache_key in self._strategy_cache:
            self._cache_hits += 1
            cached_data = self._strategy_cache[cache_key]
            
            # Use cached data to avoid expensive DataFrame operations
            if 'position_size_result' in cached_data:
                return cached_data['position_size_result']
            
            # If not cached, compute and cache the result
            if hasattr(self.strategy, 'calculate_position_size'):
                try:
                    result = self.strategy.calculate_position_size(df, index, balance)
                    cached_data['position_size_result'] = result
                    return result
                except Exception:
                    cached_data['position_size_result'] = 0.0
                    return 0.0
            return 0.0
        else:
            self._cache_misses += 1
            # Fallback to original method if not cached
            if hasattr(self.strategy, 'calculate_position_size'):
                try:
                    result = self.strategy.calculate_position_size(df, index, balance)
                    # Cache the result for future use
                    if cache_key not in self._strategy_cache:
                        self._strategy_cache[cache_key] = {
                            'current_price': df.iloc[index]['close'],
                            'current_time': df.index[index]
                        }
                    self._strategy_cache[cache_key]['position_size_result'] = result
                    self._strategy_cache_size = len(self._strategy_cache)
                    return result
                except Exception:
                    return 0.0
            return 0.0

    def _get_model_version(self) -> str:
        """Get a unique version identifier for the current model."""
        if self._model_version is not None:
            return self._model_version
            
        # Generate version based on strategy class and parameters
        strategy_info = f"{self.strategy.__class__.__name__}"
        
        # Include model-specific information if available
        if hasattr(self.strategy, 'model_hash'):
            strategy_info += f"_{self.strategy.model_hash}"
        elif hasattr(self.strategy, 'ort_session') and self.strategy.ort_session:
            # Use ONNX session info for versioning
            strategy_info += f"_onnx_{hash(str(self.strategy.ort_session.get_modelmeta()))}"
        elif hasattr(self.strategy, 'prediction_engine'):
            strategy_info += f"_pred_{hash(str(type(self.strategy.prediction_engine)))}"
        
        # Include strategy parameters for versioning
        if hasattr(self.strategy, 'get_parameters'):
            try:
                params = self.strategy.get_parameters()
                strategy_info += f"_{hash(str(sorted(params.items()) if isinstance(params, dict) else str(params)))}"
            except Exception:
                strategy_info += f"_{hash(str(type(self.strategy)))}"
        
        self._model_version = hashlib.md5(strategy_info.encode()).hexdigest()[:16]
        return self._model_version

    def _get_data_hash(self, df: pd.DataFrame) -> str:
        """Generate a hash for the dataset to detect changes."""
        if self._data_hash is not None:
            return self._data_hash
            
        # Create a hash based on data characteristics that allows cache reuse
        # for overlapping datasets (same source data, different sizes)
        # Use columns and first row only to allow cache reuse across dataset slices
        data_info = {
            'columns': list(df.columns),
            'first_row': df.iloc[0].to_dict() if len(df) > 0 else {},
            'index_start': str(df.index[0]) if len(df) > 0 else None,
            # Don't include shape or last_row to allow cache reuse for dataset slices
        }
        
        data_str = json.dumps(data_info, sort_keys=True, default=str)
        self._data_hash = hashlib.md5(data_str.encode()).hexdigest()[:16]
        return self._data_hash

    def _get_cache_key(self, index: int, df: Optional[pd.DataFrame] = None) -> str:
        """Generate a cache key that includes model version and data hash."""
        model_version = self._get_model_version()
        
        # Ensure data hash is set - if not available and df provided, generate it
        if self._data_hash is None and df is not None:
            self._get_data_hash(df)
        
        data_hash = self._data_hash or "unknown"
        return f"{model_version}_{data_hash}_{index}"

    def _get_persistent_cache_key(self, cache_type: str) -> str:
        """Generate a persistent cache key for the entire dataset."""
        model_version = self._get_model_version()
        data_hash = self._data_hash or "unknown"
        return f"{model_version}_{data_hash}_{cache_type}"

    def _update_progress(self, current: int, total: int, operation: str) -> None:
        """Update progress for long operations."""
        current_time = time.time()
        
        # Only update if enough time has passed or it's the last item
        if (current_time - self._last_progress_update > 1.0 or 
            current >= total or 
            current % PROGRESS_UPDATE_INTERVAL == 0):
            
            progress_pct = (current / total) * 100 if total > 0 else 0
            logger.info(f"{operation}: {current}/{total} ({progress_pct:.1f}%)")
            
            if self._progress_callback:
                try:
                    self._progress_callback(current, total, operation)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")
            
            self._last_progress_update = current_time

    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable limits."""
        try:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > MEMORY_THRESHOLD:
                logger.warning(f"High memory usage: {memory_percent:.1f}%. Consider reducing cache size.")
                return False
            return True
        except Exception as e:
            logger.warning(f"Failed to check memory usage: {e}")
            return True  # Continue if we can't check memory

    def _is_cache_full(self) -> bool:
        """Check if any cache has reached its maximum size."""
        return (self._feature_cache_size >= MAX_CACHE_SIZE or 
                self._strategy_cache_size >= MAX_CACHE_SIZE or 
                self._ml_predictions_cache_size >= MAX_CACHE_SIZE)

    def _cleanup_old_cache_entries(self) -> None:
        """Remove oldest cache entries when cache is full (sliding window approach)."""
        if not self._is_cache_full():
            return
            
        # Calculate how many entries to remove (keep 80% of max size)
        target_size = int(MAX_CACHE_SIZE * 0.8)
        
        # Clean up feature cache
        if self._feature_cache_size >= MAX_CACHE_SIZE:
            entries_to_remove = self._feature_cache_size - target_size
            keys_to_remove = list(self._feature_cache.keys())[:entries_to_remove]
            for key in keys_to_remove:
                del self._feature_cache[key]
            self._feature_cache_size = len(self._feature_cache)
            logger.debug(f"Cleaned up {entries_to_remove} feature cache entries")
        
        # Clean up strategy cache
        if self._strategy_cache_size >= MAX_CACHE_SIZE:
            entries_to_remove = self._strategy_cache_size - target_size
            keys_to_remove = list(self._strategy_cache.keys())[:entries_to_remove]
            for key in keys_to_remove:
                del self._strategy_cache[key]
            self._strategy_cache_size = len(self._strategy_cache)
            logger.debug(f"Cleaned up {entries_to_remove} strategy cache entries")
        
        # Clean up ML predictions cache
        if self._ml_predictions_cache_size >= MAX_CACHE_SIZE:
            entries_to_remove = self._ml_predictions_cache_size - target_size
            keys_to_remove = list(self._ml_predictions_cache.keys())[:entries_to_remove]
            for key in keys_to_remove:
                del self._ml_predictions_cache[key]
            self._ml_predictions_cache_size = len(self._ml_predictions_cache)
            logger.debug(f"Cleaned up {entries_to_remove} ML predictions cache entries")

    def set_progress_callback(self, callback: callable) -> None:
        """Set a progress callback function for long operations."""
        self._progress_callback = callback

    def enable_persistent_cache(self, enable: bool = True) -> None:
        """Enable or disable persistent disk caching."""
        self._enable_persistent_cache = enable
        logger.info(f"Persistent caching {'enabled' if enable else 'disabled'}")

    def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries from disk."""
        if not self._enable_persistent_cache:
            return 0
        return self._persistent_cache.cleanup_expired()

    def get_cache_stats(self) -> dict:
        """Get comprehensive cache statistics."""
        memory_stats = {
            'feature_cache_size': self._feature_cache_size,
            'strategy_cache_size': self._strategy_cache_size,
            'ml_predictions_cache_size': self._ml_predictions_cache_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': (self._cache_hits / (self._cache_hits + self._cache_misses) * 100) if (self._cache_hits + self._cache_misses) > 0 else 0
        }
        
        if self._enable_persistent_cache:
            disk_stats = self._persistent_cache.get_cache_stats()
            memory_stats.update(disk_stats)
        
        return memory_stats

    def _clear_feature_cache(self) -> None:
        """Clear the feature cache to free memory."""
        self._feature_cache.clear()
        self._strategy_cache.clear()
        self._ml_predictions_cache.clear()
        self._feature_cache_size = 0
        self._strategy_cache_size = 0
        self._ml_predictions_cache_size = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
    def _should_clear_cache_for_new_dataset(self, df: pd.DataFrame) -> bool:
        """Check if cache should be cleared for a new dataset."""
        if self._data_hash is None:
            return False  # No previous dataset, no need to clear
            
        # Calculate hash for new dataset
        new_hash = self._calculate_data_hash(df)
        
        # Clear cache only if dataset has actually changed
        return new_hash != self._data_hash
    
    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """Calculate data hash without storing it."""
        data_info = {
            'columns': list(df.columns),
            'first_row': df.iloc[0].to_dict() if len(df) > 0 else {},
            'index_start': str(df.index[0]) if len(df) > 0 else None,
        }
        
        data_str = json.dumps(data_info, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()[:16]

    def _extract_indicators(self, df: pd.DataFrame, index: int) -> dict:
        """Extract indicators with caching for performance."""
        if index in self._feature_cache:
            return self._feature_cache[index]['indicators']
        return util_extract_indicators(df, index)

    def _extract_sentiment_data(self, df: pd.DataFrame, index: int) -> dict:
        """Extract sentiment data with caching for performance."""
        if index in self._feature_cache:
            return self._feature_cache[index]['sentiment_data']
        return util_extract_sentiment(df, index)

    def _extract_ml_predictions(self, df: pd.DataFrame, index: int) -> dict:
        """Extract ML predictions with caching for performance."""
        if index in self._feature_cache:
            return self._feature_cache[index]['ml_predictions']
        return util_extract_ml(df, index)
