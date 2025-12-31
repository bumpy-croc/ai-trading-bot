"""
Centralized constant values for the AI Trading Bot project.
"""

DEFAULT_INITIAL_BALANCE: float = 1000  # Default starting balance in USD

# Prediction Engine Constants
DEFAULT_PREDICTION_HORIZONS = [1]  # Single horizon for MVP
DEFAULT_MIN_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_MAX_PREDICTION_LATENCY = 0.1  # seconds
# Default model registry base path (legacy flat layout). The registry also
# auto-detects a structured subdirectory at base/models when present.
DEFAULT_MODEL_REGISTRY_PATH = "src/ml/models"
DEFAULT_ENABLE_SENTIMENT = False  # Disabled by default
DEFAULT_ENABLE_MARKET_MICROSTRUCTURE = False  # MVP: disabled
DEFAULT_FEATURE_CACHE_TTL = 3600  # 1 hour
DEFAULT_MODEL_CACHE_TTL = 600  # seconds
DEFAULT_CONFIDENCE_SCALE_FACTOR = 10.0  # Scale factor for confidence calculation
DEFAULT_DIRECTION_THRESHOLD = 0.01  # 1% threshold for direction determination

# Ensemble/Regime-aware prediction enhancements
DEFAULT_ENABLE_ENSEMBLE = False
DEFAULT_ENSEMBLE_METHOD = "mean"
DEFAULT_ENABLE_REGIME_AWARE_CONFIDENCE = False

# Prediction Caching Constants
DEFAULT_PREDICTION_CACHE_TTL = 60  # seconds
DEFAULT_PREDICTION_CACHE_MAX_SIZE = 1000  # maximum number of cached predictions
DEFAULT_PREDICTION_CACHE_ENABLED = True  # enable/disable prediction caching

# Feature Engineering Constants
DEFAULT_SEQUENCE_LENGTH = 120  # LSTM sequence length for technical features
DEFAULT_NORMALIZATION_WINDOW = 120  # Window for price normalization
DEFAULT_TECHNICAL_INDICATORS_ENABLED = True
DEFAULT_NAN_THRESHOLD = 0.5  # 50% threshold for NaN validation
DEFAULT_RSI_PERIOD = 14
DEFAULT_ATR_PERIOD = 14
DEFAULT_BOLLINGER_PERIOD = 20
DEFAULT_BOLLINGER_STD_DEV = 2.0
DEFAULT_MACD_FAST_PERIOD = 12
DEFAULT_MACD_SLOW_PERIOD = 26
DEFAULT_MACD_SIGNAL_PERIOD = 9
DEFAULT_MA_PERIODS = [20, 50, 200]

# Optimization Constants (Phase 10)
DEFAULT_OPTIMIZATION_ENABLED = True
DEFAULT_OPTIMIZATION_INTERVAL = 24  # hours
DEFAULT_PERFORMANCE_LOOKBACK_DAYS = 30
DEFAULT_MIN_TRADES_FOR_OPTIMIZATION = 50
DEFAULT_OPTIMIZATION_CONFIDENCE_THRESHOLD = 0.8
DEFAULT_MAX_PARAMETER_CHANGE = 0.2  # 20% max change per cycle
DEFAULT_OPTIMIZATION_COOLDOWN_HOURS = 6
DEFAULT_WIN_RATE_THRESHOLD = 0.45
DEFAULT_SHARPE_RATIO_THRESHOLD = 0.5
DEFAULT_DRAWDOWN_THRESHOLD = 0.15

# Error Handling Constants
DEFAULT_ERROR_COOLDOWN = 30  # seconds to wait after consecutive errors

# Core Trading Defaults (used across backtest and live engines)
DEFAULT_STOP_LOSS_PCT = 0.05  # 5% stop loss
DEFAULT_MIN_STOP_LOSS_PCT = 0.01  # 1% minimum stop loss
DEFAULT_MAX_STOP_LOSS_PCT = 0.20  # 20% maximum stop loss
DEFAULT_TAKE_PROFIT_PCT = 0.04  # 4% take profit
DEFAULT_MAX_POSITION_SIZE = 0.1  # 10% max position size
DEFAULT_BASE_RISK_PER_TRADE = 0.02  # 2% risk per trade
DEFAULT_MAX_RISK_PER_TRADE = 0.03  # 3% maximum risk per trade
DEFAULT_MAX_DAILY_RISK = 0.06  # 6% maximum daily risk
DEFAULT_MAX_CORRELATED_RISK = 0.10  # 10% maximum risk for correlated positions
DEFAULT_MAX_DRAWDOWN = 0.20  # 20% maximum drawdown (fraction)
DEFAULT_FEE_RATE = 0.001  # 0.1% trading fee
DEFAULT_SLIPPAGE_RATE = 0.0005  # 0.05% slippage

# Trading Engine Constants
DEFAULT_MAX_HOLDING_HOURS = 336  # 14 days
DEFAULT_MAX_FILLED_PRICE_DEVIATION = 0.5  # 50% max deviation for suspicious fill detection
DEFAULT_END_OF_DAY_FLAT = False
DEFAULT_WEEKEND_FLAT = False
DEFAULT_MARKET_TIMEZONE = "UTC"
DEFAULT_TIME_RESTRICTIONS = {
    "no_overnight": True,
    "no_weekend": True,
    "trading_hours_only": False,
}

# Regime Detection Defaults (constants)
DEFAULT_REGIME_ADJUST_POSITION_SIZE: bool = False
DEFAULT_REGIME_HYSTERESIS_K: int = 3
DEFAULT_REGIME_MIN_DWELL: int = 12
DEFAULT_REGIME_MIN_CONFIDENCE: float = 0.5
DEFAULT_REGIME_LOOKBACK_BUFFER: int = 5  # Additional lookback buffer for regime detection
DEFAULT_REGIME_CHECK_FREQUENCY: int = 50  # Check regime every N candles
DEFAULT_REGIME_WARMUP_CANDLES: int = 60  # Minimum candles before first regime check
DEFAULT_REGIME_MIN_DATA_LENGTH: int = 60  # Minimum dataframe length for regime analysis

# Timeframe-specific regime detection configurations
# Each timeframe has tuned parameters for optimal regime detection
DEFAULT_REGIME_CONFIG_1H = {
    "slope_window": 30,
    "hysteresis_k": 3,
    "min_dwell": 10,
    "trend_threshold": 0.001,
}
DEFAULT_REGIME_CONFIG_4H = {
    "slope_window": 20,
    "hysteresis_k": 2,
    "min_dwell": 5,
    "trend_threshold": 0.002,
}
DEFAULT_REGIME_CONFIG_1D = {
    "slope_window": 15,
    "hysteresis_k": 2,
    "min_dwell": 3,
    "trend_threshold": 0.003,
}

# Prediction/Signal Defaults
DEFAULT_CONFIDENCE_SCORE: float = 0.5  # Default confidence when not provided by strategy

# Regime Position Size Multipliers (by market condition)
DEFAULT_REGIME_MULTIPLIER_BULL_LOW_VOL = 1.0  # Full size in ideal conditions
DEFAULT_REGIME_MULTIPLIER_BULL_HIGH_VOL = 0.7  # Reduced in volatile bull
DEFAULT_REGIME_MULTIPLIER_BEAR_LOW_VOL = 0.8  # Reduced in bear market
DEFAULT_REGIME_MULTIPLIER_BEAR_HIGH_VOL = 0.5  # Much reduced in volatile bear
DEFAULT_REGIME_MULTIPLIER_RANGE_LOW_VOL = 0.6  # Reduced in range market
DEFAULT_REGIME_MULTIPLIER_RANGE_HIGH_VOL = 0.3  # Very reduced in volatile range

# Regime Strategy Switching Defaults
DEFAULT_REGIME_SWITCH_COOLDOWN_MINUTES = 60  # Cooldown between switches
DEFAULT_REGIME_MIN_DURATION_BARS = 15  # Minimum bars before switching
DEFAULT_REGIME_TIMEFRAME_AGREEMENT = 0.6  # Require 60% agreement across timeframes
DEFAULT_REGIME_TRANSITION_SIZE_MULTIPLIER = 0.5  # Size multiplier during transitions
DEFAULT_REGIME_MAX_DRAWDOWN_SWITCH = 0.15  # Switch to defensive if drawdown > 15%
DEFAULT_REGIME_TIMEFRAME_WEIGHTS = {"1h": 1.0, "4h": 1.5, "1d": 2.0}  # Higher timeframes weighted more
DEFAULT_REGIME_TIMEFRAMES = ["1h", "4h", "1d"]  # Timeframes for multi-timeframe analysis

# CPU Optimization Constants
DEFAULT_CHECK_INTERVAL = 60  # Base check interval in seconds
DEFAULT_MIN_CHECK_INTERVAL = 30  # Minimum check interval (high activity)
DEFAULT_MAX_CHECK_INTERVAL = 300  # Maximum check interval (low activity)
DEFAULT_PERFORMANCE_MONITOR_INTERVAL = 30  # Performance monitoring interval
DEFAULT_SLEEP_POLL_INTERVAL = 0.5  # Sleep polling interval (reduced from 0.1s)
DEFAULT_ACCOUNT_SNAPSHOT_INTERVAL = 1800  # Account snapshot interval (30 minutes)
DEFAULT_DATA_FRESHNESS_THRESHOLD = 120  # Skip processing if data is older than 2 minutes

# Account Synchronization Constants
DEFAULT_ACCOUNT_SYNC_MIN_INTERVAL_MINUTES = 5  # Minimum minutes between account syncs
DEFAULT_BALANCE_DISCREPANCY_THRESHOLD_PCT = 1.0  # Log warning if discrepancy exceeds 1%
DEFAULT_POSITION_SIZE_COMPARISON_TOLERANCE = 0.0001  # Tolerance for position size changes

# Time Window Constants
DEFAULT_SENTIMENT_RECENT_WINDOW_HOURS = 4  # Apply live sentiment to last N hours of candles
DEFAULT_RECENT_TRADE_LOOKBACK_HOURS = 1  # Consider trades as "recent" for activity checks

# Trailing stop fallback (used in safety mode when ATR is unavailable)
DEFAULT_FALLBACK_TRAILING_PCT = 0.01  # 1% trailing distance as conservative fallback

# Dynamic Risk Management Constants
DEFAULT_DYNAMIC_RISK_ENABLED = True
DEFAULT_PERFORMANCE_WINDOW_DAYS = 30
DEFAULT_DRAWDOWN_THRESHOLDS = [0.05, 0.10, 0.15]  # 5%, 10%, 15%
DEFAULT_RISK_REDUCTION_FACTORS = [0.8, 0.6, 0.4]
DEFAULT_RECOVERY_THRESHOLDS = [0.02, 0.05]  # 2%, 5%
DEFAULT_VOLATILITY_ADJUSTMENT_ENABLED = True
DEFAULT_VOLATILITY_WINDOW_DAYS = 30
DEFAULT_HIGH_VOLATILITY_THRESHOLD = 0.03  # 3% daily volatility
DEFAULT_LOW_VOLATILITY_THRESHOLD = 0.01  # 1% daily volatility
DEFAULT_VOLATILITY_RISK_MULTIPLIERS = (0.7, 1.3)  # (high_vol, low_vol)
DEFAULT_MIN_TRADES_FOR_DYNAMIC_ADJUSTMENT = 10

# Partial operations defaults (partial exits and scale-ins)
DEFAULT_MAX_PARTIAL_EXITS_PER_CYCLE = 10  # Defense-in-depth limit for partial exits per cycle
DEFAULT_PARTIAL_EXIT_TARGETS = [0.03, 0.06, 0.10]  # 3%, 6%, 10%
DEFAULT_PARTIAL_EXIT_SIZES = [0.25, 0.25, 0.50]  # 25%, 25%, 50% of original size
DEFAULT_SCALE_IN_THRESHOLDS = [0.02, 0.05]  # 2%, 5%
DEFAULT_SCALE_IN_SIZES = [0.25, 0.25]  # 25%, 25% of original size
DEFAULT_MAX_SCALE_INS = 2

# Trailing Stop Defaults
DEFAULT_TRAILING_ACTIVATION_THRESHOLD = 0.015  # 1.5%
DEFAULT_TRAILING_DISTANCE_PCT = 0.005  # 0.5%
DEFAULT_TRAILING_DISTANCE_ATR_MULT = 1.5
DEFAULT_BREAKEVEN_THRESHOLD = 0.02  # 2.0%
DEFAULT_BREAKEVEN_BUFFER = 0.001  # 0.1%

# Correlation Control Defaults
DEFAULT_CORRELATION_WINDOW_DAYS = 30
DEFAULT_CORRELATION_THRESHOLD = 0.7
DEFAULT_MAX_CORRELATED_EXPOSURE = 0.15  # 15%
DEFAULT_CORRELATION_UPDATE_FREQUENCY_HOURS = 1
DEFAULT_CORRELATION_SAMPLE_MIN_SIZE = 20

# MFE/MAE Tracking Defaults
DEFAULT_MFE_MAE_UPDATE_FREQUENCY_SECONDS = 60
DEFAULT_MFE_MAE_PRECISION_DECIMALS = 8
DEFAULT_MFE_MAE_LOG_LEVEL = "INFO"

# Numeric Precision Constants
DEFAULT_EPSILON = 1e-9  # Small value for floating point comparisons
DEFAULT_SYMBOL_STEP_SIZE = 0.00001  # Fallback step size for quantity rounding
DEFAULT_BASIS_BALANCE_FALLBACK = 10000.0  # Fallback balance when entry_balance unavailable

# Retry Configuration Constants
DEFAULT_STOP_LOSS_MAX_RETRIES = 3  # Maximum retry attempts for stop-loss placement
DEFAULT_STOP_LOSS_RETRY_DELAY = 1.0  # Initial delay between retries (seconds)
DEFAULT_RETRY_BACKOFF_MULTIPLIER = 2  # Exponential backoff multiplier

# Regime Multiplier Fallback
DEFAULT_REGIME_UNKNOWN_MULTIPLIER = 0.5  # Conservative multiplier for unknown regimes

# Strategy Confidence Thresholds
DEFAULT_STRATEGY_MIN_CONFIDENCE = 0.3  # Standard minimum confidence for signal generation
DEFAULT_STRATEGY_MIN_CONFIDENCE_AGGRESSIVE = 0.2  # Lower threshold for aggressive strategies
DEFAULT_STRATEGY_MIN_CONFIDENCE_CONSERVATIVE = 0.4  # Higher threshold for conservative sizers

# Strategy Position Sizing Base Fractions
DEFAULT_STRATEGY_BASE_FRACTION = 0.2  # Default base fraction for ML strategies
DEFAULT_STRATEGY_BASE_FRACTION_SMALL = 0.04  # Smaller base fraction for conservative strategies

# Confidence Scale Factors
DEFAULT_CONFIDENCE_SCALE_FACTOR_MOMENTUM = 8  # Scale factor for momentum-based confidence
