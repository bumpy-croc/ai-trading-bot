"""
Centralized constant values for the AI Trading Bot project.
"""

DEFAULT_INITIAL_BALANCE: float = 1000  # Default starting balance in USD

# Prediction Engine Constants
DEFAULT_PREDICTION_HORIZONS = [1]  # Single horizon for MVP (number of time steps ahead)
DEFAULT_MIN_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence threshold for predictions
DEFAULT_MAX_PREDICTION_LATENCY = 0.1  # Maximum allowed prediction latency in seconds
DEFAULT_MODEL_REGISTRY_PATH = "src/ml"  # Path to ML model registry directory
DEFAULT_ENABLE_SENTIMENT = False  # Disabled for MVP - sentiment feature integration
DEFAULT_ENABLE_MARKET_MICROSTRUCTURE = False  # Disabled for MVP - market microstructure features
DEFAULT_FEATURE_CACHE_TTL = 300  # Feature cache time-to-live in seconds (5 minutes)
DEFAULT_MODEL_CACHE_TTL = 600  # Model cache time-to-live in seconds (10 minutes)

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
