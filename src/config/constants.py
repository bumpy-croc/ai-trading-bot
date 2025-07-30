"""
Centralized constant values for the AI Trading Bot project.
"""

DEFAULT_INITIAL_BALANCE: float = 1000  # Default starting balance in USD

# Prediction Engine Constants
DEFAULT_PREDICTION_HORIZONS = [1]  # Single horizon for MVP
DEFAULT_MIN_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_MAX_PREDICTION_LATENCY = 0.1  # seconds
DEFAULT_MODEL_REGISTRY_PATH = "src/ml"
DEFAULT_ENABLE_SENTIMENT = False  # Disabled for MVP
DEFAULT_ENABLE_MARKET_MICROSTRUCTURE = False  # Disabled for MVP
DEFAULT_FEATURE_CACHE_TTL = 300  # seconds
DEFAULT_MODEL_CACHE_TTL = 600  # seconds 