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