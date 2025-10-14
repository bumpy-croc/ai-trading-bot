from .base import BaseStrategy
from .ensemble_weighted import EnsembleWeighted
from .ml_adaptive import MlAdaptive
from .ml_basic import MlBasic
from .ml_sentiment import create_ml_sentiment_strategy
from .momentum_leverage import MomentumLeverage

# Create default instance for backward compatibility
MlSentiment = create_ml_sentiment_strategy()

__all__ = [
    "BaseStrategy",
    "MlBasic",
    "MlAdaptive",
    "MlSentiment",
    "EnsembleWeighted",
    "MomentumLeverage",
    "create_ml_sentiment_strategy",
]
