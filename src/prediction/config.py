"""
Configuration management for the prediction engine.
"""

from dataclasses import dataclass
from typing import List
from ..config.constants import (
    DEFAULT_PREDICTION_HORIZONS,
    DEFAULT_MIN_CONFIDENCE_THRESHOLD,
    DEFAULT_MAX_PREDICTION_LATENCY,
    DEFAULT_MODEL_REGISTRY_PATH,
    DEFAULT_ENABLE_SENTIMENT,
    DEFAULT_ENABLE_MARKET_MICROSTRUCTURE,
    DEFAULT_FEATURE_CACHE_TTL,
    DEFAULT_MODEL_CACHE_TTL,
)


@dataclass
class PredictionConfig:
    """Configuration for prediction engine"""
    
    prediction_horizons: List[int] = None
    min_confidence_threshold: float = DEFAULT_MIN_CONFIDENCE_THRESHOLD
    max_prediction_latency: float = DEFAULT_MAX_PREDICTION_LATENCY
    model_registry_path: str = DEFAULT_MODEL_REGISTRY_PATH
    enable_sentiment: bool = DEFAULT_ENABLE_SENTIMENT
    enable_market_microstructure: bool = DEFAULT_ENABLE_MARKET_MICROSTRUCTURE
    feature_cache_ttl: int = DEFAULT_FEATURE_CACHE_TTL
    model_cache_ttl: int = DEFAULT_MODEL_CACHE_TTL
    
    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = DEFAULT_PREDICTION_HORIZONS.copy()