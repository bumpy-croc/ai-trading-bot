"""
Configuration management for the prediction engine.

This module provides a type-safe configuration class that integrates with the
existing ConfigManager system to load prediction engine settings.
"""

from dataclasses import dataclass, field

from src.config.config_manager import get_config
from src.config.constants import (
    DEFAULT_CONFIDENCE_SCALE_FACTOR,
    DEFAULT_DIRECTION_THRESHOLD,
    DEFAULT_ENABLE_ENSEMBLE,
    DEFAULT_ENABLE_MARKET_MICROSTRUCTURE,
    DEFAULT_ENABLE_REGIME_AWARE_CONFIDENCE,
    DEFAULT_ENABLE_SENTIMENT,
    DEFAULT_ENSEMBLE_METHOD,
    DEFAULT_FEATURE_CACHE_TTL,
    DEFAULT_MAX_PREDICTION_LATENCY,
    DEFAULT_MIN_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_CACHE_TTL,
    DEFAULT_MODEL_REGISTRY_PATH,
    DEFAULT_PREDICTION_CACHE_ENABLED,
    DEFAULT_PREDICTION_CACHE_MAX_SIZE,
    DEFAULT_PREDICTION_CACHE_TTL,
    DEFAULT_PREDICTION_HORIZONS,
)


@dataclass
class PredictionConfig:
    """
    Configuration for the prediction engine.

    This class provides a type-safe way to access prediction engine configuration
    settings loaded from the ConfigManager system.
    """

    prediction_horizons: list[int] = field(
        default_factory=lambda: DEFAULT_PREDICTION_HORIZONS.copy()
    )
    min_confidence_threshold: float = DEFAULT_MIN_CONFIDENCE_THRESHOLD
    max_prediction_latency: float = DEFAULT_MAX_PREDICTION_LATENCY
    model_registry_path: str = DEFAULT_MODEL_REGISTRY_PATH
    enable_sentiment: bool = DEFAULT_ENABLE_SENTIMENT
    enable_market_microstructure: bool = DEFAULT_ENABLE_MARKET_MICROSTRUCTURE
    feature_cache_ttl: int = DEFAULT_FEATURE_CACHE_TTL
    model_cache_ttl: int = DEFAULT_MODEL_CACHE_TTL
    confidence_scale_factor: float = DEFAULT_CONFIDENCE_SCALE_FACTOR
    direction_threshold: float = DEFAULT_DIRECTION_THRESHOLD
    # New ensemble/regime-aware options
    enable_ensemble: bool = DEFAULT_ENABLE_ENSEMBLE
    ensemble_method: str = DEFAULT_ENSEMBLE_METHOD
    enable_regime_aware_confidence: bool = DEFAULT_ENABLE_REGIME_AWARE_CONFIDENCE
    
    # Prediction caching options
    prediction_cache_enabled: bool = DEFAULT_PREDICTION_CACHE_ENABLED
    prediction_cache_ttl: int = DEFAULT_PREDICTION_CACHE_TTL
    prediction_cache_max_size: int = DEFAULT_PREDICTION_CACHE_MAX_SIZE

    @classmethod
    def from_config_manager(cls) -> "PredictionConfig":
        """
        Load configuration from ConfigManager.

        Returns:
            PredictionConfig: Configured prediction engine settings
        """
        config = get_config()

        # Parse prediction horizons as list of integers
        horizons_str_list = config.get_list(
            "PREDICTION_HORIZONS", default=[str(h) for h in DEFAULT_PREDICTION_HORIZONS]
        )
        prediction_horizons = [int(h) for h in horizons_str_list]

        return cls(
            prediction_horizons=prediction_horizons,
            min_confidence_threshold=config.get_float(
                "MIN_CONFIDENCE_THRESHOLD", default=DEFAULT_MIN_CONFIDENCE_THRESHOLD
            ),
            max_prediction_latency=config.get_float(
                "MAX_PREDICTION_LATENCY", default=DEFAULT_MAX_PREDICTION_LATENCY
            ),
            model_registry_path=config.get(
                "MODEL_REGISTRY_PATH", default=DEFAULT_MODEL_REGISTRY_PATH
            )
            or DEFAULT_MODEL_REGISTRY_PATH,
            enable_sentiment=config.get_bool("ENABLE_SENTIMENT", default=DEFAULT_ENABLE_SENTIMENT),
            enable_market_microstructure=config.get_bool(
                "ENABLE_MARKET_MICROSTRUCTURE", default=DEFAULT_ENABLE_MARKET_MICROSTRUCTURE
            ),
            feature_cache_ttl=config.get_int(
                "FEATURE_CACHE_TTL", default=DEFAULT_FEATURE_CACHE_TTL
            ),
            model_cache_ttl=config.get_int("MODEL_CACHE_TTL", default=DEFAULT_MODEL_CACHE_TTL),
            confidence_scale_factor=config.get_float(
                "CONFIDENCE_SCALE_FACTOR", default=DEFAULT_CONFIDENCE_SCALE_FACTOR
            ),
            direction_threshold=config.get_float(
                "DIRECTION_THRESHOLD", default=DEFAULT_DIRECTION_THRESHOLD
            ),
            enable_ensemble=config.get_bool("ENABLE_ENSEMBLE", default=DEFAULT_ENABLE_ENSEMBLE),
            ensemble_method=config.get("ENSEMBLE_METHOD", default=DEFAULT_ENSEMBLE_METHOD)
            or DEFAULT_ENSEMBLE_METHOD,
            enable_regime_aware_confidence=config.get_bool(
                "ENABLE_REGIME_AWARE_CONFIDENCE", default=DEFAULT_ENABLE_REGIME_AWARE_CONFIDENCE
            ),
            prediction_cache_enabled=config.get_bool(
                "PREDICTION_CACHE_ENABLED", default=DEFAULT_PREDICTION_CACHE_ENABLED
            ),
            prediction_cache_ttl=config.get_int(
                "PREDICTION_CACHE_TTL", default=DEFAULT_PREDICTION_CACHE_TTL
            ),
            prediction_cache_max_size=config.get_int(
                "PREDICTION_CACHE_MAX_SIZE", default=DEFAULT_PREDICTION_CACHE_MAX_SIZE
            ),
        )

    def validate(self) -> None:
        """Validate configuration parameters"""
        if self.min_confidence_threshold < 0.0 or self.min_confidence_threshold > 1.0:
            raise ValueError("min_confidence_threshold must be between 0.0 and 1.0")

        if self.max_prediction_latency <= 0.0:
            raise ValueError("max_prediction_latency must be positive")

        if self.feature_cache_ttl <= 0:
            raise ValueError("feature_cache_ttl must be positive")

        if self.model_cache_ttl <= 0:
            raise ValueError("model_cache_ttl must be positive")

        if self.confidence_scale_factor <= 0.0:
            raise ValueError("confidence_scale_factor must be positive")

        if self.direction_threshold < 0.0:
            raise ValueError("direction_threshold must be non-negative")

        if self.prediction_cache_ttl <= 0:
            raise ValueError("prediction_cache_ttl must be positive")

        if self.prediction_cache_max_size <= 0:
            raise ValueError("prediction_cache_max_size must be positive")

        # Validate ensemble method
        valid_ensemble_methods = {"mean", "median", "weighted"}
        if self.ensemble_method not in valid_ensemble_methods:
            raise ValueError(
                f"ensemble_method must be one of {valid_ensemble_methods}, "
                f"got '{self.ensemble_method}'"
            )

        if not isinstance(self.prediction_horizons, list) or not self.prediction_horizons:
            raise ValueError("prediction_horizons must be a non-empty list")

        for horizon in self.prediction_horizons:
            if not isinstance(horizon, int) or horizon <= 0:
                raise ValueError("All prediction_horizons must be positive integers")

    def __str__(self) -> str:
        """String representation of the configuration."""
        return (
            f"PredictionConfig("
            f"horizons={self.prediction_horizons}, "
            f"confidence_threshold={self.min_confidence_threshold}, "
            f"max_latency={self.max_prediction_latency}s, "
            f"sentiment={self.enable_sentiment}, "
            f"microstructure={self.enable_market_microstructure}, "
            f"ensemble={self.enable_ensemble}/{self.ensemble_method}, "
            f"regime_conf={self.enable_regime_aware_confidence}, "
            f"cache_enabled={self.prediction_cache_enabled}, "
            f"cache_ttl={self.prediction_cache_ttl}, "
            f"cache_max_size={self.prediction_cache_max_size})"
        )
