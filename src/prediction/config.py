"""
Configuration management for the prediction engine.

This module provides a type-safe configuration class that integrates with the
existing ConfigManager system to load prediction engine settings.
"""

from dataclasses import dataclass, field
from typing import List

from config.config_manager import get_config
from config.constants import (
    DEFAULT_CONFIDENCE_SCALE_FACTOR,
    DEFAULT_DIRECTION_THRESHOLD,
    DEFAULT_ENABLE_MARKET_MICROSTRUCTURE,
    DEFAULT_ENABLE_SENTIMENT,
    DEFAULT_FEATURE_CACHE_TTL,
    DEFAULT_MAX_PREDICTION_LATENCY,
    DEFAULT_MIN_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_CACHE_TTL,
    DEFAULT_MODEL_REGISTRY_PATH,
    DEFAULT_PREDICTION_HORIZONS,
    DEFAULT_ENABLE_ENSEMBLE,
    DEFAULT_ENSEMBLE_METHOD,
    DEFAULT_ENABLE_REGIME_AWARE_CONFIDENCE,
)


@dataclass
class PredictionConfig:
    """
    Configuration for the prediction engine.

    This class provides a type-safe way to access prediction engine configuration
    settings loaded from the ConfigManager system.
    """

    prediction_horizons: List[int] = field(
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
            ),
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
            ensemble_method=config.get("ENSEMBLE_METHOD", default=DEFAULT_ENSEMBLE_METHOD),
            enable_regime_aware_confidence=config.get_bool(
                "ENABLE_REGIME_AWARE_CONFIDENCE", default=DEFAULT_ENABLE_REGIME_AWARE_CONFIDENCE
            ),
        )

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid
        """
        if not self.prediction_horizons:
            raise ValueError("At least one prediction horizon must be specified")

        if any(h <= 0 for h in self.prediction_horizons):
            raise ValueError("Prediction horizons must be positive integers")

        if not 0 <= self.min_confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")

        if self.max_prediction_latency <= 0:
            raise ValueError("Prediction latency must be positive")

        if self.feature_cache_ttl <= 0:
            raise ValueError("Feature cache TTL must be positive")

        if self.model_cache_ttl <= 0:
            raise ValueError("Model cache TTL must be positive")

        # Basic sanity for ensemble
        if self.ensemble_method not in ("mean", "median", "weighted"):
            raise ValueError("Invalid ensemble method; choose 'mean', 'median', or 'weighted'")

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
            f"regime_conf={self.enable_regime_aware_confidence})"
        )
