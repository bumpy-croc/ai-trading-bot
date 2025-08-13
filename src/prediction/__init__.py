"""
Prediction Engine Package

This package provides a modular prediction engine for the AI Trading Bot,
separating prediction logic from trading strategy logic.

Key Components:
- Configuration management
- features: Feature extraction and engineering pipeline
- models: Model loading and inference management
- ensemble: Model ensemble and aggregation (Post-MVP)
- utils: Shared utilities and caching
"""

from .config import PredictionConfig
from .engine import PredictionEngine, PredictionResult
from .exceptions import (
    FeatureExtractionError,
    InvalidInputError,
    ModelNotFoundError,
    PredictionEngineError,
    PredictionTimeoutError,
)


# Factory functions for common configurations
def create_engine(
    enable_sentiment: bool = False, enable_market_microstructure: bool = False
) -> PredictionEngine:
    """
    Create a prediction engine with common configuration.

    Args:
        enable_sentiment: Whether to enable sentiment features
        enable_market_microstructure: Whether to enable market microstructure features

    Returns:
        PredictionEngine: Configured prediction engine
    """
    config = PredictionConfig.from_config_manager()
    config.enable_sentiment = enable_sentiment
    config.enable_market_microstructure = enable_market_microstructure

    return PredictionEngine(config)


def create_minimal_engine() -> PredictionEngine:
    """
    Create a minimal prediction engine with basic configuration.

    Returns:
        PredictionEngine: Minimal prediction engine for basic use cases
    """
    return create_engine(enable_sentiment=False, enable_market_microstructure=False)


# Convenience function for quick predictions
def predict(data, model_name=None) -> PredictionResult:
    """
    Quick prediction using default engine configuration.

    Args:
        data: Market data DataFrame
        model_name: Optional model name

    Returns:
        PredictionResult: Prediction result
    """
    engine = create_minimal_engine()
    return engine.predict(data, model_name)


__version__ = "1.0.0"
__all__ = [
    "PredictionEngine",
    "PredictionResult",
    "PredictionConfig",
    "PredictionEngineError",
    "InvalidInputError",
    "ModelNotFoundError",
    "FeatureExtractionError",
    "PredictionTimeoutError",
    "create_engine",
    "create_minimal_engine",
    "predict",
]
