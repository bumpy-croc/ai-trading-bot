"""Prediction engine public API.

This module keeps imports lightweight so that configuration helpers can be
used without eagerly importing heavy numerical dependencies during module
import (important for environments where numpy linked against Accelerate may
segfault during initialization).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .config import PredictionConfig

if TYPE_CHECKING:  # pragma: no cover - type checkers only
    from .engine import PredictionEngine, PredictionResult
    from .exceptions import (
        FeatureExtractionError,
        InvalidInputError,
        ModelNotFoundError,
        PredictionEngineError,
        PredictionTimeoutError,
    )


def _load_engine() -> Any:
    """Import the prediction engine module lazily."""
    return importlib.import_module("src.prediction.engine")


def _load_exceptions() -> Any:
    """Import the exception module lazily."""
    return importlib.import_module("src.prediction.exceptions")


def __getattr__(name: str) -> Any:
    if name in {"PredictionEngine", "PredictionResult"}:
        engine_mod = _load_engine()
        return getattr(engine_mod, name)

    exception_names = {
        "PredictionEngineError",
        "InvalidInputError",
        "ModelNotFoundError",
        "FeatureExtractionError",
        "PredictionTimeoutError",
    }
    if name in exception_names:
        exc_mod = _load_exceptions()
        return getattr(exc_mod, name)

    raise AttributeError(f"module 'src.prediction' has no attribute '{name}'")


def create_engine(
    enable_sentiment: bool = False, enable_market_microstructure: bool = False
) -> PredictionEngine:
    """Create a lazily imported prediction engine with common configuration."""

    from .engine import PredictionEngine

    config = PredictionConfig.from_config_manager()
    config.enable_sentiment = enable_sentiment
    config.enable_market_microstructure = enable_market_microstructure
    return PredictionEngine(config)


def create_minimal_engine() -> PredictionEngine:
    """Create a minimal prediction engine with basic configuration."""

    return create_engine(
        enable_sentiment=False, enable_market_microstructure=False
    )


def predict(data, model_name=None) -> PredictionResult:
    """Quick prediction helper using the minimal engine configuration."""

    engine = create_minimal_engine()
    return engine.predict(data, model_name)


__version__ = "1.0.0"
__all__ = [
    "PredictionConfig",
    "PredictionEngine",
    "PredictionResult",
    "PredictionEngineError",
    "InvalidInputError",
    "ModelNotFoundError",
    "FeatureExtractionError",
    "PredictionTimeoutError",
    "create_engine",
    "create_minimal_engine",
    "predict",
]
