"""Compatibility shim for feature schema imports."""

from __future__ import annotations

import warnings

from src.tech.features.schemas import (
    SENTIMENT_FEATURES_SCHEMA,
    TECHNICAL_FEATURES_SCHEMA,
    FeatureDefinition,
    FeatureSchema,
    FeatureType,
    NormalizationMethod,
)

warnings.warn(
    "src.prediction.features.schemas is deprecated; import from "
    "src.tech.features.schemas instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "FeatureDefinition",
    "FeatureSchema",
    "FeatureType",
    "NormalizationMethod",
    "TECHNICAL_FEATURES_SCHEMA",
    "SENTIMENT_FEATURES_SCHEMA",
]
