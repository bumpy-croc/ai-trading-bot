"""Compatibility shim for legacy feature extractor imports."""

from __future__ import annotations

import warnings

from src.tech.features.base import FeatureExtractor

warnings.warn(
    "src.prediction.features.base is deprecated; import FeatureExtractor from "
    "src.tech.features.base instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["FeatureExtractor"]
