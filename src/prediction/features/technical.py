"""Compatibility shim for TechnicalFeatureExtractor imports."""

from __future__ import annotations

import warnings

from src.tech.features.technical import TechnicalFeatureExtractor

warnings.warn(
    "src.prediction.features.technical is deprecated; import from "
    "src.tech.features.technical instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["TechnicalFeatureExtractor"]
