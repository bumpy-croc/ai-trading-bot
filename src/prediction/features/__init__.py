"""
Feature Engineering Package

This package provides feature extraction and engineering capabilities
for the prediction engine.
"""

from .base import FeatureExtractor
from .market import MarketFeatureExtractor
from .pipeline import FeaturePipeline
from .schemas import (
    SENTIMENT_FEATURES_SCHEMA,
    TECHNICAL_FEATURES_SCHEMA,
    FeatureDefinition,
    FeatureSchema,
    FeatureType,
    NormalizationMethod,
)
from .sentiment import SentimentFeatureExtractor
from .technical import TechnicalFeatureExtractor

__all__ = [
    "FeatureExtractor",
    "FeaturePipeline",
    "TechnicalFeatureExtractor",
    "SentimentFeatureExtractor",
    "MarketFeatureExtractor",
    "FeatureDefinition",
    "FeatureSchema",
    "FeatureType",
    "NormalizationMethod",
    "TECHNICAL_FEATURES_SCHEMA",
    "SENTIMENT_FEATURES_SCHEMA",
]
