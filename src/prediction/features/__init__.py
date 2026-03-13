"""
Feature Engineering Package

This package provides feature extraction and engineering capabilities
for the prediction engine.
"""

from src.tech.features.base import FeatureExtractor
from src.tech.features.schemas import (
    SENTIMENT_FEATURES_SCHEMA,
    TECHNICAL_FEATURES_SCHEMA,
    FeatureDefinition,
    FeatureSchema,
    FeatureType,
    NormalizationMethod,
)
from src.tech.features.technical import TechnicalFeatureExtractor

from .enhanced_sentiment import EnhancedSentimentExtractor
from .fusion import FeatureFusionPipeline
from .macro import MacroFeatureExtractor
from .market import MarketFeatureExtractor
from .onchain import OnChainFeatureExtractor
from .pipeline import FeaturePipeline
from .sentiment import SentimentFeatureExtractor

__all__ = [
    "FeatureExtractor",
    "FeaturePipeline",
    "FeatureFusionPipeline",
    "TechnicalFeatureExtractor",
    "SentimentFeatureExtractor",
    "MarketFeatureExtractor",
    "OnChainFeatureExtractor",
    "MacroFeatureExtractor",
    "EnhancedSentimentExtractor",
    "FeatureDefinition",
    "FeatureSchema",
    "FeatureType",
    "NormalizationMethod",
    "TECHNICAL_FEATURES_SCHEMA",
    "SENTIMENT_FEATURES_SCHEMA",
]
