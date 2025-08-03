"""
Feature Engineering Package

This package provides feature extraction and engineering capabilities
for the prediction engine.
"""

from .base import FeatureExtractor
from .pipeline import FeaturePipeline
from .technical import TechnicalFeatureExtractor
from .sentiment import SentimentFeatureExtractor
from .market import MarketFeatureExtractor
from .schemas import (
    FeatureDefinition, FeatureSchema, FeatureType, NormalizationMethod,
    TECHNICAL_FEATURES_SCHEMA, SENTIMENT_FEATURES_SCHEMA
)

__all__ = [
    'FeatureExtractor',
    'FeaturePipeline', 
    'TechnicalFeatureExtractor',
    'SentimentFeatureExtractor',
    'MarketFeatureExtractor',
    'FeatureDefinition',
    'FeatureSchema', 
    'FeatureType',
    'NormalizationMethod',
    'TECHNICAL_FEATURES_SCHEMA',
    'SENTIMENT_FEATURES_SCHEMA'
]