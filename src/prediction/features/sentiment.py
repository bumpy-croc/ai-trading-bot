"""
Sentiment Feature Extractor

This module extracts sentiment-based features from market sentiment data.
For MVP, this extractor is disabled and returns neutral sentiment values.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from .base import FeatureExtractor
from .schemas import SENTIMENT_FEATURES_SCHEMA
from src.config.constants import DEFAULT_ENABLE_SENTIMENT


class SentimentFeatureExtractor(FeatureExtractor):
    """
    Extracts sentiment features from market sentiment data.
    
    For MVP, this extractor is disabled and provides neutral sentiment values
    to maintain model compatibility.
    """
    
    def __init__(self, enabled: bool = DEFAULT_ENABLE_SENTIMENT):
        """
        Initialize the sentiment feature extractor.
        
        Args:
            enabled: Whether sentiment extraction is enabled (False for MVP)
        """
        super().__init__("sentiment")
        self.enabled = enabled
        self._feature_names = SENTIMENT_FEATURES_SCHEMA.get_feature_names()
    
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract sentiment features from data.
        
        For MVP, this returns neutral sentiment values.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with original data plus sentiment features
        """
        if not self.validate_input(data):
            raise ValueError("Invalid input data: missing required OHLCV columns")
        
        df = data.copy()
        
        if not self.enabled:
            # MVP: Return neutral sentiment values
            return self._add_neutral_sentiment_features(df)
        else:
            # Post-MVP: Implement actual sentiment extraction
            return self._extract_sentiment_features(df)
    
    def _add_neutral_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add neutral sentiment feature values for MVP compatibility."""
        # Add neutral sentiment values as defined in schema
        for feature_def in SENTIMENT_FEATURES_SCHEMA.features:
            if feature_def.default_value is not None:
                df[feature_def.name] = feature_def.default_value
            else:
                # Default neutral values
                if 'primary' in feature_def.name:
                    df[feature_def.name] = 0.5  # Neutral sentiment
                elif 'momentum' in feature_def.name:
                    df[feature_def.name] = 0.0  # No momentum
                elif 'volatility' in feature_def.name:
                    df[feature_def.name] = 0.3  # Low-moderate volatility
                elif 'confidence' in feature_def.name:
                    df[feature_def.name] = 0.7  # Moderate confidence
                else:
                    df[feature_def.name] = 0.0  # Default neutral
        
        return df
    
    def _extract_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract actual sentiment features (Post-MVP implementation).
        
        This method will be implemented in post-MVP phases when sentiment
        analysis is enabled.
        """
        # Placeholder for future implementation
        # Will integrate with SentiCrypt provider and other sentiment sources
        
        # For now, return neutral values
        return self._add_neutral_sentiment_features(df)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names this extractor produces."""
        return self._feature_names.copy()
    
    def get_config(self) -> dict:
        """Get configuration parameters for this extractor."""
        config = super().get_config()
        config.update({
            'enabled': self.enabled,
            'mvp_mode': not self.enabled
        })
        return config