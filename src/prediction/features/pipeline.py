"""
Feature Pipeline

This module provides the main FeaturePipeline class that orchestrates
feature extraction from multiple extractors with caching and error handling.
"""

import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Optional, Union
from .base import FeatureExtractor
from .technical import TechnicalFeatureExtractor
from .sentiment import SentimentFeatureExtractor
from .market import MarketFeatureExtractor
from ..utils.caching import FeatureCache, get_global_feature_cache
from src.config.constants import (
    DEFAULT_ENABLE_SENTIMENT, DEFAULT_ENABLE_MARKET_MICROSTRUCTURE,
    DEFAULT_FEATURE_CACHE_TTL, DEFAULT_TECHNICAL_INDICATORS_ENABLED,
    DEFAULT_NAN_THRESHOLD
)


class FeaturePipeline:
    """
    Feature engineering pipeline that orchestrates multiple feature extractors.
    
    This pipeline manages the extraction of technical, sentiment, and market
    microstructure features from raw market data, with built-in caching and
    error handling capabilities.
    """
    
    def __init__(self,
                 enable_technical: bool = DEFAULT_TECHNICAL_INDICATORS_ENABLED,
                 enable_sentiment: bool = DEFAULT_ENABLE_SENTIMENT,
                 enable_market_microstructure: bool = DEFAULT_ENABLE_MARKET_MICROSTRUCTURE,
                 use_cache: bool = True,
                 cache_ttl: int = DEFAULT_FEATURE_CACHE_TTL,
                 custom_extractors: Optional[List[FeatureExtractor]] = None):
        """
        Initialize the feature pipeline.
        
        Args:
            enable_technical: Whether to enable technical feature extraction
            enable_sentiment: Whether to enable sentiment feature extraction
            enable_market_microstructure: Whether to enable market microstructure features
            use_cache: Whether to use feature caching
            cache_ttl: Time-to-live for cached features in seconds
            custom_extractors: Optional list of custom feature extractors
        """
        self.enable_technical = enable_technical
        self.enable_sentiment = enable_sentiment
        self.enable_market_microstructure = enable_market_microstructure
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        
        # Initialize extractors
        self.extractors: Dict[str, FeatureExtractor] = {}
        self._initialize_extractors(custom_extractors)
        
        # Initialize cache
        self.cache = get_global_feature_cache() if use_cache else None
        
        # Performance tracking
        self._performance_stats = {
            'total_transforms': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'extractor_times': {}
        }
    
    def _initialize_extractors(self, custom_extractors: Optional[List[FeatureExtractor]] = None):
        """Initialize feature extractors based on configuration."""
        # Add technical feature extractor (MVP)
        if self.enable_technical:
            self.extractors['technical'] = TechnicalFeatureExtractor()
        
        # Add sentiment feature extractor (MVP: disabled)
        if self.enable_sentiment:
            self.extractors['sentiment'] = SentimentFeatureExtractor(enabled=True)
        
        # Add market microstructure extractor (MVP: disabled)
        if self.enable_market_microstructure:
            self.extractors['market'] = MarketFeatureExtractor(enabled=True)
        
        # Add custom extractors
        if custom_extractors:
            for extractor in custom_extractors:
                self.extractors[extractor.name] = extractor
    
    def transform(self, data: pd.DataFrame, use_cache: Optional[bool] = None) -> pd.DataFrame:
        """
        Transform raw OHLCV data into ML-ready features.
        
        Args:
            data: Raw market data (OHLCV format)
            use_cache: Override default cache setting for this transform
            
        Returns:
            DataFrame with original data plus extracted features
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If feature extraction fails
        """
        start_time = time.time()
        
        # Validate input
        if data is None or data.empty:
            raise ValueError("Input data is empty or None")
        
        # Check if we should use cache
        should_use_cache = use_cache if use_cache is not None else self.use_cache
        
        try:
            # Start with original data
            result = data.copy()
            
            # Check for complete pipeline result in cache first
            cached_result = None
            if should_use_cache and self.cache:
                cached_result = self.cache.get(
                    data, "pipeline_complete", self.get_config()
                )
            
            if cached_result is not None:
                # Use cached complete result
                result = cached_result
                self._performance_stats['cache_hits'] += 1
                # Don't record extractor times when using cache since extractors weren't executed
            else:
                # Apply each feature extractor
                self._performance_stats['cache_misses'] += 1  # Increment once per pipeline miss
                
                for extractor_name, extractor in self.extractors.items():
                    extractor_start = time.time()
                    
                    # Extract features (no individual extractor caching for now)
                    result = extractor.extract(result)
                    
                    # Track extractor performance
                    extractor_time = time.time() - extractor_start
                    if extractor_name not in self._performance_stats['extractor_times']:
                        self._performance_stats['extractor_times'][extractor_name] = []
                    self._performance_stats['extractor_times'][extractor_name].append(extractor_time)
                
                # Cache the final result if caching is enabled
                if should_use_cache and self.cache:
                    self.cache.set(
                        data, "pipeline_complete", self.get_config(),
                        result, ttl=self.cache_ttl
                    )
            
            # Handle missing values
            result = self._handle_missing_values(result)
            
            # Validate output
            self._validate_output(result)
            
            # Update performance stats
            total_time = time.time() - start_time
            self._performance_stats['total_transforms'] += 1
            self._performance_stats['total_time'] += total_time
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Feature pipeline transformation failed: {str(e)}")
    
    def _handle_missing_values(self, data: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values in the feature data.
        
        Args:
            data: DataFrame with potential missing values
            method: Method to handle missing values
            
        Returns:
            DataFrame with missing values handled
        """
        if method == 'forward_fill':
            return data.ffill()
        elif method == 'backward_fill':
            return data.bfill()
        elif method == 'interpolate':
            return data.interpolate()
        elif method == 'drop':
            return data.dropna()
        else:
            # Default: forward fill then backward fill
            return data.ffill().bfill()
    
    def _validate_output(self, data: pd.DataFrame) -> None:
        """
        Validate the output of feature extraction.
        
        Args:
            data: DataFrame to validate
            
        Raises:
            ValueError: If output validation fails
        """
        if data.empty:
            raise ValueError("Feature extraction resulted in empty DataFrame")
        
        # Check for excessive NaN values
        total_values = data.size
        nan_values = data.isna().sum().sum()
        nan_ratio = nan_values / total_values
        
        # Fail if the proportion of missing values exceeds the acceptable threshold for model input quality.
        if nan_ratio > DEFAULT_NAN_THRESHOLD:
            raise ValueError(f"Too many NaN values in output: {nan_ratio:.2%}")
        
        # Check for infinite values
        infinite_values = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
        if infinite_values > 0:
            raise ValueError(f"Found {infinite_values} infinite values in output")
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names that this pipeline produces.
        
        Returns:
            List of feature column names
        """
        all_features = []
        
        for extractor in self.extractors.values():
            all_features.extend(extractor.get_feature_names())
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(all_features))
    
    def get_extractor_names(self) -> List[str]:
        """
        Get list of enabled extractor names.
        
        Returns:
            List of extractor names
        """
        return list(self.extractors.keys())
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration for the entire pipeline.
        
        Returns:
            Dictionary with pipeline configuration
        """
        extractor_configs = {}
        for name, extractor in self.extractors.items():
            extractor_configs[name] = extractor.get_config()
        
        return {
            'enable_technical': self.enable_technical,
            'enable_sentiment': self.enable_sentiment,
            'enable_market_microstructure': self.enable_market_microstructure,
            'use_cache': self.use_cache,
            'cache_ttl': self.cache_ttl,
            'extractors': extractor_configs,
            'total_features': len(self.get_feature_names())
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the pipeline.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = self._performance_stats.copy()
        
        # Calculate averages
        if stats['total_transforms'] > 0:
            stats['avg_time_per_transform'] = stats['total_time'] / stats['total_transforms']
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        else:
            stats['avg_time_per_transform'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        # Calculate extractor averages
        extractor_avg_times = {}
        for name, times in stats['extractor_times'].items():
            extractor_avg_times[name] = np.mean(times) if times else 0.0
        stats['extractor_avg_times'] = extractor_avg_times
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the feature cache."""
        if self.cache:
            self.cache.clear()
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics if caching is enabled, None otherwise
        """
        return self.cache.get_stats() if self.cache else None
    
    def validate_features(self, data: pd.DataFrame) -> Dict[str, Dict[str, bool]]:
        """
        Validate features from all extractors.
        
        Args:
            data: DataFrame with extracted features
            
        Returns:
            Nested dictionary with validation results per extractor
        """
        validation_results = {}
        
        for name, extractor in self.extractors.items():
            if hasattr(extractor, 'validate_features'):
                validation_results[name] = extractor.validate_features(data)
            else:
                # Basic validation
                feature_names = extractor.get_feature_names()
                validation_results[name] = {
                    feature: feature in data.columns for feature in feature_names
                }
        
        return validation_results
    
    def add_extractor(self, extractor: FeatureExtractor) -> None:
        """
        Add a custom feature extractor to the pipeline.
        
        Args:
            extractor: Feature extractor to add
        """
        self.extractors[extractor.name] = extractor
    
    def remove_extractor(self, name: str) -> None:
        """
        Remove a feature extractor from the pipeline.
        
        Args:
            name: Name of the extractor to remove
        """
        if name in self.extractors:
            del self.extractors[name]
    
    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self._performance_stats = {
            'total_transforms': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'extractor_times': {}
        }