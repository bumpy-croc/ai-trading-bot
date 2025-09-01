"""
Feature Pipeline

This module provides the main FeaturePipeline class that orchestrates
feature extraction from multiple extractors with caching and error handling.
"""

import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config.constants import DEFAULT_FEATURE_CACHE_TTL
from src.prediction.utils.caching import FeatureCache

from .base import FeatureExtractor
from .market import MarketFeatureExtractor
from .price_only import PriceOnlyFeatureExtractor
from .sentiment import SentimentFeatureExtractor
from .technical import TechnicalFeatureExtractor

# Default threshold for NaN values in features
DEFAULT_NAN_THRESHOLD = 0.5


class FeaturePipeline:
    """
    Feature engineering pipeline that orchestrates multiple feature extractors.

    This pipeline manages the extraction of technical, sentiment, and market
    microstructure features from raw market data, with built-in caching and
    error handling capabilities.
    """

    def __init__(
        self,
        config: dict,
        use_cache: bool = True,
        cache_ttl: int = DEFAULT_FEATURE_CACHE_TTL,
        custom_extractors: Optional[list[FeatureExtractor]] = None,
    ):
        """
        Initialize feature pipeline.

        Args:
            config: Configuration dictionary
            use_cache: Whether to use feature caching
            cache_ttl: Cache TTL in seconds
            custom_extractors: Optional list of custom feature extractors
        """
        self.config = config
        self.use_cache = use_cache
        self.cache = FeatureCache(cache_ttl) if use_cache else None

        # Initialize extractors
        self.extractors: dict[str, FeatureExtractor] = {}
        self._initialize_extractors(custom_extractors)

        # Performance tracking
        self.stats = {
            "total_extractions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "extraction_times": {},
            "total_time": 0.0,
        }

    def _initialize_extractors(self, custom_extractors: Optional[list[FeatureExtractor]] = None):
        """Initialize feature extractors based on configuration."""
        # Add technical feature extractor (MVP)
        if self.config.get("technical_features", {}).get("enabled", True):
            tech_config = self.config.get("technical_features", {}).copy()
            tech_config.pop("enabled", None)  # Remove enabled parameter
            self.extractors["technical"] = TechnicalFeatureExtractor(**tech_config)

        # Add sentiment feature extractor
        if self.config.get("sentiment_features", {}).get("enabled", False):
            self.extractors["sentiment"] = SentimentFeatureExtractor(
                **self.config.get("sentiment_features", {})
            )

        # Add market feature extractor
        if self.config.get("market_features", {}).get("enabled", False):
            self.extractors["market"] = MarketFeatureExtractor(
                **self.config.get("market_features", {})
            )

        # Add price-only feature extractor
        if self.config.get("price_only_features", {}).get("enabled", False):
            self.extractors["price_only"] = PriceOnlyFeatureExtractor(
                **self.config.get("price_only_features", {})
            )

        # Add custom extractors
        if custom_extractors:
            for extractor in custom_extractors:
                self.extractors[extractor.__class__.__name__] = extractor

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

        # Track cache hit status for this operation
        cache_hit = False

        try:
            # Start with original data
            result = data.copy()

            # Check for complete pipeline result in cache first
            cached_result = None
            if should_use_cache and self.cache:
                cached_result = self.cache.get(data, "pipeline_complete", self.get_config())

            if cached_result is not None:
                # Use cached complete result
                result = cached_result
                cache_hit = True
                self.stats["cache_hits"] += 1
                # Don't record extractor times when using cache since extractors weren't executed
            else:
                # Apply each feature extractor
                self.stats["cache_misses"] += 1  # Increment once per pipeline miss

                for extractor_name, extractor in self.extractors.items():
                    extractor_start = time.time()

                    # Extract features (no individual extractor caching for now)
                    result = extractor.extract(result)

                    # Track extractor performance
                    extractor_time = time.time() - extractor_start
                    if extractor_name not in self.stats["extraction_times"]:
                        self.stats["extraction_times"][extractor_name] = []
                    self.stats["extraction_times"][extractor_name].append(extractor_time)

                # Cache the final result if caching is enabled
                if should_use_cache and self.cache:
                    self.cache.set(
                        data,
                        "pipeline_complete",
                        self.get_config(),
                        result,
                        ttl=self.cache.default_ttl,
                    )

            # Handle missing values
            result = self._handle_missing_values(result)

            # Validate output
            self._validate_output(result)

            # Update performance stats
            total_time = time.time() - start_time
            self.stats["total_extractions"] += 1
            self.stats["total_time"] += total_time

            # Store cache hit status for this operation
            self._last_cache_hit = cache_hit

            return result

        except Exception as e:
            raise RuntimeError(f"Feature pipeline transformation failed: {str(e)}") from e

    def _handle_missing_values(
        self, data: pd.DataFrame, method: str = "forward_fill"
    ) -> pd.DataFrame:
        """
        Handle missing values in the feature data.

        Args:
            data: DataFrame with potential missing values
            method: Method to handle missing values

        Returns:
            DataFrame with missing values handled
        """
        if method == "forward_fill":
            return data.ffill()
        elif method == "backward_fill":
            return data.bfill()
        elif method == "interpolate":
            return data.interpolate()
        elif method == "drop":
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

    def get_feature_names(self) -> list[str]:
        """
        Get list of all feature names that this pipeline produces.

        Returns:
            List of feature names
        """
        all_features = []
        for extractor in self.extractors.values():
            if extractor.enabled:
                all_features.extend(extractor.get_feature_names())

        # Remove duplicates while preserving order
        return list(dict.fromkeys(all_features))

    def get_extractor_names(self) -> list[str]:
        """
        Get list of enabled extractor names.

        Returns:
            List of extractor names
        """
        return list(self.extractors.keys())

    def get_config(self) -> dict[str, Any]:
        """
        Get configuration for the entire pipeline.

        Returns:
            Configuration dictionary
        """
        return {
            "use_cache": self.use_cache,
            "cache_ttl": self.cache.default_ttl if self.cache else None,
            "extractors": {
                name: extractor.get_config() for name, extractor in self.extractors.items()
            },
        }

    def get_performance_stats(self) -> dict[str, Any]:
        """
        Get performance statistics for the pipeline.

        Returns:
            Performance statistics dictionary
        """
        stats = self.stats.copy()
        if self.cache:
            cache_stats = self.cache.get_stats()
            stats["cache"] = cache_stats
        return stats

    def clear_cache(self) -> None:
        """Clear the feature cache."""
        if self.cache:
            self.cache.clear()

    def get_cache_stats(self) -> Optional[dict[str, Any]]:
        """
        Get cache statistics.

        Returns:
            Cache statistics or None if caching is disabled
        """
        return self.cache.get_stats() if self.cache else None

    def validate_features(self, data: pd.DataFrame) -> dict[str, dict[str, bool]]:
        """
        Validate features from all extractors.

        Args:
            data: Input data

        Returns:
            Validation results for each extractor
        """
        results = {}
        for name, extractor in self.extractors.items():
            if extractor.enabled:
                try:
                    extractor.validate_features(data)
                    results[name] = {"valid": True, "error": None}
                except Exception as e:
                    results[name] = {"valid": False, "error": str(e)}
        return results

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
        self.stats = {
            "total_extractions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "extraction_times": {},
            "total_time": 0.0,
        }

    def get_last_cache_hit_status(self) -> bool:
        """
        Get the cache hit status for the last transform operation.

        Returns:
            True if the last operation was a cache hit, False otherwise
        """
        return getattr(self, "_last_cache_hit", False)
