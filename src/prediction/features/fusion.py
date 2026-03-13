"""
Feature fusion pipeline.

Combines technical, on-chain, macro, and sentiment features into a
unified feature matrix for model input.
"""

import logging
import time
from typing import Any

import pandas as pd

from src.tech.features.base import FeatureExtractor

from .enhanced_sentiment import EnhancedSentimentExtractor
from .macro import MacroFeatureExtractor
from .onchain import OnChainFeatureExtractor

logger = logging.getLogger(__name__)


class FeatureFusionPipeline:
    """
    Combines multiple feature groups into a unified feature matrix.

    Configurable feature groups allow enabling/disabling on-chain, macro,
    and enhanced sentiment features independently. Technical features are
    expected to already be present in the input DataFrame.
    """

    def __init__(
        self,
        enable_onchain: bool = False,
        enable_macro: bool = False,
        enable_enhanced_sentiment: bool = False,
        onchain_kwargs: dict[str, Any] | None = None,
        macro_kwargs: dict[str, Any] | None = None,
        sentiment_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize the feature fusion pipeline.

        Args:
            enable_onchain: Whether to include on-chain features
            enable_macro: Whether to include macro economic features
            enable_enhanced_sentiment: Whether to include enhanced sentiment features
            onchain_kwargs: Additional kwargs for OnChainFeatureExtractor
            macro_kwargs: Additional kwargs for MacroFeatureExtractor
            sentiment_kwargs: Additional kwargs for EnhancedSentimentExtractor
        """
        self.extractors: dict[str, FeatureExtractor] = {}

        if enable_onchain:
            kwargs = onchain_kwargs or {}
            self.extractors["onchain"] = OnChainFeatureExtractor(
                enabled=True, **kwargs
            )

        if enable_macro:
            kwargs = macro_kwargs or {}
            self.extractors["macro"] = MacroFeatureExtractor(
                enabled=True, **kwargs
            )

        if enable_enhanced_sentiment:
            kwargs = sentiment_kwargs or {}
            self.extractors["enhanced_sentiment"] = EnhancedSentimentExtractor(
                enabled=True, **kwargs
            )

        self.stats: dict[str, Any] = {
            "total_transforms": 0,
            "extraction_times": {},
        }

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all enabled feature extractors to the input data.

        Expects input DataFrame to already contain OHLCV columns (and
        optionally technical features from an upstream pipeline).

        Args:
            data: DataFrame with at least OHLCV columns

        Returns:
            DataFrame with all enabled feature groups appended

        Raises:
            ValueError: If input data is empty or None
        """
        if data is None or data.empty:
            raise ValueError("Input data is empty or None")

        result = data.copy()

        for name, extractor in self.extractors.items():
            start = time.time()
            result = extractor.extract(result)
            elapsed = time.time() - start

            if name not in self.stats["extraction_times"]:
                self.stats["extraction_times"][name] = []
            self.stats["extraction_times"][name].append(elapsed)

        self.stats["total_transforms"] += 1
        return result

    def get_feature_names(self) -> list[str]:
        """
        Get list of all feature names produced by enabled extractors.

        Returns:
            Combined list of feature names from all enabled extractors
        """
        names: list[str] = []
        for extractor in self.extractors.values():
            names.extend(extractor.get_feature_names())
        return names

    def get_extractor_names(self) -> list[str]:
        """Return list of enabled extractor names."""
        return list(self.extractors.keys())

    def get_config(self) -> dict[str, Any]:
        """Get configuration for the fusion pipeline."""
        return {
            "extractors": {
                name: extractor.get_config()
                for name, extractor in self.extractors.items()
            },
        }

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        return self.stats.copy()
