"""Tests for FeatureFusionPipeline."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.prediction.features.fusion import FeatureFusionPipeline


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame({
        "open": close - np.random.rand(n) * 50,
        "high": close + np.random.rand(n) * 100,
        "low": close - np.random.rand(n) * 100,
        "close": close,
        "volume": np.random.rand(n) * 1e6 + 1e5,
    })


class TestFeatureFusionPipeline:
    """Tests for FeatureFusionPipeline."""

    def test_no_extractors_returns_original(self, sample_ohlcv):
        """Pipeline with no extractors returns data unchanged."""
        pipeline = FeatureFusionPipeline()
        result = pipeline.transform(sample_ohlcv)

        pd.testing.assert_frame_equal(result, sample_ohlcv)

    def test_onchain_only(self, sample_ohlcv):
        """Pipeline with only on-chain features enabled."""
        pipeline = FeatureFusionPipeline(enable_onchain=True)
        result = pipeline.transform(sample_ohlcv)

        assert "exchange_netflow" in result.columns
        assert "whale_ratio" in result.columns
        assert "spx_trend" not in result.columns
        assert "fear_greed_normalized" not in result.columns

    def test_macro_only(self, sample_ohlcv):
        """Pipeline with only macro features enabled."""
        pipeline = FeatureFusionPipeline(enable_macro=True)
        result = pipeline.transform(sample_ohlcv)

        assert "spx_trend" in result.columns
        assert "dxy_trend" in result.columns
        assert "exchange_netflow" not in result.columns

    @patch(
        "src.prediction.features.enhanced_sentiment.FearGreedProvider",
        side_effect=Exception("no network"),
    )
    def test_enhanced_sentiment_only(self, mock_provider, sample_ohlcv):
        """Pipeline with only enhanced sentiment features enabled."""
        pipeline = FeatureFusionPipeline(enable_enhanced_sentiment=True)
        result = pipeline.transform(sample_ohlcv)

        assert "fear_greed_normalized" in result.columns
        assert "composite_sentiment" in result.columns
        assert "exchange_netflow" not in result.columns

    @patch(
        "src.prediction.features.enhanced_sentiment.FearGreedProvider",
        side_effect=Exception("no network"),
    )
    def test_all_extractors_enabled(self, mock_provider, sample_ohlcv):
        """Pipeline with all feature groups enabled."""
        pipeline = FeatureFusionPipeline(
            enable_onchain=True,
            enable_macro=True,
            enable_enhanced_sentiment=True,
        )
        result = pipeline.transform(sample_ohlcv)

        # On-chain features
        assert "exchange_netflow" in result.columns
        assert "whale_ratio" in result.columns
        # Macro features
        assert "spx_trend" in result.columns
        assert "dxy_trend" in result.columns
        # Sentiment features
        assert "fear_greed_normalized" in result.columns
        assert "composite_sentiment" in result.columns
        # OHLCV preserved
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_all_features_in_valid_range(self, sample_ohlcv):
        """All generated features are in [-1, 1] range."""
        pipeline = FeatureFusionPipeline(
            enable_onchain=True,
            enable_macro=True,
        )
        result = pipeline.transform(sample_ohlcv)

        for feature in pipeline.get_feature_names():
            assert result[feature].min() >= -1.0, f"{feature} below -1"
            assert result[feature].max() <= 1.0, f"{feature} above 1"

    def test_empty_input_raises_error(self):
        """Empty input raises ValueError."""
        pipeline = FeatureFusionPipeline(enable_onchain=True)

        with pytest.raises(ValueError, match="Input data is empty"):
            pipeline.transform(pd.DataFrame())

    def test_none_input_raises_error(self):
        """None input raises ValueError."""
        pipeline = FeatureFusionPipeline()

        with pytest.raises(ValueError, match="Input data is empty"):
            pipeline.transform(None)

    def test_get_feature_names(self):
        """get_feature_names returns combined list from all extractors."""
        pipeline = FeatureFusionPipeline(
            enable_onchain=True,
            enable_macro=True,
        )
        names = pipeline.get_feature_names()

        assert "exchange_netflow" in names
        assert "spx_trend" in names
        assert len(names) == 10  # 5 onchain + 5 macro

    def test_get_feature_names_empty(self):
        """get_feature_names returns empty list when no extractors enabled."""
        pipeline = FeatureFusionPipeline()
        assert pipeline.get_feature_names() == []

    def test_get_extractor_names(self):
        """get_extractor_names returns list of enabled extractors."""
        pipeline = FeatureFusionPipeline(
            enable_onchain=True,
            enable_macro=True,
        )
        names = pipeline.get_extractor_names()

        assert "onchain" in names
        assert "macro" in names
        assert len(names) == 2

    def test_get_config(self):
        """get_config returns configuration for all extractors."""
        pipeline = FeatureFusionPipeline(enable_onchain=True)
        config = pipeline.get_config()

        assert "extractors" in config
        assert "onchain" in config["extractors"]

    def test_stats_tracking(self, sample_ohlcv):
        """Pipeline tracks transformation statistics."""
        pipeline = FeatureFusionPipeline(enable_onchain=True)

        pipeline.transform(sample_ohlcv)
        pipeline.transform(sample_ohlcv)
        stats = pipeline.get_stats()

        assert stats["total_transforms"] == 2
        assert "onchain" in stats["extraction_times"]
        assert len(stats["extraction_times"]["onchain"]) == 2

    def test_preserves_original_data(self, sample_ohlcv):
        """Transform does not modify the input DataFrame."""
        pipeline = FeatureFusionPipeline(enable_onchain=True, enable_macro=True)
        original = sample_ohlcv.copy()
        pipeline.transform(sample_ohlcv)

        pd.testing.assert_frame_equal(sample_ohlcv, original)

    def test_custom_kwargs_passed_to_extractors(self):
        """Custom kwargs are forwarded to extractors."""
        pipeline = FeatureFusionPipeline(
            enable_onchain=True,
            onchain_kwargs={"cache_ttl": 999},
        )
        config = pipeline.get_config()

        assert config["extractors"]["onchain"]["cache_ttl"] == 999
