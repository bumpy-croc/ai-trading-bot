"""Tests for EnhancedSentimentExtractor."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.prediction.features.enhanced_sentiment import EnhancedSentimentExtractor


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


@pytest.mark.fast
class TestEnhancedSentimentExtractor:
    """Tests for EnhancedSentimentExtractor."""

    def test_disabled_returns_neutral_values(self, sample_ohlcv):
        """Disabled extractor returns zeros for all features."""
        extractor = EnhancedSentimentExtractor(enabled=False)
        result = extractor.extract(sample_ohlcv)

        for feature in extractor.get_feature_names():
            assert feature in result.columns
            assert (result[feature] == 0.0).all()

    @patch(
        "src.prediction.features.enhanced_sentiment.FearGreedProvider",
        side_effect=ConnectionError("Network error"),
    )
    def test_enabled_fallback_on_provider_failure(self, mock_provider, sample_ohlcv):
        """Falls back to price-derived proxy when FearGreedProvider fails."""
        extractor = EnhancedSentimentExtractor(enabled=True)
        result = extractor.extract(sample_ohlcv)

        # Should still produce all features even without provider
        for feature in extractor.get_feature_names():
            assert feature in result.columns
            assert not result[feature].isna().any()

    def test_enabled_produces_all_features(self, sample_ohlcv):
        """Enabled extractor produces all expected feature columns (with fallback)."""
        # Patch provider to avoid network calls
        with patch(
            "src.prediction.features.enhanced_sentiment.FearGreedProvider",
            side_effect=ConnectionError("no network"),
        ):
            extractor = EnhancedSentimentExtractor(enabled=True)
            result = extractor.extract(sample_ohlcv)

        expected = [
            "fear_greed_normalized",
            "social_volume_zscore",
            "news_sentiment_score",
            "composite_sentiment",
        ]
        for feature in expected:
            assert feature in result.columns

    def test_features_normalized_to_range(self, sample_ohlcv):
        """All features are clipped to [-1, 1] range."""
        with patch(
            "src.prediction.features.enhanced_sentiment.FearGreedProvider",
            side_effect=ConnectionError("no network"),
        ):
            extractor = EnhancedSentimentExtractor(enabled=True)
            result = extractor.extract(sample_ohlcv)

        for feature in extractor.get_feature_names():
            assert result[feature].min() >= -1.0
            assert result[feature].max() <= 1.0

    def test_no_nan_or_inf_values(self, sample_ohlcv):
        """Features contain no NaN or infinite values."""
        with patch(
            "src.prediction.features.enhanced_sentiment.FearGreedProvider",
            side_effect=ConnectionError("no network"),
        ):
            extractor = EnhancedSentimentExtractor(enabled=True)
            result = extractor.extract(sample_ohlcv)

        for feature in extractor.get_feature_names():
            assert not result[feature].isna().any(), f"{feature} has NaN values"
            assert not np.isinf(result[feature]).any(), f"{feature} has inf values"

    def test_composite_is_weighted_average(self, sample_ohlcv):
        """Composite sentiment is a weighted combination of components."""
        with patch(
            "src.prediction.features.enhanced_sentiment.FearGreedProvider",
            side_effect=ConnectionError("no network"),
        ):
            extractor = EnhancedSentimentExtractor(enabled=True)
            result = extractor.extract(sample_ohlcv)

        expected_composite = np.clip(
            0.4 * result["fear_greed_normalized"]
            + 0.3 * result["social_volume_zscore"]
            + 0.3 * result["news_sentiment_score"],
            -1.0,
            1.0,
        )
        pd.testing.assert_series_equal(
            result["composite_sentiment"],
            expected_composite,
            check_names=False,
        )

    def test_preserves_original_columns(self, sample_ohlcv):
        """Original OHLCV columns are preserved in output."""
        extractor = EnhancedSentimentExtractor(enabled=False)
        result = extractor.extract(sample_ohlcv)

        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns
            pd.testing.assert_series_equal(result[col], sample_ohlcv[col])

    def test_invalid_input_raises_error(self):
        """Invalid input raises ValueError."""
        extractor = EnhancedSentimentExtractor(enabled=True)

        with pytest.raises(ValueError, match="Invalid input data"):
            extractor.extract(pd.DataFrame({"x": [1, 2, 3]}))

    def test_get_feature_names(self):
        """get_feature_names returns expected list."""
        extractor = EnhancedSentimentExtractor()
        names = extractor.get_feature_names()

        assert len(names) == 4
        assert "fear_greed_normalized" in names
        assert "composite_sentiment" in names

    def test_get_config(self):
        """get_config returns expected configuration."""
        extractor = EnhancedSentimentExtractor(enabled=True)
        config = extractor.get_config()

        assert config["enabled"] is True
        assert config["name"] == "enhanced_sentiment"

    def test_does_not_modify_input(self, sample_ohlcv):
        """Extract does not modify the input DataFrame."""
        extractor = EnhancedSentimentExtractor(enabled=False)
        original = sample_ohlcv.copy()
        extractor.extract(sample_ohlcv)

        pd.testing.assert_frame_equal(sample_ohlcv, original)

    def test_zero_price_raises_error(self):
        """Zero prices are rejected by strict validation."""
        extractor = EnhancedSentimentExtractor(enabled=True)
        df = pd.DataFrame({
            "open": [100.0, 200.0],
            "high": [110.0, 210.0],
            "low": [90.0, 190.0],
            "close": [0.0, 200.0],
            "volume": [1000.0, 2000.0],
        })
        with pytest.raises(ValueError, match="prices must be positive"):
            extractor.extract(df)

    def test_nan_price_raises_error(self):
        """NaN prices are rejected by strict validation."""
        extractor = EnhancedSentimentExtractor(enabled=True)
        df = pd.DataFrame({
            "open": [100.0, float("nan")],
            "high": [110.0, 210.0],
            "low": [90.0, 190.0],
            "close": [105.0, 205.0],
            "volume": [1000.0, 2000.0],
        })
        with pytest.raises(ValueError, match="prices must be positive"):
            extractor.extract(df)
