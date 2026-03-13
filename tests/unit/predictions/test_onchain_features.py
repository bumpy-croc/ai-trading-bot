"""Tests for OnChainFeatureExtractor."""

import numpy as np
import pandas as pd
import pytest

from src.prediction.features.onchain import OnChainFeatureExtractor


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


class TestOnChainFeatureExtractor:
    """Tests for OnChainFeatureExtractor."""

    def test_disabled_returns_neutral_values(self, sample_ohlcv):
        """Disabled extractor returns zeros for all features."""
        extractor = OnChainFeatureExtractor(enabled=False)
        result = extractor.extract(sample_ohlcv)

        for feature in extractor.get_feature_names():
            assert feature in result.columns
            assert (result[feature] == 0.0).all()

    def test_enabled_produces_all_features(self, sample_ohlcv):
        """Enabled extractor produces all expected feature columns."""
        extractor = OnChainFeatureExtractor(enabled=True)
        result = extractor.extract(sample_ohlcv)

        expected = [
            "exchange_netflow",
            "whale_ratio",
            "supply_in_profit_pct",
            "hodl_wave_signal",
            "active_addresses_zscore",
        ]
        for feature in expected:
            assert feature in result.columns

    def test_features_normalized_to_range(self, sample_ohlcv):
        """All features are clipped to [-1, 1] range."""
        extractor = OnChainFeatureExtractor(enabled=True)
        result = extractor.extract(sample_ohlcv)

        for feature in extractor.get_feature_names():
            assert result[feature].min() >= -1.0
            assert result[feature].max() <= 1.0

    def test_no_nan_or_inf_values(self, sample_ohlcv):
        """Features contain no NaN or infinite values."""
        extractor = OnChainFeatureExtractor(enabled=True)
        result = extractor.extract(sample_ohlcv)

        for feature in extractor.get_feature_names():
            assert not result[feature].isna().any(), f"{feature} has NaN values"
            assert not np.isinf(result[feature]).any(), f"{feature} has inf values"

    def test_preserves_original_columns(self, sample_ohlcv):
        """Original OHLCV columns are preserved in output."""
        extractor = OnChainFeatureExtractor(enabled=True)
        result = extractor.extract(sample_ohlcv)

        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns
            pd.testing.assert_series_equal(result[col], sample_ohlcv[col])

    def test_invalid_input_raises_error(self):
        """Invalid input raises ValueError."""
        extractor = OnChainFeatureExtractor(enabled=True)
        bad_df = pd.DataFrame({"x": [1, 2, 3]})

        with pytest.raises(ValueError, match="Invalid input data"):
            extractor.extract(bad_df)

    def test_empty_input_raises_error(self):
        """Empty input raises ValueError."""
        extractor = OnChainFeatureExtractor(enabled=True)

        with pytest.raises(ValueError, match="Invalid input data"):
            extractor.extract(pd.DataFrame())

    def test_get_feature_names(self):
        """get_feature_names returns expected list."""
        extractor = OnChainFeatureExtractor()
        names = extractor.get_feature_names()

        assert len(names) == 5
        assert "exchange_netflow" in names
        assert "active_addresses_zscore" in names

    def test_get_config(self):
        """get_config returns expected configuration."""
        extractor = OnChainFeatureExtractor(enabled=True, cache_ttl=600)
        config = extractor.get_config()

        assert config["enabled"] is True
        assert config["cache_ttl"] == 600
        assert config["name"] == "onchain"

    def test_deterministic_output(self, sample_ohlcv):
        """Same input produces same output (deterministic simulation)."""
        extractor = OnChainFeatureExtractor(enabled=True)
        result1 = extractor.extract(sample_ohlcv)
        result2 = extractor.extract(sample_ohlcv)

        for feature in extractor.get_feature_names():
            pd.testing.assert_series_equal(result1[feature], result2[feature])

    def test_does_not_modify_input(self, sample_ohlcv):
        """Extract does not modify the input DataFrame."""
        extractor = OnChainFeatureExtractor(enabled=True)
        original = sample_ohlcv.copy()
        extractor.extract(sample_ohlcv)

        pd.testing.assert_frame_equal(sample_ohlcv, original)
