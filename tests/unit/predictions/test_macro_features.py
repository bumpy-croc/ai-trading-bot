"""Tests for MacroFeatureExtractor."""

import numpy as np
import pandas as pd
import pytest

from src.prediction.features.macro import MacroFeatureExtractor


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
class TestMacroFeatureExtractor:
    """Tests for MacroFeatureExtractor."""

    def test_disabled_returns_neutral_values(self, sample_ohlcv):
        """Disabled extractor returns zeros for all features."""
        extractor = MacroFeatureExtractor(enabled=False)
        result = extractor.extract(sample_ohlcv)

        for feature in extractor.get_feature_names():
            assert feature in result.columns
            assert (result[feature] == 0.0).all()

    def test_enabled_produces_all_features(self, sample_ohlcv):
        """Enabled extractor produces all expected feature columns."""
        extractor = MacroFeatureExtractor(enabled=True)
        result = extractor.extract(sample_ohlcv)

        expected = [
            "spx_trend",
            "dxy_trend",
            "treasury_10y_change",
            "gold_trend",
            "oil_trend",
        ]
        for feature in expected:
            assert feature in result.columns

    def test_features_normalized_to_range(self, sample_ohlcv):
        """All features are clipped to [-1, 1] range."""
        extractor = MacroFeatureExtractor(enabled=True)
        result = extractor.extract(sample_ohlcv)

        for feature in extractor.get_feature_names():
            assert result[feature].min() >= -1.0
            assert result[feature].max() <= 1.0

    def test_no_nan_or_inf_values(self, sample_ohlcv):
        """Features contain no NaN or infinite values."""
        extractor = MacroFeatureExtractor(enabled=True)
        result = extractor.extract(sample_ohlcv)

        for feature in extractor.get_feature_names():
            assert not result[feature].isna().any(), f"{feature} has NaN values"
            assert not np.isinf(result[feature]).any(), f"{feature} has inf values"

    def test_preserves_original_columns(self, sample_ohlcv):
        """Original OHLCV columns are preserved in output."""
        extractor = MacroFeatureExtractor(enabled=True)
        result = extractor.extract(sample_ohlcv)

        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns
            pd.testing.assert_series_equal(result[col], sample_ohlcv[col])

    def test_invalid_input_raises_error(self):
        """Invalid input raises ValueError."""
        extractor = MacroFeatureExtractor(enabled=True)

        with pytest.raises(ValueError, match="Invalid input data"):
            extractor.extract(pd.DataFrame({"x": [1, 2, 3]}))

    def test_get_feature_names(self):
        """get_feature_names returns expected list."""
        extractor = MacroFeatureExtractor()
        names = extractor.get_feature_names()

        assert len(names) == 5
        assert "spx_trend" in names
        assert "oil_trend" in names

    def test_get_config(self):
        """get_config returns expected configuration."""
        extractor = MacroFeatureExtractor(enabled=True, cache_ttl=1800)
        config = extractor.get_config()

        assert config["enabled"] is True
        assert config["cache_ttl"] == 1800
        assert config["name"] == "macro"

    def test_momentum_signal_helper(self, sample_ohlcv):
        """Internal _momentum_signal produces values in [-1, 1]."""
        extractor = MacroFeatureExtractor(enabled=True)
        signal = extractor._momentum_signal(sample_ohlcv["close"])

        assert signal.min() >= -1.0
        assert signal.max() <= 1.0
        assert not signal.isna().any()

    def test_dxy_inversely_correlated(self, sample_ohlcv):
        """DXY trend should be inversely related to SPX trend."""
        extractor = MacroFeatureExtractor(enabled=True)
        result = extractor.extract(sample_ohlcv)

        # DXY uses negative momentum, so correlation should be negative
        correlation = result["spx_trend"].corr(result["dxy_trend"])
        assert correlation < 0, "DXY should be inversely correlated with SPX proxy"

    def test_deterministic_output(self, sample_ohlcv):
        """Same input produces same output."""
        extractor = MacroFeatureExtractor(enabled=True)
        result1 = extractor.extract(sample_ohlcv)
        result2 = extractor.extract(sample_ohlcv)

        for feature in extractor.get_feature_names():
            pd.testing.assert_series_equal(result1[feature], result2[feature])

    def test_does_not_modify_input(self, sample_ohlcv):
        """Extract does not modify the input DataFrame."""
        extractor = MacroFeatureExtractor(enabled=True)
        original = sample_ohlcv.copy()
        extractor.extract(sample_ohlcv)

        pd.testing.assert_frame_equal(sample_ohlcv, original)

    def test_zero_price_raises_error(self):
        """Zero prices are rejected by strict validation."""
        extractor = MacroFeatureExtractor(enabled=True)
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
        extractor = MacroFeatureExtractor(enabled=True)
        df = pd.DataFrame({
            "open": [100.0, float("nan")],
            "high": [110.0, 210.0],
            "low": [90.0, 190.0],
            "close": [105.0, 205.0],
            "volume": [1000.0, 2000.0],
        })
        with pytest.raises(ValueError, match="prices must be positive"):
            extractor.extract(df)
