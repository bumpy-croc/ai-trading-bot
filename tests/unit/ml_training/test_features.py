"""Unit tests for ML training pipeline feature engineering."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.ml.training_pipeline.features import (
    COVERAGE_WEIGHT,
    DAYS_PER_YEAR,
    FRESHNESS_WEIGHT,
    MAX_DATA_FRESHNESS_DAYS,
    QUALITY_THRESHOLD_HIGH,
    QUALITY_THRESHOLD_MEDIUM,
    RSI_MAX,
    RSI_WINDOW,
    SMA_WINDOWS,
    assess_sentiment_data_quality,
    create_robust_features,
    merge_price_sentiment_data,
)


@pytest.mark.fast
class TestConstants:
    """Test that constants are defined with expected values."""

    def test_sentiment_quality_constants(self):
        assert MAX_DATA_FRESHNESS_DAYS == 999
        assert COVERAGE_WEIGHT == 0.6
        assert FRESHNESS_WEIGHT == 0.4
        assert DAYS_PER_YEAR == 365
        assert QUALITY_THRESHOLD_HIGH == 0.8
        assert QUALITY_THRESHOLD_MEDIUM == 0.4

    def test_technical_indicator_constants(self):
        assert SMA_WINDOWS == [7, 14, 30]
        assert RSI_WINDOW == 14
        assert RSI_MAX == 100


@pytest.mark.fast
class TestAssessSentimentDataQuality:
    """Test sentiment data quality assessment."""

    def test_empty_sentiment_returns_price_only(self):
        # Arrange
        price_df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2025-01-01", periods=3, freq="D"),
        )
        sentiment_df = pd.DataFrame()

        # Act
        result = assess_sentiment_data_quality(sentiment_df, price_df)

        # Assert
        assert result["recommendation"] == "price_only"
        assert result["total_sentiment_points"] == 0
        assert result["data_freshness_days"] == MAX_DATA_FRESHNESS_DAYS
        assert "reason" in result
        assert result["reason"] == "No sentiment data available"

    def test_full_overlap_high_quality(self):
        # Arrange - sentiment covers full price period with recent data
        start = datetime(2025, 1, 1)
        price_df = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104]},
            index=pd.date_range(start, periods=5, freq="D"),
        )
        sentiment_df = pd.DataFrame(
            {"sentiment_score": [0.5, 0.6, 0.7, 0.8, 0.9]},
            index=pd.date_range(start, periods=5, freq="D"),
        )

        # Act
        result = assess_sentiment_data_quality(sentiment_df, price_df)

        # Assert
        assert result["recommendation"] in ["full_sentiment", "hybrid_with_fallback"]
        assert result["coverage_ratio"] == 1.0
        assert result["total_sentiment_points"] == 5
        assert result["total_price_points"] == 5
        assert result["quality_score"] > 0

    def test_partial_overlap_medium_quality(self):
        # Arrange - sentiment covers only half of price period
        start = datetime(2025, 1, 1)
        price_df = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104]},
            index=pd.date_range(start, periods=5, freq="D"),
        )
        # Sentiment only for last 2 days
        sentiment_df = pd.DataFrame(
            {"sentiment_score": [0.7, 0.8]},
            index=pd.date_range(start + timedelta(days=3), periods=2, freq="D"),
        )

        # Act
        result = assess_sentiment_data_quality(sentiment_df, price_df)

        # Assert
        assert result["coverage_ratio"] < 1.0
        assert result["coverage_ratio"] > 0.0
        assert result["total_sentiment_points"] == 2

    def test_no_overlap_returns_early(self):
        # Arrange - no temporal overlap
        price_df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2025-01-01", periods=3, freq="D"),
        )
        sentiment_df = pd.DataFrame(
            {"sentiment_score": [0.5, 0.6]},
            index=pd.date_range("2025-02-01", periods=2, freq="D"),
        )

        # Act
        result = assess_sentiment_data_quality(sentiment_df, price_df)

        # Assert - returns "unknown" when assessment cannot complete
        assert result["recommendation"] == "unknown"
        assert "reason" in result
        assert "no temporal overlap" in result["reason"].lower()

    def test_timezone_handling(self):
        # Arrange - mixed timezone-aware and naive timestamps
        price_df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC"),
        )
        sentiment_df = pd.DataFrame(
            {"sentiment_score": [0.5, 0.6, 0.7]},
            index=pd.date_range("2025-01-01", periods=3, freq="D"),  # No timezone
        )

        # Act - should not raise exception
        result = assess_sentiment_data_quality(sentiment_df, price_df)

        # Assert
        assert "recommendation" in result
        assert result["coverage_ratio"] >= 0


@pytest.mark.fast
class TestMergePriceSentimentData:
    """Test price and sentiment data merging."""

    def test_empty_sentiment_returns_price_only(self):
        # Arrange
        price_df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2025-01-01", periods=3, freq="D"),
        )
        sentiment_df = pd.DataFrame()

        # Act
        result = merge_price_sentiment_data(price_df, sentiment_df, "1d")

        # Assert
        pd.testing.assert_frame_equal(result, price_df)

    def test_merge_with_matching_timeframe(self):
        # Arrange
        index = pd.date_range("2025-01-01", periods=3, freq="D")
        price_df = pd.DataFrame({"close": [100, 101, 102]}, index=index)
        sentiment_df = pd.DataFrame({"sentiment_score": [0.5, 0.6, 0.7]}, index=index)

        # Act
        result = merge_price_sentiment_data(price_df, sentiment_df, "1d")

        # Assert
        assert "close" in result.columns
        assert "sentiment_score" in result.columns
        assert len(result) == 3

    def test_merge_with_resampling(self):
        # Arrange - hourly price data, daily sentiment data
        price_df = pd.DataFrame(
            {"close": [100, 101, 102, 103]},
            index=pd.date_range("2025-01-01", periods=4, freq="H"),
        )
        sentiment_df = pd.DataFrame(
            {"sentiment_score": [0.5]},
            index=pd.date_range("2025-01-01", periods=1, freq="D"),
        )

        # Act
        result = merge_price_sentiment_data(price_df, sentiment_df, "1h")

        # Assert
        assert "close" in result.columns
        assert "sentiment_score" in result.columns
        assert len(result) == 4


@pytest.mark.fast
class TestCreateRobustFeatures:
    """Test feature engineering."""

    def test_creates_scaled_price_features(self):
        # Arrange
        data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104] * 20,
                "high": [101, 102, 103, 104, 105] * 20,
                "low": [99, 100, 101, 102, 103] * 20,
                "close": [100.5, 101.5, 102.5, 103.5, 104.5] * 20,
                "volume": [1000, 1100, 1200, 1300, 1400] * 20,
            },
            index=pd.date_range("2025-01-01", periods=100, freq="D"),
        )
        sentiment_assessment = {"recommendation": "price_only"}

        # Act
        result_df, scalers, feature_names = create_robust_features(data, sentiment_assessment, 60)

        # Assert
        assert "open_scaled" in result_df.columns
        assert "high_scaled" in result_df.columns
        assert "low_scaled" in result_df.columns
        assert "close_scaled" in result_df.columns
        assert "volume_scaled" in result_df.columns
        assert "open_scaled" in feature_names
        assert "open" in scalers
        assert len(scalers) == 5  # 5 price features

    def test_creates_sma_features(self):
        # Arrange
        data = pd.DataFrame(
            {"close": list(range(100, 200))},
            index=pd.date_range("2025-01-01", periods=100, freq="D"),
        )
        data["open"] = data["close"]
        data["high"] = data["close"] + 1
        data["low"] = data["close"] - 1
        data["volume"] = 1000
        sentiment_assessment = {"recommendation": "price_only"}

        # Act
        result_df, scalers, feature_names = create_robust_features(data, sentiment_assessment, 60)

        # Assert
        for window in SMA_WINDOWS:
            assert f"sma_{window}" in result_df.columns
            assert f"sma_{window}_scaled" in result_df.columns
            assert f"sma_{window}_scaled" in feature_names

    def test_creates_rsi_features(self):
        # Arrange
        np.random.seed(42)
        close_prices = 100 + np.cumsum(np.random.randn(100))
        data = pd.DataFrame(
            {
                "open": close_prices,
                "high": close_prices + 1,
                "low": close_prices - 1,
                "close": close_prices,
                "volume": 1000,
            },
            index=pd.date_range("2025-01-01", periods=100, freq="D"),
        )
        sentiment_assessment = {"recommendation": "price_only"}

        # Act
        result_df, scalers, feature_names = create_robust_features(data, sentiment_assessment, 60)

        # Assert
        assert "rsi" in result_df.columns
        assert "rsi_scaled" in result_df.columns
        assert "rsi_scaled" in feature_names
        # RSI should be between 0 and 100
        rsi_values = result_df["rsi"].dropna()
        assert (rsi_values >= 0).all() and (rsi_values <= RSI_MAX).all()

    def test_handles_sentiment_features_when_recommended(self):
        # Arrange
        data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104] * 20,
                "high": [101, 102, 103, 104, 105] * 20,
                "low": [99, 100, 101, 102, 103] * 20,
                "close": [100.5, 101.5, 102.5, 103.5, 104.5] * 20,
                "volume": [1000, 1100, 1200, 1300, 1400] * 20,
                "sentiment_score": [0.5, 0.6, 0.7, 0.8, 0.9] * 20,
                "sentiment_volume": [100, 110, 120, 130, 140] * 20,
            },
            index=pd.date_range("2025-01-01", periods=100, freq="D"),
        )
        sentiment_assessment = {"recommendation": "full_sentiment"}

        # Act
        result_df, scalers, feature_names = create_robust_features(data, sentiment_assessment, 60)

        # Assert
        assert (
            "sentiment_score_filled" in result_df.columns
            or "sentiment_score_scaled" in result_df.columns
        )

    def test_drops_nans_from_rolling_windows(self):
        # Arrange
        data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104] * 20,
                "high": [101, 102, 103, 104, 105] * 20,
                "low": [99, 100, 101, 102, 103] * 20,
                "close": [100.5, 101.5, 102.5, 103.5, 104.5] * 20,
                "volume": [1000, 1100, 1200, 1300, 1400] * 20,
            },
            index=pd.date_range("2025-01-01", periods=100, freq="D"),
        )
        sentiment_assessment = {"recommendation": "price_only"}

        # Act
        result_df, scalers, feature_names = create_robust_features(data, sentiment_assessment, 60)

        # Assert - no NaNs should remain in feature columns
        for feature_name in feature_names:
            assert not result_df[feature_name].isna().any(), f"Feature {feature_name} contains NaNs"
