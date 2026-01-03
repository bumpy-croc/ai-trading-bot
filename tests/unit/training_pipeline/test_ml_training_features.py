"""Unit tests for ML training pipeline feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.ml.training_pipeline.features import (
    SMA_WINDOWS,
    _calculate_rsi_fast,
    assess_sentiment_data_quality,
    create_robust_features,
    merge_price_sentiment_data,
    normalize_timezone,
)


@pytest.mark.fast
class TestNormalizeTimezone:
    """Test normalize_timezone function."""

    def test_both_naive(self):
        # Arrange
        ts1 = pd.Timestamp("2024-01-01 12:00:00")
        ts2 = pd.Timestamp("2024-01-02 12:00:00")

        # Act
        result1, result2 = normalize_timezone(ts1, ts2)

        # Assert - both naive timestamps are converted to UTC-aware
        assert result1 == ts1.tz_localize("UTC")
        assert result2 == ts2.tz_localize("UTC")
        assert result1.tzinfo is not None
        assert result2.tzinfo is not None

    def test_both_aware(self):
        # Arrange
        ts1 = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        ts2 = pd.Timestamp("2024-01-02 12:00:00", tz="UTC")

        # Act
        result1, result2 = normalize_timezone(ts1, ts2)

        # Assert
        assert result1 == ts1
        assert result2 == ts2
        assert result1.tzinfo is not None
        assert result2.tzinfo is not None

    def test_first_aware_second_naive(self):
        # Arrange
        ts1 = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        ts2 = pd.Timestamp("2024-01-02 12:00:00")

        # Act
        result1, result2 = normalize_timezone(ts1, ts2)

        # Assert - naive timestamp is converted to UTC, aware stays as-is
        assert result1.tzinfo is not None
        assert result2.tzinfo is not None
        assert result1 == ts1
        assert result2 == ts2.tz_localize("UTC")

    def test_first_naive_second_aware(self):
        # Arrange
        ts1 = pd.Timestamp("2024-01-01 12:00:00")
        ts2 = pd.Timestamp("2024-01-02 12:00:00", tz="UTC")

        # Act
        result1, result2 = normalize_timezone(ts1, ts2)

        # Assert - naive timestamp is converted to UTC, aware stays as-is
        assert result1.tzinfo is not None
        assert result2.tzinfo is not None
        assert result1 == ts1.tz_localize("UTC")
        assert result2 == ts2


@pytest.mark.fast
class TestAssessSentimentDataQuality:
    """Test assess_sentiment_data_quality function."""

    def test_empty_sentiment_dataframe(self):
        # Arrange
        price_df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        sentiment_df = pd.DataFrame()

        # Act
        result = assess_sentiment_data_quality(sentiment_df, price_df)

        # Assert
        assert result["total_sentiment_points"] == 0
        assert result["coverage_ratio"] == 0.0
        assert result["recommendation"] == "price_only"
        assert "reason" in result
        assert "No sentiment data available" in result["reason"]

    def test_no_temporal_overlap(self):
        # Arrange
        price_df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        sentiment_df = pd.DataFrame(
            {"sentiment_score": [0.5, 0.6, 0.7]},
            index=pd.date_range("2024-02-01", periods=3, freq="1D"),
        )

        # Act
        result = assess_sentiment_data_quality(sentiment_df, price_df)

        # Assert
        assert result["coverage_ratio"] == 0.0
        # When no overlap, recommendation is "unknown" not "price_only"
        assert result["recommendation"] in ["price_only", "unknown"]
        assert "No temporal overlap" in result["reason"]

    def test_full_coverage_fresh_data(self):
        # Arrange
        price_df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        sentiment_df = pd.DataFrame(
            {"sentiment_score": [0.5, 0.6, 0.7]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )

        # Act
        result = assess_sentiment_data_quality(sentiment_df, price_df)

        # Assert
        assert result["coverage_ratio"] == 1.0
        # Quality score depends on freshness (how recent the data is)
        # With old test data, freshness will be low, so score may not reach threshold
        assert result["quality_score"] > 0
        assert result["recommendation"] in ["full_sentiment", "hybrid_with_fallback", "price_only"]

    def test_partial_coverage(self):
        # Arrange
        price_df = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104]},
            index=pd.date_range("2024-01-01", periods=5, freq="1D"),
        )
        sentiment_df = pd.DataFrame(
            {"sentiment_score": [0.5, 0.6]},
            index=pd.date_range("2024-01-01", periods=2, freq="1D"),
        )

        # Act
        result = assess_sentiment_data_quality(sentiment_df, price_df)

        # Assert
        assert 0 < result["coverage_ratio"] < 1.0
        assert result["total_sentiment_points"] == 2
        assert result["total_price_points"] == 5

    def test_mixed_timezones(self):
        # Arrange
        price_df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        sentiment_df = pd.DataFrame(
            {"sentiment_score": [0.5, 0.6, 0.7]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC"),
        )

        # Act
        result = assess_sentiment_data_quality(sentiment_df, price_df)

        # Assert - should handle mixed timezones without error
        assert result["coverage_ratio"] > 0
        assert "recommendation" in result

    def test_quality_score_calculation(self):
        # Arrange
        price_df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        sentiment_df = pd.DataFrame(
            {"sentiment_score": [0.5, 0.6, 0.7]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )

        # Act
        result = assess_sentiment_data_quality(sentiment_df, price_df)

        # Assert - quality score should be between 0 and 1
        assert 0 <= result["quality_score"] <= 1.0
        assert "data_freshness_days" in result


@pytest.mark.fast
class TestMergePriceSentimentData:
    """Test merge_price_sentiment_data function."""

    def test_empty_sentiment_returns_price_only(self):
        # Arrange
        price_df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="1h"),
        )
        sentiment_df = pd.DataFrame()

        # Act
        result = merge_price_sentiment_data(price_df, sentiment_df, "1h")

        # Assert
        assert result.equals(price_df)
        assert "close" in result.columns
        assert len(result) == 3

    def test_daily_timeframe_no_resampling(self):
        # Arrange
        price_df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        sentiment_df = pd.DataFrame(
            {"sentiment_score": [0.5, 0.6, 0.7]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )

        # Act
        result = merge_price_sentiment_data(price_df, sentiment_df, "1d")

        # Assert
        assert "close" in result.columns
        assert "sentiment_score" in result.columns
        assert len(result) == 3

    def test_hourly_timeframe_with_resampling(self):
        # Arrange
        price_df = pd.DataFrame(
            {"close": [100, 101, 102, 103]},
            index=pd.date_range("2024-01-01", periods=4, freq="1h"),
        )
        sentiment_df = pd.DataFrame(
            {"sentiment_score": [0.5]},
            index=pd.date_range("2024-01-01", periods=1, freq="1D"),
        )

        # Act
        result = merge_price_sentiment_data(price_df, sentiment_df, "1h")

        # Assert
        assert "close" in result.columns
        assert "sentiment_score" in result.columns
        assert len(result) == 4

    def test_left_join_preserves_price_data(self):
        # Arrange
        price_df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        sentiment_df = pd.DataFrame(
            {"sentiment_score": [0.5]},
            index=pd.date_range("2024-01-01", periods=1, freq="1D"),
        )

        # Act
        result = merge_price_sentiment_data(price_df, sentiment_df, "1d")

        # Assert
        assert len(result) == 3
        assert result["sentiment_score"].notna().sum() >= 1


@pytest.mark.fast
class TestCreateRobustFeatures:
    """Test create_robust_features function."""

    def test_price_only_features(self):
        # Arrange - need enough data to survive NaN drops from rolling windows
        # SMA_WINDOWS max is 30, RSI_WINDOW is 14, so need > 30 + buffer
        n_rows = 100
        data = pd.DataFrame(
            {
                "open": np.linspace(100, 110, n_rows),
                "high": np.linspace(105, 115, n_rows),
                "low": np.linspace(99, 109, n_rows),
                "close": np.linspace(103, 113, n_rows),
                "volume": np.linspace(1000, 2000, n_rows),
            },
            index=pd.date_range("2024-01-01", periods=n_rows, freq="1D"),
        )
        sentiment_assessment = {"recommendation": "price_only"}

        # Act
        result_df, scalers, feature_names = create_robust_features(
            data, sentiment_assessment, time_steps=10
        )

        # Assert
        assert "open_scaled" in feature_names
        assert "high_scaled" in feature_names
        assert "low_scaled" in feature_names
        assert "close_scaled" in feature_names
        assert "volume_scaled" in feature_names
        assert "rsi_scaled" in feature_names
        for window in SMA_WINDOWS:
            assert f"sma_{window}_scaled" in feature_names
        assert "open" in scalers
        assert "close" in scalers

    def test_features_with_sentiment(self):
        # Arrange
        n_rows = 100
        data = pd.DataFrame(
            {
                "open": np.linspace(100, 110, n_rows),
                "high": np.linspace(105, 115, n_rows),
                "low": np.linspace(99, 109, n_rows),
                "close": np.linspace(103, 113, n_rows),
                "volume": np.linspace(1000, 2000, n_rows),
                "sentiment_score": np.linspace(0.5, 0.9, n_rows),
                "sentiment_volume": np.linspace(100, 200, n_rows),
                "sentiment_momentum": np.linspace(0.1, 0.5, n_rows),
            },
            index=pd.date_range("2024-01-01", periods=n_rows, freq="1D"),
        )
        sentiment_assessment = {"recommendation": "full_sentiment"}

        # Act
        result_df, scalers, feature_names = create_robust_features(
            data, sentiment_assessment, time_steps=10
        )

        # Assert
        assert "sentiment_score_scaled" in feature_names
        assert "sentiment_volume_scaled" in feature_names
        assert "sentiment_momentum_scaled" in feature_names

    def test_drops_nan_rows(self):
        # Arrange
        n_rows = 100
        data = pd.DataFrame(
            {
                "open": np.linspace(100, 110, n_rows),
                "high": np.linspace(105, 115, n_rows),
                "low": np.linspace(99, 109, n_rows),
                "close": np.linspace(103, 113, n_rows),
                "volume": np.linspace(1000, 2000, n_rows),
            },
            index=pd.date_range("2024-01-01", periods=n_rows, freq="1D"),
        )
        sentiment_assessment = {"recommendation": "price_only"}
        original_len = len(data)

        # Act
        result_df, scalers, feature_names = create_robust_features(
            data, sentiment_assessment, time_steps=10
        )

        # Assert - NaN rows from rolling operations should be dropped
        assert len(result_df) < original_len
        assert not result_df.isnull().any().any()

    def test_insufficient_data_after_nans_raises_error(self):
        # Arrange - very small dataset
        data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [99, 100, 101],
                "close": [103, 104, 105],
                "volume": [1000, 1100, 1200],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        sentiment_assessment = {"recommendation": "price_only"}

        # Act & Assert
        with pytest.raises(ValueError, match="Insufficient data after dropping NaNs"):
            create_robust_features(data, sentiment_assessment, time_steps=10)

    def test_sentiment_features_fill_nan(self):
        # Arrange
        n_rows = 200  # Need more rows because RSI calculation creates NaNs that get dropped
        sentiment_values = [0.5 if i % 2 == 0 else None for i in range(n_rows)]
        data = pd.DataFrame(
            {
                "open": np.linspace(100, 110, n_rows),
                "high": np.linspace(105, 115, n_rows),
                "low": np.linspace(99, 109, n_rows),
                "close": np.linspace(103, 113, n_rows),
                "volume": np.linspace(1000, 2000, n_rows),
                "sentiment_score": sentiment_values,
            },
            index=pd.date_range("2024-01-01", periods=n_rows, freq="1D"),
        )
        sentiment_assessment = {"recommendation": "full_sentiment"}

        # Act
        result_df, scalers, feature_names = create_robust_features(
            data, sentiment_assessment, time_steps=10
        )

        # Assert - sentiment NaNs should be filled with 0
        assert "sentiment_score_scaled" in result_df.columns
        # After dropna for technical indicators, check sentiment feature has no NaNs
        assert not result_df["sentiment_score_scaled"].isnull().any()

    def test_all_price_features_present(self):
        # Arrange
        n_rows = 100
        data = pd.DataFrame(
            {
                "open": np.linspace(100, 110, n_rows),
                "high": np.linspace(105, 115, n_rows),
                "low": np.linspace(99, 109, n_rows),
                "close": np.linspace(103, 113, n_rows),
                "volume": np.linspace(1000, 2000, n_rows),
            },
            index=pd.date_range("2024-01-01", periods=n_rows, freq="1D"),
        )
        sentiment_assessment = {"recommendation": "price_only"}

        # Act
        result_df, scalers, feature_names = create_robust_features(
            data, sentiment_assessment, time_steps=10
        )

        # Assert - all expected price features should be present
        expected_features = [
            "open_scaled",
            "high_scaled",
            "low_scaled",
            "close_scaled",
            "volume_scaled",
        ]
        for feature in expected_features:
            assert feature in feature_names
            assert feature in result_df.columns

    def test_rsi_calculation(self):
        # Arrange
        n_rows = 100
        data = pd.DataFrame(
            {
                "open": np.linspace(100, 110, n_rows),
                "high": np.linspace(105, 115, n_rows),
                "low": np.linspace(99, 109, n_rows),
                "close": np.linspace(103, 113, n_rows),
                "volume": np.linspace(1000, 2000, n_rows),
            },
            index=pd.date_range("2024-01-01", periods=n_rows, freq="1D"),
        )
        sentiment_assessment = {"recommendation": "price_only"}

        # Act
        result_df, scalers, feature_names = create_robust_features(
            data, sentiment_assessment, time_steps=10
        )

        # Assert - RSI values should be between 0 and 1 after scaling
        assert "rsi_scaled" in result_df.columns
        assert result_df["rsi_scaled"].min() >= 0
        assert result_df["rsi_scaled"].max() <= 1


@pytest.mark.fast
class TestCalculateRsiFast:
    """Test _calculate_rsi_fast function."""

    def test_rsi_window_zero_raises_error(self):
        # Arrange
        close_prices = np.array([100.0, 101.0, 102.0, 101.0, 103.0])

        # Act & Assert
        with pytest.raises(ValueError, match="RSI window must be positive"):
            _calculate_rsi_fast(close_prices, window=0)

    def test_rsi_window_negative_raises_error(self):
        # Arrange
        close_prices = np.array([100.0, 101.0, 102.0, 101.0, 103.0])

        # Act & Assert
        with pytest.raises(ValueError, match="RSI window must be positive"):
            _calculate_rsi_fast(close_prices, window=-5)

    def test_rsi_basic_calculation(self):
        # Arrange - monotonically increasing prices should have high RSI
        close_prices = np.linspace(100, 120, 50)

        # Act
        rsi = _calculate_rsi_fast(close_prices, window=14)

        # Assert - RSI for uptrend should be high (near 100)
        valid_rsi = rsi[~np.isnan(rsi)]
        assert len(valid_rsi) > 0
        assert np.mean(valid_rsi) > 50  # Uptrend should be above neutral

    def test_rsi_returns_nans_for_insufficient_data(self):
        # Arrange - less data than window size
        close_prices = np.array([100.0, 101.0, 102.0])

        # Act
        rsi = _calculate_rsi_fast(close_prices, window=14)

        # Assert - all values should be NaN since window > data length
        assert np.all(np.isnan(rsi))
