import numpy as np
import pandas as pd
import pytest

from src.tech.indicators.core import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_moving_averages,
    calculate_rsi,
    calculate_support_resistance,
    clear_indicator_cache,
    detect_market_regime,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast, pytest.mark.mock_only]


class TestMovingAverages:
    def test_ema_calculation(self):
        data = pd.Series([1, 2, 3, 4, 5])
        ema = calculate_ema(data, period=3)
        assert not ema.isna().all()
        assert len(ema) == len(data)
        assert ema.iloc[0] == data.iloc[0]
        assert ema.std() <= data.std()

    def test_moving_average_edge_cases(self):
        empty_data = pd.Series([], dtype=float)
        ema_empty = calculate_ema(empty_data, period=3)
        assert len(ema_empty) == 0
        single_data = pd.Series([5])
        ema_single = calculate_ema(single_data, period=3)
        assert len(ema_single) == 1
        assert ema_single.iloc[0] == 5

    def test_calculate_moving_averages(self):
        """Test calculate_moving_averages with multiple periods."""
        data = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        periods = [3, 5]
        result = calculate_moving_averages(data, periods)

        assert isinstance(result, pd.DataFrame)
        assert "ma_3" in result.columns
        assert "ma_5" in result.columns
        assert len(result) == len(data)

        # Verify MA calculations
        expected_ma_3 = data["close"].rolling(window=3).mean()
        pd.testing.assert_series_equal(result["ma_3"], expected_ma_3, check_names=False)

    def test_calculate_moving_averages_validation(self):
        """Test input validation for calculate_moving_averages."""
        data = pd.DataFrame({"close": [1, 2, 3, 4, 5]})

        # Test missing close column
        with pytest.raises(ValueError, match="must contain 'close' column"):
            calculate_moving_averages(pd.DataFrame({"high": [1, 2, 3]}), [5])

        # Test empty periods list
        with pytest.raises(ValueError, match="periods list cannot be empty"):
            calculate_moving_averages(data, [])

        # Test negative period
        with pytest.raises(ValueError, match="Period must be positive"):
            calculate_moving_averages(data, [5, -1])

        # Test zero period
        with pytest.raises(ValueError, match="Period must be positive"):
            calculate_moving_averages(data, [0])

    def test_ema_validation(self):
        """Test input validation for calculate_ema."""
        data = pd.Series([1, 2, 3, 4, 5])

        # Test negative period
        with pytest.raises(ValueError, match="EMA period must be positive"):
            calculate_ema(data, period=-1)

        # Test zero period
        with pytest.raises(ValueError, match="EMA period must be positive"):
            calculate_ema(data, period=0)


class TestRSI:
    def test_rsi_calculation(self):
        # Use enough data points (> period + 1) so Wilder's smoothing produces values
        data = pd.Series(
            [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.23, 44.57, 44.15, 43.42, 44.8, 45.1]
        )
        rsi = calculate_rsi(data, period=10)
        valid_rsi = rsi.dropna()
        assert len(valid_rsi) > 0
        assert valid_rsi.min() >= 0
        assert valid_rsi.max() <= 100
        assert len(rsi) == len(data)
        # Wilder's smoothing: first valid value is at index `period` (0-based)
        assert rsi.iloc[:10].isna().all()

    def test_rsi_extreme_values(self):
        increasing_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        rsi_increasing = calculate_rsi(increasing_data, period=14)
        assert rsi_increasing.iloc[-1] > 70
        decreasing_data = pd.Series([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        rsi_decreasing = calculate_rsi(decreasing_data, period=14)
        assert rsi_decreasing.iloc[-1] < 30

    def test_rsi_validation(self):
        """Test input validation for calculate_rsi."""
        data = pd.Series([1, 2, 3, 4, 5])

        # Test negative period
        with pytest.raises(ValueError, match="RSI period must be positive"):
            calculate_rsi(data, period=-1)

        # Test zero period
        with pytest.raises(ValueError, match="RSI period must be positive"):
            calculate_rsi(data, period=0)

        # Test DataFrame missing close column
        with pytest.raises(ValueError, match="must contain 'close' column"):
            calculate_rsi(pd.DataFrame({"high": [1, 2, 3]}), period=5)

        # Test invalid smoothing method
        with pytest.raises(ValueError, match="smoothing_method must be one of"):
            calculate_rsi(data, period=3, smoothing_method="invalid")

    def test_rsi_wilder_vs_sma_differ(self):
        """Wilder's smoothing and SMA produce different RSI values."""
        np.random.seed(42)
        prices = pd.Series(np.cumsum(np.random.randn(50)) + 100)
        rsi_wilder = calculate_rsi(prices, period=14, smoothing_method="wilder")
        rsi_sma = calculate_rsi(prices, period=14, smoothing_method="sma")
        # Both should be valid RSI
        valid_wilder = rsi_wilder.dropna()
        valid_sma = rsi_sma.dropna()
        assert len(valid_wilder) > 0
        assert len(valid_sma) > 0
        # They should differ (different algorithms)
        common_idx = valid_wilder.index.intersection(valid_sma.index)
        assert not np.allclose(
            rsi_wilder.loc[common_idx].values, rsi_sma.loc[common_idx].values
        )

    def test_rsi_training_inference_parity(self):
        """Verify calculate_rsi and _calculate_rsi_fast produce identical values."""
        from src.ml.training_pipeline.features import _calculate_rsi_fast

        np.random.seed(123)
        prices = np.cumsum(np.random.randn(100)) + 500

        # Inference path (pandas Series)
        rsi_inference = calculate_rsi(pd.Series(prices), period=14)
        # Training path (numpy array)
        rsi_training = _calculate_rsi_fast(prices, window=14)

        # Both should produce identical values where valid
        valid_mask = ~np.isnan(rsi_training)
        assert valid_mask.sum() > 0
        np.testing.assert_allclose(
            rsi_inference.values[valid_mask],
            rsi_training[valid_mask],
            rtol=1e-5,
            err_msg="Training and inference RSI values diverge - algorithm mismatch",
        )


class TestATR:
    def test_atr_calculation(self):
        data = pd.DataFrame(
            {
                "high": [10, 12, 11, 13, 14],
                "low": [8, 9, 10, 11, 12],
                "close": [9, 11, 10.5, 12, 13],
            }
        )
        result = calculate_atr(data, period=3)
        atr = result["atr"]
        assert (atr.dropna() >= 0).all()
        assert len(atr) == len(data)
        assert atr.iloc[:2].isna().all()

    def test_atr_volatility_measurement(self):
        low_vol_data = pd.DataFrame(
            {
                "high": [100, 100.1, 100.05, 100.2, 100.15],
                "low": [99.9, 99.95, 100, 100.1, 100.05],
                "close": [100, 100.05, 100.02, 100.15, 100.1],
            }
        )
        low_vol_result = calculate_atr(low_vol_data, period=3)
        high_vol_data = pd.DataFrame(
            {
                "high": [100, 105, 95, 110, 90],
                "low": [95, 100, 90, 105, 85],
                "close": [98, 102, 92, 108, 88],
            }
        )
        high_vol_result = calculate_atr(high_vol_data, period=3)
        assert high_vol_result["atr"].iloc[-1] > low_vol_result["atr"].iloc[-1]

    def test_atr_validation(self):
        """Test input validation for calculate_atr."""
        data = pd.DataFrame(
            {
                "high": [10, 12, 11],
                "low": [8, 9, 10],
                "close": [9, 11, 10.5],
            }
        )

        # Test missing columns
        with pytest.raises(ValueError, match="missing required columns"):
            calculate_atr(pd.DataFrame({"close": [1, 2, 3]}), period=3)

        # Test negative period
        with pytest.raises(ValueError, match="ATR period must be positive"):
            calculate_atr(data, period=-1)

        # Test zero period
        with pytest.raises(ValueError, match="ATR period must be positive"):
            calculate_atr(data, period=0)


class TestBollingerBands:
    def test_bollinger_bands_calculation(self):
        data = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        bb = calculate_bollinger_bands(data, period=5, std_dev=2)
        assert isinstance(bb, pd.DataFrame)
        assert "bb_upper" in bb.columns and "bb_middle" in bb.columns and "bb_lower" in bb.columns
        valid_mask = bb["bb_upper"].notna() & bb["bb_middle"].notna()
        assert (bb.loc[valid_mask, "bb_upper"] >= bb.loc[valid_mask, "bb_middle"]).all()
        valid_mask = bb["bb_lower"].notna() & bb["bb_middle"].notna()
        assert (bb.loc[valid_mask, "bb_lower"] <= bb.loc[valid_mask, "bb_middle"]).all()
        expected_middle = data["close"].rolling(window=5).mean()
        pd.testing.assert_series_equal(bb["bb_middle"], expected_middle, check_names=False)

    def test_bollinger_bands_volatility(self):
        low_vol_data = pd.DataFrame({"close": [100, 100.1, 100.05, 100.2, 100.15, 100.1, 100.05]})
        high_vol_data = pd.DataFrame({"close": [100, 105, 95, 110, 90, 105, 95]})
        bb_low = calculate_bollinger_bands(low_vol_data, period=5, std_dev=2)
        bb_high = calculate_bollinger_bands(high_vol_data, period=5, std_dev=2)
        low_width = bb_low["bb_upper"].iloc[-1] - bb_low["bb_lower"].iloc[-1]
        high_width = bb_high["bb_upper"].iloc[-1] - bb_high["bb_lower"].iloc[-1]
        assert high_width > low_width

    def test_bollinger_bands_validation(self):
        """Test input validation for calculate_bollinger_bands."""
        data = pd.DataFrame({"close": [1, 2, 3, 4, 5]})

        # Test missing close column
        with pytest.raises(ValueError, match="must contain 'close' column"):
            calculate_bollinger_bands(pd.DataFrame({"high": [1, 2, 3]}))

        # Test negative period
        with pytest.raises(ValueError, match="Bollinger Bands period must be positive"):
            calculate_bollinger_bands(data, period=-1)

        # Test negative std_dev
        with pytest.raises(ValueError, match="Standard deviation multiplier must be positive"):
            calculate_bollinger_bands(data, period=5, std_dev=-1)


class TestMACD:
    def test_macd_calculation(self):
        data = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]})
        macd = calculate_macd(data, fast_period=12, slow_period=26, signal_period=9)
        assert isinstance(macd, pd.DataFrame)
        assert (
            "macd" in macd.columns and "macd_signal" in macd.columns and "macd_hist" in macd.columns
        )
        ema_fast = calculate_ema(data["close"], period=12)
        ema_slow = calculate_ema(data["close"], period=26)
        expected_macd = ema_fast - ema_slow
        valid_mask = ~expected_macd.isna()
        if valid_mask.any():
            pd.testing.assert_series_equal(
                macd["macd"][valid_mask], expected_macd[valid_mask], check_names=False
            )

    def test_macd_validation(self):
        """Test input validation for calculate_macd."""
        data = pd.DataFrame({"close": [1, 2, 3, 4, 5]})

        # Test missing close column
        with pytest.raises(ValueError, match="must contain 'close' column"):
            calculate_macd(pd.DataFrame({"high": [1, 2, 3]}))

        # Test negative fast period
        with pytest.raises(ValueError, match="Fast period must be positive"):
            calculate_macd(data, fast_period=-1)

        # Test negative slow period
        with pytest.raises(ValueError, match="Slow period must be positive"):
            calculate_macd(data, slow_period=0)

        # Test negative signal period
        with pytest.raises(ValueError, match="Signal period must be positive"):
            calculate_macd(data, signal_period=-1)

        # Test fast period >= slow period (invalid MACD configuration)
        with pytest.raises(ValueError, match="Fast period.*must be less than slow period"):
            calculate_macd(data, fast_period=26, slow_period=12)


class TestMarketRegime:
    def test_market_regime_detection(self):
        data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            }
        )
        result = detect_market_regime(data)
        assert isinstance(result, pd.DataFrame)
        assert (
            "regime" in result.columns
            and "volatility" in result.columns
            and "trend" in result.columns
        )
        assert all(
            regime in ["trending", "ranging", "volatile"] for regime in result["regime"].dropna()
        )

    def test_market_regime_constant_prices(self):
        """Test regime detection with constant prices (edge case)."""
        # Constant prices should trigger division by zero protection
        data = pd.DataFrame({"close": [100] * 100})
        result = detect_market_regime(data, volatility_lookback=20, trend_lookback=50)

        # Should not raise division by zero error
        assert isinstance(result, pd.DataFrame)
        assert "regime" in result.columns
        # All regimes should be valid values (no NaN from division errors)
        assert all(
            regime in ["trending", "ranging", "volatile"] for regime in result["regime"].dropna()
        )

    def test_market_regime_short_data(self):
        """Test regime detection with data shorter than lookback windows."""
        # Data shorter than lookback windows
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})
        result = detect_market_regime(data, volatility_lookback=20, trend_lookback=50)

        # Should not crash, but most values will be NaN
        assert isinstance(result, pd.DataFrame)
        assert "regime" in result.columns

    def test_market_regime_validation(self):
        """Test input validation for detect_market_regime."""
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})

        # Test missing close column
        with pytest.raises(ValueError, match="must contain 'close' column"):
            detect_market_regime(pd.DataFrame({"high": [1, 2, 3]}))

        # Test negative volatility lookback
        with pytest.raises(ValueError, match="Volatility lookback must be positive"):
            detect_market_regime(data, volatility_lookback=-1)

        # Test negative trend lookback
        with pytest.raises(ValueError, match="Trend lookback must be positive"):
            detect_market_regime(data, trend_lookback=0)

        # Test negative regime threshold
        with pytest.raises(ValueError, match="Regime threshold must be non-negative"):
            detect_market_regime(data, regime_threshold=-0.1)


class TestSupportResistance:
    def test_support_resistance_calculation(self):
        data = pd.DataFrame(
            {
                "high": [10, 12, 11, 13, 14, 13, 15, 14, 16, 15],
                "low": [8, 9, 10, 11, 12, 11, 13, 12, 14, 13],
                "close": [9, 11, 10.5, 12, 13, 12.5, 14, 13.5, 15, 14.5],
            }
        )
        support, resistance = calculate_support_resistance(data, period=5, num_points=3)
        assert isinstance(support, pd.Series)
        assert isinstance(resistance, pd.Series)
        assert len(support) <= 3
        assert len(resistance) <= 3

    def test_support_resistance_validation(self):
        """Test input validation for calculate_support_resistance."""
        data = pd.DataFrame(
            {
                "high": [10, 12, 11],
                "low": [8, 9, 10],
                "close": [9, 11, 10.5],
            }
        )

        # Test missing columns
        with pytest.raises(ValueError, match="missing required columns"):
            calculate_support_resistance(pd.DataFrame({"close": [1, 2, 3]}))

        # Test negative period
        with pytest.raises(ValueError, match="Period must be positive"):
            calculate_support_resistance(data, period=-1)

        # Test zero num_points
        with pytest.raises(ValueError, match="num_points must be positive"):
            calculate_support_resistance(data, period=5, num_points=0)


class TestIndicatorIntegration:
    def test_indicators_with_realistic_data(self):
        np.random.seed(42)
        n_points = 100
        base_price = 100
        prices = []
        for _ in range(n_points):
            change = np.random.normal(0, 0.02)
            base_price *= 1 + change
            prices.append(base_price)
        price_series = pd.Series(prices)
        ema = calculate_ema(price_series, period=20)
        rsi = calculate_rsi(price_series, period=14)
        assert not ema.isna().all()
        assert not rsi.isna().all()
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_indicators_with_missing_data(self):
        data_with_nan = pd.Series([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10])
        ema = calculate_ema(data_with_nan, period=3)
        rsi = calculate_rsi(data_with_nan, period=3)
        assert len(ema) == len(data_with_nan)
        assert len(rsi) == len(data_with_nan)

    def test_indicators_performance(self):
        large_data = pd.Series(np.random.randn(10000))
        import time

        start_time = time.time()
        _ = calculate_ema(large_data, period=20)
        _ = calculate_rsi(large_data, period=14)
        assert time.time() - start_time < 3.0


class TestIndicatorCaching:
    """Test caching functionality for technical indicators."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_indicator_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        clear_indicator_cache()

    def test_cache_hit_returns_identical_results(self):
        """Test that cached results match original calculations."""
        data = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        periods = [3, 5]

        # First call - cache miss
        result1 = calculate_moving_averages(data, periods)
        # Second call - cache hit
        result2 = calculate_moving_averages(data, periods)

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_cache_key_includes_positional_arguments(self):
        """Test that different positional arguments generate different cache keys."""
        data = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        # Call with different positional arguments
        result1 = calculate_moving_averages(data, [5, 10])
        result2 = calculate_moving_averages(data, [20, 50])

        # Results should be different (different periods)
        assert not result1["ma_5"].equals(result2.get("ma_5", pd.Series()))
        assert "ma_20" in result2.columns
        assert "ma_20" not in result1.columns

    def test_cache_key_includes_keyword_arguments(self):
        """Test that different keyword arguments generate different cache keys."""
        data = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        # Call with different keyword arguments
        result1 = calculate_bollinger_bands(data, period=5, std_dev=2.0)
        result2 = calculate_bollinger_bands(data, period=5, std_dev=3.0)

        # Results should be different (different std_dev)
        assert not result1["bb_upper"].equals(result2["bb_upper"])

    def test_cache_key_respects_dataframe_changes(self):
        """Test that cache invalidates when DataFrame changes."""
        data1 = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        data2 = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]})  # Last value different

        result1 = calculate_moving_averages(data1, [5])
        result2 = calculate_moving_averages(data2, [5])

        # Results should be different (different last value)
        assert not result1["ma_5"].equals(result2["ma_5"])

    def test_cache_key_respects_earlier_row_changes(self):
        """Test that cache invalidates when earlier rows change even if last row is the same.

        Regression test for the Codex P1 finding: exchange backfills or corrected
        historical candles change earlier rows without affecting the final candle.
        All indicator functions (MA, ATR, BB, MACD) depend on prior rows via rolling
        windows, so the cache must detect these changes.
        """
        # Same last row (close=5), completely different earlier rows
        data1 = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
        data2 = pd.DataFrame({"close": [10, 20, 30, 40, 5]})

        result1 = calculate_moving_averages(data1, [3])
        result2 = calculate_moving_averages(data2, [3])

        # MA values should differ because earlier rows are different
        assert not result1["ma_3"].equals(result2["ma_3"])

    def test_cache_key_respects_row_count(self):
        """Test that cache invalidates when row count changes."""
        data1 = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
        data2 = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6]})

        result1 = calculate_moving_averages(data1, [3])
        result2 = calculate_moving_averages(data2, [3])

        # Results should have different lengths
        assert len(result1) != len(result2)

    def test_empty_dataframe_skips_caching(self):
        """Test that empty DataFrames skip caching without errors."""
        empty_data = pd.DataFrame({"close": []})

        # Should not raise errors
        result = calculate_moving_averages(empty_data, [5])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_clear_cache_function(self):
        """Test that clear_indicator_cache() clears the cache."""
        data = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        # Populate cache
        _ = calculate_moving_averages(data, [5])
        _ = calculate_atr(
            pd.DataFrame(
                {
                    "high": [10, 12, 11, 13, 14],
                    "low": [8, 9, 10, 11, 12],
                    "close": [9, 11, 10.5, 12, 13],
                }
            ),
            period=3,
        )

        # Clear cache
        clear_indicator_cache()

        # Cache should be empty (we can't directly verify, but function should not error)
        result = calculate_moving_averages(data, [5])
        assert isinstance(result, pd.DataFrame)

    def test_cache_eviction_when_exceeding_max_size(self):
        """Test that cache evicts oldest entries when exceeding max size."""
        # Note: _CACHE_MAX_SIZE is 1000, we can't easily test full eviction
        # but we can verify the function handles large numbers of cache entries
        data = pd.DataFrame({"close": list(range(100))})

        # Create multiple cache entries with different parameters
        for i in range(50):
            calculate_moving_averages(data, [i + 1])

        # Should not raise memory errors and still work correctly
        result = calculate_moving_averages(data, [5])
        assert isinstance(result, pd.DataFrame)
        assert "ma_5" in result.columns

    def test_cached_result_is_copy_not_reference(self):
        """Test that cached results are copies to prevent mutation."""
        data = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        result1 = calculate_moving_averages(data, [5])
        result2 = calculate_moving_averages(data, [5])

        # Mutate result1
        result1.loc[0, "ma_5"] = 999

        # result2 should not be affected (it's a copy, not a reference)
        assert result2.loc[0, "ma_5"] != 999

    def test_atr_caching(self):
        """Test caching works correctly for calculate_atr."""
        data = pd.DataFrame(
            {
                "high": [10, 12, 11, 13, 14, 15, 16, 17, 18, 19],
                "low": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                "close": [9, 11, 10.5, 12, 13, 14, 15, 16, 17, 18],
            }
        )

        result1 = calculate_atr(data, period=3)
        result2 = calculate_atr(data, period=3)

        pd.testing.assert_frame_equal(result1, result2)

    def test_bollinger_bands_caching(self):
        """Test caching works correctly for calculate_bollinger_bands."""
        data = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        result1 = calculate_bollinger_bands(data, period=5, std_dev=2.0)
        result2 = calculate_bollinger_bands(data, period=5, std_dev=2.0)

        pd.testing.assert_frame_equal(result1, result2)

    def test_macd_caching(self):
        """Test caching works correctly for calculate_macd."""
        data = pd.DataFrame({"close": list(range(1, 31))})

        result1 = calculate_macd(data, fast_period=12, slow_period=26, signal_period=9)
        result2 = calculate_macd(data, fast_period=12, slow_period=26, signal_period=9)

        pd.testing.assert_frame_equal(result1, result2)
