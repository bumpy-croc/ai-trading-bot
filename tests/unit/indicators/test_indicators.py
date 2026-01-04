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
        data = pd.Series([44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.23, 44.57, 44.15, 43.42])
        rsi = calculate_rsi(data, period=10)
        assert rsi.min() >= 0
        assert rsi.max() <= 100
        assert len(rsi) == len(data)
        assert rsi.iloc[:9].isna().all()

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
        assert time.time() - start_time < 1.0
