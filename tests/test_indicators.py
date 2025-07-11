"""
Tests for technical indicators.

Technical indicators are critical for strategy signal generation. Tests cover:
- Indicator calculation accuracy
- Edge cases and data validation
- Performance with large datasets
- Mathematical correctness
- Integration with pandas DataFrames
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from indicators.technical import (
    calculate_ema, calculate_rsi, calculate_atr,
    calculate_bollinger_bands, calculate_macd, detect_market_regime,
    calculate_support_resistance
)


class TestMovingAverages:
    """Test moving average calculations"""

    def test_ema_calculation(self):
        """Test Exponential Moving Average calculation"""
        # Test data
        data = pd.Series([1, 2, 3, 4, 5])
        
        # Calculate EMA with period 3
        ema = calculate_ema(data, period=3)
        
        # EMA should be calculated and not all NaN
        assert not ema.isna().all()
        assert len(ema) == len(data)
        
        # First value should equal first data point
        assert ema.iloc[0] == data.iloc[0]
        
        # EMA should be smoother than original data
        assert ema.std() <= data.std()

    def test_moving_average_edge_cases(self):
        """Test moving averages with edge cases"""
        # Empty series
        empty_data = pd.Series([])
        ema_empty = calculate_ema(empty_data, period=3)
        assert len(ema_empty) == 0
        
        # Single value
        single_data = pd.Series([5])
        ema_single = calculate_ema(single_data, period=3)
        assert len(ema_single) == 1
        assert ema_single.iloc[0] == 5


class TestRSI:
    """Test Relative Strength Index calculation"""

    def test_rsi_calculation(self):
        """Test RSI calculation with known values"""
        # Test data with clear trends
        data = pd.Series([44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.23, 44.57, 44.15, 43.42])
        
        rsi = calculate_rsi(data, period=14)
        
        # RSI should be between 0 and 100
        assert rsi.min() >= 0
        assert rsi.max() <= 100
        
        # Should have same length as input
        assert len(rsi) == len(data)
        
        # First period-1 values should be NaN
        assert rsi.iloc[:13].isna().all()

    def test_rsi_extreme_values(self):
        """Test RSI with extreme price movements"""
        # All increasing prices (should give high RSI)
        increasing_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        rsi_increasing = calculate_rsi(increasing_data, period=14)
        
        # Last value should be high (close to 100)
        assert rsi_increasing.iloc[-1] > 70
        
        # All decreasing prices (should give low RSI)
        decreasing_data = pd.Series([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        rsi_decreasing = calculate_rsi(decreasing_data, period=14)
        
        # Last value should be low (close to 0)
        assert rsi_decreasing.iloc[-1] < 30


class TestATR:
    """Test Average True Range calculation"""

    def test_atr_calculation(self):
        """Test ATR calculation with sample data"""
        # Create sample OHLCV data
        data = pd.DataFrame({
            'high': [10, 12, 11, 13, 14],
            'low': [8, 9, 10, 11, 12],
            'close': [9, 11, 10.5, 12, 13]
        })
        
        atr = calculate_atr(data, period=3)
        
        # ATR should be positive
        assert (atr >= 0).all()
        
        # Should have same length as input
        assert len(atr) == len(data)
        
        # First period-1 values should be NaN
        assert atr.iloc[:2].isna().all()

    def test_atr_volatility_measurement(self):
        """Test that ATR correctly measures volatility"""
        # Low volatility data
        low_vol_data = pd.DataFrame({
            'high': [100, 100.1, 100.05, 100.2, 100.15],
            'low': [99.9, 99.95, 100, 100.1, 100.05],
            'close': [100, 100.05, 100.02, 100.15, 100.1]
        })
        
        low_vol_atr = calculate_atr(low_vol_data, period=3)
        
        # High volatility data
        high_vol_data = pd.DataFrame({
            'high': [100, 105, 95, 110, 90],
            'low': [95, 100, 90, 105, 85],
            'close': [98, 102, 92, 108, 88]
        })
        
        high_vol_atr = calculate_atr(high_vol_data, period=3)
        
        # High volatility ATR should be larger
        assert high_vol_atr.iloc[-1] > low_vol_atr.iloc[-1]


class TestBollingerBands:
    """Test Bollinger Bands calculation"""

    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        bb = calculate_bollinger_bands(data, period=5, std_dev=2)
        
        # Should return DataFrame with upper, middle, lower bands
        assert isinstance(bb, pd.DataFrame)
        assert 'upper' in bb.columns
        assert 'middle' in bb.columns
        assert 'lower' in bb.columns
        
        # Upper band should be above middle band
        assert (bb['upper'] >= bb['middle']).all()
        
        # Lower band should be below middle band
        assert (bb['lower'] <= bb['middle']).all()
        
        # Middle band should be rolling mean
        expected_middle = data.rolling(window=5).mean()
        pd.testing.assert_series_equal(bb['middle'], expected_middle, check_names=False)

    def test_bollinger_bands_volatility(self):
        """Test that Bollinger Bands respond to volatility"""
        # Low volatility data
        low_vol_data = pd.Series([100, 100.1, 100.05, 100.2, 100.15, 100.1, 100.05])
        
        # High volatility data
        high_vol_data = pd.Series([100, 105, 95, 110, 90, 105, 95])
        
        bb_low = calculate_bollinger_bands(low_vol_data, period=5, std_dev=2)
        bb_high = calculate_bollinger_bands(high_vol_data, period=5, std_dev=2)
        
        # High volatility should have wider bands
        low_width = bb_low['upper'].iloc[-1] - bb_low['lower'].iloc[-1]
        high_width = bb_high['upper'].iloc[-1] - bb_high['lower'].iloc[-1]
        
        assert high_width > low_width


class TestMACD:
    """Test MACD calculation"""

    def test_macd_calculation(self):
        """Test MACD calculation"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        
        macd = calculate_macd(data, fast_period=12, slow_period=26, signal_period=9)
        
        # Should return DataFrame with MACD line, signal line, and histogram
        assert isinstance(macd, pd.DataFrame)
        assert 'macd' in macd.columns
        assert 'signal' in macd.columns
        assert 'histogram' in macd.columns
        
        # MACD line should be EMA difference
        ema_fast = calculate_ema(data, period=12)
        ema_slow = calculate_ema(data, period=26)
        expected_macd = ema_fast - ema_slow
        
        # Compare non-NaN values
        valid_mask = ~expected_macd.isna()
        if valid_mask.any():
            pd.testing.assert_series_equal(
                macd['macd'][valid_mask], 
                expected_macd[valid_mask], 
                check_names=False
            )


class TestMarketRegime:
    """Test market regime detection"""

    def test_market_regime_detection(self):
        """Test market regime detection"""
        # Create sample data
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        })
        
        result = detect_market_regime(data)
        
        # Should return DataFrame with regime column
        assert isinstance(result, pd.DataFrame)
        assert 'regime' in result.columns
        assert 'volatility' in result.columns
        assert 'trend' in result.columns
        
        # Regime should be one of the expected values
        valid_regimes = ['trending', 'ranging', 'volatile']
        assert all(regime in valid_regimes for regime in result['regime'].dropna())


class TestSupportResistance:
    """Test support and resistance calculation"""

    def test_support_resistance_calculation(self):
        """Test support and resistance calculation"""
        # Create sample data
        data = pd.DataFrame({
            'high': [10, 12, 11, 13, 14, 13, 15, 14, 16, 15],
            'low': [8, 9, 10, 11, 12, 11, 13, 12, 14, 13],
            'close': [9, 11, 10.5, 12, 13, 12.5, 14, 13.5, 15, 14.5]
        })
        
        support, resistance = calculate_support_resistance(data, period=5, num_points=3)
        
        # Should return Series
        assert isinstance(support, pd.Series)
        assert isinstance(resistance, pd.Series)
        
        # Should have reasonable number of points
        assert len(support) <= 3
        assert len(resistance) <= 3


class TestIndicatorIntegration:
    """Test indicator integration and edge cases"""

    def test_indicators_with_realistic_data(self):
        """Test indicators with realistic market data"""
        # Generate realistic price data
        np.random.seed(42)
        n_points = 100
        base_price = 100
        
        prices = []
        for i in range(n_points):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            base_price *= (1 + change)
            prices.append(base_price)
        
        price_series = pd.Series(prices)
        
        # Test all indicators with realistic data
        ema = calculate_ema(price_series, period=20)
        rsi = calculate_rsi(price_series, period=14)
        
        # All should be calculated without errors
        assert not ema.isna().all()
        assert not rsi.isna().all()
        
        # RSI should be within bounds
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_indicators_with_missing_data(self):
        """Test indicators handle missing data gracefully"""
        data_with_nan = pd.Series([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10])
        
        # Should not crash with NaN values
        ema = calculate_ema(data_with_nan, period=3)
        rsi = calculate_rsi(data_with_nan, period=3)
        
        # Should return Series with same length
        assert len(ema) == len(data_with_nan)
        assert len(rsi) == len(data_with_nan)

    def test_indicators_performance(self):
        """Test indicators performance with large datasets"""
        # Large dataset
        large_data = pd.Series(np.random.randn(10000))
        
        # Should complete within reasonable time
        import time
        start_time = time.time()
        
        ema = calculate_ema(large_data, period=20)
        rsi = calculate_rsi(large_data, period=14)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete in under 1 second
        assert execution_time < 1.0
        
        # Results should be correct
        assert len(ema) == len(large_data)
        assert len(rsi) == len(large_data)