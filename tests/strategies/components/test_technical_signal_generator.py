"""
Unit tests for Technical Signal Generator components
"""


import numpy as np
import pandas as pd
import pytest

from src.regime.detector import TrendLabel, VolLabel
from src.strategies.components.regime_context import RegimeContext
from src.strategies.components.signal_generator import Signal, SignalDirection
from src.strategies.components.technical_signal_generator import (
    MACDSignalGenerator,
    RSISignalGenerator,
    TechnicalSignalGenerator,
)


class TestTechnicalSignalGenerator:
    """Test TechnicalSignalGenerator implementation"""
    
    def create_test_dataframe(self, length=100, trend='neutral'):
        """Create test DataFrame with OHLCV data"""
        dates = pd.date_range('2023-01-01', periods=length, freq='1H')
        
        # Create price data with specified trend
        np.random.seed(42)
        base_price = 100.0
        
        if trend == 'uptrend':
            trend_component = np.linspace(0, 20, length)
            noise = np.random.normal(0, 1, length)
        elif trend == 'downtrend':
            trend_component = np.linspace(0, -20, length)
            noise = np.random.normal(0, 1, length)
        else:  # neutral
            trend_component = np.zeros(length)
            noise = np.random.normal(0, 2, length)
        
        prices = base_price + trend_component + noise
        prices = np.maximum(prices, 10)  # Minimum price
        
        data = {
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, length))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, length))),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, length)
        }
        
        return pd.DataFrame(data, index=dates)
    
    def create_regime_context(self, trend=TrendLabel.TREND_UP, volatility=VolLabel.LOW):
        """Create test regime context"""
        return RegimeContext(
            trend=trend,
            volatility=volatility,
            confidence=0.8,
            duration=15,
            strength=0.7
        )
    
    def test_technical_signal_generator_initialization(self):
        """Test TechnicalSignalGenerator initialization"""
        generator = TechnicalSignalGenerator(
            name="test_technical",
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30,
            ma_short=20,
            ma_long=50
        )
        
        assert generator.name == "test_technical"
        assert generator.rsi_period == 14
        assert generator.rsi_overbought == 70
        assert generator.rsi_oversold == 30
        assert generator.ma_short == 20
        assert generator.ma_long == 50
        assert generator.min_periods == max(14, 26+9, 50, 20, 14)  # max of all periods
    
    def test_generate_signal_insufficient_history(self):
        """Test signal generation with insufficient history"""
        generator = TechnicalSignalGenerator()
        df = self.create_test_dataframe(30)  # Less than min_periods
        
        signal = generator.generate_signal(df, 20)
        
        assert signal.direction == SignalDirection.HOLD
        assert signal.strength == 0.0
        assert signal.confidence == 0.0
        assert signal.metadata['reason'] == 'insufficient_history'
    
    def test_generate_signal_with_indicators(self):
        """Test signal generation with calculated indicators"""
        generator = TechnicalSignalGenerator(
            rsi_period=14,
            ma_short=10,
            ma_long=20,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9
        )
        df = self.create_test_dataframe(100, trend='uptrend')
        
        signal = generator.generate_signal(df, 80)
        
        assert isinstance(signal.direction, SignalDirection)
        assert 0.0 <= signal.strength <= 1.0
        assert 0.0 <= signal.confidence <= 1.0
        assert 'rsi_signal' in signal.metadata
        assert 'macd_signal' in signal.metadata
        assert 'ma_signal' in signal.metadata
        assert 'bb_signal' in signal.metadata
    
    def test_generate_signal_with_regime_awareness(self):
        """Test signal generation with regime-aware weighting"""
        generator = TechnicalSignalGenerator()
        df = self.create_test_dataframe(100, trend='uptrend')
        
        # Test trending regime
        trending_regime = self.create_regime_context(TrendLabel.TREND_UP, VolLabel.LOW)
        signal_trending = generator.generate_signal(df, 80, trending_regime)
        
        # Test ranging regime
        ranging_regime = self.create_regime_context(TrendLabel.RANGE, VolLabel.LOW)
        signal_ranging = generator.generate_signal(df, 80, ranging_regime)
        
        assert 'regime_trend' in signal_trending.metadata
        assert 'regime_volatility' in signal_trending.metadata
        assert 'regime_confidence' in signal_trending.metadata
        
        assert 'regime_trend' in signal_ranging.metadata
        assert 'regime_volatility' in signal_ranging.metadata
        assert 'regime_confidence' in signal_ranging.metadata
    
    def test_rsi_signal_generation(self):
        """Test RSI signal component"""
        generator = TechnicalSignalGenerator(rsi_period=14)
        
        # Create DataFrame with extreme RSI conditions
        df = pd.DataFrame({
            'open': [100] * 20,
            'high': [105] * 20,
            'low': [95] * 20,
            'close': [100] * 20,
            'volume': [1000] * 20,
            'rsi': [20, 30, 50, 70, 80] + [50] * 15  # Oversold, neutral, overbought
        })
        
        # Test oversold condition
        rsi_signal = generator._get_rsi_signal(df, 0)
        assert rsi_signal == 1  # Buy signal
        
        # Test neutral condition
        rsi_signal = generator._get_rsi_signal(df, 2)
        assert rsi_signal == 0  # Hold signal
        
        # Test overbought condition
        rsi_signal = generator._get_rsi_signal(df, 4)
        assert rsi_signal == -1  # Sell signal
    
    def test_macd_signal_generation(self):
        """Test MACD signal component"""
        generator = TechnicalSignalGenerator()
        
        # Create DataFrame with MACD crossover conditions
        df = pd.DataFrame({
            'open': [100] * 5,
            'high': [105] * 5,
            'low': [95] * 5,
            'close': [100] * 5,
            'volume': [1000] * 5,
            'macd': [-0.5, -0.2, 0.1, 0.3, 0.2],
            'macd_signal': [0.0, 0.0, 0.0, 0.0, 0.0]
        })
        
        # Test bullish crossover (MACD crosses above signal)
        macd_signal = generator._get_macd_signal(df, 2)
        assert macd_signal == 1  # Buy signal
        
        # Test no crossover
        macd_signal = generator._get_macd_signal(df, 3)
        assert macd_signal == 0  # Hold signal
    
    def test_ma_signal_generation(self):
        """Test Moving Average signal component"""
        generator = TechnicalSignalGenerator(ma_short=10, ma_long=20)
        
        # Create DataFrame with MA conditions
        df = pd.DataFrame({
            'open': [100] * 5,
            'high': [105] * 5,
            'low': [95] * 5,
            'close': [105, 100, 95, 90, 85],  # Declining prices
            'volume': [1000] * 5,
            'ma_10': [102, 101, 100, 99, 98],  # Short MA
            'ma_20': [100, 100, 100, 100, 100]  # Long MA
        })
        
        # Test bullish condition (price > short MA > long MA)
        ma_signal = generator._get_ma_signal(df, 0)
        assert ma_signal == 1  # Buy signal
        
        # Test bearish condition (price < short MA < long MA)
        ma_signal = generator._get_ma_signal(df, 4)
        assert ma_signal == -1  # Sell signal
    
    def test_bb_signal_generation(self):
        """Test Bollinger Bands signal component"""
        generator = TechnicalSignalGenerator()
        
        # Create DataFrame with BB conditions
        df = pd.DataFrame({
            'open': [100] * 5,
            'high': [105] * 5,
            'low': [95] * 5,
            'close': [85, 100, 115, 100, 100],  # Prices at different BB levels
            'volume': [1000] * 5,
            'bb_upper': [110] * 5,
            'bb_middle': [100] * 5,
            'bb_lower': [90] * 5
        })
        
        # Test oversold condition (price at lower band)
        bb_signal = generator._get_bb_signal(df, 0)
        assert bb_signal == 1  # Buy signal
        
        # Test overbought condition (price at upper band)
        bb_signal = generator._get_bb_signal(df, 2)
        assert bb_signal == -1  # Sell signal
        
        # Test neutral condition (price in middle)
        bb_signal = generator._get_bb_signal(df, 1)
        assert bb_signal == 0  # Hold signal
    
    def test_signal_combination(self):
        """Test signal combination logic"""
        generator = TechnicalSignalGenerator()
        
        # Test all bullish signals
        combined = generator._combine_signals(1, 1, 1, 1)
        assert combined == SignalDirection.BUY
        
        # Test all bearish signals
        combined = generator._combine_signals(-1, -1, -1, -1)
        assert combined == SignalDirection.SELL
        
        # Test mixed signals (should be hold or weak signal)
        combined = generator._combine_signals(1, -1, 0, 0)
        assert combined in [SignalDirection.HOLD, SignalDirection.BUY, SignalDirection.SELL]
    
    def test_signal_strength_calculation(self):
        """Test signal strength calculation"""
        generator = TechnicalSignalGenerator()
        
        # Test all agreeing signals
        strength = generator._calculate_signal_strength(1, 1, 1, 1)
        assert strength == 1.0
        
        # Test no signals
        strength = generator._calculate_signal_strength(0, 0, 0, 0)
        assert strength == 0.0
        
        # Test mixed signals
        strength = generator._calculate_signal_strength(1, 1, -1, 0)
        assert 0.0 <= strength <= 1.0
    
    def test_get_confidence(self):
        """Test get_confidence method"""
        generator = TechnicalSignalGenerator()
        df = self.create_test_dataframe(100)
        
        confidence = generator.get_confidence(df, 80)
        
        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)
    
    def test_get_parameters(self):
        """Test get_parameters method"""
        generator = TechnicalSignalGenerator(
            name="test_tech",
            rsi_period=21,
            ma_short=15,
            ma_long=45
        )
        
        params = generator.get_parameters()
        
        assert params['name'] == "test_tech"
        assert params['rsi_period'] == 21
        assert params['ma_short'] == 15
        assert params['ma_long'] == 45
        assert 'min_periods' in params


class TestRSISignalGenerator:
    """Test RSISignalGenerator implementation"""
    
    def create_test_dataframe_with_rsi(self, rsi_values):
        """Create test DataFrame with specific RSI values"""
        length = len(rsi_values)
        dates = pd.date_range('2023-01-01', periods=length, freq='1H')
        
        data = {
            'open': [100] * length,
            'high': [105] * length,
            'low': [95] * length,
            'close': [100] * length,
            'volume': [1000] * length,
            'rsi': rsi_values
        }
        
        return pd.DataFrame(data, index=dates)
    
    def test_rsi_signal_generator_initialization(self):
        """Test RSISignalGenerator initialization"""
        generator = RSISignalGenerator(
            name="test_rsi",
            period=21,
            overbought=75,
            oversold=25
        )
        
        assert generator.name == "test_rsi"
        assert generator.period == 21
        assert generator.overbought == 75
        assert generator.oversold == 25
    
    def test_generate_signal_insufficient_history(self):
        """Test signal generation with insufficient history"""
        generator = RSISignalGenerator(period=14)
        df = pd.DataFrame({
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        })
        
        signal = generator.generate_signal(df, 5)
        
        assert signal.direction == SignalDirection.HOLD
        assert signal.strength == 0.0
        assert signal.confidence == 0.0
        assert signal.metadata['reason'] == 'insufficient_history'
    
    def test_generate_signal_oversold(self):
        """Test signal generation in oversold condition"""
        generator = RSISignalGenerator(period=14, oversold=30)
        df = self.create_test_dataframe_with_rsi([50] * 14 + [25])  # Enough history + oversold
        
        signal = generator.generate_signal(df, 14)  # Index after minimum periods
        
        assert signal.direction == SignalDirection.BUY
        assert signal.strength > 0.0
        assert signal.confidence >= 0.7  # High confidence at extreme
        assert signal.metadata['rsi_value'] == 25
    
    def test_generate_signal_overbought(self):
        """Test signal generation in overbought condition"""
        generator = RSISignalGenerator(period=14, overbought=70)
        df = self.create_test_dataframe_with_rsi([75])  # Overbought
        
        signal = generator.generate_signal(df, 0)
        
        assert signal.direction == SignalDirection.SELL
        assert signal.strength > 0.0
        assert signal.confidence >= 0.7  # High confidence at extreme
        assert signal.metadata['rsi_value'] == 75
    
    def test_generate_signal_neutral(self):
        """Test signal generation in neutral condition"""
        generator = RSISignalGenerator(period=14)
        df = self.create_test_dataframe_with_rsi([50])  # Neutral
        
        signal = generator.generate_signal(df, 0)
        
        assert signal.direction == SignalDirection.HOLD
        assert signal.strength == 0.0
        assert signal.confidence == 0.3  # Lower confidence in neutral zone
    
    def test_generate_signal_extreme_oversold(self):
        """Test signal generation in extreme oversold condition"""
        generator = RSISignalGenerator(period=14, oversold=30)
        df = self.create_test_dataframe_with_rsi([15])  # Extreme oversold
        
        signal = generator.generate_signal(df, 0)
        
        assert signal.direction == SignalDirection.BUY
        assert signal.strength > 0.0
        assert signal.confidence == 0.9  # Very high confidence at extreme
    
    def test_generate_signal_extreme_overbought(self):
        """Test signal generation in extreme overbought condition"""
        generator = RSISignalGenerator(period=14, overbought=70)
        df = self.create_test_dataframe_with_rsi([85])  # Extreme overbought
        
        signal = generator.generate_signal(df, 0)
        
        assert signal.direction == SignalDirection.SELL
        assert signal.strength > 0.0
        assert signal.confidence == 0.9  # Very high confidence at extreme
    
    def test_get_confidence_levels(self):
        """Test confidence levels at different RSI values"""
        generator = RSISignalGenerator(period=14)
        
        # Test extreme levels (high confidence)
        df_extreme = self.create_test_dataframe_with_rsi([15])
        confidence_extreme = generator.get_confidence(df_extreme, 0)
        assert confidence_extreme == 0.9
        
        # Test moderate levels (medium confidence)
        df_moderate = self.create_test_dataframe_with_rsi([25])
        confidence_moderate = generator.get_confidence(df_moderate, 0)
        assert confidence_moderate == 0.7
        
        # Test neutral levels (low confidence)
        df_neutral = self.create_test_dataframe_with_rsi([50])
        confidence_neutral = generator.get_confidence(df_neutral, 0)
        assert confidence_neutral == 0.3
    
    def test_get_parameters(self):
        """Test get_parameters method"""
        generator = RSISignalGenerator(
            name="custom_rsi",
            period=21,
            overbought=75,
            oversold=25
        )
        
        params = generator.get_parameters()
        
        assert params['name'] == "custom_rsi"
        assert params['period'] == 21
        assert params['overbought'] == 75
        assert params['oversold'] == 25


class TestMACDSignalGenerator:
    """Test MACDSignalGenerator implementation"""
    
    def create_test_dataframe_with_macd(self, macd_values, macd_signal_values, macd_hist_values):
        """Create test DataFrame with specific MACD values"""
        length = len(macd_values)
        dates = pd.date_range('2023-01-01', periods=length, freq='1H')
        
        data = {
            'open': [100] * length,
            'high': [105] * length,
            'low': [95] * length,
            'close': [100] * length,
            'volume': [1000] * length,
            'macd': macd_values,
            'macd_signal': macd_signal_values,
            'macd_hist': macd_hist_values
        }
        
        return pd.DataFrame(data, index=dates)
    
    def test_macd_signal_generator_initialization(self):
        """Test MACDSignalGenerator initialization"""
        generator = MACDSignalGenerator(
            name="test_macd",
            fast_period=10,
            slow_period=20,
            signal_period=5
        )
        
        assert generator.name == "test_macd"
        assert generator.fast_period == 10
        assert generator.slow_period == 20
        assert generator.signal_period == 5
        assert generator.min_periods == 25  # slow_period + signal_period
    
    def test_generate_signal_insufficient_history(self):
        """Test signal generation with insufficient history"""
        generator = MACDSignalGenerator()
        df = pd.DataFrame({
            'open': [100] * 20,
            'high': [105] * 20,
            'low': [95] * 20,
            'close': [100] * 20,
            'volume': [1000] * 20
        })
        
        signal = generator.generate_signal(df, 10)
        
        assert signal.direction == SignalDirection.HOLD
        assert signal.strength == 0.0
        assert signal.confidence == 0.0
        assert signal.metadata['reason'] == 'insufficient_history'
    
    def test_generate_signal_bullish_crossover(self):
        """Test signal generation for bullish MACD crossover"""
        generator = MACDSignalGenerator()
        
        # Create enough history + crossover: MACD crosses above signal line
        base_values = [0.0] * 35  # Enough for min_periods
        df = self.create_test_dataframe_with_macd(
            macd_values=base_values + [-0.1, 0.1],
            macd_signal_values=base_values + [0.0, 0.0],
            macd_hist_values=base_values + [-0.1, 0.1]
        )
        
        signal = generator.generate_signal(df, 36)  # Index after crossover
        
        assert signal.direction == SignalDirection.BUY
        assert signal.strength > 0.0
        assert signal.confidence > 0.0
        assert signal.metadata['crossover_detected'] == True
        assert signal.metadata['macd_value'] == 0.1
        assert signal.metadata['macd_signal_value'] == 0.0
    
    def test_generate_signal_bearish_crossover(self):
        """Test signal generation for bearish MACD crossover"""
        generator = MACDSignalGenerator()
        
        # Create enough history + crossover: MACD crosses below signal line
        base_values = [0.0] * 35  # Enough for min_periods
        df = self.create_test_dataframe_with_macd(
            macd_values=base_values + [0.1, -0.1],
            macd_signal_values=base_values + [0.0, 0.0],
            macd_hist_values=base_values + [0.1, -0.1]
        )
        
        signal = generator.generate_signal(df, 36)  # Index after crossover
        
        assert signal.direction == SignalDirection.SELL
        assert signal.strength > 0.0
        assert signal.confidence > 0.0
        assert signal.metadata['crossover_detected'] == True
    
    def test_generate_signal_no_crossover(self):
        """Test signal generation when no crossover occurs"""
        generator = MACDSignalGenerator()
        
        # No crossover: MACD stays above signal line
        base_values = [0.0] * 35  # Enough for min_periods
        df = self.create_test_dataframe_with_macd(
            macd_values=base_values + [0.1, 0.2],
            macd_signal_values=base_values + [0.0, 0.0],
            macd_hist_values=base_values + [0.1, 0.2]
        )
        
        signal = generator.generate_signal(df, 36)  # Index after no crossover
        
        assert signal.direction == SignalDirection.HOLD
        assert signal.strength == 0.0
        assert signal.metadata['crossover_detected'] == False
    
    def test_confidence_based_on_histogram(self):
        """Test confidence calculation based on MACD histogram strength"""
        generator = MACDSignalGenerator()
        
        # Strong histogram
        base_values = [0.0] * 35
        df_strong = self.create_test_dataframe_with_macd(
            macd_values=base_values + [-0.1, 0.1],
            macd_signal_values=base_values + [0.0, 0.0],
            macd_hist_values=base_values + [-0.1, 0.1]  # Strong histogram
        )
        
        signal_strong = generator.generate_signal(df_strong, 36)
        
        # Weak histogram
        df_weak = self.create_test_dataframe_with_macd(
            macd_values=base_values + [-0.01, 0.01],
            macd_signal_values=base_values + [0.0, 0.0],
            macd_hist_values=base_values + [-0.01, 0.01]  # Weak histogram
        )
        
        signal_weak = generator.generate_signal(df_weak, 36)
        
        # Strong histogram should have higher confidence
        assert signal_strong.confidence > signal_weak.confidence
    
    def test_get_confidence(self):
        """Test get_confidence method"""
        generator = MACDSignalGenerator()
        df = self.create_test_dataframe_with_macd(
            macd_values=[0.1],
            macd_signal_values=[0.0],
            macd_hist_values=[0.1]
        )
        
        confidence = generator.get_confidence(df, 0)
        
        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)
    
    def test_get_parameters(self):
        """Test get_parameters method"""
        generator = MACDSignalGenerator(
            name="custom_macd",
            fast_period=8,
            slow_period=21,
            signal_period=7
        )
        
        params = generator.get_parameters()
        
        assert params['name'] == "custom_macd"
        assert params['fast_period'] == 8
        assert params['slow_period'] == 21
        assert params['signal_period'] == 7
        assert params['min_periods'] == 28  # slow_period + signal_period


class TestTechnicalSignalGeneratorEdgeCases:
    """Test edge cases and error conditions for technical signal generators"""
    
    def test_invalid_dataframe(self):
        """Test behavior with invalid DataFrame"""
        generator = TechnicalSignalGenerator()
        
        # Test empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            generator.generate_signal(empty_df, 0)
        
        # Test DataFrame missing required columns
        invalid_df = pd.DataFrame({'price': [100, 101, 102]})
        with pytest.raises(ValueError, match="DataFrame missing required columns"):
            generator.generate_signal(invalid_df, 0)
    
    def test_invalid_index(self):
        """Test behavior with invalid index"""
        generator = TechnicalSignalGenerator()
        df = pd.DataFrame({
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        })
        
        # Test negative index
        with pytest.raises(IndexError, match="Index -1 is out of bounds"):
            generator.generate_signal(df, -1)
        
        # Test index >= length
        with pytest.raises(IndexError, match="Index 10 is out of bounds"):
            generator.generate_signal(df, 10)
    
    def test_nan_indicator_handling(self):
        """Test handling of NaN indicator values"""
        generator = RSISignalGenerator(period=14)
        
        df = pd.DataFrame({
            'open': [100] * 20,
            'high': [105] * 20,
            'low': [95] * 20,
            'close': [100] * 20,
            'volume': [1000] * 20,
            'rsi': [np.nan] * 20  # All NaN RSI values
        })
        
        signal = generator.generate_signal(df, 15)
        
        assert signal.direction == SignalDirection.HOLD
        assert signal.strength == 0.0
        assert signal.confidence == 0.0
        assert signal.metadata['reason'] == 'rsi_calculation_failed'
    
    def test_zero_division_handling(self):
        """Test handling of potential zero division in calculations"""
        generator = TechnicalSignalGenerator()
        
        # Create DataFrame with constant prices (could cause division by zero)
        df = pd.DataFrame({
            'open': [100] * 100,
            'high': [100] * 100,
            'low': [100] * 100,
            'close': [100] * 100,
            'volume': [1000] * 100
        })
        
        # Should handle gracefully without crashing
        signal = generator.generate_signal(df, 80)
        
        assert isinstance(signal, Signal)
        assert isinstance(signal.direction, SignalDirection)
        assert 0.0 <= signal.strength <= 1.0
        assert 0.0 <= signal.confidence <= 1.0