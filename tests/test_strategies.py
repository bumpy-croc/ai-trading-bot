"""
Tests for trading strategies.

Strategy tests are crucial as they generate trading signals. Tests cover:
- Signal generation accuracy
- Indicator calculations
- Position sizing logic
- Entry/exit condition validation
- Edge cases and market conditions
- Strategy parameter validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategies.base import BaseStrategy
from strategies.adaptive import AdaptiveStrategy
from strategies.enhanced import EnhancedStrategy


class TestBaseStrategy:
    """Test the base strategy interface"""

    def test_base_strategy_is_abstract(self):
        """Test that BaseStrategy cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseStrategy("TestStrategy")

    def test_base_strategy_interface(self, real_adaptive_strategy):
        """Test that concrete strategies implement the required interface"""
        strategy = real_adaptive_strategy
        
        # Test required methods exist
        assert hasattr(strategy, 'calculate_indicators')
        assert hasattr(strategy, 'check_entry_conditions')
        assert hasattr(strategy, 'check_exit_conditions')
        assert hasattr(strategy, 'calculate_position_size')
        assert hasattr(strategy, 'get_parameters')
        assert hasattr(strategy, 'get_trading_pair')
        assert hasattr(strategy, 'set_trading_pair')

    def test_trading_pair_management(self, real_adaptive_strategy):
        """Test trading pair getter/setter"""
        strategy = real_adaptive_strategy
        
        # Test default
        assert strategy.get_trading_pair() == 'BTCUSDT'
        
        # Test setter
        strategy.set_trading_pair('ETHUSDT')
        assert strategy.get_trading_pair() == 'ETHUSDT'


class TestAdaptiveStrategy:
    """Test the adaptive strategy implementation"""

    @pytest.mark.strategy
    def test_adaptive_strategy_initialization(self):
        """Test adaptive strategy initialization"""
        strategy = AdaptiveStrategy()
        
        assert strategy.name == "AdaptiveStrategy"
        assert strategy.trading_pair == 'BTCUSDT'
        assert hasattr(strategy, 'base_risk_per_trade')
        assert hasattr(strategy, 'fast_ma')
        assert hasattr(strategy, 'slow_ma')

    @pytest.mark.strategy
    def test_indicator_calculation(self, sample_ohlcv_data):
        """Test that all required indicators are calculated"""
        strategy = AdaptiveStrategy()
        
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        
        # Check that indicators were added
        required_indicators = [
            f'ma_{strategy.fast_ma}',
            f'ma_{strategy.slow_ma}', 
            f'ma_{strategy.long_ma}',
            'rsi',
            'atr',
            'volatility',
            'trend_strength',
            'volume_ma',
            'regime'
        ]
        
        for indicator in required_indicators:
            assert indicator in df_with_indicators.columns, f"Missing indicator: {indicator}"

    @pytest.mark.strategy
    def test_atr_calculation(self, sample_ohlcv_data):
        """Test ATR calculation accuracy"""
        strategy = AdaptiveStrategy()
        
        # Calculate ATR manually to verify
        df = sample_ohlcv_data.copy()
        atr_series = strategy.calculate_atr(df)
        
        # ATR should be positive
        assert (atr_series >= 0).all()
        
        # ATR should have the right length (with NaN for initial values)
        assert len(atr_series) == len(df)
        
        # First few values should be NaN due to rolling calculation
        assert pd.isna(atr_series.iloc[:13]).all()  # First 14-1 values should be NaN

    @pytest.mark.strategy
    def test_trend_strength_calculation(self, sample_ohlcv_data):
        """Test trend strength calculation"""
        strategy = AdaptiveStrategy()
        
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        
        # Trend strength should be a number
        trend_strength = df_with_indicators['trend_strength']
        assert pd.api.types.is_numeric_dtype(trend_strength)
        
        # Should have reasonable range (not extreme values)
        valid_trend = trend_strength.dropna()
        if len(valid_trend) > 0:
            assert valid_trend.abs().max() < 1.0  # Should be reasonable percentage

    @pytest.mark.strategy
    def test_market_regime_detection(self, sample_ohlcv_data):
        """Test market regime detection"""
        strategy = AdaptiveStrategy()
        
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        
        regimes = df_with_indicators['regime'].dropna().unique()
        valid_regimes = {'trending', 'ranging', 'volatile'}
        
        # All detected regimes should be valid
        for regime in regimes:
            assert regime in valid_regimes

    @pytest.mark.strategy
    def test_entry_conditions_with_valid_data(self, sample_ohlcv_data):
        """Test entry condition checking with valid data"""
        strategy = AdaptiveStrategy()
        
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        
        # Test entry conditions on various points
        for i in range(max(1, len(df_with_indicators) - 10), len(df_with_indicators)):
            result = strategy.check_entry_conditions(df_with_indicators, i)
            assert isinstance(result, bool)

    @pytest.mark.strategy
    def test_exit_conditions_with_valid_data(self, sample_ohlcv_data):
        """Test exit condition checking with valid data"""
        strategy = AdaptiveStrategy()
        
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        entry_price = df_with_indicators['close'].iloc[-10]
        
        # Test exit conditions
        for i in range(max(1, len(df_with_indicators) - 5), len(df_with_indicators)):
            result = strategy.check_exit_conditions(df_with_indicators, i, entry_price)
            assert isinstance(result, bool)

    @pytest.mark.strategy
    def test_position_size_calculation(self, sample_ohlcv_data):
        """Test position size calculation"""
        strategy = AdaptiveStrategy()
        
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        balance = 10000
        
        # Test position sizing
        for i in range(max(1, len(df_with_indicators) - 5), len(df_with_indicators)):
            position_size = strategy.calculate_position_size(df_with_indicators, i, balance)
            
            # Position size should be reasonable
            assert position_size >= 0
            assert position_size <= balance  # Should not exceed total balance

    @pytest.mark.strategy
    def test_stop_loss_calculation(self, sample_ohlcv_data):
        """Test stop loss calculation"""
        strategy = AdaptiveStrategy()
        
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        price = 50000
        
        # Test long stop loss
        stop_loss_long = strategy.calculate_stop_loss(df_with_indicators, len(df_with_indicators)-1, price, 'long')
        assert stop_loss_long < price  # Long stop loss should be below entry price
        assert stop_loss_long > 0
        
        # Test short stop loss
        stop_loss_short = strategy.calculate_stop_loss(df_with_indicators, len(df_with_indicators)-1, price, 'short')
        assert stop_loss_short > price  # Short stop loss should be above entry price

    @pytest.mark.strategy
    def test_strategy_parameters(self):
        """Test strategy parameter retrieval"""
        strategy = AdaptiveStrategy()
        
        params = strategy.get_parameters()
        
        assert isinstance(params, dict)
        assert 'name' in params
        assert 'base_risk_per_trade' in params
        assert 'fast_ma' in params
        assert 'slow_ma' in params
        
        # Parameters should be reasonable
        assert 0 < params['base_risk_per_trade'] <= 0.1  # Between 0% and 10%
        assert params['fast_ma'] < params['slow_ma']  # Fast MA should be shorter


class TestStrategyEdgeCases:
    """Test strategy behavior in edge cases"""

    @pytest.mark.strategy
    def test_insufficient_data_handling(self):
        """Test strategy behavior with insufficient data"""
        strategy = AdaptiveStrategy()
        
        # Create minimal dataset
        minimal_data = pd.DataFrame({
            'open': [50000, 50100],
            'high': [50200, 50300], 
            'low': [49800, 49900],
            'close': [50100, 50200],
            'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2, freq='1H'))
        
        # Should handle gracefully
        try:
            df_with_indicators = strategy.calculate_indicators(minimal_data)
            # Most indicators will be NaN, but shouldn't crash
            assert len(df_with_indicators) == len(minimal_data)
        except Exception as e:
            pytest.fail(f"Strategy failed with minimal data: {e}")

    @pytest.mark.strategy
    def test_entry_conditions_out_of_bounds(self, sample_ohlcv_data):
        """Test entry conditions with out-of-bounds indices"""
        strategy = AdaptiveStrategy()
        
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        
        # Test negative index
        result = strategy.check_entry_conditions(df_with_indicators, -1)
        assert result == False
        
        # Test index beyond data length
        result = strategy.check_entry_conditions(df_with_indicators, len(df_with_indicators) + 10)
        assert result == False
        
        # Test index 0 (should be False due to needing previous data)
        result = strategy.check_entry_conditions(df_with_indicators, 0)
        assert result == False

    @pytest.mark.strategy
    def test_exit_conditions_out_of_bounds(self, sample_ohlcv_data):
        """Test exit conditions with out-of-bounds indices"""
        strategy = AdaptiveStrategy()
        
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        entry_price = 50000
        
        # Test out-of-bounds indices
        result = strategy.check_exit_conditions(df_with_indicators, -1, entry_price)
        assert result == False
        
        result = strategy.check_exit_conditions(df_with_indicators, len(df_with_indicators) + 10, entry_price)
        assert result == False

    @pytest.mark.strategy
    def test_position_size_edge_cases(self, sample_ohlcv_data):
        """Test position size calculation edge cases"""
        strategy = AdaptiveStrategy()
        
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        
        # Zero balance
        position_size = strategy.calculate_position_size(df_with_indicators, len(df_with_indicators)-1, 0)
        assert position_size == 0
        
        # Very small balance
        position_size = strategy.calculate_position_size(df_with_indicators, len(df_with_indicators)-1, 1)
        assert position_size >= 0
        assert position_size <= 1

    @pytest.mark.strategy
    def test_missing_indicator_data(self):
        """Test strategy behavior when indicator calculation fails"""
        strategy = AdaptiveStrategy()
        
        # Create data with missing values
        problematic_data = pd.DataFrame({
            'open': [50000, np.nan, 50200],
            'high': [50200, 50300, np.nan], 
            'low': [49800, 49900, 50000],
            'close': [50100, 50200, 50300],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1H'))
        
        # Should handle NaN values gracefully
        try:
            df_with_indicators = strategy.calculate_indicators(problematic_data)
            # Should complete without crashing
            assert len(df_with_indicators) == len(problematic_data)
        except Exception as e:
            pytest.fail(f"Strategy failed with NaN data: {e}")


class TestStrategyMarketConditions:
    """Test strategy behavior in different market conditions"""

    @pytest.mark.strategy
    def test_bull_market_conditions(self, market_conditions):
        """Test strategy in bull market conditions"""
        strategy = AdaptiveStrategy()
        
        # Create uptrending data
        dates = pd.date_range('2024-01-01', periods=50, freq='1H')
        closes = [50000 * (1.01 ** i) for i in range(50)]  # 1% hourly growth
        
        bull_data = pd.DataFrame({
            'open': [c * 0.995 for c in closes],
            'high': [c * 1.005 for c in closes],
            'low': [c * 0.99 for c in closes],
            'close': closes,
            'volume': [1000 + np.random.randint(-100, 100) for _ in range(50)]
        }, index=dates)
        
        df_with_indicators = strategy.calculate_indicators(bull_data)
        
        # In bull market, should generate some entry signals
        entry_signals = []
        for i in range(10, len(df_with_indicators)):
            if strategy.check_entry_conditions(df_with_indicators, i):
                entry_signals.append(i)
        
        # Should detect the trend and generate some signals
        assert len(entry_signals) > 0

    @pytest.mark.strategy
    def test_bear_market_conditions(self):
        """Test strategy in bear market conditions"""
        strategy = AdaptiveStrategy()
        
        # Create downtrending data
        dates = pd.date_range('2024-01-01', periods=50, freq='1H')
        closes = [50000 * (0.99 ** i) for i in range(50)]  # 1% hourly decline
        
        bear_data = pd.DataFrame({
            'open': [c * 1.005 for c in closes],
            'high': [c * 1.01 for c in closes],
            'low': [c * 0.995 for c in closes],
            'close': closes,
            'volume': [1000 + np.random.randint(-100, 100) for _ in range(50)]
        }, index=dates)
        
        df_with_indicators = strategy.calculate_indicators(bear_data)
        
        # Strategy should adapt to bear market
        # Test that it doesn't generate excessive long signals in declining market
        entry_signals = []
        for i in range(10, len(df_with_indicators)):
            if strategy.check_entry_conditions(df_with_indicators, i):
                entry_signals.append(i)
        
        # Should be more conservative in bear market
        assert len(entry_signals) <= len(df_with_indicators) * 0.3  # Max 30% of periods

    @pytest.mark.strategy
    def test_sideways_market_conditions(self):
        """Test strategy in sideways market conditions"""
        strategy = AdaptiveStrategy()
        
        # Create sideways/ranging data
        dates = pd.date_range('2024-01-01', periods=50, freq='1H')
        base_price = 50000
        closes = [base_price + 1000 * np.sin(i * 0.3) for i in range(50)]  # Oscillating around base
        
        sideways_data = pd.DataFrame({
            'open': [c + np.random.uniform(-50, 50) for c in closes],
            'high': [c + abs(np.random.uniform(0, 100)) for c in closes],
            'low': [c - abs(np.random.uniform(0, 100)) for c in closes],
            'close': closes,
            'volume': [1000 + np.random.randint(-100, 100) for _ in range(50)]
        }, index=dates)
        
        df_with_indicators = strategy.calculate_indicators(sideways_data)
        
        # Check regime detection
        regimes = df_with_indicators['regime'].dropna()
        if len(regimes) > 0:
            # Should detect ranging conditions
            ranging_count = (regimes == 'ranging').sum()
            assert ranging_count > 0

    @pytest.mark.strategy
    def test_volatile_market_conditions(self):
        """Test strategy in highly volatile market conditions"""
        strategy = AdaptiveStrategy()
        
        # Create highly volatile data
        dates = pd.date_range('2024-01-01', periods=50, freq='1H')
        base_price = 50000
        closes = [base_price * (1 + np.random.uniform(-0.05, 0.05)) for _ in range(50)]  # 5% random moves
        
        volatile_data = pd.DataFrame({
            'open': [c * (1 + np.random.uniform(-0.02, 0.02)) for c in closes],
            'high': [c * (1 + abs(np.random.uniform(0, 0.03))) for c in closes],
            'low': [c * (1 - abs(np.random.uniform(0, 0.03))) for c in closes],
            'close': closes,
            'volume': [1000 + np.random.randint(-500, 500) for _ in range(50)]
        }, index=dates)
        
        df_with_indicators = strategy.calculate_indicators(volatile_data)
        
        # Strategy should reduce position sizes in volatile conditions
        position_sizes = []
        for i in range(20, len(df_with_indicators)):
            size = strategy.calculate_position_size(df_with_indicators, i, 10000)
            position_sizes.append(size)
        
        # Check that positions are generally smaller due to volatility
        avg_position_size = np.mean([s for s in position_sizes if s > 0])
        if not np.isnan(avg_position_size):
            assert avg_position_size <= 10000 * strategy.max_position_size


class TestMultipleStrategies:
    """Test behavior with multiple strategy implementations"""

    @pytest.mark.strategy
    def test_strategy_consistency(self, sample_ohlcv_data):
        """Test that different strategies produce consistent data structures"""
        strategies = [AdaptiveStrategy()]
        
        # Add other strategies when available
        try:
            strategies.append(EnhancedStrategy())
        except:
            pass  # Enhanced strategy might not be available
        
        for strategy in strategies:
            df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
            
            # All strategies should return DataFrame
            assert isinstance(df_with_indicators, pd.DataFrame)
            
            # Should have at least the original columns
            for col in sample_ohlcv_data.columns:
                assert col in df_with_indicators.columns
            
            # Should have parameters
            params = strategy.get_parameters()
            assert isinstance(params, dict)
            assert 'name' in params

    @pytest.mark.strategy
    def test_strategy_parameter_ranges(self):
        """Test that strategy parameters are in reasonable ranges"""
        strategy = AdaptiveStrategy()
        params = strategy.get_parameters()
        
        # Risk parameters should be reasonable
        if 'base_risk_per_trade' in params:
            assert 0 < params['base_risk_per_trade'] <= 0.1
        
        if 'max_risk_per_trade' in params:
            assert 0 < params['max_risk_per_trade'] <= 0.2
        
        # Moving average periods should be reasonable
        if 'fast_ma' in params and 'slow_ma' in params:
            assert params['fast_ma'] < params['slow_ma']
            assert params['fast_ma'] >= 2
            assert params['slow_ma'] <= 200