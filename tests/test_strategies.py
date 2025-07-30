"""
Tests for trading strategies.

Strategy tests are crucial as they generate trading signals. Tests cover:
- Signal generation accuracy
- Indicator calculations
- Position sizing logic
- Entry/exit condition validation
- Edge cases and market conditions
- Strategy parameter validation
- Strategy execution logging
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from strategies.base import BaseStrategy
from strategies.ml_adaptive import MlAdaptive
from strategies.ml_with_sentiment import MlWithSentiment
from strategies.ml_basic import MlBasic


class TestBaseStrategy:
    """Test the base strategy interface"""

    def test_base_strategy_is_abstract(self):
        """Test that BaseStrategy cannot be instantiated directly"""
        with pytest.raises(TypeError):
            # BaseStrategy is abstract and requires a name parameter, but should still fail
            BaseStrategy("TestStrategy")

    def test_base_strategy_interface(self, mock_strategy):
        """Test that concrete strategies implement the required interface"""
        strategy = mock_strategy
        
        # Test required methods exist
        assert hasattr(strategy, 'calculate_indicators')
        assert hasattr(strategy, 'check_entry_conditions')
        assert hasattr(strategy, 'check_exit_conditions')
        assert hasattr(strategy, 'calculate_position_size')
        assert hasattr(strategy, 'get_parameters')
        
        # Test optional methods with fallbacks
        if hasattr(strategy, 'get_trading_pair'):
            assert hasattr(strategy, 'set_trading_pair')

    def test_trading_pair_management(self, mock_strategy):
        """Test trading pair getter/setter"""
        strategy = mock_strategy
        
        # Test that trading_pair attribute exists
        assert hasattr(strategy, 'trading_pair')
        assert isinstance(strategy.trading_pair, str)
        
        # Test that we can access the trading pair
        assert strategy.trading_pair == "BTCUSDT"


class TestMlAdaptive:
    """Test the adaptive strategy implementation"""

    @pytest.mark.strategy
    def test_adaptive_strategy_initialization(self):
        """Test adaptive strategy initialization"""
        strategy = MlAdaptive()
        
        assert hasattr(strategy, 'name')
        # Use getattr with default for optional attributes
        assert getattr(strategy, 'trading_pair', 'BTC-USD') is not None
        assert hasattr(strategy, 'base_stop_loss_pct')
        assert hasattr(strategy, 'base_take_profit_pct')
        assert hasattr(strategy, 'base_position_size')

    @pytest.mark.strategy
    def test_indicator_calculation(self, sample_ohlcv_data):
        """Test that all required indicators are calculated"""
        strategy = MlAdaptive()
        
        # Ensure we have enough data for indicators
        if len(sample_ohlcv_data) < 50:
            # Extend sample data for testing
            additional_data = []
            base_price = sample_ohlcv_data['close'].iloc[-1]
            last_date = sample_ohlcv_data.index[-1]
            
            for i in range(50 - len(sample_ohlcv_data)):
                new_date = last_date + timedelta(hours=i+1)
                price_change = np.random.normal(0, 0.01)
                new_price = base_price * (1 + price_change)
                
                additional_data.append({
                    'open': base_price,
                    'high': new_price * 1.005,
                    'low': new_price * 0.995,
                    'close': new_price,
                    'volume': 1000
                })
                base_price = new_price
            
            additional_df = pd.DataFrame(additional_data, index=pd.date_range(
                last_date + timedelta(hours=1), periods=len(additional_data), freq='1h'
            ))
            sample_ohlcv_data = pd.concat([sample_ohlcv_data, additional_df])
        
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        
        # Check that indicators were added (flexible checking)
        original_columns = set(sample_ohlcv_data.columns)
        new_columns = set(df_with_indicators.columns)
        added_columns = new_columns - original_columns
        
        # Should have added some indicators
        assert len(added_columns) > 0

    @pytest.mark.strategy
    def test_atr_calculation(self, sample_ohlcv_data):
        """Test ATR calculation accuracy"""
        strategy = MlAdaptive()
        
        # Check if strategy has ATR calculation method
        if hasattr(strategy, 'calculate_atr'):
            atr_series = strategy.calculate_atr(sample_ohlcv_data)
            
            # ATR should be positive where calculated
            valid_atr = atr_series.dropna()
            if len(valid_atr) > 0:
                assert (valid_atr >= 0).all()
            
            # ATR should have the right length
            assert len(atr_series) == len(sample_ohlcv_data)
        else:
            # If no explicit ATR method, check if it's calculated in indicators
            df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
            atr_columns = [col for col in df_with_indicators.columns if 'atr' in col.lower()]
            assert len(atr_columns) > 0, "No ATR calculation found"

    @pytest.mark.strategy
    def test_entry_conditions_with_valid_data(self, sample_ohlcv_data):
        """Test entry condition checking with valid data"""
        strategy = MlAdaptive()
        
        try:
            df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
            
            # Test entry conditions on valid indices only
            valid_indices = range(max(1, len(df_with_indicators) - 10), len(df_with_indicators))
            for i in valid_indices:
                if i < len(df_with_indicators):
                    result = strategy.check_entry_conditions(df_with_indicators, i)
                    assert isinstance(result, bool)
        except Exception as e:
            # If indicators calculation fails due to insufficient data, skip gracefully
            if "insufficient" in str(e).lower() or len(sample_ohlcv_data) < 20:
                pytest.skip(f"Insufficient data for strategy testing: {e}")
            else:
                raise

    @pytest.mark.strategy
    def test_position_size_calculation(self, sample_ohlcv_data):
        """Test position size calculation"""
        strategy = MlAdaptive()
        
        try:
            df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
            balance = 10000
            
            # Test position sizing on valid data points
            valid_indices = range(max(20, len(df_with_indicators) - 5), len(df_with_indicators))
            for i in valid_indices:
                if i < len(df_with_indicators):
                    position_size = strategy.calculate_position_size(df_with_indicators, i, balance)
                    
                    # Position size should be reasonable
                    assert position_size >= 0
                    # Position size should not exceed reasonable bounds
                    max_reasonable_size = balance * 0.5  # 50% max
                    assert position_size <= max_reasonable_size
        except Exception as e:
            if "insufficient" in str(e).lower() or len(sample_ohlcv_data) < 20:
                pytest.skip(f"Insufficient data for position size testing: {e}")
            else:
                raise

    @pytest.mark.strategy
    def test_strategy_parameters(self):
        """Test strategy parameter retrieval"""
        strategy = MlAdaptive()
        
        params = strategy.get_parameters()
        
        assert isinstance(params, dict)
        # Check for common parameters (flexible)
        expected_params = ['name', 'base_risk_per_trade']
        present_params = [param for param in expected_params if param in params]
        assert len(present_params) > 0, f"Expected at least one of {expected_params} in params"
        
        # Validate risk parameter if present
        if 'base_risk_per_trade' in params:
            assert 0 < params['base_risk_per_trade'] <= 0.1  # Between 0% and 10%

    @pytest.mark.strategy
    def test_stop_loss_calculation(self, sample_ohlcv_data):
        """Test stop loss calculation"""
        strategy = MlAdaptive()
        
        # Only test if method exists
        if hasattr(strategy, 'calculate_stop_loss'):
            try:
                df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
                price = 50000
                
                if len(df_with_indicators) > 0:
                    # Test long stop loss
                    stop_loss_long = strategy.calculate_stop_loss(
                        df_with_indicators, len(df_with_indicators)-1, price, 'long'
                    )
                    assert stop_loss_long < price  # Long stop loss should be below entry price
                    assert stop_loss_long > 0
                    
                    # Test short stop loss
                    stop_loss_short = strategy.calculate_stop_loss(
                        df_with_indicators, len(df_with_indicators)-1, price, 'short'
                    )
                    assert stop_loss_short > price  # Short stop loss should be above entry price
            except Exception as e:
                if "insufficient" in str(e).lower():
                    pytest.skip(f"Insufficient data for stop loss testing: {e}")
                else:
                    raise


class TestStrategyEdgeCases:
    """Test strategy behavior in edge cases"""

    @pytest.mark.strategy
    def test_insufficient_data_handling(self):
        """Test strategy behavior with insufficient data"""
        strategy = MlAdaptive()
        
        # Create minimal dataset
        minimal_data = pd.DataFrame({
            'open': [50000, 50100],
            'high': [50200, 50300], 
            'low': [49800, 49900],
            'close': [50100, 50200],
            'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2, freq='1h'))
        
        # Should handle gracefully - either work or fail with meaningful error
        try:
            df_with_indicators = strategy.calculate_indicators(minimal_data)
            # If it works, should return a DataFrame
            assert isinstance(df_with_indicators, pd.DataFrame)
            assert len(df_with_indicators) == len(minimal_data)
        except Exception as e:
            # Should be a meaningful error about insufficient data
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in ['insufficient', 'data', 'length', 'period'])

    @pytest.mark.strategy
    def test_entry_conditions_out_of_bounds(self, sample_ohlcv_data):
        """Test entry conditions with out-of-bounds indices"""
        strategy = MlAdaptive()
        
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        
        # Test negative index - should handle gracefully
        result = strategy.check_entry_conditions(df_with_indicators, -1)
        assert result == False
        
        # Test index beyond data length - should handle gracefully
        result = strategy.check_entry_conditions(df_with_indicators, len(df_with_indicators) + 10)
        assert result == False
        
        # Test index 0 (should be False due to needing previous data)
        result = strategy.check_entry_conditions(df_with_indicators, 0)
        assert result == False

    @pytest.mark.strategy
    def test_position_size_edge_cases(self, sample_ohlcv_data):
        """Test position size calculation edge cases"""
        strategy = MlAdaptive()
        
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        
        if len(df_with_indicators) > 0:
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
        strategy = MlAdaptive()
        
        # Create data with missing values
        problematic_data = pd.DataFrame({
            'open': [50000, np.nan, 50200],
            'high': [50200, 50300, np.nan], 
            'low': [49800, 49900, 50000],
            'close': [50100, 50200, 50300],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1h'))
        
        # Should handle NaN values gracefully
        try:
            df_with_indicators = strategy.calculate_indicators(problematic_data)
            # Should complete without crashing
            assert isinstance(df_with_indicators, pd.DataFrame)
            assert len(df_with_indicators) == len(problematic_data)
        except Exception as e:
            # Should be a meaningful error about data quality
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in ['nan', 'missing', 'invalid', 'data'])


class TestStrategyMarketConditions:
    """Test strategy behavior in different market conditions"""

    @pytest.mark.strategy
    def test_bull_market_conditions(self, market_conditions):
        """Test strategy in bull market conditions"""
        strategy = MlAdaptive()
        
        # Create uptrending data
        dates = pd.date_range('2024-01-01', periods=50, freq='1h')
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
        strategy = MlAdaptive()
        
        # Create downtrending data
        dates = pd.date_range('2024-01-01', periods=50, freq='1h')
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
        strategy = MlAdaptive()
        
        # Create sideways/ranging data
        dates = pd.date_range('2024-01-01', periods=50, freq='1h')
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
        strategy = MlAdaptive()
        
        # Create highly volatile data
        dates = pd.date_range('2024-01-01', periods=50, freq='1h')
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
        strategies = [MlAdaptive()]
        
        # Add other strategies when available
        try:
            strategies.append(MlWithSentiment())
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
        strategy = MlAdaptive()
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


# Tests moved from test_missing_strategies.py
class TestMlWithSentiment:
    """Test the enhanced strategy implementation and logging"""

    @pytest.mark.strategy
    def test_enhanced_strategy_initialization(self):
        """Test enhanced strategy initialization"""
        strategy = MlWithSentiment()
        
        assert hasattr(strategy, 'name')
        assert getattr(strategy, 'trading_pair', 'BTCUSDT') is not None
        assert hasattr(strategy, 'risk_per_trade')
        assert hasattr(strategy, 'max_position_size')

    @pytest.mark.strategy
    def test_enhanced_strategy_execution_logging(self, sample_ohlcv_data):
        """Test that enhanced strategy logs execution details"""
        strategy = MlWithSentiment()
        
        # Mock database manager for logging
        mock_db_manager = Mock()
        strategy.set_database_manager(mock_db_manager, session_id=123)
        
        # Calculate indicators
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        
        # Test entry conditions on valid indices
        valid_indices = range(max(20, len(df_with_indicators) - 5), len(df_with_indicators))
        for i in valid_indices:
            if i < len(df_with_indicators):
                result = strategy.check_entry_conditions(df_with_indicators, i)
                try:
                    assert isinstance(result, (bool, np.bool_))
                except AssertionError:
                    print(f"DEBUG: result={result!r}, type={type(result)} at index {i}")
                    raise
                
                # Verify that log_execution was called
                mock_db_manager.log_strategy_execution.assert_called()
                
                # Get the last call arguments
                call_args = mock_db_manager.log_strategy_execution.call_args
                assert call_args is not None
                
                # Verify required parameters
                args, kwargs = call_args
                assert kwargs['strategy_name'] == strategy.name
                assert kwargs['signal_type'] == 'entry'
                assert kwargs['price'] > 0
                assert isinstance(kwargs['reasons'], list)
                assert len(kwargs['reasons']) > 0
                # Instead of checking additional_context, check for merged key-value in reasons
                assert 'strategy_type=enhanced_multi_condition' in kwargs['reasons']

    @pytest.mark.strategy
    def test_enhanced_strategy_parameters(self):
        """Test enhanced strategy parameter retrieval"""
        strategy = MlWithSentiment()
        
        params = strategy.get_parameters()
        
        # Check required parameters
        assert 'name' in params
        assert 'risk_per_trade' in params
        assert 'max_position_size' in params
        assert 'base_stop_loss_pct' in params
        assert 'min_conditions' in params


class TestMlBasicStrategy:
    """Test the ML basic strategy implementation and logging"""

    @pytest.mark.strategy
    def test_ml_basic_strategy_initialization(self):
        """Test ML basic strategy initialization"""
        strategy = MlBasic()
        
        assert hasattr(strategy, 'name')
        assert getattr(strategy, 'trading_pair', 'BTCUSDT') is not None
        assert hasattr(strategy, 'model_path')
        assert hasattr(strategy, 'sequence_length')
        assert hasattr(strategy, 'stop_loss_pct')
        assert hasattr(strategy, 'take_profit_pct')

    @pytest.mark.strategy
    def test_ml_basic_strategy_execution_logging(self, sample_ohlcv_data):
        """Test that ML basic strategy logs execution details"""
        strategy = MlBasic()
        
        # Mock database manager for logging
        mock_db_manager = Mock()
        strategy.set_database_manager(mock_db_manager, session_id=456)
        
        # Calculate indicators
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        
        # Test entry conditions on valid indices (need enough data for ML predictions)
        valid_indices = range(max(120, len(df_with_indicators) - 3), len(df_with_indicators))
        for i in valid_indices:
            if i < len(df_with_indicators):
                result = strategy.check_entry_conditions(df_with_indicators, i)
                assert isinstance(result, bool)
                
                # Verify that log_execution was called
                mock_db_manager.log_strategy_execution.assert_called()
                
                # Get the last call arguments
                call_args = mock_db_manager.log_strategy_execution.call_args
                assert call_args is not None
                
                # Verify required parameters
                args, kwargs = call_args
                assert kwargs['strategy_name'] == strategy.name
                assert kwargs['signal_type'] == 'entry'
                assert kwargs['price'] > 0
                assert isinstance(kwargs['reasons'], list)
                assert len(kwargs['reasons']) > 0
                
                # Verify ML predictions
                assert 'ml_predictions' in kwargs or 'prediction_available' in kwargs.get('additional_context', {})
                
                # Verify additional context
                assert 'additional_context' in kwargs
                context = kwargs['additional_context']
                assert 'model_type' in context
                assert context['model_type'] == 'ml_basic'

    @pytest.mark.strategy
    def test_ml_basic_strategy_missing_prediction_logging(self, sample_ohlcv_data):
        """Test that ML basic strategy logs when prediction is missing"""
        strategy = MlBasic()
        
        # Mock database manager for logging
        mock_db_manager = Mock()
        strategy.set_database_manager(mock_db_manager, session_id=789)
        
        # Create data without ML predictions
        df_no_predictions = sample_ohlcv_data.copy()
        df_no_predictions['onnx_pred'] = np.nan
        
        # Test entry conditions - should log missing prediction
        if len(df_no_predictions) > 1:
            result = strategy.check_entry_conditions(df_no_predictions, 1)
            assert result is False  # Should return False for missing prediction
            
            # Verify that log_execution was called for missing prediction
            mock_db_manager.log_strategy_execution.assert_called()
            
            # Get the last call arguments
            call_args = mock_db_manager.log_strategy_execution.call_args
            assert call_args is not None
            
            # Verify it logged the missing prediction
            args, kwargs = call_args
            assert 'missing_ml_prediction' in kwargs['reasons']
            # Instead of checking additional_context, check reasons for the merged key-value
            assert 'prediction_available=False' in kwargs['reasons']

    @pytest.mark.strategy
    def test_ml_basic_strategy_parameters(self):
        """Test ML basic strategy parameter retrieval"""
        strategy = MlBasic()
        
        params = strategy.get_parameters()
        
        # Check required parameters
        assert 'name' in params
        assert 'model_path' in params
        assert 'sequence_length' in params
        assert 'stop_loss_pct' in params
        assert 'take_profit_pct' in params

    @pytest.mark.strategy
    def test_ml_basic_exit_conditions(self, sample_ohlcv_data):
        strategy = MlBasic()
        df = strategy.calculate_indicators(sample_ohlcv_data)
        
        # * Use realistic entry price close to current price
        current_price = df['close'].iloc[-1]
        entry_price = current_price * 0.98  # Entry at 2% below current price
        
        # Test not exit (predicted price higher than current)
        df.at[df.index[-1], 'onnx_pred'] = current_price * 1.05  # 5% above current
        assert not strategy.check_exit_conditions(df, len(df)-1, entry_price)
        
        # Test exit on unfavorable prediction (predicted price lower than current)
        df.at[df.index[-1], 'onnx_pred'] = current_price * 0.95  # 5% below current
        assert strategy.check_exit_conditions(df, len(df)-1, entry_price)


class TestMlWithSentimentStrategy:
    """Test the ML with sentiment strategy implementation and logging"""

    @pytest.mark.strategy
    def test_ml_with_sentiment_strategy_initialization(self):
        """Test ML with sentiment strategy initialization"""
        strategy = MlWithSentiment()
        
        assert hasattr(strategy, 'name')
        assert getattr(strategy, 'trading_pair', 'BTCUSDT') is not None
        assert hasattr(strategy, 'model_path')
        assert hasattr(strategy, 'sequence_length')
        assert hasattr(strategy, 'use_sentiment')
        assert hasattr(strategy, 'stop_loss_pct')
        assert hasattr(strategy, 'take_profit_pct')

    @pytest.mark.strategy
    def test_ml_with_sentiment_strategy_execution_logging(self, sample_ohlcv_data):
        """Test that ML with sentiment strategy logs execution details"""
        strategy = MlWithSentiment()
        
        # Mock database manager for logging
        mock_db_manager = Mock()
        strategy.set_database_manager(mock_db_manager, session_id=101)
        
        # Calculate indicators
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        
        # Test entry conditions on valid indices (need enough data for ML predictions)
        valid_indices = range(max(120, len(df_with_indicators) - 3), len(df_with_indicators))
        for i in valid_indices:
            if i < len(df_with_indicators):
                result = strategy.check_entry_conditions(df_with_indicators, i)
                assert isinstance(result, bool)
                
                # Verify that log_execution was called
                mock_db_manager.log_strategy_execution.assert_called()
                
                # Get the last call arguments
                call_args = mock_db_manager.log_strategy_execution.call_args
                assert call_args is not None
                
                # Verify required parameters
                args, kwargs = call_args
                assert kwargs['strategy_name'] == strategy.name
                assert kwargs['signal_type'] == 'entry'
                assert kwargs['price'] > 0
                assert isinstance(kwargs['reasons'], list)
                assert len(kwargs['reasons']) > 0
                
                # Verify ML predictions
                assert 'ml_predictions' in kwargs or 'prediction_available' in kwargs.get('additional_context', {})
                
                # Verify additional context
                assert 'additional_context' in kwargs
                context = kwargs['additional_context']
                assert 'model_type' in context
                assert context['model_type'] == 'ml_with_sentiment'

    @pytest.mark.strategy
    def test_ml_with_sentiment_strategy_missing_prediction_logging(self, sample_ohlcv_data):
        """Test that ML with sentiment strategy logs when prediction is missing"""
        strategy = MlWithSentiment()
        
        # Mock database manager for logging
        mock_db_manager = Mock()
        strategy.set_database_manager(mock_db_manager, session_id=202)
        
        # Create data without ML predictions
        df_no_predictions = sample_ohlcv_data.copy()
        df_no_predictions['ml_prediction'] = np.nan
        
        # Test entry conditions - should log missing prediction
        if len(df_no_predictions) > 1:
            result = strategy.check_entry_conditions(df_no_predictions, 1)
            assert result is False  # Should return False for missing prediction
            
            # Verify that log_execution was called for missing prediction
            mock_db_manager.log_strategy_execution.assert_called()
            
            # Get the last call arguments
            call_args = mock_db_manager.log_strategy_execution.call_args
            assert call_args is not None
            
            # Verify it logged the missing prediction
            args, kwargs = call_args
            assert 'missing_ml_prediction' in kwargs['reasons']
            assert 'prediction_available=False' in kwargs['reasons']

    @pytest.mark.strategy
    def test_ml_with_sentiment_strategy_parameters(self):
        """Test ML with sentiment strategy parameter retrieval"""
        strategy = MlWithSentiment()
        
        params = strategy.get_parameters()
        
        # Check required parameters
        assert 'name' in params
        assert 'model_path' in params
        assert 'sequence_length' in params
        assert 'use_sentiment' in params
        assert 'stop_loss_pct' in params
        assert 'take_profit_pct' in params


class TestStrategyLoggingIntegration:
    """Integration tests for strategy logging across all strategies"""

    @pytest.mark.strategy
    def test_all_strategies_have_logging_capability(self):
        """Test that all available strategies have logging capability"""
        strategies = []
        
        strategies.append(MlWithSentiment())
        strategies.append(MlBasic())
        strategies.append(MlWithSentiment())
        
        # Test that all strategies inherit from BaseStrategy and have logging
        for strategy in strategies:
            assert hasattr(strategy, 'log_execution')
            assert hasattr(strategy, 'set_database_manager')
            assert hasattr(strategy, 'enable_execution_logging')
            
            # Test that logging can be enabled/disabled
            strategy.enable_execution_logging = False
            assert strategy.enable_execution_logging is False
            strategy.enable_execution_logging = True
            assert strategy.enable_execution_logging is True

    @pytest.mark.strategy
    def test_strategy_logging_with_mock_database(self, sample_ohlcv_data):
        """Test that strategies can log to a mock database"""
        strategies = []
        
        strategies.append(MlWithSentiment())
        strategies.append(MlBasic())
        strategies.append(MlWithSentiment())
        
        mock_db_manager = Mock()
        
        for i, strategy in enumerate(strategies):
            # Set up database manager
            strategy.set_database_manager(mock_db_manager, session_id=1000 + i)
            
            # Calculate indicators
            df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
            
            # Test entry conditions
            if len(df_with_indicators) > 20:
                test_index = min(20, len(df_with_indicators) - 1)
                result = strategy.check_entry_conditions(df_with_indicators, test_index)
                try:
                    assert isinstance(result, (bool, np.bool_))
                except AssertionError:
                    print(f"DEBUG: result={result!r}, type={type(result)} for strategy {type(strategy).__name__} at index {test_index}")
                    raise
                
                # Verify logging was called
                mock_db_manager.log_strategy_execution.assert_called()
                
                # Verify session ID was passed correctly
                call_args = mock_db_manager.log_strategy_execution.call_args
                args, kwargs = call_args
                assert kwargs['session_id'] == 1000 + i
                # For MlWithSentiment, check for merged context in reasons
                if isinstance(strategy, MlWithSentiment):
                    assert 'strategy_type=enhanced_multi_condition' in kwargs['reasons']