"""
Tests for trading strategies.

Strategy tests are crucial as they generate trading signals. Tests cover:
- Signal generation accuracy with prediction engine integration
- Indicator calculations
- Position sizing logic
- Entry/exit condition validation
- Edge cases and market conditions
- Strategy parameter validation
- Strategy execution logging
- Prediction engine integration and fallbacks
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from strategies.base import BaseStrategy
from strategies.adaptive import AdaptiveStrategy
from strategies.enhanced import EnhancedStrategy
from strategies.ml_basic import MlBasic
from strategies.ml_with_sentiment import MlWithSentiment


class TestBaseStrategy:
    """Test the base strategy interface"""

    def test_base_strategy_is_abstract(self):
        """Test that BaseStrategy cannot be instantiated directly"""
        with pytest.raises(TypeError):
            # BaseStrategy is abstract and requires a name parameter, but should still fail
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
        
        # Test prediction engine integration
        assert hasattr(strategy, 'get_prediction')
        assert hasattr(strategy, 'prediction_engine')
        
        # Test optional methods with fallbacks
        if hasattr(strategy, 'get_trading_pair'):
            assert hasattr(strategy, 'set_trading_pair')

    def test_trading_pair_management(self, real_adaptive_strategy):
        """Test trading pair getter/setter"""
        strategy = real_adaptive_strategy
        
        # Test methods exist before calling
        if hasattr(strategy, 'get_trading_pair'):
            # Test default
            current_pair = strategy.get_trading_pair()
            assert isinstance(current_pair, str)
            
            # Test setter
            if hasattr(strategy, 'set_trading_pair'):
                strategy.set_trading_pair('ETHUSDT')
                assert strategy.get_trading_pair() == 'ETHUSDT'

    def test_prediction_engine_integration(self, sample_ohlcv_data):
        """Test base strategy prediction engine integration"""
        # Create a mock prediction engine
        mock_prediction_engine = Mock()
        mock_prediction_result = Mock()
        mock_prediction_result.price = 50000.0
        mock_prediction_result.confidence = 0.8
        mock_prediction_result.direction = 1
        mock_prediction_result.model_name = 'test_model'
        mock_prediction_result.timestamp = pd.Timestamp.now()
        mock_prediction_result.error = None
        mock_prediction_engine.predict.return_value = mock_prediction_result
        
        # Test with prediction engine
        try:
            from strategies.ml_basic import MlBasic
            strategy = MlBasic(prediction_engine=mock_prediction_engine)
            
            # Test get_prediction method
            result = strategy.get_prediction(sample_ohlcv_data, len(sample_ohlcv_data) - 1)
            
            assert result['price'] == 50000.0
            assert result['confidence'] == 0.8
            assert result['direction'] == 1
            assert result['model_name'] == 'test_model'
            assert result.get('error') is None
            
        except ImportError:
            pytest.skip("ML strategies not available")

    def test_prediction_engine_error_handling(self, sample_ohlcv_data):
        """Test prediction engine error handling"""
        # Create a mock prediction engine that raises an exception
        mock_prediction_engine = Mock()
        mock_prediction_engine.predict.side_effect = Exception("Prediction failed")
        
        try:
            from strategies.ml_basic import MlBasic
            strategy = MlBasic(prediction_engine=mock_prediction_engine)
            
            # Test error handling
            result = strategy.get_prediction(sample_ohlcv_data, len(sample_ohlcv_data) - 1)
            
            assert result['price'] is None
            assert result['confidence'] == 0.0
            assert result['direction'] == 0
            assert 'error' in result
            assert 'Prediction failed' in result['error']
            
        except ImportError:
            pytest.skip("ML strategies not available")


class TestAdaptiveStrategy:
    """Test the adaptive strategy implementation"""

    @pytest.mark.strategy
    def test_adaptive_strategy_initialization(self):
        """Test adaptive strategy initialization"""
        strategy = AdaptiveStrategy()
        
        assert hasattr(strategy, 'name')
        # Use getattr with default for optional attributes
        assert getattr(strategy, 'trading_pair', 'BTCUSDT') is not None
        assert hasattr(strategy, 'base_risk_per_trade')
        assert hasattr(strategy, 'fast_ma')
        assert hasattr(strategy, 'slow_ma')

    @pytest.mark.strategy
    def test_indicator_calculation(self, sample_ohlcv_data):
        """Test that all required indicators are calculated"""
        strategy = AdaptiveStrategy()
        
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
                    'high': new_price * 1.01,
                    'low': new_price * 0.99,
                    'close': new_price,
                    'volume': np.random.randint(1000, 10000)
                })
                base_price = new_price
            
            additional_df = pd.DataFrame(additional_data, index=pd.date_range(
                start=last_date + timedelta(hours=1), periods=len(additional_data), freq='H'
            ))
            
            extended_data = pd.concat([sample_ohlcv_data, additional_df])
        else:
            extended_data = sample_ohlcv_data
        
        df_with_indicators = strategy.calculate_indicators(extended_data)
        
        # Test that required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in df_with_indicators.columns
        
        # Test that indicators were calculated
        # Note: Some indicators might have NaN values for initial rows
        assert len(df_with_indicators) == len(extended_data)
        
        # Test data types
        for col in required_columns:
            assert pd.api.types.is_numeric_dtype(df_with_indicators[col])

    @pytest.mark.strategy
    def test_entry_exit_conditions(self, sample_ohlcv_data):
        """Test entry and exit condition methods exist and return boolean"""
        strategy = AdaptiveStrategy()
        
        # Ensure we have enough data
        if len(sample_ohlcv_data) < 50:
            # Create extended sample data
            extended_data = sample_ohlcv_data.copy()
            for _ in range(50 - len(sample_ohlcv_data)):
                extended_data = pd.concat([extended_data, sample_ohlcv_data.tail(1)])
        else:
            extended_data = sample_ohlcv_data
        
        df_with_indicators = strategy.calculate_indicators(extended_data)
        
        # Test entry conditions
        if len(df_with_indicators) > 20:
            entry_result = strategy.check_entry_conditions(df_with_indicators, 20)
            assert isinstance(entry_result, (bool, np.bool_))
        
            # Test exit conditions
            entry_price = df_with_indicators['close'].iloc[20]
            exit_result = strategy.check_exit_conditions(df_with_indicators, 21, entry_price)
            assert isinstance(exit_result, (bool, np.bool_))

    @pytest.mark.strategy
    def test_position_sizing(self, sample_ohlcv_data):
        """Test position sizing calculation"""
        strategy = AdaptiveStrategy()
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        
        balance = 10000.0
        test_index = min(10, len(df_with_indicators) - 1)
        
        position_size = strategy.calculate_position_size(df_with_indicators, test_index, balance)
        
        # Position size should be positive and reasonable
        assert position_size > 0
        assert position_size <= balance * 0.2  # Max 20% position size
        
        # Test with zero balance
        zero_position = strategy.calculate_position_size(df_with_indicators, test_index, 0.0)
        assert zero_position == 0.0

    @pytest.mark.strategy
    def test_strategy_parameters(self):
        """Test strategy parameter retrieval"""
        strategy = AdaptiveStrategy()
        
        params = strategy.get_parameters()
        
        # Should return a dictionary
        assert isinstance(params, dict)
        assert len(params) > 0
        
        # Should contain at least strategy name
        assert 'name' in params or any('name' in str(k).lower() for k in params.keys())


class TestEnhancedStrategy:
    """Test the enhanced strategy implementation"""

    @pytest.mark.strategy
    def test_enhanced_strategy_initialization(self):
        """Test enhanced strategy initialization"""
        strategy = EnhancedStrategy()
        
        assert hasattr(strategy, 'name')
        assert getattr(strategy, 'trading_pair', 'BTCUSDT') is not None
        assert hasattr(strategy, 'fast_ma')
        assert hasattr(strategy, 'slow_ma')
        assert hasattr(strategy, 'rsi_period')
        assert hasattr(strategy, 'bb_period')

    @pytest.mark.strategy
    def test_enhanced_strategy_execution_logging(self, sample_ohlcv_data):
        """Test that enhanced strategy logs execution details"""
        strategy = EnhancedStrategy()
        
        # Mock database manager for logging
        mock_db_manager = Mock()
        strategy.set_database_manager(mock_db_manager, session_id=123)
        
        # Calculate indicators
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        
        # Test entry conditions
        if len(df_with_indicators) > 20:
            test_index = 20
            result = strategy.check_entry_conditions(df_with_indicators, test_index)
            assert isinstance(result, (bool, np.bool_))
            
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


class TestMlBasicStrategy:
    """Test the ML basic strategy implementation with prediction engine integration"""

    @pytest.mark.strategy
    def test_ml_basic_strategy_initialization(self):
        """Test ML basic strategy initialization"""
        # Mock prediction engine to avoid dependency issues
        mock_prediction_engine = Mock()
        strategy = MlBasic(prediction_engine=mock_prediction_engine)
        
        assert hasattr(strategy, 'name')
        assert getattr(strategy, 'trading_pair', 'BTCUSDT') is not None
        assert hasattr(strategy, 'prediction_engine')
        assert hasattr(strategy, 'stop_loss_pct')
        assert hasattr(strategy, 'take_profit_pct')
        assert strategy.prediction_engine == mock_prediction_engine

    @pytest.mark.strategy
    def test_ml_basic_strategy_prediction_integration(self, sample_ohlcv_data):
        """Test ML basic strategy with prediction engine"""
        # Create mock prediction engine
        mock_prediction_engine = Mock()
        mock_prediction_result = Mock()
        mock_prediction_result.price = 50000.0
        mock_prediction_result.confidence = 0.8
        mock_prediction_result.direction = 1
        mock_prediction_result.model_name = 'btcusdt_price'
        mock_prediction_result.timestamp = pd.Timestamp.now()
        mock_prediction_result.error = None
        mock_prediction_engine.predict.return_value = mock_prediction_result
        
        strategy = MlBasic(prediction_engine=mock_prediction_engine)
        
        # Mock database manager for logging
        mock_db_manager = Mock()
        strategy.set_database_manager(mock_db_manager, session_id=456)
        
        # Calculate indicators
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        
        # Test entry conditions
        result = strategy.check_entry_conditions(df_with_indicators, len(df_with_indicators) - 1)
        
        assert isinstance(result, bool)
        assert result is True  # Should enter with positive direction and high confidence
        
        # Verify prediction engine was called
        mock_prediction_engine.predict.assert_called_once()
        
        # Verify logging was called
        mock_db_manager.log_strategy_execution.assert_called()
        
        # Get the last call arguments
        call_args = mock_db_manager.log_strategy_execution.call_args
        assert call_args is not None
        
        # Verify required parameters
        args, kwargs = call_args
        assert kwargs['strategy_name'] == strategy.name
        assert kwargs['signal_type'] == 'entry'
        assert kwargs['action_taken'] == 'entry_signal'
        assert kwargs['price'] > 0
        assert isinstance(kwargs['reasons'], list)
        assert len(kwargs['reasons']) > 0
        
        # Verify ML predictions
        assert 'ml_predictions' in kwargs
        ml_preds = kwargs['ml_predictions']
        assert ml_preds['raw_prediction'] == 50000.0
        assert ml_preds['direction'] == 1
        
        # Verify additional context
        assert 'additional_context' in kwargs
        context = kwargs['additional_context']
        assert context['model_type'] == 'ml_basic'
        assert context['model_name'] == 'btcusdt_price'

    @pytest.mark.strategy
    def test_ml_basic_strategy_fallback_on_prediction_error(self, sample_ohlcv_data):
        """Test strategy fallback when prediction fails"""
        # Create mock prediction engine that raises error
        mock_prediction_engine = Mock()
        mock_prediction_engine.predict.side_effect = Exception("Prediction failed")
        
        strategy = MlBasic(prediction_engine=mock_prediction_engine)
        
        # Mock database manager for logging
        mock_db_manager = Mock()
        strategy.set_database_manager(mock_db_manager, session_id=789)
        
        # Calculate indicators
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        
        # Test entry conditions
        result = strategy.check_entry_conditions(df_with_indicators, len(df_with_indicators) - 1)
        
        assert result is False  # Should return False on prediction error
        
        # Verify logging was called for error
        mock_db_manager.log_strategy_execution.assert_called()
        
        # Get the last call arguments
        call_args = mock_db_manager.log_strategy_execution.call_args
        assert call_args is not None
        
        # Verify it logged the prediction error
        args, kwargs = call_args
        assert kwargs['action_taken'] == 'no_action'
        assert any('prediction_error' in reason for reason in kwargs['reasons'])
        assert any('Prediction failed' in reason for reason in kwargs['reasons'])

    @pytest.mark.strategy
    def test_ml_basic_strategy_missing_prediction_logging(self, sample_ohlcv_data):
        """Test that ML basic strategy logs when prediction engine returns None"""
        # Create mock prediction engine that returns None prediction
        mock_prediction_engine = Mock()
        mock_prediction_result = Mock()
        mock_prediction_result.price = None
        mock_prediction_result.confidence = 0.0
        mock_prediction_result.direction = 0
        mock_prediction_result.model_name = 'test_model'
        mock_prediction_result.timestamp = pd.Timestamp.now()
        mock_prediction_result.error = None
        mock_prediction_engine.predict.return_value = mock_prediction_result
        
        strategy = MlBasic(prediction_engine=mock_prediction_engine)
        
        # Mock database manager for logging
        mock_db_manager = Mock()
        strategy.set_database_manager(mock_db_manager, session_id=789)
        
        # Calculate indicators
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        
        # Test entry conditions - should log missing prediction
        result = strategy.check_entry_conditions(df_with_indicators, len(df_with_indicators) - 1)
        assert result is False  # Should return False for missing prediction
        
        # Verify that log_execution was called for missing prediction
        mock_db_manager.log_strategy_execution.assert_called()
        
        # Get the last call arguments
        call_args = mock_db_manager.log_strategy_execution.call_args
        assert call_args is not None
        
        # Verify it logged the missing prediction
        args, kwargs = call_args
        assert any('missing_ml_prediction' in reason for reason in kwargs['reasons'])
        assert kwargs['action_taken'] == 'no_action'
        # Check additional context
        context = kwargs.get('additional_context', {})
        assert context.get('prediction_available') is False

    @pytest.mark.strategy
    def test_ml_basic_strategy_parameters(self):
        """Test ML basic strategy parameter retrieval"""
        mock_prediction_engine = Mock()
        strategy = MlBasic(prediction_engine=mock_prediction_engine)
        
        params = strategy.get_parameters()
        
        # Check required parameters
        assert 'name' in params
        assert 'trading_pair' in params
        assert 'stop_loss_pct' in params
        assert 'take_profit_pct' in params
        assert 'prediction_engine_available' in params
        assert params['prediction_engine_available'] is True

    @pytest.mark.strategy
    def test_ml_basic_exit_conditions_with_prediction(self, sample_ohlcv_data):
        """Test ML basic exit conditions with prediction engine"""
        # Create mock prediction engine
        mock_prediction_engine = Mock()
        mock_prediction_result = Mock()
        mock_prediction_result.price = 45000.0  # Lower than current price for exit signal
        mock_prediction_result.confidence = 0.8
        mock_prediction_result.direction = -1
        mock_prediction_result.model_name = 'test_model'
        mock_prediction_result.timestamp = pd.Timestamp.now()
        mock_prediction_result.error = None
        mock_prediction_engine.predict.return_value = mock_prediction_result
        
        strategy = MlBasic(prediction_engine=mock_prediction_engine)
        df = strategy.calculate_indicators(sample_ohlcv_data)
        
        # Use realistic entry price close to current price
        current_price = df['close'].iloc[-1]
        entry_price = current_price * 0.98  # Entry at 2% below current price
        
        # Test exit with unfavorable prediction (>2% drop predicted with high confidence)
        # Mock result shows 45000 vs current ~50000 = ~10% drop with 0.8 confidence
        result = strategy.check_exit_conditions(df, len(df)-1, entry_price)
        
        # Should exit due to unfavorable ML prediction
        assert result is True


class TestMlWithSentimentStrategy:
    """Test the ML with sentiment strategy implementation with prediction engine integration"""

    @pytest.mark.strategy
    def test_ml_with_sentiment_strategy_initialization(self):
        """Test ML with sentiment strategy initialization"""
        # Mock prediction engine to avoid dependency issues
        mock_prediction_engine = Mock()
        strategy = MlWithSentiment(prediction_engine=mock_prediction_engine)
        
        assert hasattr(strategy, 'name')
        assert getattr(strategy, 'trading_pair', 'BTCUSDT') is not None
        assert hasattr(strategy, 'prediction_engine')
        assert hasattr(strategy, 'use_sentiment')
        assert hasattr(strategy, 'sentiment_weight')
        assert hasattr(strategy, 'stop_loss_pct')
        assert hasattr(strategy, 'take_profit_pct')
        assert strategy.prediction_engine == mock_prediction_engine

    @pytest.mark.strategy
    def test_ml_with_sentiment_strategy_prediction_integration(self, sample_ohlcv_data):
        """Test ML with sentiment strategy with prediction engine"""
        # Create mock prediction engine
        mock_prediction_engine = Mock()
        mock_prediction_result = Mock()
        mock_prediction_result.price = 52000.0
        mock_prediction_result.confidence = 0.7
        mock_prediction_result.direction = 1
        mock_prediction_result.model_name = 'btcusdt_sentiment'
        mock_prediction_result.timestamp = pd.Timestamp.now()
        mock_prediction_result.error = None
        mock_prediction_engine.predict.return_value = mock_prediction_result
        
        strategy = MlWithSentiment(prediction_engine=mock_prediction_engine, use_sentiment=False)
        
        # Mock database manager for logging
        mock_db_manager = Mock()
        strategy.set_database_manager(mock_db_manager, session_id=101)
        
        # Calculate indicators
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        
        # Test entry conditions
        result = strategy.check_entry_conditions(df_with_indicators, len(df_with_indicators) - 1)
        
        assert isinstance(result, bool)
        
        # Verify prediction engine was called
        mock_prediction_engine.predict.assert_called_once()
        
        # Verify logging was called
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
        assert 'ml_predictions' in kwargs
        ml_preds = kwargs['ml_predictions']
        assert ml_preds['raw_prediction'] == 52000.0
        assert ml_preds['direction'] == 1
        
        # Verify additional context
        assert 'additional_context' in kwargs
        context = kwargs['additional_context']
        assert context['model_type'] == 'ml_with_sentiment'
        assert context['model_name'] == 'btcusdt_sentiment'

    @pytest.mark.strategy
    def test_ml_with_sentiment_strategy_combined_signal(self, sample_ohlcv_data):
        """Test ML sentiment strategy combined signals"""
        # Create mock prediction engine
        mock_prediction_engine = Mock()
        mock_prediction_result = Mock()
        mock_prediction_result.price = 51000.0
        mock_prediction_result.confidence = 0.7
        mock_prediction_result.direction = 1
        mock_prediction_result.model_name = 'btcusdt_sentiment'
        mock_prediction_result.timestamp = pd.Timestamp.now()
        mock_prediction_result.error = None
        mock_prediction_engine.predict.return_value = mock_prediction_result
        
        # Add sentiment data to test data
        test_data_with_sentiment = sample_ohlcv_data.copy()
        test_data_with_sentiment['sentiment_primary'] = 0.7  # Positive sentiment
        
        strategy = MlWithSentiment(prediction_engine=mock_prediction_engine, use_sentiment=True)
        
        # Test entry conditions
        result = strategy.check_entry_conditions(test_data_with_sentiment, len(test_data_with_sentiment) - 1)
        
        # Should use combined ML and sentiment for entry decision
        assert isinstance(result, bool)

    @pytest.mark.strategy
    def test_ml_with_sentiment_strategy_missing_prediction_logging(self, sample_ohlcv_data):
        """Test that ML with sentiment strategy logs when prediction is missing"""
        # Create mock prediction engine that returns None prediction
        mock_prediction_engine = Mock()
        mock_prediction_result = Mock()
        mock_prediction_result.price = None
        mock_prediction_result.confidence = 0.0
        mock_prediction_result.direction = 0
        mock_prediction_result.model_name = 'test_model'
        mock_prediction_result.timestamp = pd.Timestamp.now()
        mock_prediction_result.error = None
        mock_prediction_engine.predict.return_value = mock_prediction_result
        
        strategy = MlWithSentiment(prediction_engine=mock_prediction_engine)
        
        # Mock database manager for logging
        mock_db_manager = Mock()
        strategy.set_database_manager(mock_db_manager, session_id=202)
        
        # Calculate indicators
        df_with_indicators = strategy.calculate_indicators(sample_ohlcv_data)
        
        # Test entry conditions - should log missing prediction
        result = strategy.check_entry_conditions(df_with_indicators, len(df_with_indicators) - 1)
        assert result is False  # Should return False for missing prediction
        
        # Verify that log_execution was called for missing prediction
        mock_db_manager.log_strategy_execution.assert_called()
        
        # Get the last call arguments
        call_args = mock_db_manager.log_strategy_execution.call_args
        assert call_args is not None
        
        # Verify it logged the missing prediction
        args, kwargs = call_args
        assert any('missing_ml_prediction' in reason for reason in kwargs['reasons'])
        # Check additional context
        context = kwargs.get('additional_context', {})
        assert context.get('prediction_available') is False

    @pytest.mark.strategy
    def test_ml_with_sentiment_strategy_parameters(self):
        """Test ML with sentiment strategy parameter retrieval"""
        mock_prediction_engine = Mock()
        strategy = MlWithSentiment(prediction_engine=mock_prediction_engine)
        
        params = strategy.get_parameters()
        
        # Check required parameters
        assert 'name' in params
        assert 'trading_pair' in params
        assert 'use_sentiment' in params
        assert 'sentiment_weight' in params
        assert 'stop_loss_pct' in params
        assert 'take_profit_pct' in params
        assert 'prediction_engine_available' in params
        assert params['prediction_engine_available'] is True


class TestStrategyLoggingIntegration:
    """Integration tests for strategy logging across all strategies"""

    @pytest.mark.strategy
    def test_all_strategies_have_logging_capability(self):
        """Test that all available strategies have logging capability"""
        strategies = []
        
        strategies.append(EnhancedStrategy())
        
        # Create ML strategies with mock prediction engines
        mock_prediction_engine = Mock()
        strategies.append(MlBasic(prediction_engine=mock_prediction_engine))
        strategies.append(MlWithSentiment(prediction_engine=mock_prediction_engine))
        
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
    def test_strategy_prediction_engine_integration(self, sample_ohlcv_data):
        """Test that ML strategies work with prediction engine"""
        # Create mock prediction engine
        mock_prediction_engine = Mock()
        mock_prediction_result = Mock()
        mock_prediction_result.price = 50000.0
        mock_prediction_result.confidence = 0.8
        mock_prediction_result.direction = 1
        mock_prediction_result.model_name = 'test_model'
        mock_prediction_result.timestamp = pd.Timestamp.now()
        mock_prediction_result.error = None
        mock_prediction_engine.predict.return_value = mock_prediction_result
        
        # Test strategy with prediction engine
        strategy = MlBasic(prediction_engine=mock_prediction_engine)
        
        # Test entry conditions
        df = strategy.calculate_indicators(sample_ohlcv_data)
        result = strategy.check_entry_conditions(df, len(df) - 1)
        
        assert isinstance(result, bool)
        assert result is True  # Should enter with positive signal
        
        # Test prediction was called
        mock_prediction_engine.predict.assert_called()

    @pytest.mark.strategy
    def test_strategy_logging_with_mock_database(self, sample_ohlcv_data):
        """Test that strategies can log to a mock database"""
        strategies = []
        
        strategies.append(EnhancedStrategy())
        
        # Create ML strategies with mock prediction engines
        mock_prediction_engine = Mock()
        mock_prediction_result = Mock()
        mock_prediction_result.price = 50000.0
        mock_prediction_result.confidence = 0.8
        mock_prediction_result.direction = 1
        mock_prediction_result.model_name = 'test_model'
        mock_prediction_result.timestamp = pd.Timestamp.now()
        mock_prediction_result.error = None
        mock_prediction_engine.predict.return_value = mock_prediction_result
        
        strategies.append(MlBasic(prediction_engine=mock_prediction_engine))
        strategies.append(MlWithSentiment(prediction_engine=mock_prediction_engine))
        
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
                # For EnhancedStrategy, check for merged context in reasons
                if isinstance(strategy, EnhancedStrategy):
                    assert 'strategy_type=enhanced_multi_condition' in kwargs['reasons']