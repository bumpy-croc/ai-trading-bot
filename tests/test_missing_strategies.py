"""
Tests for missing strategies: enhanced, ml_basic, ml_with_sentiment.

These tests focus on strategy execution logging with log_execution method.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Import strategies conditionally to avoid test failures if not available
try:
    from strategies.enhanced import EnhancedStrategy
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

try:
    from strategies.ml_basic import MlBasic
    ML_BASIC_AVAILABLE = True
except ImportError:
    ML_BASIC_AVAILABLE = False

try:
    from strategies.ml_with_sentiment import MlWithSentiment
    ML_WITH_SENTIMENT_AVAILABLE = True
except ImportError:
    ML_WITH_SENTIMENT_AVAILABLE = False


class TestEnhancedStrategy:
    """Test the enhanced strategy implementation and logging"""

    @pytest.mark.skipif(not ENHANCED_AVAILABLE, reason="EnhancedStrategy not available")
    @pytest.mark.strategy
    def test_enhanced_strategy_initialization(self):
        """Test enhanced strategy initialization"""
        strategy = EnhancedStrategy()
        
        assert hasattr(strategy, 'name')
        assert getattr(strategy, 'trading_pair', 'BTCUSDT') is not None
        assert hasattr(strategy, 'risk_per_trade')
        assert hasattr(strategy, 'max_position_size')

    @pytest.mark.skipif(not ENHANCED_AVAILABLE, reason="EnhancedStrategy not available")
    @pytest.mark.strategy
    def test_enhanced_strategy_execution_logging(self, sample_ohlcv_data):
        """Test that enhanced strategy logs execution details"""
        strategy = EnhancedStrategy()
        
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
                
                # Verify additional context
                assert 'additional_context' in kwargs
                context = kwargs['additional_context']
                assert 'strategy_type' in context
                assert context['strategy_type'] == 'enhanced_multi_condition'

    @pytest.mark.skipif(not ENHANCED_AVAILABLE, reason="EnhancedStrategy not available")
    @pytest.mark.strategy
    def test_enhanced_strategy_parameters(self):
        """Test enhanced strategy parameter retrieval"""
        strategy = EnhancedStrategy()
        
        params = strategy.get_parameters()
        
        # Check required parameters
        assert 'name' in params
        assert 'risk_per_trade' in params
        assert 'max_position_size' in params
        assert 'base_stop_loss_pct' in params
        assert 'min_conditions' in params


class TestMlBasicStrategy:
    """Test the ML basic strategy implementation and logging"""

    @pytest.mark.skipif(not ML_BASIC_AVAILABLE, reason="MlBasic not available")
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

    @pytest.mark.skipif(not ML_BASIC_AVAILABLE, reason="MlBasic not available")
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

    @pytest.mark.skipif(not ML_BASIC_AVAILABLE, reason="MlBasic not available")
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
            assert kwargs['additional_context']['prediction_available'] is False

    @pytest.mark.skipif(not ML_BASIC_AVAILABLE, reason="MlBasic not available")
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


class TestMlWithSentimentStrategy:
    """Test the ML with sentiment strategy implementation and logging"""

    @pytest.mark.skipif(not ML_WITH_SENTIMENT_AVAILABLE, reason="MlWithSentiment not available")
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

    @pytest.mark.skipif(not ML_WITH_SENTIMENT_AVAILABLE, reason="MlWithSentiment not available")
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

    @pytest.mark.skipif(not ML_WITH_SENTIMENT_AVAILABLE, reason="MlWithSentiment not available")
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
            assert kwargs['additional_context']['prediction_available'] is False

    @pytest.mark.skipif(not ML_WITH_SENTIMENT_AVAILABLE, reason="MlWithSentiment not available")
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
        
        if ENHANCED_AVAILABLE:
            strategies.append(EnhancedStrategy())
        if ML_BASIC_AVAILABLE:
            strategies.append(MlBasic())
        if ML_WITH_SENTIMENT_AVAILABLE:
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
        
        if ENHANCED_AVAILABLE:
            strategies.append(EnhancedStrategy())
        if ML_BASIC_AVAILABLE:
            strategies.append(MlBasic())
        if ML_WITH_SENTIMENT_AVAILABLE:
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
                assert isinstance(result, bool)
                
                # Verify logging was called
                mock_db_manager.log_strategy_execution.assert_called()
                
                # Verify session ID was passed correctly
                call_args = mock_db_manager.log_strategy_execution.call_args
                args, kwargs = call_args
                assert kwargs['session_id'] == 1000 + i