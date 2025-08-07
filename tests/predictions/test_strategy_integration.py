"""
Prediction-Strategy Integration Tests

This module tests the integration between the refactored strategies and the prediction engine.
It verifies that strategies correctly use the prediction engine for ML predictions instead of
embedded logic.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from datetime import datetime, timezone

from src.strategies.ml_basic import MlBasic
from src.strategies.ml_adaptive import MlAdaptive
from src.strategies.ml_with_sentiment import MlWithSentiment


class TestStrategyPredictionIntegration:
    """Test suite for strategy-prediction engine integration"""

    def setup_method(self):
        """Set up test fixtures"""
        # Create sample market data
        dates = pd.date_range('2023-01-01', periods=150, freq='1H')
        self.test_data = pd.DataFrame({
            'open': np.random.uniform(45000, 55000, 150),
            'high': np.random.uniform(46000, 56000, 150),
            'low': np.random.uniform(44000, 54000, 150),
            'close': np.random.uniform(45000, 55000, 150),
            'volume': np.random.uniform(100, 1000, 150)
        }, index=dates)
        
        # Ensure price consistency (high >= close >= low, etc.)
        for i in range(len(self.test_data)):
            row = self.test_data.iloc[i]
            low = min(row['open'], row['high'], row['low'], row['close'])
            high = max(row['open'], row['high'], row['low'], row['close'])
            self.test_data.iloc[i, self.test_data.columns.get_loc('low')] = low
            self.test_data.iloc[i, self.test_data.columns.get_loc('high')] = high

    def _create_mock_prediction_engine(self, prediction_result=None):
        """Create a mock prediction engine with configurable results"""
        mock_engine = Mock()
        
        # Default prediction result
        if prediction_result is None:
            prediction_result = Mock(
                price=50000.0,
                confidence=0.8,
                direction=1,
                model_name='test_model',
                timestamp=datetime.now(timezone.utc),
                inference_time=0.1,
                features_used=5,
                cache_hit=False,
                error=None
            )
        
        mock_engine.predict.return_value = prediction_result
        return mock_engine

    def test_ml_basic_strategy_prediction_integration(self):
        """Test ML basic strategy with prediction engine"""
        # Create mock prediction engine with bullish prediction
        mock_engine = self._create_mock_prediction_engine()
        
        strategy = MlBasic(prediction_engine=mock_engine)
        
        # Calculate indicators (should not generate predictions anymore)
        df_with_indicators = strategy.calculate_indicators(self.test_data)
        
        # Test entry conditions
        result = strategy.check_entry_conditions(df_with_indicators, 140)
        
        # Verify prediction engine was called
        mock_engine.predict.assert_called_once()
        
        # Should enter since mock prediction is bullish with high confidence
        assert result is True
        
        # Test position sizing
        position_size = strategy.calculate_position_size(df_with_indicators, 140, 10000.0)
        assert position_size > 0
        
        # Test exit conditions
        exit_result = strategy.check_exit_conditions(df_with_indicators, 140, 49000.0)
        # Should not exit immediately with favorable prediction
        assert exit_result is False

    def test_ml_basic_strategy_prediction_error_handling(self):
        """Test ML basic strategy fallback when prediction fails"""
        # Create mock prediction engine that returns error
        prediction_result = Mock(
            price=None,
            confidence=0.0,
            direction=0,
            model_name='test_model',
            timestamp=datetime.now(timezone.utc),
            inference_time=0.0,
            features_used=0,
            cache_hit=False,
            error='Prediction failed'
        )
        
        mock_engine = self._create_mock_prediction_engine(prediction_result)
        strategy = MlBasic(prediction_engine=mock_engine)
        
        df_with_indicators = strategy.calculate_indicators(self.test_data)
        
        # Test entry conditions - should return False on error
        result = strategy.check_entry_conditions(df_with_indicators, 140)
        assert result is False
        
        # Test position sizing - should return 0 on error
        position_size = strategy.calculate_position_size(df_with_indicators, 140, 10000.0)
        assert position_size == 0.0

    def test_ml_adaptive_strategy_prediction_integration(self):
        """Test ML adaptive strategy with prediction engine"""
        mock_engine = self._create_mock_prediction_engine()
        
        strategy = MlAdaptive(prediction_engine=mock_engine)
        
        # Calculate indicators (includes market regime detection)
        df_with_indicators = strategy.calculate_indicators(self.test_data)
        
        # Verify market regime detection still works
        assert 'market_regime' in df_with_indicators.columns
        assert 'volatility_20' in df_with_indicators.columns
        
        # Test entry conditions
        result = strategy.check_entry_conditions(df_with_indicators, 140)
        
        # Verify prediction engine was called
        mock_engine.predict.assert_called_once()
        
        # Should consider entry based on market regime and prediction
        assert isinstance(result, bool)
        
        # Test position sizing with market conditions
        position_size = strategy.calculate_position_size(df_with_indicators, 140, 10000.0)
        assert position_size >= 0  # Should be non-negative

    def test_ml_adaptive_strategy_market_regime_integration(self):
        """Test that ML adaptive strategy properly uses both prediction and market regime"""
        # Create bearish prediction
        bearish_prediction = Mock(
            price=48000.0,  # Lower than typical current price
            confidence=0.9,
            direction=-1,
            model_name='test_model',
            timestamp=datetime.now(timezone.utc),
            inference_time=0.1,
            features_used=5,
            cache_hit=False,
            error=None
        )
        
        mock_engine = self._create_mock_prediction_engine(bearish_prediction)
        strategy = MlAdaptive(prediction_engine=mock_engine)
        
        df_with_indicators = strategy.calculate_indicators(self.test_data)
        
        # Test short entry conditions (should work with bearish prediction)
        short_result = strategy.check_short_entry_conditions(df_with_indicators, 140)
        
        # Verify prediction engine was called
        mock_engine.predict.assert_called()
        
        # Result depends on market regime, but should handle the bearish prediction
        assert isinstance(short_result, bool)

    def test_ml_sentiment_strategy_prediction_integration(self):
        """Test ML sentiment strategy with prediction engine"""
        mock_engine = self._create_mock_prediction_engine()
        
        strategy = MlWithSentiment(prediction_engine=mock_engine, use_sentiment=False)
        
        # Calculate indicators
        df_with_indicators = strategy.calculate_indicators(self.test_data)
        
        # Test entry conditions
        result = strategy.check_entry_conditions(df_with_indicators, 140)
        
        # Verify prediction engine was called
        mock_engine.predict.assert_called_once()
        
        # Should enter with bullish prediction
        assert result is True
        
        # Test position sizing
        position_size = strategy.calculate_position_size(df_with_indicators, 140, 10000.0)
        assert position_size > 0

    def test_ml_sentiment_strategy_with_sentiment_features(self):
        """Test ML sentiment strategy maintains sentiment feature processing"""
        mock_engine = self._create_mock_prediction_engine()
        
        strategy = MlWithSentiment(
            prediction_engine=mock_engine,
            use_sentiment=True,
            sentiment_csv_path=None  # Will fail to load, testing fallback
        )
        
        df_with_indicators = strategy.calculate_indicators(self.test_data)
        
        # Should have sentiment features even if provider failed
        assert 'sentiment_freshness' in df_with_indicators.columns
        
        # Test entry conditions still work
        result = strategy.check_entry_conditions(df_with_indicators, 140)
        mock_engine.predict.assert_called_once()
        
        assert isinstance(result, bool)

    def test_strategy_without_prediction_engine(self):
        """Test strategy behavior when prediction engine is None"""
        # Test with None prediction engine (should create default if available)
        with pytest.raises(Exception):  # May raise import error if prediction engine not available
            strategy = MlBasic(prediction_engine=None)
        
        # Or test with explicit None and ensure graceful handling
        strategy = MlBasic.__new__(MlBasic)  # Bypass __init__
        strategy.name = "test"
        strategy.prediction_engine = None
        strategy.trading_pair = 'BTCUSDT'
        strategy.stop_loss_pct = 0.02
        strategy.take_profit_pct = 0.04
        
        # Manually initialize other required attributes
        strategy.logger = Mock()
        strategy.db_manager = None
        strategy.session_id = None
        strategy.enable_execution_logging = False
        
        df_with_indicators = strategy.calculate_indicators(self.test_data)
        
        # Test entry conditions - should handle None prediction engine gracefully
        result = strategy.check_entry_conditions(df_with_indicators, 140)
        assert result is False  # Should not enter without valid predictions

    def test_strategy_prediction_caching(self):
        """Test that strategies properly handle prediction caching"""
        # Create mock with cache hit
        cached_prediction = Mock(
            price=50000.0,
            confidence=0.8,
            direction=1,
            model_name='test_model',
            timestamp=datetime.now(timezone.utc),
            inference_time=0.001,  # Very fast due to cache
            features_used=5,
            cache_hit=True,
            error=None
        )
        
        mock_engine = self._create_mock_prediction_engine(cached_prediction)
        strategy = MlBasic(prediction_engine=mock_engine)
        
        df_with_indicators = strategy.calculate_indicators(self.test_data)
        
        # Call entry conditions twice - should use same prediction
        result1 = strategy.check_entry_conditions(df_with_indicators, 140)
        result2 = strategy.check_entry_conditions(df_with_indicators, 140)
        
        # Both should succeed
        assert result1 is True
        assert result2 is True
        
        # Engine should be called for each check
        assert mock_engine.predict.call_count == 2

    def test_strategy_confidence_based_position_sizing(self):
        """Test that strategies properly scale position size based on confidence"""
        # Test with high confidence
        high_conf_prediction = Mock(
            price=52000.0,
            confidence=0.9,
            direction=1,
            model_name='test_model',
            timestamp=datetime.now(timezone.utc),
            inference_time=0.1,
            features_used=5,
            cache_hit=False,
            error=None
        )
        
        mock_engine_high = self._create_mock_prediction_engine(high_conf_prediction)
        strategy_high = MlBasic(prediction_engine=mock_engine_high)
        df_with_indicators = strategy_high.calculate_indicators(self.test_data)
        
        size_high = strategy_high.calculate_position_size(df_with_indicators, 140, 10000.0)
        
        # Test with low confidence
        low_conf_prediction = Mock(
            price=50500.0,
            confidence=0.3,
            direction=1,
            model_name='test_model',
            timestamp=datetime.now(timezone.utc),
            inference_time=0.1,
            features_used=5,
            cache_hit=False,
            error=None
        )
        
        mock_engine_low = self._create_mock_prediction_engine(low_conf_prediction)
        strategy_low = MlBasic(prediction_engine=mock_engine_low)
        
        size_low = strategy_low.calculate_position_size(df_with_indicators, 140, 10000.0)
        
        # High confidence should result in larger position size
        assert size_high > size_low

    def test_all_strategies_have_prediction_engine_parameter(self):
        """Test that all strategies properly accept prediction_engine parameter"""
        mock_engine = self._create_mock_prediction_engine()
        
        strategies = [
            MlBasic(prediction_engine=mock_engine),
            MlAdaptive(prediction_engine=mock_engine),
            MlWithSentiment(prediction_engine=mock_engine)
        ]
        
        for strategy in strategies:
            assert strategy.prediction_engine is not None
            assert hasattr(strategy, 'get_prediction')
            
            # Test get_prediction method works
            prediction = strategy.get_prediction(self.test_data, 140)
            assert 'price' in prediction
            assert 'confidence' in prediction
            assert 'direction' in prediction

    def test_strategy_get_parameters_includes_prediction_info(self):
        """Test that strategy parameters include prediction engine information"""
        mock_engine = self._create_mock_prediction_engine()
        
        strategies = [
            MlBasic(prediction_engine=mock_engine),
            MlAdaptive(prediction_engine=mock_engine),
            MlWithSentiment(prediction_engine=mock_engine)
        ]
        
        for strategy in strategies:
            params = strategy.get_parameters()
            assert 'prediction_engine_available' in params
            assert params['prediction_engine_available'] is True