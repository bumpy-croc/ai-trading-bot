"""
Tests for strategy-prediction engine integration.

This module tests that the refactored ML strategies work correctly with the
prediction engine and produce expected results.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from src.strategies.ml_basic import MlBasic
from src.strategies.ml_adaptive import MlAdaptive
from src.strategies.ml_with_sentiment import MlWithSentiment
from src.prediction.engine import PredictionEngine, PredictionResult


class TestStrategyPredictionIntegration:
    """Test strategy integration with prediction engine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create sample market data
        dates = pd.date_range('2023-01-01', periods=150, freq='1h')
        np.random.seed(42)  # For reproducible tests
        
        self.test_data = pd.DataFrame({
            'open': 50000 + np.random.randn(150) * 100,
            'high': 50100 + np.random.randn(150) * 100,
            'low': 49900 + np.random.randn(150) * 100,
            'close': 50000 + np.random.randn(150) * 100,
            'volume': 1000 + np.random.randn(150) * 50
        }, index=dates)
        
        # Ensure proper OHLC relationships
        for i in range(len(self.test_data)):
            o, h, l, c = (self.test_data.iloc[i]['open'], 
                         self.test_data.iloc[i]['high'],
                         self.test_data.iloc[i]['low'], 
                         self.test_data.iloc[i]['close'])
            
            # Adjust to maintain OHLC logic
            min_val = min(o, c)
            max_val = max(o, c)
            self.test_data.iloc[i, self.test_data.columns.get_loc('high')] = max(h, max_val)
            self.test_data.iloc[i, self.test_data.columns.get_loc('low')] = min(l, min_val)
    
    def test_ml_basic_strategy_prediction_integration(self):
        """Test ML basic strategy with prediction engine"""
        # Create mock prediction engine
        mock_engine = Mock()
        mock_engine.predict.return_value = PredictionResult(
            price=50500.0,
            confidence=0.8,
            direction=1,
            model_name='btcusdt_price',
            timestamp=pd.Timestamp.now()
        )
        
        strategy = MlBasic(prediction_engine=mock_engine)
        
        # Calculate indicators
        df = strategy.calculate_indicators(self.test_data)
        
        # Test entry conditions
        result = strategy.check_entry_conditions(df, 140)
        
        assert result is True  # Should enter based on high confidence and positive direction
        assert strategy.prediction_engine == mock_engine
        
        # Verify prediction engine was called
        mock_engine.predict.assert_called_once()
        
        # Test that the call was made with the right data slice
        call_args = mock_engine.predict.call_args
        assert len(call_args[0][0]) == 141  # Data up to index 140 (inclusive)
    
    def test_ml_basic_strategy_low_confidence_no_entry(self):
        """Test ML basic strategy doesn't enter with low confidence"""
        # Create mock prediction engine with low confidence
        mock_engine = Mock()
        mock_engine.predict.return_value = PredictionResult(
            price=50100.0,
            confidence=0.3,  # Below default threshold of 0.6
            direction=1,
            model_name='btcusdt_price',
            timestamp=pd.Timestamp.now()
        )
        
        strategy = MlBasic(prediction_engine=mock_engine)
        df = strategy.calculate_indicators(self.test_data)
        
        # Test entry conditions
        result = strategy.check_entry_conditions(df, 140)
        
        assert result is False  # Should not enter due to low confidence
    
    def test_ml_adaptive_strategy_prediction_integration(self):
        """Test ML adaptive strategy with prediction engine"""
        mock_engine = Mock()
        mock_engine.predict.return_value = PredictionResult(
            price=50800.0,
            confidence=0.85,
            direction=1,
            model_name='btcusdt_price',
            timestamp=pd.Timestamp.now()
        )
        
        strategy = MlAdaptive(prediction_engine=mock_engine)
        
        # Calculate indicators (includes market regime detection)
        df = strategy.calculate_indicators(self.test_data)
        
        # Test entry conditions
        result = strategy.check_entry_conditions(df, 140)
        
        assert result is True  # Should enter with high confidence
        assert 'market_regime' in df.columns  # Should have regime detection
        
        # Test adaptive threshold calculation
        threshold = strategy._calculate_adaptive_threshold(df, 140)
        assert threshold > 0  # Should return a positive threshold
        
        # Verify prediction engine was called
        mock_engine.predict.assert_called_once()
    
    def test_ml_adaptive_strategy_crisis_mode(self):
        """Test ML adaptive strategy in crisis mode (higher threshold)"""
        mock_engine = Mock()
        mock_engine.predict.return_value = PredictionResult(
            price=50400.0,
            confidence=0.75,  # Good confidence but might not be enough in crisis
            direction=1,
            model_name='btcusdt_price',
            timestamp=pd.Timestamp.now()
        )
        
        strategy = MlAdaptive(prediction_engine=mock_engine)
        
        # Create high volatility data to trigger crisis mode
        crisis_data = self.test_data.copy()
        crisis_data['close'] = crisis_data['close'] * (1 + np.random.randn(150) * 0.25)  # 25% volatility
        
        df = strategy.calculate_indicators(crisis_data)
        
        # Check if crisis mode is detected
        if 'market_regime' in df.columns:
            has_crisis = (df['market_regime'] == 'crisis').any()
            if has_crisis:
                # In crisis mode, threshold should be higher
                threshold = strategy._calculate_adaptive_threshold(df, 140)
                assert threshold > strategy.adaptive_threshold  # Should be higher than base
    
    def test_ml_sentiment_strategy_prediction_integration(self):
        """Test ML sentiment strategy with prediction engine"""
        mock_engine = Mock()
        mock_engine.predict.return_value = PredictionResult(
            price=50600.0,
            confidence=0.7,
            direction=1,
            model_name='btcusdt_sentiment',
            timestamp=pd.Timestamp.now()
        )
        
        # Test with sentiment disabled to focus on prediction engine integration
        strategy = MlWithSentiment(use_sentiment=False, prediction_engine=mock_engine)
        
        df = strategy.calculate_indicators(self.test_data)
        
        # Test entry conditions
        result = strategy.check_entry_conditions(df, 140)
        
        assert result is True  # Should enter based on ML prediction
        
        # Test sentiment score fallback
        sentiment_score = strategy._get_sentiment_score(df, 140)
        assert sentiment_score == 0.0  # Should be neutral when no sentiment
        
        # Test confidence combination
        combined = strategy._combine_confidence(0.7, 0.0)
        assert combined == 0.7 * (1 - strategy.sentiment_weight)  # Should reduce ML confidence by sentiment weight
    
    def test_ml_sentiment_strategy_with_sentiment_data(self):
        """Test ML sentiment strategy with actual sentiment data"""
        mock_engine = Mock()
        mock_engine.predict.return_value = PredictionResult(
            price=50300.0,
            confidence=0.65,
            direction=1,
            model_name='btcusdt_sentiment',
            timestamp=pd.Timestamp.now()
        )
        
        strategy = MlWithSentiment(use_sentiment=True, prediction_engine=mock_engine)
        
        # Add sentiment data to test DataFrame
        sentiment_data = self.test_data.copy()
        sentiment_data['sentiment_primary'] = 0.8  # Positive sentiment
        
        df = strategy.calculate_indicators(sentiment_data)
        
        # Test sentiment score retrieval
        sentiment_score = strategy._get_sentiment_score(df, 140)
        assert sentiment_score > 0  # Should have positive sentiment
        
        # Test confidence combination
        combined = strategy._combine_confidence(0.65, 0.8)
        assert combined > 0.65  # Should be higher than ML confidence alone
    
    def test_strategy_fallback_on_prediction_error(self):
        """Test strategy fallback when prediction fails"""
        # Create mock prediction engine that returns error
        mock_engine = Mock()
        mock_engine.predict.return_value = PredictionResult(
            price=None,
            confidence=0.0,
            direction=0,
            model_name=None,
            timestamp=pd.Timestamp.now(),
            error="Model not available"
        )
        
        strategy = MlBasic(prediction_engine=mock_engine)
        df = strategy.calculate_indicators(self.test_data)
        
        # Test entry conditions
        result = strategy.check_entry_conditions(df, 140)
        
        assert result is False  # Should not enter on prediction error
        assert strategy.prediction_engine == mock_engine
    
    def test_strategy_without_prediction_engine(self):
        """Test strategy behavior when prediction engine fails to initialize"""
        with patch('src.prediction.engine.PredictionEngine') as mock_engine_class:
            mock_engine_class.side_effect = Exception("Failed to initialize")
            
            # Strategy should handle this gracefully
            strategy = MlBasic()
            assert strategy.prediction_engine is None
            
            df = strategy.calculate_indicators(self.test_data)
            result = strategy.check_entry_conditions(df, 140)
            
            # Should not enter when no prediction engine available
            assert result is False
    
    def test_position_sizing_with_prediction_engine(self):
        """Test position sizing based on prediction confidence"""
        mock_engine = Mock()
        mock_engine.predict.return_value = PredictionResult(
            price=50800.0,
            confidence=0.9,  # High confidence
            direction=1,
            model_name='btcusdt_price',
            timestamp=pd.Timestamp.now()
        )
        
        strategy = MlBasic(prediction_engine=mock_engine)
        df = strategy.calculate_indicators(self.test_data)
        
        balance = 10000
        position_size = strategy.calculate_position_size(df, 140, balance)
        
        assert position_size > 0  # Should calculate positive position size
        assert position_size <= balance * 0.2  # Should respect maximum position size
        assert position_size >= balance * 0.05  # Should respect minimum position size
    
    def test_exit_conditions_with_prediction_engine(self):
        """Test exit conditions using prediction engine"""
        mock_engine = Mock()
        mock_engine.predict.return_value = PredictionResult(
            price=49000.0,  # Significant drop predicted
            confidence=0.85,
            direction=-1,  # Negative direction
            model_name='btcusdt_price',
            timestamp=pd.Timestamp.now()
        )
        
        strategy = MlBasic(prediction_engine=mock_engine)
        df = strategy.calculate_indicators(self.test_data)
        
        entry_price = 50000
        
        # Test exit conditions
        result = strategy.check_exit_conditions(df, 140, entry_price)
        
        assert result is True  # Should exit based on negative ML prediction
    
    def test_prediction_engine_integration_end_to_end(self):
        """Test full integration from strategy to prediction engine"""
        # Create a real prediction engine (will use mock models)
        with patch('src.prediction.models.onnx_runner.OnnxRunner') as mock_runner_class:
            mock_runner = Mock()
            mock_runner.predict.return_value = Mock(
                price=50400.0,
                confidence=0.75,
                direction=1,
                model_name='btcusdt_price',
                inference_time=0.05
            )
            mock_runner_class.return_value = mock_runner
            
            # Create strategy with real prediction engine
            strategy = MlBasic()
            
            # Should have initialized prediction engine
            assert strategy.prediction_engine is not None
            
            df = strategy.calculate_indicators(self.test_data)
            result = strategy.check_entry_conditions(df, 140)
            
            # Should work end-to-end
            assert isinstance(result, bool)
    
    def test_strategy_parameters_include_prediction_info(self):
        """Test that strategy parameters include prediction-related info"""
        mock_engine = Mock()
        strategy = MlBasic(prediction_engine=mock_engine)
        
        params = strategy.get_parameters()
        
        # Should include basic strategy parameters
        assert 'name' in params
        assert 'symbol' in params
        assert 'timeframe' in params
        assert 'min_confidence_threshold' in params
    
    def test_multiple_strategies_share_prediction_engine(self):
        """Test that multiple strategies can share the same prediction engine"""
        mock_engine = Mock()
        mock_engine.predict.return_value = PredictionResult(
            price=50500.0,
            confidence=0.8,
            direction=1,
            model_name='btcusdt_price',
            timestamp=pd.Timestamp.now()
        )
        
        strategy1 = MlBasic(prediction_engine=mock_engine)
        strategy2 = MlAdaptive(prediction_engine=mock_engine)
        
        # Both should use the same engine
        assert strategy1.prediction_engine == mock_engine
        assert strategy2.prediction_engine == mock_engine
        
        df1 = strategy1.calculate_indicators(self.test_data)
        df2 = strategy2.calculate_indicators(self.test_data)
        
        result1 = strategy1.check_entry_conditions(df1, 140)
        result2 = strategy2.check_entry_conditions(df2, 140)
        
        # Both should be able to make predictions
        assert isinstance(result1, bool)
        assert isinstance(result2, bool)
        
        # Engine should have been called multiple times
        assert mock_engine.predict.call_count >= 2