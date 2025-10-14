"""
Unit tests for MlSentiment strategy - Component-Based Implementation
"""

import pytest

from src.strategies.ml_sentiment import create_ml_sentiment_strategy
from src.strategies.components import Strategy, SignalDirection

pytestmark = pytest.mark.unit


class TestMlSentimentStrategy:
    """Test ML Sentiment strategy component-based implementation."""

    def test_create_ml_sentiment_strategy_factory(self):
        """Test that create_ml_sentiment_strategy() factory function works"""
        strategy = create_ml_sentiment_strategy()
        
        assert isinstance(strategy, Strategy)
        assert strategy.name == "MlSentiment"
        assert strategy.signal_generator is not None
        assert strategy.risk_manager is not None
        assert strategy.position_sizer is not None
        assert strategy.regime_detector is not None

    def test_ml_sentiment_strategy_initialization(self):
        """Test ML Sentiment strategy initialization with custom parameters"""
        strategy = create_ml_sentiment_strategy(
            name="CustomMlSentiment",
            sequence_length=100,
        )
        
        assert strategy.name == "CustomMlSentiment"
        assert strategy.signal_generator.sequence_length == 100

    def test_ml_sentiment_process_candle_returns_valid_decision(self, sample_ohlcv_data):
        """Test that process_candle() returns valid TradingDecision"""
        strategy = create_ml_sentiment_strategy()
        balance = 10000.0
        
        # Need sufficient data for sequence length (120)
        if len(sample_ohlcv_data) < 150:
            pytest.skip("Insufficient data for ML Sentiment strategy")
        
        decision = strategy.process_candle(sample_ohlcv_data, index=130, balance=balance)
        
        # Validate TradingDecision structure
        assert decision is not None
        assert hasattr(decision, 'signal')
        assert hasattr(decision, 'position_size')
        assert hasattr(decision, 'regime')
        assert hasattr(decision, 'risk_metrics')
        assert hasattr(decision, 'metadata')
        
        # Validate signal
        assert decision.signal.direction in [SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD]
        assert 0 <= decision.signal.confidence <= 1
        
        # Validate position size
        assert decision.position_size >= 0
        assert decision.position_size <= balance

    def test_ml_sentiment_integration(self, sample_ohlcv_data):
        """Test sentiment integration in ML Sentiment strategy"""
        strategy = create_ml_sentiment_strategy()
        balance = 10000.0
        
        if len(sample_ohlcv_data) < 150:
            pytest.skip("Insufficient data for ML Sentiment strategy")
        
        decision = strategy.process_candle(sample_ohlcv_data, index=130, balance=balance)
        
        # Decision should be made (even if sentiment data is unavailable)
        assert decision is not None
        assert decision.signal is not None

    def test_ml_sentiment_signal_generation(self, sample_ohlcv_data):
        """Test ML Sentiment signal generation logic"""
        strategy = create_ml_sentiment_strategy()
        balance = 10000.0
        
        if len(sample_ohlcv_data) < 150:
            pytest.skip("Insufficient data for ML Sentiment strategy")
        
        decision = strategy.process_candle(sample_ohlcv_data, index=130, balance=balance)
        
        # Signal should have confidence and strength
        assert hasattr(decision.signal, 'confidence')
        assert hasattr(decision.signal, 'strength')
        assert decision.signal.confidence >= 0
        assert decision.signal.confidence <= 1

    def test_ml_sentiment_risk_management(self, sample_ohlcv_data):
        """Test ML Sentiment risk management"""
        strategy = create_ml_sentiment_strategy()
        balance = 10000.0
        
        if len(sample_ohlcv_data) < 150:
            pytest.skip("Insufficient data for ML Sentiment strategy")
        
        decision = strategy.process_candle(sample_ohlcv_data, index=130, balance=balance)
        
        # Risk metrics should be present
        assert decision.risk_metrics is not None
        assert isinstance(decision.risk_metrics, dict)

    def test_ml_sentiment_position_sizing(self, sample_ohlcv_data):
        """Test ML Sentiment position sizing"""
        strategy = create_ml_sentiment_strategy()
        balance = 10000.0
        
        if len(sample_ohlcv_data) < 150:
            pytest.skip("Insufficient data for ML Sentiment strategy")
        
        decision = strategy.process_candle(sample_ohlcv_data, index=130, balance=balance)
        
        # Position size should be reasonable
        if decision.signal.direction != SignalDirection.HOLD:
            assert decision.position_size > 0
            assert decision.position_size <= balance * 0.5  # Should not exceed 50% of balance
