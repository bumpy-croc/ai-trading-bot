import pytest

from src.strategies.ml_basic import create_ml_basic_strategy
from src.strategies.components import SignalDirection

pytestmark = pytest.mark.unit


class TestMlBasicLogging:
    def test_ml_basic_strategy_execution_metadata(self, sample_ohlcv_data):
        """Test that ML Basic strategy includes execution metadata in decisions"""
        strategy = create_ml_basic_strategy()
        balance = 10000.0
        
        if len(sample_ohlcv_data) < 150:
            pytest.skip("Insufficient data for ML Basic strategy")
        
        decision = strategy.process_candle(sample_ohlcv_data, index=130, balance=balance)
        
        # Validate metadata is present
        assert decision.metadata is not None
        assert isinstance(decision.metadata, dict)
        
        # Check for execution time
        assert decision.execution_time_ms >= 0
        
        # Metadata should contain useful information
        assert len(decision.metadata) > 0

    def test_ml_basic_strategy_decision_context(self, sample_ohlcv_data):
        """Test that decisions include proper context"""
        strategy = create_ml_basic_strategy()
        balance = 10000.0
        
        if len(sample_ohlcv_data) < 150:
            pytest.skip("Insufficient data for ML Basic strategy")
        
        decision = strategy.process_candle(sample_ohlcv_data, index=130, balance=balance)
        
        # Decision should have timestamp
        assert decision.timestamp is not None
        
        # Signal should have direction and confidence
        assert decision.signal.direction in [SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD]
        assert 0 <= decision.signal.confidence <= 1
        
        # Risk metrics should be present
        assert decision.risk_metrics is not None
