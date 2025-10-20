from types import MethodType

import pytest

from src.strategies.components import SignalDirection
from src.strategies.ml_basic import create_ml_basic_strategy

pytestmark = pytest.mark.unit


def _process_with_fixture(strategy, sample_ohlcv_data, balance=10000.0):
    """Run the strategy with enough history to populate logging metadata."""
    index = strategy.signal_generator.sequence_length + 10
    assert len(sample_ohlcv_data) > index, "sample_ohlcv_data must provide enough candles for ML tests"
    strategy.signal_generator._get_ml_prediction = MethodType(
        lambda self, df, idx: float(df["close"].iloc[idx - 1] * 1.01),
        strategy.signal_generator,
    )
    return strategy.process_candle(sample_ohlcv_data, index=index, balance=balance)


class TestMlBasicLogging:
    def test_ml_basic_strategy_execution_metadata(self, sample_ohlcv_data):
        """Test that ML Basic strategy includes execution metadata in decisions"""
        strategy = create_ml_basic_strategy()
        balance = 10000.0

        decision = _process_with_fixture(strategy, sample_ohlcv_data, balance)
        
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

        decision = _process_with_fixture(strategy, sample_ohlcv_data, balance)
        
        # Decision should have timestamp
        assert decision.timestamp is not None
        
        # Signal should have direction and confidence
        assert decision.signal.direction in [SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD]
        assert 0 <= decision.signal.confidence <= 1
        
        # Risk metrics should be present
        assert decision.risk_metrics is not None
