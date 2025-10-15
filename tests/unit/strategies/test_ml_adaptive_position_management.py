"""
Unit tests for MlAdaptive strategy - Component-Based Implementation
"""

from types import MethodType

from src.strategies.components import SignalDirection, Strategy
from src.strategies.ml_adaptive import create_ml_adaptive_strategy


def _process_with_fixture(strategy, sample_ohlcv_data, balance=10000.0):
    """Run the adaptive strategy with adequate historical context for sequence models."""
    index = strategy.signal_generator.sequence_length + 10
    assert len(sample_ohlcv_data) > index, "sample_ohlcv_data must provide enough candles for ML tests"
    strategy.signal_generator._get_ml_prediction = MethodType(
        lambda self, df, idx: float(df["close"].iloc[idx - 1] * 1.01),
        strategy.signal_generator,
    )
    return strategy.process_candle(sample_ohlcv_data, index=index, balance=balance)


class TestMlAdaptiveStrategy:
    """Test ML Adaptive strategy component-based implementation."""

    def test_create_ml_adaptive_strategy_factory(self):
        """Test that create_ml_adaptive_strategy() factory function works"""
        strategy = create_ml_adaptive_strategy()
        
        assert isinstance(strategy, Strategy)
        assert strategy.name == "MlAdaptive"
        assert strategy.signal_generator is not None
        assert strategy.risk_manager is not None
        assert strategy.position_sizer is not None
        assert strategy.regime_detector is not None

    def test_ml_adaptive_strategy_initialization(self):
        """Test ML Adaptive strategy initialization with custom parameters"""
        strategy = create_ml_adaptive_strategy(
            name="CustomMlAdaptive",
            sequence_length=100,
        )
        
        assert strategy.name == "CustomMlAdaptive"
        assert strategy.signal_generator.sequence_length == 100

    def test_ml_adaptive_process_candle_returns_valid_decision(self, sample_ohlcv_data):
        """Test that process_candle() returns valid TradingDecision"""
        strategy = create_ml_adaptive_strategy()
        balance = 10000.0
        
        decision = _process_with_fixture(strategy, sample_ohlcv_data, balance)
        
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

    def test_ml_adaptive_regime_aware_behavior(self, sample_ohlcv_data):
        """Test that ML Adaptive strategy is regime-aware"""
        strategy = create_ml_adaptive_strategy()
        balance = 10000.0
        
        decision = _process_with_fixture(strategy, sample_ohlcv_data, balance)
        
        # Regime context should be present with trend/volatility annotations
        assert decision.regime is not None
        assert hasattr(decision.regime, 'trend')
        assert hasattr(decision.regime, 'volatility')

    def test_ml_adaptive_signal_generation(self, sample_ohlcv_data):
        """Test ML Adaptive signal generation logic"""
        strategy = create_ml_adaptive_strategy()
        balance = 10000.0
        
        decision = _process_with_fixture(strategy, sample_ohlcv_data, balance)
        
        # Signal should have confidence and strength
        assert hasattr(decision.signal, 'confidence')
        assert hasattr(decision.signal, 'strength')
        assert decision.signal.confidence >= 0
        assert decision.signal.confidence <= 1

    def test_ml_adaptive_risk_management(self, sample_ohlcv_data):
        """Test ML Adaptive regime-adaptive risk management"""
        strategy = create_ml_adaptive_strategy()
        balance = 10000.0
        
        decision = _process_with_fixture(strategy, sample_ohlcv_data, balance)
        
        # Risk metrics should be present
        assert decision.risk_metrics is not None
        assert isinstance(decision.risk_metrics, dict)

    def test_ml_adaptive_position_sizing(self, sample_ohlcv_data):
        """Test ML Adaptive position sizing"""
        strategy = create_ml_adaptive_strategy()
        balance = 10000.0
        
        decision = _process_with_fixture(strategy, sample_ohlcv_data, balance)
        
        # Position size should be reasonable
        if decision.signal.direction != SignalDirection.HOLD:
            assert decision.position_size > 0
            assert decision.position_size <= balance * 0.5  # Should not exceed 50% of balance
