"""
Unit tests for MomentumLeverage strategy - Component-Based Implementation
"""

import pytest

from src.strategies.momentum_leverage import create_momentum_leverage_strategy
from src.strategies.components import Strategy, SignalDirection

pytestmark = pytest.mark.unit


class TestMomentumLeverageStrategy:
    """Test Momentum Leverage strategy component-based implementation."""

    def test_create_momentum_leverage_strategy_factory(self):
        """Test that create_momentum_leverage_strategy() factory function works"""
        strategy = create_momentum_leverage_strategy()
        
        assert isinstance(strategy, Strategy)
        assert strategy.name == "MomentumLeverage"
        assert strategy.signal_generator is not None
        assert strategy.risk_manager is not None
        assert strategy.position_sizer is not None
        assert strategy.regime_detector is not None

    def test_momentum_leverage_strategy_initialization(self):
        """Test Momentum Leverage strategy initialization with custom parameters"""
        strategy = create_momentum_leverage_strategy(
            name="CustomMomentum",
            momentum_entry_threshold=0.02,
            base_risk=0.08,
        )
        
        assert strategy.name == "CustomMomentum"
        assert strategy.signal_generator.momentum_entry_threshold == 0.02

    def test_momentum_leverage_process_candle_returns_valid_decision(self, sample_ohlcv_data):
        """Test that process_candle() returns valid TradingDecision"""
        strategy = create_momentum_leverage_strategy()
        balance = 10000.0
        
        # Need sufficient data for momentum calculation
        if len(sample_ohlcv_data) < 50:
            pytest.skip("Insufficient data for Momentum Leverage strategy")
        
        decision = strategy.process_candle(sample_ohlcv_data, index=30, balance=balance)
        
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

    def test_momentum_signal_generation(self, sample_ohlcv_data):
        """Test momentum signal generation logic"""
        strategy = create_momentum_leverage_strategy()
        balance = 10000.0
        
        if len(sample_ohlcv_data) < 50:
            pytest.skip("Insufficient data for Momentum Leverage strategy")
        
        decision = strategy.process_candle(sample_ohlcv_data, index=30, balance=balance)
        
        # Signal should have confidence and strength
        assert hasattr(decision.signal, 'confidence')
        assert hasattr(decision.signal, 'strength')
        assert decision.signal.confidence >= 0
        assert decision.signal.confidence <= 1

    def test_momentum_leverage_risk_management(self, sample_ohlcv_data):
        """Test Momentum Leverage volatility-based risk management"""
        strategy = create_momentum_leverage_strategy()
        balance = 10000.0
        
        if len(sample_ohlcv_data) < 50:
            pytest.skip("Insufficient data for Momentum Leverage strategy")
        
        decision = strategy.process_candle(sample_ohlcv_data, index=30, balance=balance)
        
        # Risk metrics should be present
        assert decision.risk_metrics is not None
        assert isinstance(decision.risk_metrics, dict)

    def test_momentum_leverage_position_sizing(self, sample_ohlcv_data):
        """Test Momentum Leverage aggressive position sizing"""
        strategy = create_momentum_leverage_strategy()
        balance = 10000.0
        
        if len(sample_ohlcv_data) < 50:
            pytest.skip("Insufficient data for Momentum Leverage strategy")
        
        decision = strategy.process_candle(sample_ohlcv_data, index=30, balance=balance)
        
        # Position size should be aggressive but reasonable
        if decision.signal.direction != SignalDirection.HOLD:
            assert decision.position_size > 0
            assert decision.position_size <= balance * 0.5  # Should not exceed 50% of balance

    def test_momentum_leverage_aggressive_parameters(self):
        """Test that Momentum Leverage uses aggressive parameters"""
        strategy = create_momentum_leverage_strategy()
        
        # Check that position sizer has aggressive base fraction
        assert strategy.position_sizer.base_fraction == 0.5  # 50% base allocation
        
        # Check that risk manager has wide stop loss
        assert strategy.risk_manager.base_risk == 0.10  # 10% stop loss
