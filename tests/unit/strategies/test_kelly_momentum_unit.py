"""
Unit tests for KellyMomentum strategy - Component-Based Implementation
"""

from __future__ import annotations

import pytest

from src.strategies.components import SignalDirection, Strategy
from src.strategies.components.position_sizer import KellyCriterionSizer
from src.strategies.kelly_momentum import create_kelly_momentum_strategy

pytestmark = pytest.mark.unit


class TestKellyMomentumStrategy:
    """Test Kelly Momentum strategy component-based implementation."""

    def test_create_kelly_momentum_strategy_factory(self):
        """Test that create_kelly_momentum_strategy() factory function works."""
        strategy = create_kelly_momentum_strategy()

        assert isinstance(strategy, Strategy)
        assert strategy.name == "KellyMomentum"
        assert strategy.signal_generator is not None
        assert strategy.risk_manager is not None
        assert strategy.position_sizer is not None
        assert strategy.regime_detector is not None

    def test_position_sizer_is_kelly_criterion(self):
        """Test that the strategy uses KellyCriterionSizer."""
        strategy = create_kelly_momentum_strategy()
        assert isinstance(strategy.position_sizer, KellyCriterionSizer)

    def test_custom_name(self):
        """Test strategy initialization with custom name."""
        strategy = create_kelly_momentum_strategy(name="CustomKelly")
        assert strategy.name == "CustomKelly"

    def test_custom_kelly_fraction(self):
        """Test strategy initialization with custom kelly_fraction."""
        strategy = create_kelly_momentum_strategy(kelly_fraction=0.25)
        assert strategy.position_sizer.kelly_fraction == 0.25

    def test_custom_momentum_threshold(self):
        """Test strategy initialization with custom momentum threshold."""
        strategy = create_kelly_momentum_strategy(momentum_entry_threshold=0.02)
        assert strategy.signal_generator.momentum_entry_threshold == 0.02

    def test_risk_overrides_set(self):
        """Test that risk overrides are properly configured."""
        strategy = create_kelly_momentum_strategy()
        overrides = strategy.get_risk_overrides()

        assert overrides is not None
        assert overrides["position_sizer"] == "kelly_criterion"
        assert "dynamic_risk" in overrides
        assert "partial_operations" in overrides
        assert "trailing_stop" in overrides

    def test_process_candle_returns_valid_decision(self, sample_ohlcv_data):
        """Test that process_candle() returns valid TradingDecision."""
        strategy = create_kelly_momentum_strategy()
        balance = 10000.0

        if len(sample_ohlcv_data) < 50:
            pytest.skip("Insufficient data for Kelly Momentum strategy")

        decision = strategy.process_candle(sample_ohlcv_data, index=30, balance=balance)

        assert decision is not None
        assert hasattr(decision, "signal")
        assert hasattr(decision, "position_size")
        assert hasattr(decision, "regime")
        assert hasattr(decision, "risk_metrics")

        assert decision.signal.direction in [
            SignalDirection.BUY,
            SignalDirection.SELL,
            SignalDirection.HOLD,
        ]
        assert 0 <= decision.signal.confidence <= 1
        assert decision.position_size >= 0
        assert decision.position_size <= balance

    def test_strategy_uses_cold_start_initially(self, sample_ohlcv_data):
        """Test that strategy uses cold start fallback before trades are recorded."""
        strategy = create_kelly_momentum_strategy()
        sizer = strategy.position_sizer

        assert isinstance(sizer, KellyCriterionSizer)
        assert not sizer.has_sufficient_history

    def test_strategy_attributes_set(self):
        """Test that strategy exposes expected configuration attributes."""
        strategy = create_kelly_momentum_strategy(
            take_profit_pct=0.15,
            fallback_fraction=0.04,
        )

        assert strategy.base_position_size == 0.04
        assert strategy.take_profit_pct == 0.15

    def test_strategy_registered_in_module(self):
        """Test that strategy factory is importable from strategies package."""
        from src.strategies import create_kelly_momentum_strategy as factory

        strategy = factory()
        assert isinstance(strategy, Strategy)
