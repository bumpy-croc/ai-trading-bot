"""Tests for adaptive trend strategy factory and custom components.

Validates TrendFollowingRiskManager, TrendFollowingPositionSizer,
and the create_adaptive_trend_strategy factory function.
"""


import pytest

from src.strategies.adaptive_trend import (
    TrendFollowingPositionSizer,
    TrendFollowingRiskManager,
    create_adaptive_trend_strategy,
)
from src.strategies.components.signal_generator import Signal, SignalDirection


def _make_signal(
    direction: SignalDirection = SignalDirection.BUY,
    strength: float = 0.8,
    confidence: float = 0.9,
) -> Signal:
    """Create a test signal."""
    return Signal(
        direction=direction,
        strength=strength,
        confidence=confidence,
        metadata={"generator": "test"},
    )


class TestTrendFollowingRiskManager:
    """Test TrendFollowingRiskManager."""

    def test_default_initialization(self):
        """Test default parameter values."""
        rm = TrendFollowingRiskManager()
        assert rm.target_allocation == 0.90
        assert rm.stop_loss_pct == 0.15

    def test_custom_initialization(self):
        """Test custom parameter values."""
        rm = TrendFollowingRiskManager(target_allocation=0.99, stop_loss_pct=0.40)
        assert rm.target_allocation == 0.99
        assert rm.stop_loss_pct == 0.40

    def test_invalid_allocation_raises(self):
        """Test that invalid allocation raises ValueError."""
        with pytest.raises(ValueError, match="target_allocation"):
            TrendFollowingRiskManager(target_allocation=1.5)
        with pytest.raises(ValueError, match="target_allocation"):
            TrendFollowingRiskManager(target_allocation=0.0)

    def test_invalid_stop_loss_raises(self):
        """Test that invalid stop loss raises ValueError."""
        with pytest.raises(ValueError, match="stop_loss_pct"):
            TrendFollowingRiskManager(stop_loss_pct=0.0)
        with pytest.raises(ValueError, match="stop_loss_pct"):
            TrendFollowingRiskManager(stop_loss_pct=0.6)

    def test_position_size_for_buy(self):
        """Test position sizing returns fixed allocation for BUY signal."""
        rm = TrendFollowingRiskManager(target_allocation=0.95)
        signal = _make_signal(SignalDirection.BUY)
        size = rm.calculate_position_size(signal, balance=10000.0)
        assert size == pytest.approx(9500.0)

    def test_position_size_for_hold(self):
        """Test position sizing returns 0 for HOLD signal."""
        rm = TrendFollowingRiskManager(target_allocation=0.95)
        signal = _make_signal(SignalDirection.HOLD)
        size = rm.calculate_position_size(signal, balance=10000.0)
        assert size == 0.0

    def test_position_size_ignores_confidence(self):
        """Test that position size is the same regardless of confidence."""
        rm = TrendFollowingRiskManager(target_allocation=0.95)
        low_conf = _make_signal(SignalDirection.BUY, confidence=0.1)
        high_conf = _make_signal(SignalDirection.BUY, confidence=0.99)
        assert rm.calculate_position_size(low_conf, 10000.0) == rm.calculate_position_size(
            high_conf, 10000.0
        )

    def test_stop_loss_for_buy(self):
        """Test stop loss calculation for long position."""
        rm = TrendFollowingRiskManager(stop_loss_pct=0.40)
        signal = _make_signal(SignalDirection.BUY)
        sl = rm.get_stop_loss(entry_price=50000.0, signal=signal)
        assert sl == pytest.approx(30000.0)

    def test_stop_loss_for_sell(self):
        """Test stop loss calculation for short position."""
        rm = TrendFollowingRiskManager(stop_loss_pct=0.40)
        signal = _make_signal(SignalDirection.SELL)
        sl = rm.get_stop_loss(entry_price=50000.0, signal=signal)
        assert sl == pytest.approx(70000.0)

    def test_stop_loss_invalid_price_raises(self):
        """Test that zero/negative entry price raises ValueError."""
        rm = TrendFollowingRiskManager()
        signal = _make_signal(SignalDirection.BUY)
        with pytest.raises(ValueError, match="entry_price must be positive"):
            rm.get_stop_loss(entry_price=0.0, signal=signal)
        with pytest.raises(ValueError, match="entry_price must be positive"):
            rm.get_stop_loss(entry_price=-100.0, signal=signal)

    def test_get_parameters(self):
        """Test parameter dict contains expected keys."""
        rm = TrendFollowingRiskManager(target_allocation=0.99, stop_loss_pct=0.40)
        params = rm.get_parameters()
        assert params["target_allocation"] == 0.99
        assert params["stop_loss_pct"] == 0.40


class TestTrendFollowingPositionSizer:
    """Test TrendFollowingPositionSizer."""

    def test_passthrough_sizing(self):
        """Test that risk_amount passes through bounded by max_fraction."""
        ps = TrendFollowingPositionSizer(max_fraction=0.95)
        signal = _make_signal(SignalDirection.BUY)
        size = ps.calculate_size(signal, balance=10000.0, risk_amount=9500.0)
        assert size == 9500.0

    def test_caps_at_max_fraction(self):
        """Test that size is capped at max_fraction * balance."""
        ps = TrendFollowingPositionSizer(max_fraction=0.50)
        signal = _make_signal(SignalDirection.BUY)
        size = ps.calculate_size(signal, balance=10000.0, risk_amount=9500.0)
        assert size == 5000.0

    def test_hold_returns_zero(self):
        """Test that HOLD signal returns 0."""
        ps = TrendFollowingPositionSizer()
        signal = _make_signal(SignalDirection.HOLD)
        size = ps.calculate_size(signal, balance=10000.0, risk_amount=5000.0)
        assert size == 0.0

    def test_zero_risk_amount_returns_zero(self):
        """Test that zero risk amount returns 0."""
        ps = TrendFollowingPositionSizer()
        signal = _make_signal(SignalDirection.BUY)
        size = ps.calculate_size(signal, balance=10000.0, risk_amount=0.0)
        assert size == 0.0

    def test_get_parameters(self):
        """Test parameter dict contains max_fraction."""
        ps = TrendFollowingPositionSizer(max_fraction=0.95)
        params = ps.get_parameters()
        assert params["max_fraction"] == 0.95


class TestCreateAdaptiveTrendStrategy:
    """Test the factory function."""

    def test_creates_strategy(self):
        """Test that factory creates a Strategy instance."""
        strategy = create_adaptive_trend_strategy()
        assert strategy.name == "AdaptiveTrend"

    def test_default_parameters(self):
        """Test that default parameters match expected values."""
        strategy = create_adaptive_trend_strategy()
        params = strategy.signal_generator.get_parameters()
        assert params["trend_ema_period"] == 90
        assert params["exit_confirmation_days"] == 18
        assert params["exit_buffer_pct"] == 0.08
        assert params["ema_slope_lookback"] == 35

    def test_custom_name(self):
        """Test custom strategy name."""
        strategy = create_adaptive_trend_strategy(name="MyTrend")
        assert strategy.name == "MyTrend"

    def test_risk_overrides_set(self):
        """Test that risk overrides are properly configured."""
        strategy = create_adaptive_trend_strategy(target_allocation=0.99, max_position_pct=0.99)
        overrides = strategy.get_risk_overrides()
        assert overrides["max_fraction"] == 0.99
        assert overrides["base_fraction"] == 0.99
        assert "stop_loss_pct" in overrides
        assert "take_profit_pct" in overrides

    def test_no_trailing_stop(self):
        """Test that trailing stop is not configured."""
        strategy = create_adaptive_trend_strategy()
        overrides = strategy.get_risk_overrides()
        assert "trailing_stop_distance" not in overrides

    def test_max_position_pct_exposed(self):
        """Test that _max_position_pct is set on strategy."""
        strategy = create_adaptive_trend_strategy(max_position_pct=0.99)
        assert strategy._max_position_pct == 0.99

    def test_custom_parameters_flow_through(self):
        """Test that custom parameters reach the signal generator."""
        strategy = create_adaptive_trend_strategy(
            trend_ema_period=75,
            entry_confirmation_days=3,
            exit_confirmation_days=25,
            ema_slope_lookback=30,
        )
        params = strategy.signal_generator.get_parameters()
        assert params["trend_ema_period"] == 75
        assert params["entry_confirmation_days"] == 3
        assert params["exit_confirmation_days"] == 25
        assert params["ema_slope_lookback"] == 30
