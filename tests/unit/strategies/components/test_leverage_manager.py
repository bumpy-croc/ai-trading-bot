"""Tests for LeverageManager - regime-based dynamic leverage."""

from __future__ import annotations

import pytest

from src.strategies.components.leverage_manager import (
    DEFAULT_LEVERAGE_MAP,
    LeverageManager,
    LeverageState,
)
from src.strategies.components.regime_context import RegimeContext, TrendLabel, VolLabel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_regime(
    trend: TrendLabel = TrendLabel.RANGE,
    vol: VolLabel = VolLabel.LOW,
    confidence: float = 0.8,
    duration: int = 20,
    strength: float = 0.7,
) -> RegimeContext:
    """Create a RegimeContext with sensible defaults."""
    return RegimeContext(
        trend=trend,
        volatility=vol,
        confidence=confidence,
        duration=duration,
        strength=strength,
    )


# ---------------------------------------------------------------------------
# Construction / Validation
# ---------------------------------------------------------------------------


class TestLeverageManagerInit:
    """Test LeverageManager construction and parameter validation."""

    def test_default_construction(self) -> None:
        mgr = LeverageManager()
        assert mgr.max_leverage == 3.0
        assert 0.0 < mgr.decay_rate <= 1.0
        assert mgr.min_regime_bars >= 0

    def test_custom_parameters(self) -> None:
        mgr = LeverageManager(max_leverage=2.0, decay_rate=0.3, min_regime_bars=10)
        assert mgr.max_leverage == 2.0
        assert mgr.decay_rate == 0.3
        assert mgr.min_regime_bars == 10

    def test_invalid_max_leverage_raises(self) -> None:
        with pytest.raises(ValueError, match="max_leverage must be positive"):
            LeverageManager(max_leverage=0)
        with pytest.raises(ValueError, match="max_leverage must be positive"):
            LeverageManager(max_leverage=-1)

    def test_invalid_decay_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="decay_rate must be in"):
            LeverageManager(decay_rate=0.0)
        with pytest.raises(ValueError, match="decay_rate must be in"):
            LeverageManager(decay_rate=1.5)

    def test_invalid_min_regime_bars_raises(self) -> None:
        with pytest.raises(ValueError, match="min_regime_bars must be non-negative"):
            LeverageManager(min_regime_bars=-1)

    def test_leverage_map_clamped_to_max(self) -> None:
        """Leverage map values exceeding max_leverage are clamped."""
        mgr = LeverageManager(max_leverage=1.5)
        for value in mgr.leverage_map.values():
            assert value <= 1.5

    def test_custom_leverage_map(self) -> None:
        custom = {
            (TrendLabel.TREND_UP, VolLabel.LOW): 5.0,
            (TrendLabel.RANGE, VolLabel.LOW): 1.0,
        }
        mgr = LeverageManager(max_leverage=4.0, leverage_map=custom)
        # 5.0 should be clamped to 4.0
        assert mgr.leverage_map[(TrendLabel.TREND_UP, VolLabel.LOW)] == 4.0
        assert mgr.leverage_map[(TrendLabel.RANGE, VolLabel.LOW)] == 1.0


# ---------------------------------------------------------------------------
# Leverage multiplier for each regime
# ---------------------------------------------------------------------------


class TestRegimeLeverage:
    """Test leverage values for each regime combination."""

    def test_bull_low_vol_high_leverage(self) -> None:
        """Bull + low vol should produce leverage > 1.0."""
        mgr = LeverageManager(decay_rate=1.0)  # instant transition
        regime = _make_regime(TrendLabel.TREND_UP, VolLabel.LOW, duration=30)
        lev = mgr.get_leverage_multiplier(regime)
        assert lev > 1.0

    def test_bull_high_vol_moderate_leverage(self) -> None:
        """Bull + high vol should produce moderate leverage."""
        mgr = LeverageManager(decay_rate=1.0)
        regime = _make_regime(TrendLabel.TREND_UP, VolLabel.HIGH, duration=30)
        lev = mgr.get_leverage_multiplier(regime)
        assert lev > 1.0
        # Should be less than bull + low vol
        mgr2 = LeverageManager(decay_rate=1.0)
        bull_low = mgr2.get_leverage_multiplier(
            _make_regime(TrendLabel.TREND_UP, VolLabel.LOW, duration=30)
        )
        assert lev < bull_low

    def test_range_neutral_leverage(self) -> None:
        """Range regime should produce leverage near 1.0."""
        mgr = LeverageManager(decay_rate=1.0)
        regime = _make_regime(TrendLabel.RANGE, VolLabel.LOW, duration=30)
        lev = mgr.get_leverage_multiplier(regime)
        assert 0.9 <= lev <= 1.1

    def test_mild_bear_reduced_leverage(self) -> None:
        """Bear + low vol should produce leverage < 1.0."""
        mgr = LeverageManager(decay_rate=1.0)
        regime = _make_regime(TrendLabel.TREND_DOWN, VolLabel.LOW, duration=30)
        lev = mgr.get_leverage_multiplier(regime)
        assert lev < 1.0

    def test_bear_high_vol_near_zero(self) -> None:
        """Bear + high vol should produce leverage near 0.0."""
        mgr = LeverageManager(decay_rate=1.0)
        regime = _make_regime(TrendLabel.TREND_DOWN, VolLabel.HIGH, duration=30)
        lev = mgr.get_leverage_multiplier(regime)
        assert lev < 0.5


# ---------------------------------------------------------------------------
# Smooth transitions
# ---------------------------------------------------------------------------


class TestSmoothTransitions:
    """Test that leverage transitions smoothly between regimes."""

    def test_no_instant_jump(self) -> None:
        """Switching from bull to bear should not jump instantly."""
        mgr = LeverageManager(decay_rate=0.15)

        # Establish bull regime
        bull = _make_regime(TrendLabel.TREND_UP, VolLabel.LOW, duration=30)
        for _ in range(20):
            mgr.get_leverage_multiplier(bull)
        bull_lev = mgr.current_leverage
        assert bull_lev > 1.5  # Should be well above neutral

        # Switch to bear
        bear = _make_regime(TrendLabel.TREND_DOWN, VolLabel.HIGH, duration=1)
        lev_after_switch = mgr.get_leverage_multiplier(bear)

        # Should not jump to near-zero instantly
        assert lev_after_switch > 0.5
        # But should be moving toward target
        assert lev_after_switch < bull_lev

    def test_converges_over_time(self) -> None:
        """Leverage should converge to target over multiple bars."""
        mgr = LeverageManager(decay_rate=0.2)

        regime = _make_regime(TrendLabel.TREND_UP, VolLabel.LOW, duration=50)
        values = []
        for _ in range(50):
            values.append(mgr.get_leverage_multiplier(regime))

        # Should be increasing toward target
        assert values[-1] > values[0]
        # Should approach but not exceed max_leverage
        assert values[-1] <= mgr.max_leverage

    def test_decay_rate_1_instant_transition(self) -> None:
        """With decay_rate=1.0, transition should be immediate."""
        mgr = LeverageManager(decay_rate=1.0)
        regime = _make_regime(TrendLabel.RANGE, VolLabel.LOW, duration=50)
        lev = mgr.get_leverage_multiplier(regime)
        # With high conviction range regime, should be near 1.0
        assert 0.9 <= lev <= 1.1

    def test_gradual_ramp_up(self) -> None:
        """Leverage should gradually ramp when entering a new regime."""
        mgr = LeverageManager(decay_rate=0.1, min_regime_bars=3)

        # Start from neutral
        prev = mgr.current_leverage
        regime = _make_regime(TrendLabel.TREND_UP, VolLabel.LOW, duration=20)
        for i in range(10):
            curr = mgr.get_leverage_multiplier(regime)
            # Each step should move toward target (increasing for bull)
            assert curr >= prev - 0.01  # Allow tiny float rounding
            prev = curr


# ---------------------------------------------------------------------------
# Regime duration conviction
# ---------------------------------------------------------------------------


class TestConviction:
    """Test that conviction increases with regime duration."""

    def test_short_duration_low_conviction(self) -> None:
        """Very short regime duration produces lower leverage."""
        mgr = LeverageManager(decay_rate=1.0, min_regime_bars=10)
        short = _make_regime(TrendLabel.TREND_UP, VolLabel.LOW, duration=2)
        lev_short = mgr.get_leverage_multiplier(short)

        mgr2 = LeverageManager(decay_rate=1.0, min_regime_bars=10)
        long = _make_regime(TrendLabel.TREND_UP, VolLabel.LOW, duration=50)
        lev_long = mgr2.get_leverage_multiplier(long)

        assert lev_short < lev_long

    def test_zero_duration_minimal_leverage(self) -> None:
        """Duration 0 should produce leverage near neutral."""
        mgr = LeverageManager(decay_rate=1.0, min_regime_bars=5)
        regime = _make_regime(TrendLabel.TREND_UP, VolLabel.LOW, duration=0)
        lev = mgr.get_leverage_multiplier(regime)
        # Should be near 1.0 (neutral) due to zero conviction
        assert 0.9 <= lev <= 1.1

    def test_low_confidence_reduces_conviction(self) -> None:
        """Low regime confidence should reduce effective leverage."""
        mgr1 = LeverageManager(decay_rate=1.0)
        high_conf = _make_regime(
            TrendLabel.TREND_UP, VolLabel.LOW, confidence=0.9, duration=30
        )
        lev_high = mgr1.get_leverage_multiplier(high_conf)

        mgr2 = LeverageManager(decay_rate=1.0)
        low_conf = _make_regime(
            TrendLabel.TREND_UP, VolLabel.LOW, confidence=0.2, duration=30
        )
        lev_low = mgr2.get_leverage_multiplier(low_conf)

        assert lev_low < lev_high

    def test_conviction_plateaus(self) -> None:
        """Very long durations should not produce excessively different leverage."""
        mgr1 = LeverageManager(decay_rate=1.0)
        d100 = _make_regime(TrendLabel.TREND_UP, VolLabel.LOW, duration=100)
        lev_100 = mgr1.get_leverage_multiplier(d100)

        mgr2 = LeverageManager(decay_rate=1.0)
        d1000 = _make_regime(TrendLabel.TREND_UP, VolLabel.LOW, duration=1000)
        lev_1000 = mgr2.get_leverage_multiplier(d1000)

        # Should be very close (within 10%)
        assert abs(lev_1000 - lev_100) / max(lev_100, 0.01) < 0.10


# ---------------------------------------------------------------------------
# Bounds and caps
# ---------------------------------------------------------------------------


class TestBounds:
    """Test that leverage stays within configured bounds."""

    def test_never_exceeds_max_leverage(self) -> None:
        """Leverage should never exceed max_leverage."""
        mgr = LeverageManager(max_leverage=2.0, decay_rate=1.0)
        regime = _make_regime(
            TrendLabel.TREND_UP, VolLabel.LOW, confidence=1.0, duration=1000
        )
        for _ in range(100):
            lev = mgr.get_leverage_multiplier(regime)
            assert lev <= mgr.max_leverage + 1e-9

    def test_never_goes_negative(self) -> None:
        """Leverage should never go below 0.0."""
        mgr = LeverageManager(decay_rate=1.0)
        regime = _make_regime(
            TrendLabel.TREND_DOWN, VolLabel.HIGH, confidence=1.0, duration=1000
        )
        for _ in range(100):
            lev = mgr.get_leverage_multiplier(regime)
            assert lev >= 0.0

    def test_max_leverage_cap_with_high_conviction(self) -> None:
        """Even perfect conditions should respect max_leverage."""
        mgr = LeverageManager(max_leverage=1.5, decay_rate=1.0)
        regime = _make_regime(
            TrendLabel.TREND_UP, VolLabel.LOW, confidence=1.0, duration=500
        )
        lev = mgr.get_leverage_multiplier(regime)
        assert lev <= 1.5 + 1e-9


# ---------------------------------------------------------------------------
# Reset and state
# ---------------------------------------------------------------------------


class TestState:
    """Test state management and reset."""

    def test_reset_returns_to_defaults(self) -> None:
        mgr = LeverageManager(decay_rate=1.0)
        regime = _make_regime(TrendLabel.TREND_UP, VolLabel.LOW, duration=30)
        mgr.get_leverage_multiplier(regime)
        assert mgr.current_leverage != 1.0

        mgr.reset()
        assert mgr.current_leverage == 1.0
        assert mgr.target_leverage == 1.0

    def test_get_parameters(self) -> None:
        mgr = LeverageManager(max_leverage=2.5, decay_rate=0.2, min_regime_bars=8)
        params = mgr.get_parameters()
        assert params["max_leverage"] == 2.5
        assert params["decay_rate"] == 0.2
        assert params["min_regime_bars"] == 8
        assert isinstance(params["leverage_map"], dict)

    def test_regime_change_resets_duration(self) -> None:
        """Switching regimes should reset internal duration tracking."""
        mgr = LeverageManager(decay_rate=0.5)

        bull = _make_regime(TrendLabel.TREND_UP, VolLabel.LOW, duration=20)
        for _ in range(5):
            mgr.get_leverage_multiplier(bull)

        # Switch regime
        bear = _make_regime(TrendLabel.TREND_DOWN, VolLabel.HIGH, duration=1)
        mgr.get_leverage_multiplier(bear)

        assert mgr._state.regime_bars_held == 0
        assert mgr._state.last_trend == TrendLabel.TREND_DOWN
