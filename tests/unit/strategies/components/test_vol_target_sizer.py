"""Unit tests for volatility-targeted position sizing (#805)."""

from __future__ import annotations

import pytest

from src.strategies.components.position_sizer import (
    VolatilityTargetSizer,
    maybe_wrap_with_vol_target,
)
from src.strategies.components.signal_generator import Signal, SignalDirection

pytestmark = pytest.mark.unit


class _Base:
    """Constant base sizer returning a fixed notional."""

    def __init__(self, size=100.0):
        self._size = size
        self.warmup_period = 7

    def calculate_size(self, signal, balance, risk_amount, regime=None):
        return self._size

    def get_feature_generators(self):
        return ["fg"]


class _Regime:
    def __init__(self, atr_percentile):
        self.metadata = {"atr_percentile": atr_percentile}


def _sig():
    return Signal(SignalDirection.BUY, 0.8, 0.8, {})


# --- multiplier ------------------------------------------------------------


def test_high_vol_reduces_multiplier():
    sizer = VolatilityTargetSizer(_Base())
    assert sizer.volatility_multiplier(_Regime(0.9)) < 1.0


def test_median_vol_is_unit_multiplier():
    sizer = VolatilityTargetSizer(_Base())
    assert sizer.volatility_multiplier(_Regime(0.5)) == pytest.approx(1.0)


def test_low_vol_boosts_but_is_capped():
    sizer = VolatilityTargetSizer(_Base(), max_multiplier=1.5)
    # target 0.5 / max(0.02, 0.1) = 5.0, capped to 1.5
    assert sizer.volatility_multiplier(_Regime(0.02)) == pytest.approx(1.5)


def test_multiplier_floor_applied():
    sizer = VolatilityTargetSizer(_Base(), min_multiplier=0.6)
    # target 0.5 / 1.0 = 0.5, floored to 0.6
    assert sizer.volatility_multiplier(_Regime(1.0)) == pytest.approx(0.6)


def test_multiplier_none_when_no_metadata():
    sizer = VolatilityTargetSizer(_Base())
    assert sizer.volatility_multiplier(None) is None

    class NoMeta:
        metadata = None

    assert sizer.volatility_multiplier(NoMeta()) is None


def test_multiplier_none_on_non_finite_atr():
    sizer = VolatilityTargetSizer(_Base())
    assert sizer.volatility_multiplier(_Regime(float("nan"))) is None


# --- calculate_size --------------------------------------------------------


def test_size_scaled_down_in_high_vol():
    sizer = VolatilityTargetSizer(_Base(size=100.0))
    sized = sizer.calculate_size(_sig(), 1000.0, 20.0, _Regime(0.9))
    assert sized < 100.0
    assert sized > 0.0


def test_size_passthrough_without_regime():
    sizer = VolatilityTargetSizer(_Base(size=100.0))
    assert sizer.calculate_size(_sig(), 1000.0, 20.0, None) == 100.0


def test_zero_base_size_unchanged():
    sizer = VolatilityTargetSizer(_Base(size=0.0))
    assert sizer.calculate_size(_sig(), 1000.0, 20.0, _Regime(0.9)) == 0.0


def test_warmup_and_feature_generators_delegate():
    sizer = VolatilityTargetSizer(_Base())
    assert sizer.warmup_period == 7
    assert sizer.get_feature_generators() == ["fg"]


# --- validation ------------------------------------------------------------


def test_rejects_bad_target_percentile():
    with pytest.raises(ValueError, match="target_atr_percentile"):
        VolatilityTargetSizer(_Base(), target_atr_percentile=1.5)


def test_rejects_inverted_multiplier_bounds():
    with pytest.raises(ValueError, match="min_multiplier"):
        VolatilityTargetSizer(_Base(), min_multiplier=2.0, max_multiplier=1.0)


# --- factory flag gating ---------------------------------------------------


def test_maybe_wrap_respects_flag(monkeypatch):
    base = _Base()
    monkeypatch.delenv("FEATURE_ENABLE_VOL_TARGET_SIZING", raising=False)
    assert maybe_wrap_with_vol_target(base) is base
    monkeypatch.setenv("FEATURE_ENABLE_VOL_TARGET_SIZING", "true")
    assert isinstance(maybe_wrap_with_vol_target(base), VolatilityTargetSizer)
