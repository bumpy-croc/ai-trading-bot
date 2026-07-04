"""Unit tests for shared gross-exposure accounting and the pre-order gate (#802)."""

from __future__ import annotations

import pytest

from src.engines.shared.execution.entry_handler_mixin import SharedEntryHandlerMixin
from src.engines.shared.exposure import gross_exposure_fraction, position_notional
from src.strategies.components.exposure_governor import ExposureGovernor

pytestmark = pytest.mark.unit


class _Pos:
    def __init__(self, size, entry_balance):
        self.size = size
        self.entry_balance = entry_balance


class _Regime:
    class trend:
        value = "trend_down"

    class volatility:
        value = "high_vol"


# --- exposure helper -------------------------------------------------------


def test_position_notional_size_times_entry_balance():
    assert position_notional(_Pos(0.25, 1000.0)) == 250.0


def test_position_notional_missing_fields_is_zero():
    class Bad:
        pass

    assert position_notional(Bad()) == 0.0


def test_position_notional_non_finite_is_zero():
    assert position_notional(_Pos(float("inf"), 1000.0)) == 0.0


def test_position_notional_absolute_value():
    # A short recorded with negative size still contributes positive gross.
    assert position_notional(_Pos(-0.25, 1000.0)) == 250.0


def test_gross_exposure_fraction_sums_over_positions():
    positions = [_Pos(0.2, 1000.0), _Pos(0.1, 1000.0)]  # 200 + 100 = 300
    assert gross_exposure_fraction(positions, 1500.0) == pytest.approx(0.2)


def test_gross_exposure_fraction_zero_equity_is_zero():
    assert gross_exposure_fraction([_Pos(0.2, 1000.0)], 0.0) == 0.0


def test_gross_exposure_fraction_empty():
    assert gross_exposure_fraction([], 1000.0) == 0.0


# --- mixin gate ------------------------------------------------------------


class _Handler(SharedEntryHandlerMixin):
    """Minimal concrete handler exercising only the shared gate."""

    def __init__(self, governor=None, positions=None):
        self.configure_exposure_gate(
            governor, (lambda: positions) if positions is not None else None
        )


def test_gate_inert_when_no_governor():
    h = _Handler()
    assert h.apply_pre_order_gates(0.3, regime=None, equity=1000.0) == (0.3, None)


def test_gate_inert_when_disabled():
    h = _Handler(governor=ExposureGovernor(enabled=False), positions=[])
    assert h.apply_pre_order_gates(0.3, regime=_Regime(), equity=1000.0) == (0.3, None)


def test_gate_caps_using_current_exposure():
    # bear high_vol cap 0.15; existing exposure 0.10 (100/1000); propose 0.10
    positions = [_Pos(0.10, 1000.0)]
    h = _Handler(governor=ExposureGovernor(enabled=True), positions=positions)
    allowed, reason = h.apply_pre_order_gates(0.10, regime=_Regime(), equity=1000.0)
    assert allowed == pytest.approx(0.05)
    assert reason is not None


def test_gate_regime_none_uses_conservative_cap():
    h = _Handler(governor=ExposureGovernor(enabled=True), positions=[])
    # unknown cap 0.15; propose 0.30 -> trimmed to 0.15
    allowed, reason = h.apply_pre_order_gates(0.30, regime=None, equity=1000.0)
    assert allowed == pytest.approx(0.15)
    assert reason is not None


def test_current_gross_exposure_degrades_to_zero_on_source_error():
    def boom():
        raise RuntimeError("tracker unavailable")

    h = _Handler()
    h.configure_exposure_gate(ExposureGovernor(enabled=True), boom)
    # Must not raise; exposure treated as 0 so entries are never broken by it.
    assert h.current_gross_exposure_fraction(1000.0) == 0.0


def test_parity_backtest_and_live_handlers_share_gate():
    """The backtest and live entry handlers must cap identically — guaranteed by
    both inheriting the same mixin method. Assert both classes resolve the same
    unbound method object (no per-engine override)."""
    from src.engines.backtest.execution.entry_handler import EntryHandler as BtHandler
    from src.engines.live.execution.entry_handler import LiveEntryHandler as LiveHandler

    assert BtHandler.apply_pre_order_gates is SharedEntryHandlerMixin.apply_pre_order_gates
    assert LiveHandler.apply_pre_order_gates is SharedEntryHandlerMixin.apply_pre_order_gates
    assert (
        BtHandler.current_gross_exposure_fraction
        is SharedEntryHandlerMixin.current_gross_exposure_fraction
    )
