"""Unit tests for LiveDynamicRiskCoordinator (#486).

The dynamic-risk sizing adjustment + audit logging were extracted from
LiveTradingEngine into ``LiveDynamicRiskCoordinator`` (the engine keeps thin
delegating wrappers, still called by the trading loop and by
``LiveEntryCoordinator`` via ``state._apply_dynamic_risk_adjustment``). These
tests drive the coordinator directly via an autospec'd engine-state backref.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, create_autospec

import pytest

from src.engines.live.dynamic_risk_coordinator import (
    LiveDynamicRiskCoordinator,
    LiveDynamicRiskEngineState,
)

pytestmark = pytest.mark.fast

# Sentinel: default means "an enabled manager" (a fresh mock); pass None to disable.
_ENABLED = object()


def _make_state(*, manager=_ENABLED, session_id=1, balance=1000.0) -> MagicMock:
    """Autospec'd engine-state backref with dynamic-risk defaults."""
    state = create_autospec(LiveDynamicRiskEngineState, instance=True)
    state.dynamic_risk_manager = MagicMock() if manager is _ENABLED else manager
    state.current_balance = balance
    state.initial_balance = 1000.0
    state.trading_session_id = session_id
    state._durable_peak_balance.return_value = 1200.0
    state.db_manager = MagicMock()
    state._dynamic_risk_handler = MagicMock()
    state._dynamic_risk_handler.get_adjustment_objects.return_value = []
    return state


def test_apply_passthrough_when_manager_disabled():
    state = _make_state(manager=None)
    result = LiveDynamicRiskCoordinator(state).apply_dynamic_risk_adjustment(0.1, datetime.now(UTC))
    assert result == 0.1
    state._dynamic_risk_handler.apply_dynamic_risk.assert_not_called()


def test_apply_returns_handler_adjusted_size_and_logs():
    state = _make_state()
    state._dynamic_risk_handler.apply_dynamic_risk.return_value = 0.05

    result = LiveDynamicRiskCoordinator(state).apply_dynamic_risk_adjustment(0.1, datetime.now(UTC))

    assert result == 0.05
    # peak_balance passed = max(durable peak, balance) = max(1200, 1000) = 1200
    kwargs = state._dynamic_risk_handler.apply_dynamic_risk.call_args.kwargs
    assert kwargs["balance"] == 1000.0
    assert kwargs["peak_balance"] == 1200.0
    assert kwargs["trading_session_id"] == 1
    # adjustment objects were drained for logging
    state._dynamic_risk_handler.get_adjustment_objects.assert_called_once_with(clear=True)


def test_apply_peak_never_below_balance_when_durable_source_lags():
    """A durable peak of 0.0 (guard not yet seeded, tracker empty) must not
    produce a phantom 100% drawdown — the balance floors the peak."""
    state = _make_state()
    state._durable_peak_balance.return_value = 0.0
    state._dynamic_risk_handler.apply_dynamic_risk.return_value = 0.08

    LiveDynamicRiskCoordinator(state).apply_dynamic_risk_adjustment(0.1, datetime.now(UTC))

    kwargs = state._dynamic_risk_handler.apply_dynamic_risk.call_args.kwargs
    assert kwargs["peak_balance"] == 1000.0


def test_apply_falls_back_to_original_on_error():
    state = _make_state()
    state._dynamic_risk_handler.apply_dynamic_risk.side_effect = RuntimeError("boom")
    result = LiveDynamicRiskCoordinator(state).apply_dynamic_risk_adjustment(0.1, datetime.now(UTC))
    assert result == 0.1  # original size preserved on failure


def test_apply_uses_initial_balance_when_current_nonpositive():
    state = _make_state(balance=0.0)
    state._dynamic_risk_handler.apply_dynamic_risk.return_value = 0.07
    LiveDynamicRiskCoordinator(state).apply_dynamic_risk_adjustment(0.1, datetime.now(UTC))
    kwargs = state._dynamic_risk_handler.apply_dynamic_risk.call_args.kwargs
    assert kwargs["balance"] == 1000.0  # fell back to initial_balance


def test_log_writes_db_row_per_adjustment_when_session_active():
    state = _make_state(session_id=7)
    adj = MagicMock(
        position_size_factor=0.5,
        primary_reason="drawdown_reduction",
        original_size=0.1,
        adjusted_size=0.05,
        current_drawdown=0.2,
    )
    state._dynamic_risk_handler.get_adjustment_objects.return_value = [adj]

    LiveDynamicRiskCoordinator(state).log_dynamic_risk_adjustments()

    state.db_manager.log_risk_adjustment.assert_called_once()
    kwargs = state.db_manager.log_risk_adjustment.call_args.kwargs
    assert kwargs["session_id"] == 7
    assert kwargs["adjustment_type"] == "drawdown"  # split on "_"
    assert kwargs["adjusted_value"] == 0.5


def test_log_skips_db_when_no_session():
    state = _make_state(session_id=None)
    adj = MagicMock(primary_reason="volatility_scale")
    state._dynamic_risk_handler.get_adjustment_objects.return_value = [adj]

    LiveDynamicRiskCoordinator(state).log_dynamic_risk_adjustments()

    state.db_manager.log_risk_adjustment.assert_not_called()
