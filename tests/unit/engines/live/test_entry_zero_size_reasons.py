"""Zero-size entry decisions are explained in logs and strategy_executions (#1045)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.engines.live.execution.entry_coordinator import LiveEntryCoordinator
from src.engines.live.execution.entry_handler import LiveEntrySignal
from src.strategies.components import SignalDirection

pytestmark = [pytest.mark.unit, pytest.mark.fast, pytest.mark.mock_only]


@dataclass
class _Signal:
    direction: SignalDirection
    strength: float = 0.5
    confidence: float = 0.02
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Decision:
    signal: _Signal
    position_size: float
    metadata: dict[str, Any]
    regime: Any = None
    risk_metrics: Any = None


def _state(handler_result: LiveEntrySignal) -> MagicMock:
    state = MagicMock()
    state._is_runtime_strategy.return_value = True
    state._close_only_mode = False
    state.current_balance = 1000.0
    state.max_position_size = 0.5
    state.timeframe = "1h"
    state.trading_session_id = 7
    state.db_manager = MagicMock()
    state._extract_indicators.return_value = {}
    state._extract_sentiment_data.return_value = {}
    state._extract_ml_predictions.return_value = {}
    state.live_entry_handler.process_runtime_decision.return_value = handler_result
    state.live_position_tracker.has_position_for_symbol.return_value = False
    state.live_position_tracker.position_count = 0
    state.risk_manager.get_max_concurrent_positions.return_value = 1
    return state


def _check_entry(state: MagicMock, decision: _Decision) -> None:
    df = pd.DataFrame(
        {"open": [2000.0], "high": [2010.0], "low": [1990.0], "close": [2000.0], "volume": [1.0]},
        index=pd.date_range("2026-07-14", periods=1, freq="1h"),
    )
    LiveEntryCoordinator(state).check_entry_conditions(
        df=df,
        current_index=0,
        symbol="ETHUSDT",
        current_price=2000.0,
        current_time=pd.Timestamp("2026-07-14T12:00:00Z").to_pydatetime(),
        runtime_decision=decision,
    )


def _logged_reasons(state: MagicMock) -> list[str]:
    return state.db_manager.log_strategy_execution.call_args.kwargs["reasons"]


def test_strategy_zero_reason_is_persisted_with_the_execution_row():
    state = _state(LiveEntrySignal(should_enter=False, reasons=["runtime_hold", "balance_1000.00"]))
    decision = _Decision(
        signal=_Signal(SignalDirection.BUY),
        position_size=0.0,
        metadata={"size_zero_reason": "risk_manager(flat):confidence_0.020_below_min_0.050"},
    )

    _check_entry(state, decision)

    reasons = _logged_reasons(state)
    assert "size_zero_risk_manager(flat):confidence_0.020_below_min_0.050" in reasons
    assert "runtime_hold" in reasons
    assert "no_position_size" in reasons


def test_gate_that_zeroes_a_sized_entry_is_logged_and_persisted(caplog):
    state = _state(
        LiveEntrySignal(
            should_enter=False,
            reasons=["exposure_cap_reached_0.1000", "size_reduced_to_zero"],
        )
    )
    decision = _Decision(
        signal=_Signal(SignalDirection.BUY, confidence=0.4), position_size=100.0, metadata={}
    )

    with caplog.at_level(logging.INFO):
        _check_entry(state, decision)

    assert "Entry size reduced to zero for ETHUSDT: exposure_cap_reached_0.1000" in caplog.text
    assert "exposure_cap_reached_0.1000" in _logged_reasons(state)


def test_plain_hold_adds_no_zero_size_noise(caplog):
    state = _state(LiveEntrySignal(should_enter=False, reasons=["runtime_hold"]))
    decision = _Decision(signal=_Signal(SignalDirection.HOLD), position_size=0.0, metadata={})

    with caplog.at_level(logging.INFO):
        _check_entry(state, decision)

    assert "reduced to zero" not in caplog.text
    assert not any(r.startswith("size_zero_") for r in _logged_reasons(state))
