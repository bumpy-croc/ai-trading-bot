"""Same-iteration enforcement of the max-drawdown hard cap at entry chokepoints.

The loop-level ``_check_max_drawdown`` runs AFTER entry evaluation, so on the
exact bar where a stop-loss fill pushed drawdown to the cap, a fresh entry
could execute before close-only latched (one-iteration leak; 2026-07-12 risk
audit P1). Every exposure-increase chokepoint therefore calls the in-line
pre-order gate ``_refresh_drawdown_gate`` first — mirroring the #807 in-line
circuit-breaker gate — so the cap binds on the SAME iteration.

Fixture mirrors the audit's numeric trace: session peak $100 from
account_history, a stop-loss fill drops the balance to $80 (exactly the 20%
cap) mid-iteration, and the entry evaluated later in that iteration must be
blocked.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytestmark = pytest.mark.fast


def _entry_df() -> pd.DataFrame:
    return pd.DataFrame({"close": [50000.0]})


def _no_entry_signal() -> MagicMock:
    signal = MagicMock()
    signal.should_enter = False
    signal.side = None
    return signal


def _runtime_decision() -> MagicMock:
    decision = MagicMock()
    decision.signal.strength = 0.8
    decision.signal.confidence = 0.7
    decision.regime = None
    decision.risk_metrics = None
    decision.metadata = {}
    return decision


def _check_entry(engine) -> None:
    engine.entry_coordinator.check_entry_conditions(
        df=_entry_df(),
        current_index=0,
        symbol="BTCUSDT",
        current_price=50000.0,
        current_time=datetime(2024, 1, 1, tzinfo=UTC),
        runtime_decision=_runtime_decision(),
    )


def test_entry_evaluation_blocked_on_the_breach_bar(make_live_engine):
    """Audit P1 trace: SL fill drops balance to the cap mid-iteration → the
    SAME iteration's entry evaluation must be blocked, not the next one."""
    engine = make_live_engine(db_peak=100.0, resumed_balance=100.0)
    engine.current_balance = 80.0  # stop-loss realized earlier this iteration
    engine.live_entry_handler = MagicMock()
    engine.live_entry_handler.process_runtime_decision.return_value = _no_entry_signal()

    with patch.object(engine, "_is_runtime_strategy", return_value=True):
        _check_entry(engine)

    assert engine._close_only_mode is True
    engine.live_entry_handler.process_runtime_decision.assert_not_called()
    engine.strategy.process_candle.assert_not_called()


def test_entry_evaluation_proceeds_below_the_cap(make_live_engine):
    """No false positive: 15% drawdown is below the cap — evaluation runs."""
    engine = make_live_engine(db_peak=100.0, resumed_balance=100.0)
    engine.current_balance = 85.0
    engine.live_entry_handler = MagicMock()
    engine.live_entry_handler.process_runtime_decision.return_value = _no_entry_signal()

    with patch.object(engine, "_is_runtime_strategy", return_value=True):
        _check_entry(engine)

    assert engine._close_only_mode is False
    engine.live_entry_handler.process_runtime_decision.assert_called_once()


def test_execute_entry_locked_blocked_on_the_breach_bar(make_live_engine):
    """The execution chokepoint re-assesses the guard too, so direct callers
    and the legacy short path cannot add exposure on the breach bar."""
    from src.engines.shared.models import PositionSide

    engine = make_live_engine(db_peak=100.0, resumed_balance=100.0)
    engine.current_balance = 80.0
    engine.live_entry_handler = MagicMock()

    engine.entry_coordinator.execute_entry_locked(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=0.2,
        price=50000.0,
        stop_loss=45000.0,
        take_profit=55000.0,
        signal_strength=0.8,
        signal_confidence=0.7,
    )

    assert engine._close_only_mode is True
    engine.live_entry_handler.execute_entry.assert_not_called()


def test_legacy_short_evaluation_blocked_on_the_breach_bar(make_live_engine):
    engine = make_live_engine(db_peak=100.0, resumed_balance=100.0)
    engine.current_balance = 80.0
    engine.strategy.check_short_entry_conditions = MagicMock(return_value=True)

    engine.entry_coordinator.process_legacy_short_entry(
        df=_entry_df(),
        current_index=0,
        symbol="BTCUSDT",
        current_price=50000.0,
        current_time=datetime(2024, 1, 1, tzinfo=UTC),
    )

    assert engine._close_only_mode is True
    engine.strategy.check_short_entry_conditions.assert_not_called()


def test_scale_in_close_only_provider_reassesses_drawdown(make_live_engine):
    """Scale-ins read close-only through the engine's provider; it must run
    the in-line gate so a breach earlier in the iteration blocks them too."""
    engine = make_live_engine(db_peak=100.0, resumed_balance=100.0)
    engine.current_balance = 80.0

    assert engine.live_exit_handler._close_only_provider() is True
    assert engine._close_only_mode is True


def test_scale_in_provider_stays_open_below_the_cap(make_live_engine):
    engine = make_live_engine(db_peak=100.0, resumed_balance=100.0)
    engine.current_balance = 85.0

    assert engine.live_exit_handler._close_only_provider() is False
    assert engine._close_only_mode is False


def test_resume_mid_breach_reblocks_before_the_next_entry_evaluation(make_live_engine):
    """resume_trading() while still in breach must not leak even one entry
    evaluation: the in-line gate re-latches close-only before the check."""
    engine = make_live_engine(db_peak=100.0, resumed_balance=100.0)
    engine.current_balance = 80.0
    engine.live_entry_handler = MagicMock()
    engine.live_entry_handler.process_runtime_decision.return_value = _no_entry_signal()
    engine._check_max_drawdown()  # loop check trips close-only
    assert engine._close_only_mode is True

    engine.resume_trading()  # operator resumes while drawdown persists
    assert engine._close_only_mode is False

    with patch.object(engine, "_is_runtime_strategy", return_value=True):
        _check_entry(engine)

    assert engine._close_only_mode is True
    engine.live_entry_handler.process_runtime_decision.assert_not_called()
