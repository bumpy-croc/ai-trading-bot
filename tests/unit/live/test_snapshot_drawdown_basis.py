"""Stored ``account_history.drawdown`` shares the hard cap's equity basis (#1267)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.engines.live.execution.position_tracker import LivePosition, PositionSide
from src.engines.live.logging.event_logger import LiveEventLogger
from src.engines.live.monitoring import LiveAccountMonitor

pytestmark = pytest.mark.fast


def _logger() -> tuple[LiveEventLogger, MagicMock]:
    db_manager = MagicMock()
    db_manager.get_first_snapshot_of_day.return_value = None
    return LiveEventLogger(db_manager=db_manager, log_to_database=True, session_id=1), db_manager


def _underwater_position(unrealized_pnl: float) -> dict[str, LivePosition]:
    position = LivePosition(
        symbol="ETHUSDT",
        side=PositionSide.LONG,
        size=0.2,
        entry_price=100.0,
        entry_time=datetime.now(UTC),
        entry_balance=100.0,
        order_id="dd-1",
        unrealized_pnl=unrealized_pnl,
    )
    return {"dd-1": position}


def test_drawdown_column_moves_with_unrealized_loss_when_peak_equity_given():
    event_logger, db_manager = _logger()

    event_logger.log_account_snapshot(
        balance=100.0,
        positions=_underwater_position(-12.0),
        total_pnl=0.0,
        peak_balance=100.0,
        peak_equity=100.0,
    )

    assert db_manager.log_account_snapshot.call_args.kwargs["drawdown"] == pytest.approx(12.0)


def test_drawdown_column_is_the_worse_of_balance_and_equity_legs():
    event_logger, db_manager = _logger()

    # Balance is 20% off its peak; open profit lifts equity to only 15% off.
    event_logger.log_account_snapshot(
        balance=80.0,
        positions=_underwater_position(5.0),
        total_pnl=-20.0,
        peak_balance=100.0,
        peak_equity=100.0,
    )

    assert db_manager.log_account_snapshot.call_args.kwargs["drawdown"] == pytest.approx(20.0)


def test_drawdown_column_is_balance_based_without_peak_equity():
    event_logger, db_manager = _logger()

    event_logger.log_account_snapshot(
        balance=100.0,
        positions=_underwater_position(-12.0),
        total_pnl=0.0,
        peak_balance=100.0,
    )

    assert db_manager.log_account_snapshot.call_args.kwargs["drawdown"] == pytest.approx(0.0)


def _monitor_state(peak_equity) -> MagicMock:
    state = MagicMock()
    state.current_balance = 100.0
    perf = MagicMock()
    perf.total_pnl = 0.0
    perf.peak_balance = 100.0
    state.performance_tracker.get_metrics.return_value = perf
    state.live_position_tracker.positions = {}
    state._durable_peak_equity.return_value = peak_equity
    return state


def test_monitor_forwards_the_guards_peak_equity():
    state = _monitor_state(112.5)

    LiveAccountMonitor(engine_state=state).log_account_snapshot()

    assert state.event_logger.log_account_snapshot.call_args.kwargs["peak_equity"] == 112.5


@pytest.mark.parametrize("bad", [0.0, float("nan"), "n/a", None])
def test_monitor_drops_an_unusable_peak_equity(bad):
    state = _monitor_state(bad)

    LiveAccountMonitor(engine_state=state).log_account_snapshot()

    assert state.event_logger.log_account_snapshot.call_args.kwargs["peak_equity"] is None
