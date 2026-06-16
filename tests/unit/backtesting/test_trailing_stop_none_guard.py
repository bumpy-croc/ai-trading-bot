"""Regression tests for #761: trailing-stop update with a None stop price.

``TrailingStopManager.update`` can return ``updated=True`` with
``new_stop_price=None`` (trailing just activated without a stop improvement,
e.g. ATR unavailable on the activation candle). The backtest
``PositionTracker.update_trailing_stop`` previously compared that ``None``
against the current stop (``None > float`` → ``TypeError``), crashing the
whole backtest run. It now mirrors the live tracker: flag updates apply, the
price comparison is skipped.
"""

from datetime import UTC, datetime

import pytest

from src.engines.backtest.execution.position_tracker import PositionTracker
from src.engines.backtest.models import ActiveTrade
from src.engines.shared.models import PositionSide

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _open_tracker(stop_loss: float | None) -> PositionTracker:
    tracker = PositionTracker()
    tracker.open_position(
        ActiveTrade(
            symbol="TEST",
            side=PositionSide.LONG,
            entry_price=100.0,
            entry_time=datetime(2024, 1, 1, tzinfo=UTC),
            size=0.1,
            stop_loss=stop_loss,
        )
    )
    return tracker


class TestUpdateTrailingStopNoneGuard:
    def test_none_stop_with_existing_stop_does_not_raise(self):
        """Regression: None vs existing float stop crashed with TypeError."""
        tracker = _open_tracker(stop_loss=95.0)

        changed = tracker.update_trailing_stop(
            new_stop_loss=None, activated=True, breakeven_triggered=False
        )

        trade = tracker.current_trade
        assert trade is not None
        assert changed is True  # activation flag changed
        assert trade.stop_loss == 95.0  # price untouched
        assert trade.trailing_stop_activated is True

    def test_none_stop_without_existing_stop_keeps_none(self):
        """Activation without a price must not fabricate a stop price."""
        tracker = _open_tracker(stop_loss=None)

        changed = tracker.update_trailing_stop(
            new_stop_loss=None, activated=True, breakeven_triggered=False
        )

        trade = tracker.current_trade
        assert trade is not None
        assert changed is True
        assert trade.stop_loss is None
        assert trade.trailing_stop_price is None
        assert trade.trailing_stop_activated is True

    def test_none_stop_with_no_flag_change_reports_unchanged(self):
        tracker = _open_tracker(stop_loss=95.0)

        changed = tracker.update_trailing_stop(
            new_stop_loss=None, activated=False, breakeven_triggered=False
        )

        assert changed is False

    def test_real_stop_improvement_still_applies(self):
        """Existing improvement semantics are untouched."""
        tracker = _open_tracker(stop_loss=95.0)

        changed = tracker.update_trailing_stop(
            new_stop_loss=97.0, activated=True, breakeven_triggered=False
        )

        trade = tracker.current_trade
        assert trade is not None
        assert changed is True
        assert trade.stop_loss == 97.0
        assert trade.trailing_stop_price == 97.0

    def test_worse_stop_is_rejected_but_flags_apply(self):
        tracker = _open_tracker(stop_loss=95.0)

        changed = tracker.update_trailing_stop(
            new_stop_loss=90.0, activated=True, breakeven_triggered=True
        )

        trade = tracker.current_trade
        assert trade is not None
        assert changed is True  # flags changed
        assert trade.stop_loss == 95.0  # worse stop rejected for a long
