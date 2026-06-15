"""Unit tests for LiveOrderFillCoordinator (#486).

The OrderTracker callbacks were extracted from LiveTradingEngine into
``LiveOrderFillCoordinator`` (the engine keeps thin delegating wrappers and
still registers those wrappers with OrderTracker). These tests drive the
coordinator directly via a mocked engine-state backref, covering the
deferred-close handoff, the stop-loss-cancel escalation seam, the cancel-refund
math, and the fail-closed tracking-lost contract. They also pin the CODE.md
hardening applied after the extraction (the cancel-refund uses an explicit
``quantity is not None`` guard rather than ``or 0.0``).
"""

from __future__ import annotations

import queue
from unittest.mock import MagicMock

import pytest

from src.database.models import EventType
from src.engines.live.execution.order_fill_coordinator import (
    LiveOrderFillCoordinator,
    LiveOrderFillEngineState,
)

pytestmark = pytest.mark.fast


def _make_position(*, stop_loss_order_id: str | None = None, entry_fee: float = 0.0, quantity=1.0):
    """Return a minimal tracked-position mock for callback tests."""
    pos = MagicMock()
    pos.stop_loss_order_id = stop_loss_order_id
    pos.quantity = quantity
    pos.metadata = {"entry_fee": entry_fee}
    return pos


def _make_state(*, positions: dict | None = None, session_id=None, balance: float = 1000.0):
    """Build a mocked engine-state backref with a real fill-exit queue."""
    state = MagicMock(spec=LiveOrderFillEngineState)
    state.current_balance = balance
    state.trading_session_id = session_id
    state.live_position_tracker = MagicMock()
    state.db_manager = MagicMock()
    state._pending_fill_exits = queue.SimpleQueue()
    state.live_position_tracker.positions = positions or {}
    return state


def test_order_fill_enqueues_stop_loss_close_for_the_loop():
    """A stop-loss full fill is deferred to the loop via the queue, not closed inline."""
    pos = _make_position(stop_loss_order_id="sl1")
    state = _make_state(positions={"pos1": pos})

    LiveOrderFillCoordinator(state).handle_order_fill("sl1", "BTCUSDT", 0.5, 100.0)

    assert state._pending_fill_exits.get_nowait() == ("pos1", 100.0)


def test_order_fill_non_stop_loss_enqueues_nothing():
    pos = _make_position(stop_loss_order_id="sl1")
    state = _make_state(positions={"pos1": pos})

    LiveOrderFillCoordinator(state).handle_order_fill("some-entry-order", "BTCUSDT", 0.5, 100.0)

    assert state._pending_fill_exits.empty()


def test_partial_stop_loss_fill_raises_critical_alert():
    pos = _make_position(stop_loss_order_id="sl1")
    state = _make_state(positions={"pos1": pos})

    LiveOrderFillCoordinator(state).handle_partial_fill("sl1", "BTCUSDT", 0.25, 100.0)

    state._send_alert.assert_called_once()


def test_partial_fill_non_stop_loss_no_alert():
    pos = _make_position(stop_loss_order_id="sl1")
    state = _make_state(positions={"pos1": pos})

    LiveOrderFillCoordinator(state).handle_partial_fill("entry-order", "BTCUSDT", 0.25, 100.0)

    state._send_alert.assert_not_called()


def test_stop_loss_cancelled_clears_id_and_escalates():
    pos = _make_position(stop_loss_order_id="sl1")
    state = _make_state(positions={"pos1": pos})

    matched = LiveOrderFillCoordinator(state).handle_stop_loss_cancelled("sl1", "BTCUSDT")

    assert matched is True
    state.live_position_tracker.set_stop_loss_order_id.assert_called_once_with("pos1", None)
    state._record_event.assert_called_once()
    assert state._record_event.call_args.kwargs["error_code"] == "STOP_LOSS_CANCELLED"
    assert state._record_event.call_args.kwargs["alert"] is True


def test_stop_loss_cancelled_no_match_returns_false():
    state = _make_state(positions={})

    matched = LiveOrderFillCoordinator(state).handle_stop_loss_cancelled("sl-unknown", "BTCUSDT")

    assert matched is False
    state.live_position_tracker.set_stop_loss_order_id.assert_not_called()
    state._record_event.assert_not_called()


def test_order_cancel_routes_stop_loss_escalation_and_returns_early():
    """If the cancelled order is a tracked stop-loss, escalate and do NOT pop a position."""
    pos = _make_position(stop_loss_order_id="sl1")
    state = _make_state(positions={"pos1": pos})

    LiveOrderFillCoordinator(state).handle_order_cancel("sl1", "BTCUSDT")

    state.live_position_tracker.set_stop_loss_order_id.assert_called_once_with("pos1", None)
    state.live_position_tracker.pop_position.assert_not_called()


def test_order_cancel_removes_phantom_and_refunds_full_fee_sessionless():
    removed = _make_position(entry_fee=5.0, quantity=1.0)
    state = _make_state(positions={}, session_id=None, balance=1000.0)
    state.live_position_tracker.pop_position.return_value = removed

    # filled_qty=0.0 -> fully unfilled -> full fee refunded; sessionless -> direct balance add.
    LiveOrderFillCoordinator(state).handle_order_cancel("entry-order", "BTCUSDT", filled_qty=0.0)

    assert state.current_balance == 1005.0


def test_order_cancel_partial_fill_refunds_only_unfilled_fraction_sessionless():
    removed = _make_position(entry_fee=10.0, quantity=1.0)
    state = _make_state(positions={}, session_id=None, balance=1000.0)
    state.live_position_tracker.pop_position.return_value = removed

    # 0.4 of 1.0 filled -> 60% unfilled -> refund 6.0 of the 10.0 fee.
    LiveOrderFillCoordinator(state).handle_order_cancel("entry-order", "BTCUSDT", filled_qty=0.4)

    assert state.current_balance == pytest.approx(1006.0)


def test_order_cancel_none_quantity_does_not_crash_and_refunds_full_fee():
    """Regression for the `or 0.0` -> `is not None` hardening: a None quantity
    must not raise and falls back to the full-fee refund path."""
    removed = _make_position(entry_fee=3.0, quantity=None)
    state = _make_state(positions={}, session_id=None, balance=1000.0)
    state.live_position_tracker.pop_position.return_value = removed

    LiveOrderFillCoordinator(state).handle_order_cancel("entry-order", "BTCUSDT", filled_qty=0.5)

    # original_qty resolves to 0.0 (not >0), so the full fee is refunded.
    assert state.current_balance == 1003.0


def test_tracking_lost_keeps_position_and_escalates():
    """Fail-closed: tracking-lost must NOT pop/refund; it keeps the position tracked."""
    state = _make_state(positions={})
    state.live_position_tracker.get_position.return_value = _make_position()

    LiveOrderFillCoordinator(state).handle_order_tracking_lost("order-1", "BTCUSDT", 3)

    state.live_position_tracker.pop_position.assert_not_called()
    state._record_event.assert_called_once()
    assert state._record_event.call_args.args[0] == EventType.ERROR
    assert state._record_event.call_args.kwargs["error_code"] == "ORDER_TRACKING_LOST"
    assert state._record_event.call_args.kwargs["alert"] is True
