"""#1240: a close that normalizes to zero records one durable event per cause."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.engines.live.execution.execution_engine import LiveExecutionEngine
from src.engines.shared.models import PositionSide

SYMBOL = "ETHUSDT"
PRICE = 2000.0


class _Harness:
    def __init__(self, *, free_base: float, symbol_info: dict):
        self.exchange = Mock()
        self.exchange.get_symbol_info.return_value = symbol_info
        balance = Mock()
        balance.free = free_base
        self.exchange.get_balance.return_value = balance
        self.engine = LiveExecutionEngine(
            enable_live_trading=True, exchange_interface=self.exchange
        )
        self.events: list[tuple[str, dict]] = []
        self.alerts: list[str] = []
        self.engine._log_execution_event = (  # type: ignore[method-assign]
            lambda event_type, message, error_code, *, severity="error", details=None: (
                self.events.append((error_code, {"severity": severity, **(details or {})}))
            )
        )
        self.engine.alert_dispatcher = self.alerts.append

    def close(self, quantity: float):
        return self.engine._close_live_order(
            symbol=SYMBOL,
            side=PositionSide.LONG,
            quantity=quantity,
            position_notional=quantity * PRICE,
            reference_price=PRICE,
        )

    def unsellable_events(self) -> list[dict]:
        return [d for code, d in self.events if code == "CLOSE_QUANTITY_UNSELLABLE"]


def _info(**overrides) -> dict:
    return {"step_size": 0.0001, "min_qty": 0.0, "min_notional": 5.0, **overrides}


def test_min_notional_abort_records_one_critical_event_and_books_no_close():
    h = _Harness(free_base=0.001, symbol_info=_info())  # $2 < $5

    for _ in range(4):
        assert h.close(0.001) is None

    h.exchange.place_order.assert_not_called()
    [event] = h.unsellable_events()
    assert event["abort_reason"] == "min_notional"
    assert event["severity"] == "critical"
    assert event["min_notional"] == 5.0
    assert event["notional"] == pytest.approx(2.0)
    assert event["intended_quantity"] == pytest.approx(0.001)
    assert len(h.alerts) == 1


def test_lot_sizing_abort_is_distinguished():
    h = _Harness(free_base=0.00005, symbol_info=_info(min_notional=0.0))

    assert h.close(0.00005) is None

    [event] = h.unsellable_events()
    assert event["abort_reason"] == "lot_sizing"


def test_min_qty_abort_is_distinguished():
    h = _Harness(free_base=0.001, symbol_info=_info(min_qty=0.01, min_notional=0.0))

    assert h.close(0.001) is None

    [event] = h.unsellable_events()
    assert event["abort_reason"] == "min_qty"


def test_zero_free_balance_is_attributed_to_holdings_cap():
    h = _Harness(free_base=0.0, symbol_info=_info())

    assert h.close(0.5) is None

    [event] = h.unsellable_events()
    assert event["abort_reason"] == "holdings_cap"
    assert event["free_base_balance"] == 0.0


def test_changed_reason_records_a_new_event():
    h = _Harness(free_base=0.001, symbol_info=_info())
    h.close(0.001)
    h.exchange.get_symbol_info.return_value = _info(min_notional=0.0, min_qty=0.01)

    h.close(0.001)

    assert [e["abort_reason"] for e in h.unsellable_events()] == ["min_notional", "min_qty"]


def test_event_rearms_after_a_close_clears_the_gate():
    h = _Harness(free_base=0.001, symbol_info=_info())
    h.close(0.001)
    h.exchange.get_symbol_info.return_value = _info(min_notional=0.0)
    h.close(0.001)  # gate passes; the condition cleared
    h.exchange.get_symbol_info.return_value = _info()

    h.close(0.001)

    assert len(h.unsellable_events()) == 2


def test_alert_failure_does_not_raise():
    h = _Harness(free_base=0.001, symbol_info=_info())
    h.engine.alert_dispatcher = Mock(side_effect=RuntimeError("boom"))

    assert h.close(0.001) is None
