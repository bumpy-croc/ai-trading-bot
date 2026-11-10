"""Tests for src/trading/close_sizing.cap_closing_sell_quantity (#989).

Direct unit tests for the shared guard the three emergency-close sites route
through, independent of any single call site's surrounding logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.trading.close_sizing import cap_closing_sell_quantity

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _exchange(*, free_base: float | None, step_size: float = 0.00001):
    exchange = MagicMock()
    exchange.get_symbol_info.return_value = {
        "step_size": step_size,
        "min_qty": step_size,
        "min_notional": 1.0,
    }
    if free_base is None:
        exchange.get_balance.return_value = None
    else:
        balance = MagicMock()
        balance.free = free_base
        exchange.get_balance.return_value = balance
    return exchange


def test_caps_to_free_base_when_commission_eats_base():
    """Audit case: held 0.0099 after the entry fee haircut vs. an intended 0.01
    (commission taken from the base fill) -> sized down to what's held."""
    exchange = _exchange(free_base=0.0099)

    result = cap_closing_sell_quantity(exchange, symbol="BTCUSDT", quantity=0.01)

    exchange.get_balance.assert_called_once_with("BTC")
    assert result <= 0.0099
    assert result == pytest.approx(0.0099)


def test_floors_instead_of_rounding_up_past_holdings():
    """A nearest-lot snap that would round UP past the free balance must floor."""
    exchange = _exchange(free_base=0.049978, step_size=0.00001)

    result = cap_closing_sell_quantity(exchange, symbol="BTCUSDT", quantity=0.05)

    assert result <= 0.049978
    assert result == pytest.approx(0.04997)


def test_uncapped_when_holdings_are_sufficient():
    """Free balance already covers the intended quantity -- no cap applied
    beyond the (no-op) lot-step floor."""
    exchange = _exchange(free_base=1.0)

    result = cap_closing_sell_quantity(exchange, symbol="BTCUSDT", quantity=0.01)

    assert result == pytest.approx(0.01)


def test_aborts_when_holdings_locked_far_below_intended():
    """Free balance covers far less than the intended close -> refuse (0.0),
    never submit a partial that would book a full close on success."""
    exchange = _exchange(free_base=0.001)

    result = cap_closing_sell_quantity(exchange, symbol="BTCUSDT", quantity=0.01)

    assert result == 0.0


def test_zero_free_base_aborts():
    exchange = _exchange(free_base=0.0)

    result = cap_closing_sell_quantity(exchange, symbol="BTCUSDT", quantity=0.01)

    assert result == 0.0


def test_balance_lookup_failure_fails_open():
    """A transient balance-lookup failure must not block an already-critical
    emergency close -- skip the cap rather than aborting on a lookup error."""
    exchange = _exchange(free_base=None)
    exchange.get_balance.side_effect = ConnectionError("balance unavailable")

    result = cap_closing_sell_quantity(exchange, symbol="BTCUSDT", quantity=0.01)

    assert result == pytest.approx(0.01)


def test_missing_symbol_info_fails_open_on_lot_step():
    """Symbol info unavailable -> skip lot-step flooring, keep the (possibly
    balance-capped) quantity rather than blocking the close."""
    exchange = _exchange(free_base=1.0)
    exchange.get_symbol_info.return_value = None

    result = cap_closing_sell_quantity(exchange, symbol="BTCUSDT", quantity=0.01)

    assert result == pytest.approx(0.01)


def test_non_positive_quantity_returns_zero():
    exchange = _exchange(free_base=1.0)

    assert cap_closing_sell_quantity(exchange, symbol="BTCUSDT", quantity=0.0) == 0.0
    assert cap_closing_sell_quantity(exchange, symbol="BTCUSDT", quantity=-0.01) == 0.0


def test_no_exchange_interface_returns_zero():
    assert cap_closing_sell_quantity(None, symbol="BTCUSDT", quantity=0.01) == 0.0
