"""Regression tests for #626: SHORT-guard rejection must not leak a journal row.

Before the fix, `_execute_live_order` wrote a `PENDING_SUBMIT` journal row BEFORE
the SHORT inventory guard, so every guard-rejected short leaked an orphan row
(4,786 accumulated in production and jammed startup reconciliation). The guard
now runs before journaling.
"""

from unittest.mock import Mock

import pytest

from src.engines.live.execution.execution_engine import LiveExecutionEngine
from src.engines.shared.models import PositionSide


def _engine(balance_free: float) -> tuple[LiveExecutionEngine, Mock]:
    exchange = Mock()
    exchange.is_margin_mode = True
    exchange.get_balance.return_value = Mock(free=balance_free)
    exchange.place_order.return_value = None
    engine = LiveExecutionEngine(enable_live_trading=True, exchange_interface=exchange)
    engine.db_manager = Mock()
    engine.session_id = 1
    engine.strategy_name = "test"
    # Bypass exchange LOT_SIZE/notional lookups — not under test here.
    engine._normalize_quantity = Mock(return_value=0.05)
    return engine, exchange


@pytest.mark.fast
def test_short_rejected_by_inventory_guard_does_not_journal():
    """A short blocked by the inventory guard must NOT write a journal row (#626)."""
    # free base value = 0.05 * 2000 = $100 (> $1 dust threshold) -> guard rejects.
    engine, exchange = _engine(balance_free=0.05)

    result = engine._execute_live_order(
        symbol="ETHUSDT", side=PositionSide.SHORT, value=100.0, price=2000.0
    )

    assert result == (None, None)
    engine.db_manager.create_order_journal_entry.assert_not_called()  # no leaked row
    exchange.place_order.assert_not_called()  # order never sent


@pytest.mark.fast
def test_short_accepted_when_no_inventory_is_journaled():
    """With no significant inventory the guard passes and the order is journaled."""
    engine, exchange = _engine(balance_free=0.0)  # no inventory -> guard passes

    engine._execute_live_order(symbol="ETHUSDT", side=PositionSide.SHORT, value=100.0, price=2000.0)

    # Journal anchor is still written for an order that is actually sent.
    engine.db_manager.create_order_journal_entry.assert_called_once()
    exchange.place_order.assert_called_once()


@pytest.mark.fast
def test_long_entry_is_journaled():
    """Long entries are unaffected — journaled then placed (no guard)."""
    engine, exchange = _engine(balance_free=0.0)

    engine._execute_live_order(symbol="ETHUSDT", side=PositionSide.LONG, value=100.0, price=2000.0)

    engine.db_manager.create_order_journal_entry.assert_called_once()
    exchange.place_order.assert_called_once()
