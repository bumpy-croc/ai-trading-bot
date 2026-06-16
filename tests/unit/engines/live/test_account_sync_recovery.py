"""Unit tests for AccountSynchronizer.recover_missing_trades.

Regression guard for a signature-mismatch bug: recover_missing_trades called
``DatabaseManager.log_trade`` with ``order_id=...``, but log_trade's parameter
is ``exit_order_id`` (it has no ``order_id`` kwarg and no ``**kwargs``). Against
the real signature this raised ``TypeError`` at runtime whenever a missing trade
was actually recovered (the ``emergency_sync`` path), and the error was silently
swallowed by the per-trade try/except -- so trades failed to recover with no
crash.

The DatabaseManager is ``create_autospec``'d so the mock enforces the real
log_trade signature: the old ``order_id=...`` call raises ``TypeError`` here
exactly as it did in production, and a successful recovery proves the fixed
call is signature-compatible.
"""

from datetime import UTC, datetime
from unittest.mock import Mock, create_autospec

import pytest

from src.database.manager import DatabaseManager
from src.engines.live.account_sync import AccountSynchronizer

pytestmark = pytest.mark.fast


def _make_exchange_trade():
    """A single exchange trade with an order id, not present in the DB."""
    trade = Mock()
    trade.trade_id = "T-123"
    trade.order_id = "O-456"
    trade.symbol = "BTCUSDT"
    trade.side = Mock(value="BUY")
    trade.price = 50_000.0
    trade.quantity = 0.01
    trade.time = datetime.now(UTC)
    return trade


def test_recover_missing_trades_logs_with_exit_order_id():
    """A missing exchange trade is recovered via ``log_trade(exit_order_id=...)``.

    With an autospec'd DatabaseManager, calling log_trade with the old wrong
    kwarg (``order_id=...``) raises TypeError exactly as in production, so
    ``recovered_trades == 1`` proves the call is signature-compatible and does
    not raise.
    """
    exchange = Mock()
    exchange.get_recent_trades.return_value = [_make_exchange_trade()]

    db = create_autospec(DatabaseManager, instance=True)
    db.get_trades_by_symbol_and_date.return_value = []  # nothing logged yet

    sync = AccountSynchronizer(exchange=exchange, db_manager=db, session_id=1)

    result = sync.recover_missing_trades("BTCUSDT")

    # 0 if log_trade raised TypeError and was swallowed by the per-trade except.
    assert result["recovered_trades"] == 1

    db.log_trade.assert_called_once()
    call_kwargs = db.log_trade.call_args.kwargs
    assert call_kwargs["exit_order_id"] == "O-456"
    assert "order_id" not in call_kwargs
