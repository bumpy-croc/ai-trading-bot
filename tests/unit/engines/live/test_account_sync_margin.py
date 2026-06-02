"""Unit tests for margin balance reconciliation (prevention layer 2).

Margin mode used to skip balance sync entirely, so realized losses that were
never booked to the tracked balance drifted from true equity unnoticed (a ~15%
gap was observed in production). These cover the reconcile that uses the
exchange's account-level net equity (`get_account_equity`) and only corrects
while flat.
"""

from unittest.mock import Mock, patch

import pytest

from src.engines.live.account_sync import AccountSynchronizer

pytestmark = pytest.mark.unit


@patch("src.data_providers.binance_provider.Client")
@patch("src.data_providers.binance_provider.get_config")
def test_get_account_equity_margin_uses_total_net_asset(mock_config, mock_client_class):
    """Margin equity = totalNetAssetOfBtc (net of liabilities) valued in USDT."""
    from src.data_providers.binance_provider import BinanceProvider

    mock_config.return_value = Mock(get_required=Mock(return_value="fake_key"))
    mock_client = Mock()
    mock_client_class.return_value = mock_client
    mock_client.get_margin_account.return_value = {"totalNetAssetOfBtc": "0.00125"}
    mock_client.get_symbol_ticker.return_value = {"price": "67000.0"}

    provider = BinanceProvider()
    provider._use_margin = True

    assert provider.get_account_equity() == pytest.approx(0.00125 * 67000.0)


@patch("src.data_providers.binance_provider.Client")
@patch("src.data_providers.binance_provider.get_config")
def test_get_account_equity_spot_uses_usdt_total(mock_config, mock_client_class):
    from src.data_providers.binance_provider import BinanceProvider

    mock_config.return_value = Mock(get_required=Mock(return_value="fake_key"))
    mock_client_class.return_value = Mock()

    provider = BinanceProvider()
    provider._use_margin = False
    provider.get_balance = Mock(return_value=Mock(total=84.5))

    assert provider.get_account_equity() == pytest.approx(84.5)


def _make_sync(equity, db_balance, usdt_total):
    exchange = Mock()
    exchange.get_account_equity.return_value = equity
    exchange.get_balance.return_value = (
        Mock(total=usdt_total) if usdt_total is not None else None
    )
    db = Mock()
    db.get_current_balance.return_value = db_balance
    sync = AccountSynchronizer(
        exchange=exchange, db_manager=db, session_id=1, use_margin=True
    )
    return sync, db


def test_margin_equity_corrects_when_flat_and_divergent():
    """Flat (equity == USDT cash) + >1% divergence -> correct down to true equity."""
    sync, db = _make_sync(equity=84.22, db_balance=99.89, usdt_total=84.22)

    res = sync._sync_margin_equity()

    assert res["corrected"] is True
    assert res["new_balance"] == pytest.approx(84.22)
    assert res["old_balance"] == pytest.approx(99.89)
    db.update_balance.assert_called_once()


def test_margin_equity_no_correct_within_threshold():
    """Flat but divergence under the 1% threshold -> no correction."""
    sync, db = _make_sync(equity=99.50, db_balance=99.89, usdt_total=99.50)

    res = sync._sync_margin_equity()

    assert res["corrected"] is False
    db.update_balance.assert_not_called()


def test_margin_equity_no_correct_when_position_held():
    """Divergent AND a position is held (equity != USDT) -> warn, do NOT correct.

    Guards against an open/unreconciled position folding its value into cash.
    """
    sync, db = _make_sync(equity=84.22, db_balance=99.89, usdt_total=65.41)

    res = sync._sync_margin_equity()

    assert res.get("corrected") is not True
    assert res["reason"] == "position held"
    db.update_balance.assert_not_called()


def test_margin_equity_no_sync_when_equity_unavailable():
    """Equity unreadable -> no sync, no correction (fail safe)."""
    sync, db = _make_sync(equity=None, db_balance=99.89, usdt_total=99.89)

    res = sync._sync_margin_equity()

    assert res["synced"] is False
    db.update_balance.assert_not_called()


def test_margin_equity_no_correct_when_equity_zero_or_negative():
    """Non-positive equity is treated as unavailable -> no correction."""
    sync, db = _make_sync(equity=0.0, db_balance=99.89, usdt_total=99.89)

    res = sync._sync_margin_equity()

    assert res["synced"] is False
    db.update_balance.assert_not_called()
