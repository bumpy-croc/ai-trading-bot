"""Unit tests for margin side_effect_type wiring through order placement paths.

Verifies that:
- Long entries pass side_effect_type=None (use existing USDT, no borrow)
- Short entries pass side_effect_type="MARGIN_BUY" (borrow asset to sell)
- All exits pass side_effect_type="AUTO_REPAY" (repay margin debt)
- Stop-loss orders pass side_effect_type="AUTO_REPAY"
- AccountSynchronizer skips balance/position sync in margin mode
- PeriodicReconciler skips asset holdings check in margin mode
"""

from unittest.mock import MagicMock

import pytest

from src.engines.shared.models import PositionSide

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_live_execution_engine(*, exchange: MagicMock | None = None):
    """Create a LiveExecutionEngine with mocked exchange interface."""
    from src.engines.live.execution.execution_engine import LiveExecutionEngine

    mock_exchange = exchange or MagicMock()
    mock_exchange.place_order.return_value = MagicMock(
        order_id="test_order_123",
        average_price=50000.0,
        filled_quantity=0.001,
        commission=0.0,
        status="FILLED",
    )

    engine = LiveExecutionEngine(
        enable_live_trading=True,
        exchange_interface=mock_exchange,
    )
    engine.db_manager = MagicMock()
    engine.session_id = 1
    engine.strategy_name = "test_strategy"
    return engine


# ---------------------------------------------------------------------------
# C1: execution_engine.py — Entry and exit order calls
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_execution_engine_entry_long_no_side_effect():
    """Long entry (BUY) should pass side_effect_type=None — no margin borrow needed."""
    engine = _make_live_execution_engine()
    engine._normalize_quantity = MagicMock(return_value=0.001)

    engine._execute_live_order(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        price=50000.0,
        value=50.0,
    )

    engine.exchange_interface.place_order.assert_called_once()
    call_kwargs = engine.exchange_interface.place_order.call_args.kwargs
    assert call_kwargs.get("side_effect_type") is None


@pytest.mark.fast
def test_execution_engine_entry_short_margin_buy():
    """Short entry (SELL) should pass side_effect_type='MARGIN_BUY' — borrow asset to sell."""
    engine = _make_live_execution_engine()
    engine._normalize_quantity = MagicMock(return_value=0.001)

    engine._execute_live_order(
        symbol="BTCUSDT",
        side=PositionSide.SHORT,
        price=50000.0,
        value=50.0,
    )

    engine.exchange_interface.place_order.assert_called_once()
    call_kwargs = engine.exchange_interface.place_order.call_args.kwargs
    assert call_kwargs.get("side_effect_type") == "MARGIN_BUY"


@pytest.mark.fast
def test_execution_engine_exit_auto_repay():
    """Exit orders should pass side_effect_type='AUTO_REPAY' — repay margin debt."""
    engine = _make_live_execution_engine()
    engine._normalize_quantity = MagicMock(return_value=0.001)

    engine._close_live_order(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        quantity=0.001,
        position_notional=50.0,
        order_id="entry_order_123",
        position_db_id=42,
    )

    engine.exchange_interface.place_order.assert_called_once()
    call_kwargs = engine.exchange_interface.place_order.call_args.kwargs
    assert call_kwargs.get("side_effect_type") == "AUTO_REPAY"


# ---------------------------------------------------------------------------
# C2: trading_engine.py — Stop-loss placement
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_trading_engine_stop_loss_auto_repay():
    """Stop-loss call in trading_engine._execute_entry includes side_effect_type='AUTO_REPAY'.

    The stop-loss is placed inline in _execute_entry with retry logic. Rather than
    constructing the full entry flow (which requires extensive mocking), we verify
    the source code contains the kwarg on the place_stop_loss_order call.
    """
    import inspect

    from src.engines.live.trading_engine import LiveTradingEngine

    source = inspect.getsource(LiveTradingEngine._execute_entry)
    assert 'side_effect_type="AUTO_REPAY"' in source, (
        "trading_engine._execute_entry must pass side_effect_type='AUTO_REPAY' "
        "to place_stop_loss_order"
    )


# ---------------------------------------------------------------------------
# C4: account_sync.py — Skip balance/position sync in margin mode
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_account_sync_skips_balance_and_position_in_margin_mode():
    """AccountSynchronizer skips both balance and position sync in margin mode.

    USDT netAsset excludes cross-asset liabilities (borrowed ETH from shorts),
    so syncing USDT alone would inflate capital and oversize the next trade.
    """
    from src.engines.live.account_sync import AccountSynchronizer

    mock_exchange = MagicMock()
    mock_exchange.sync_account_data.return_value = {
        "sync_successful": True,
        "balances": [],
        "positions": [],
        "open_orders": [],
    }
    mock_db = MagicMock()

    syncer = AccountSynchronizer(
        exchange=mock_exchange,
        db_manager=mock_db,
        session_id=1,
        use_margin=True,
    )

    result = syncer.sync_account_data(force=True)

    assert result.success is True
    assert result.data["balance_sync"]["synced"] is False
    assert "margin" in result.data["balance_sync"]["reason"]
    assert result.data["position_sync"]["synced"] is False
    assert "margin" in result.data["position_sync"]["reason"]


@pytest.mark.fast
def test_account_sync_runs_normally_without_margin():
    """AccountSynchronizer should run balance/position sync when use_margin=False."""
    from src.engines.live.account_sync import AccountSynchronizer

    mock_exchange = MagicMock()
    mock_exchange.sync_account_data.return_value = {
        "sync_successful": True,
        "balances": [],
        "positions": [],
        "open_orders": [],
    }
    mock_db = MagicMock()

    syncer = AccountSynchronizer(
        exchange=mock_exchange,
        db_manager=mock_db,
        session_id=1,
        use_margin=False,
    )

    # Patch internal sync methods to track calls
    syncer._sync_balances = MagicMock(return_value={"synced": True})
    syncer._sync_positions = MagicMock(return_value={"synced": True})
    syncer._sync_orders = MagicMock(return_value={"synced": True})

    syncer.sync_account_data(force=True)

    syncer._sync_balances.assert_called_once()
    syncer._sync_positions.assert_called_once()


# ---------------------------------------------------------------------------
# C3: reconciliation.py — PeriodicReconciler margin mode + stop-loss AUTO_REPAY
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_reconciler_accepts_use_margin_param():
    """PeriodicReconciler should accept and store use_margin parameter."""
    from src.engines.live.reconciliation import PeriodicReconciler

    reconciler = PeriodicReconciler(
        exchange_interface=MagicMock(),
        position_tracker=MagicMock(),
        db_manager=MagicMock(),
        session_id=1,
        use_margin=True,
    )

    assert reconciler._use_margin is True


@pytest.mark.fast
def test_reconciler_skips_spot_checks_in_margin_mode():
    """PeriodicReconciler should skip asset holdings check in margin mode."""
    from src.engines.live.reconciliation import PeriodicReconciler

    mock_exchange = MagicMock()
    mock_exchange.get_order.return_value = None  # No orders found
    mock_exchange.get_open_orders.return_value = []
    mock_tracker = MagicMock()
    mock_db = MagicMock()

    # Create a position that would trigger asset holdings check
    mock_position = MagicMock()
    mock_position.symbol = "BTCUSDT"
    mock_position.stop_loss = 48000.0
    mock_position.stop_loss_order_id = None
    mock_position.entry_price = 50000.0
    mock_position.side = PositionSide.LONG
    mock_position.quantity = 0.001
    mock_position.current_size = 1.0
    mock_position.original_size = 1.0
    mock_position.exchange_order_id = "test_123"
    mock_position.db_position_id = 42

    mock_tracker.positions = {"BTCUSDT_test": mock_position}

    reconciler = PeriodicReconciler(
        exchange_interface=mock_exchange,
        position_tracker=mock_tracker,
        db_manager=mock_db,
        session_id=1,
        use_margin=True,
    )

    # Run a reconciliation cycle
    reconciler._reconcile_cycle()

    # get_balance should NOT be called — spot-specific asset check is skipped
    mock_exchange.get_balance.assert_not_called()


@pytest.mark.fast
def test_reconciler_stop_loss_has_auto_repay():
    """Stop-loss placement in _place_missing_stop_loss should use AUTO_REPAY."""
    from src.engines.live.reconciliation import PeriodicReconciler

    mock_exchange = MagicMock()
    mock_exchange.place_stop_loss_order.return_value = "new_sl_123"
    mock_db = MagicMock()

    reconciler = PeriodicReconciler(
        exchange_interface=mock_exchange,
        position_tracker=MagicMock(),
        db_manager=mock_db,
        session_id=1,
    )

    mock_position = MagicMock()
    mock_position.symbol = "BTCUSDT"
    mock_position.stop_loss = 48000.0
    mock_position.entry_price = 50000.0
    mock_position.side = "long"
    mock_position.quantity = 0.001
    mock_position.current_size = 1.0
    mock_position.original_size = 1.0
    mock_position.db_position_id = 42

    reconciler._place_missing_stop_loss(mock_position, "test_order_key")

    mock_exchange.place_stop_loss_order.assert_called_once()
    call_kwargs = mock_exchange.place_stop_loss_order.call_args.kwargs
    assert call_kwargs.get("side_effect_type") == "AUTO_REPAY"


# ---------------------------------------------------------------------------
# C5: PositionReconciler (startup) — use_margin guard on _verify_asset_holdings
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_startup_reconciler_accepts_use_margin():
    """PositionReconciler must accept use_margin param and store it."""
    from src.engines.live.reconciliation import PositionReconciler

    reconciler = PositionReconciler(
        exchange_interface=MagicMock(),
        position_tracker=MagicMock(),
        db_manager=MagicMock(),
        session_id=1,
        max_position_size=0.1,
        use_margin=True,
    )
    assert reconciler._use_margin is True


@pytest.mark.fast
def test_startup_reconciler_defaults_margin_false():
    """PositionReconciler defaults to use_margin=False."""
    from src.engines.live.reconciliation import PositionReconciler

    reconciler = PositionReconciler(
        exchange_interface=MagicMock(),
        position_tracker=MagicMock(),
        db_manager=MagicMock(),
        session_id=1,
    )
    assert reconciler._use_margin is False


@pytest.mark.fast
def test_startup_reconciler_skips_verify_asset_holdings_in_margin_mode():
    """_verify_asset_holdings should be a no-op in margin mode."""
    from src.engines.live.reconciliation import (
        PositionReconciler,
        ReconciliationResult,
    )

    exchange = MagicMock()
    reconciler = PositionReconciler(
        exchange_interface=exchange,
        position_tracker=MagicMock(),
        db_manager=MagicMock(),
        session_id=1,
        use_margin=True,
    )

    position = MagicMock()
    position.symbol = "BTCUSDT"
    position.side = "long"
    position.quantity = 1.0
    position.current_size = 1.0
    position.original_size = 1.0

    result = ReconciliationResult(
        entity_type="position",
        entity_id="BTCUSDT",
        status="ok",
        corrections=[],
    )

    reconciler._verify_asset_holdings(position, result)
    exchange.get_balance.assert_not_called()


# ---------------------------------------------------------------------------
# C6: get_balances() / get_balance() — netAsset in margin mode
# ---------------------------------------------------------------------------


def _make_binance_provider(*, use_margin: bool = False):
    """Create a BinanceProvider with mocked client and optional margin mode."""
    from unittest.mock import patch

    with patch("src.data_providers.binance_provider.get_config") as mock_config:
        mock_config_obj = MagicMock()
        mock_config_obj.get.return_value = None
        mock_config_obj.get_required.return_value = "fake_key"
        mock_config.return_value = mock_config_obj

        with patch("src.data_providers.binance_provider.Client"):
            from src.data_providers.binance_provider import BinanceProvider

            provider = BinanceProvider(
                api_key="test_key_1234567890abcdef",
                api_secret="test_secret_1234567890abcdef",
                testnet=True,
            )
            provider._use_margin = use_margin
            return provider


@pytest.mark.fast
def test_margin_get_balances_uses_net_asset():
    """In margin mode, total should be netAsset, not free+locked."""
    provider = _make_binance_provider(use_margin=True)
    provider._call_get_account = MagicMock(return_value={
        "balances": [
            {
                "asset": "BTC",
                "free": "1.5",
                "locked": "0.5",
                # netAsset = free + locked - borrowed - interest = 1.0
                "netAsset": "1.0",
            },
            {
                "asset": "USDT",
                "free": "10000.0",
                "locked": "0.0",
                "netAsset": "8000.0",
            },
        ]
    })

    balances = provider.get_balances()
    btc = next(b for b in balances if b.asset == "BTC")
    usdt = next(b for b in balances if b.asset == "USDT")

    assert btc.total == 1.0
    assert usdt.total == 8000.0
    assert btc.free == 1.5
    assert btc.locked == 0.5


@pytest.mark.fast
def test_spot_get_balances_uses_free_plus_locked():
    """In spot mode, total should remain free+locked (no regression)."""
    provider = _make_binance_provider(use_margin=False)
    provider._call_get_account = MagicMock(return_value={
        "balances": [
            {"asset": "BTC", "free": "1.5", "locked": "0.5", "netAsset": "1.0"},
        ]
    })

    balances = provider.get_balances()
    btc = next(b for b in balances if b.asset == "BTC")
    assert btc.total == 2.0  # free + locked, NOT netAsset


@pytest.mark.fast
def test_margin_get_balance_uses_net_asset():
    """In margin mode, get_balance() should use netAsset for total."""
    provider = _make_binance_provider(use_margin=True)
    provider._call_get_account = MagicMock(return_value={
        "balances": [
            {"asset": "BTC", "free": "1.5", "locked": "0.5", "netAsset": "1.0"},
        ]
    })

    balance = provider.get_balance("BTC")
    assert balance is not None
    assert balance.total == 1.0


@pytest.mark.fast
def test_spot_get_balance_uses_free_plus_locked():
    """In spot mode, get_balance() should use free+locked."""
    provider = _make_binance_provider(use_margin=False)
    provider._call_get_account = MagicMock(return_value={
        "balances": [
            {"asset": "BTC", "free": "1.5", "locked": "0.5"},
        ]
    })

    balance = provider.get_balance("BTC")
    assert balance is not None
    assert balance.total == 2.0


@pytest.mark.fast
def test_margin_get_balances_fallback_when_no_net_asset():
    """If netAsset is missing in margin mode, fall back to free+locked."""
    provider = _make_binance_provider(use_margin=True)
    provider._call_get_account = MagicMock(return_value={
        "balances": [
            {"asset": "BTC", "free": "1.5", "locked": "0.5"},
        ]
    })

    balances = provider.get_balances()
    btc = next(b for b in balances if b.asset == "BTC")
    assert btc.total == 2.0  # Fallback: free + locked
