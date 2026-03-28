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
def test_account_sync_skips_in_margin_mode():
    """AccountSynchronizer should skip balance/position sync when use_margin=True."""
    from src.engines.live.account_sync import AccountSynchronizer

    mock_exchange = MagicMock()
    mock_exchange.sync_account_data.return_value = {
        "sync_successful": True,
        "balances": [{"asset": "USDT", "free": "1000"}],
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
