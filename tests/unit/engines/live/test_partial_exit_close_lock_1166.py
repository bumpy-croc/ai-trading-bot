"""#1166: a partial exit that fully closes a position must route through the
same machinery as every other close — the #710 cancel-then-close sequence
(cancel a resting stop-loss before a market close, so the base asset is
actually free) and the #703 base-asset lock (serialises closes per base
asset).

``LiveExitHandler._execute_partial_exit`` used to call the handler's own raw
``execute_exit`` directly when a partial exit fully closed a position,
bypassing ``LiveExitCoordinator.execute_exit_locked`` (and therefore both the
cancel-then-close sequence and the lock) entirely. That is the same failure
class #1165 fixed for the primary close path, reachable through a second,
un-gated door.
"""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.fast

from src.engines.live.trading_engine import LiveTradingEngine, Position, PositionSide
from tests.mocks import MockDatabaseManager


@pytest.fixture(autouse=True)
def _fast_env(monkeypatch):
    """In-memory DB + no real sleeps (re-protect retry uses time.sleep)."""
    monkeypatch.setattr("src.engines.live.trading_engine.DatabaseManager", MockDatabaseManager)
    monkeypatch.setattr("src.engines.live.trading_engine.time.sleep", lambda *_a, **_k: None)


def _order(filled):
    """A stop order mock with the given filled base quantity (None -> unreadable)."""
    if filled is None:
        return None
    return Mock(filled_quantity=filled, status="NEW")


def _make_engine(sl_fills=(0.0, 0.0), cancel_returns=True):
    """A live-mode LiveTradingEngine with a mocked exchange and a REAL
    (unmocked) exit handler + exit coordinator, so the actual production
    wiring between them is exercised.

    Only the exchange interface and the low-level execution engine's
    ``execute_exit`` are stubbed; everything above that (handler, tracker,
    coordinator, base-asset lock) is the real production code. Returns
    ``(engine, calls)`` recording the cancel/close order.
    """
    strategy = Mock()
    strategy.get_risk_overrides.return_value = None
    data_provider = Mock()
    data_provider.get_current_price.return_value = 100.0

    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=data_provider,
        initial_balance=1_000.0,
        enable_live_trading=False,
        log_trades=False,
        fee_rate=0.0,
        slippage_rate=0.0,
    )

    engine.enable_live_trading = True
    exchange = Mock()
    exchange.is_margin_mode = True
    exchange.get_open_orders_checked.return_value = []
    engine.exchange_interface = exchange
    engine.order_tracker = Mock()
    engine.performance_tracker.record_trade = Mock()
    engine._check_stop_loss_filled = Mock(return_value=(False, None))

    fills = list(sl_fills)

    def _get_order(*_a, **_k):
        f = fills.pop(0) if len(fills) > 1 else fills[0]
        return _order(f)

    exchange.get_order.side_effect = _get_order

    calls: list[str] = []

    def _cancel(*_a, **_k):
        calls.append("cancel")
        return cancel_returns

    exchange.cancel_order.side_effect = _cancel

    # Stub only the lowest-level order placement so the close doesn't need a
    # real exchange fill; everything above it (handler.execute_exit,
    # coordinator, lock, cancel-then-close) runs for real.
    def _exec_exit(*_a, **_k):
        calls.append("close")
        return Mock(success=True, error=None, executed_price=110.0, exit_fee=0.0, slippage_cost=0.0)

    engine.live_execution_engine.execute_exit = Mock(side_effect=_exec_exit)

    return engine, calls


def _track(engine, *, current_size=0.1, original_size=0.2):
    position = Position(
        symbol="ETHUSDT",
        side=PositionSide.LONG,
        size=original_size,
        entry_price=100.0,
        entry_time=datetime(2025, 1, 1, tzinfo=UTC),
        order_id="order-1",
        original_size=original_size,
        current_size=current_size,
    )
    position.stop_loss_order_id = "sl-1"
    position.stop_loss = 95.0
    position.quantity = 1.0
    engine.live_position_tracker.track_recovered_position(position, db_id=None)
    return position


def _finish_via_partials(engine, position, *, delta_fraction):
    """Drive the private method under test directly, as the real caller
    (check_partial_operations) does once a partial-exit policy signals the
    final slice."""
    engine.live_exit_handler._execute_partial_exit(
        order_id="order-1",
        position=position,
        delta_fraction=delta_fraction,
        price=110.0,
        target_level=1,
        current_balance=1_000.0,
    )


def test_partial_exit_full_close_cancels_resting_stop_before_closing():
    """The #710 cancel-then-close sequence must run: a resting, clean stop
    is cancelled BEFORE the market close is submitted."""
    engine, calls = _make_engine(sl_fills=(0.0, 0.0))
    position = _track(engine, current_size=0.1, original_size=0.2)

    _finish_via_partials(engine, position, delta_fraction=0.1)

    assert calls == ["cancel", "close"]
    engine.exchange_interface.cancel_order.assert_called_once_with("sl-1", "ETHUSDT")


def test_partial_exit_full_close_serialises_on_base_asset_lock():
    """The #703 base-asset lock must be acquired for the position's base
    asset — proof the close routes through the coordinator's execute_exit,
    not the handler's raw (unlocked) execute_exit."""
    engine, _calls = _make_engine(sl_fills=(0.0, 0.0))
    position = _track(engine, current_size=0.1, original_size=0.2)

    lock_for_spy = Mock(wraps=engine._base_asset_locks.lock_for)
    engine._base_asset_locks.lock_for = lock_for_spy

    _finish_via_partials(engine, position, delta_fraction=0.1)

    lock_for_spy.assert_called_once_with("ETH")


def test_partial_exit_full_close_defers_on_mid_fill_stop():
    """Inventory-awareness (#710): if the resting stop has ANY fill, the
    close must be deferred to the reconciler — not submitted regardless."""
    engine, calls = _make_engine(sl_fills=(0.3,))
    position = _track(engine, current_size=0.1, original_size=0.2)

    _finish_via_partials(engine, position, delta_fraction=0.1)

    engine.exchange_interface.cancel_order.assert_not_called()
    assert calls == []
