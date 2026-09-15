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

import logging
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

    Only the exchange interface and the lowest-level order-placement calls on
    the execution engine are stubbed; everything above that (handler,
    tracker, coordinator, base-asset lock, and ``LiveExecutionEngine
    .execute_exit`` itself, notional guard included) is the real production
    code. Returns ``(engine, calls, close_orders)``: ``calls`` records the
    cancel/close order, ``close_orders`` records the quantity/notional each
    submitted close order was sized with.
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
    # ``LiveExecutionEngine.enable_live_trading`` is set at construction from
    # the (False) constructor arg above and doesn't follow the engine-level
    # flag flipped here — without this, execute_exit takes its "paper trade"
    # branch and never reaches (or validates) real order placement at all.
    engine.live_execution_engine.enable_live_trading = True
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
    close_orders: list[dict] = []

    def _cancel(*_a, **_k):
        calls.append("cancel")
        return cancel_returns

    exchange.cancel_order.side_effect = _cancel

    # Stub only the lowest-level order placement (the actual exchange call),
    # so ``LiveExecutionEngine.execute_exit`` runs for real — including its
    # ``position_notional <= 0`` guard. A test that stubs execute_exit itself
    # (as this used to) never exercises that guard, so it can't catch a
    # notional-sizing bug upstream (#1183).
    def _close_live_order(symbol, side, quantity, position_notional, **_k):
        calls.append("close")
        close_orders.append({"quantity": quantity, "position_notional": position_notional})
        return "close-order-1"

    engine.live_execution_engine._close_live_order = Mock(side_effect=_close_live_order)
    # None short-circuits the fill-enrichment branch in execute_exit, so the
    # simulated price/fee/slippage computed from position_notional stand.
    engine.live_execution_engine._fetch_order_details = Mock(return_value=None)

    return engine, calls, close_orders


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
    engine, calls, _close_orders = _make_engine(sl_fills=(0.0, 0.0))
    position = _track(engine, current_size=0.1, original_size=0.2)

    _finish_via_partials(engine, position, delta_fraction=0.1)

    assert calls == ["cancel", "close"]
    engine.exchange_interface.cancel_order.assert_called_once_with("sl-1", "ETHUSDT")


def test_partial_exit_full_close_serialises_on_base_asset_lock():
    """The #703 base-asset lock must be acquired for the position's base
    asset — proof the close routes through the coordinator's execute_exit,
    not the handler's raw (unlocked) execute_exit."""
    engine, _calls, _close_orders = _make_engine(sl_fills=(0.0, 0.0))
    position = _track(engine, current_size=0.1, original_size=0.2)

    lock_for_spy = Mock(wraps=engine._base_asset_locks.lock_for)
    engine._base_asset_locks.lock_for = lock_for_spy

    _finish_via_partials(engine, position, delta_fraction=0.1)

    lock_for_spy.assert_called_once_with("ETH")


def test_partial_exit_full_close_defers_on_mid_fill_stop():
    """Inventory-awareness (#710): if the resting stop has ANY fill, the
    close must be deferred to the reconciler — not submitted regardless."""
    engine, calls, _close_orders = _make_engine(sl_fills=(0.3,))
    position = _track(engine, current_size=0.1, original_size=0.2)

    _finish_via_partials(engine, position, delta_fraction=0.1)

    engine.exchange_interface.cancel_order.assert_not_called()
    assert calls == []


def test_partial_exit_full_close_submits_real_order_sized_off_full_quantity():
    """#1183: ``apply_partial_exit`` zeroes ``current_size`` before this final
    leg routes through the coordinator. Live partial exits never place a real
    exchange order (#734), so the exchange still holds the FULL original
    quantity at this point — the close must be sized off that, not off the
    zeroed ``current_size``.

    Before the fix, ``_calculate_position_notional`` re-derived the notional
    from the already-zeroed ``current_size``, computing 0.0.
    ``LiveExecutionEngine.execute_exit`` rejects a non-positive notional
    outright ("Invalid position notional: 0.0"), so no close order was ever
    submitted — this test would have failed with ``calls == ["cancel"]`` and
    ``close_orders == []`` (and, in a live run, the cancelled stop then gets
    re-placed and the position sits open at size 0 forever).
    """
    engine, calls, close_orders = _make_engine(sl_fills=(0.0, 0.0))
    position = _track(engine, current_size=0.1, original_size=0.2)
    assert position.current_size == pytest.approx(0.1)

    _finish_via_partials(engine, position, delta_fraction=0.1)

    assert calls == ["cancel", "close"]
    assert position.current_size == 0.0
    assert len(close_orders) == 1
    # _track sets position.quantity=1.0; the close price is 110.0.
    assert close_orders[0]["position_notional"] == pytest.approx(110.0)
    assert close_orders[0]["quantity"] == pytest.approx(1.0)


def test_partial_exit_full_close_without_coordinator_wired_logs_critical_and_skips(caplog):
    """If ``execute_full_exit`` isn't wired (tests, standalone use without an
    engine), the final close must be skipped and logged as CRITICAL rather
    than falling back to an unlocked, un-cancelled raw close (#1166)."""
    engine, calls, close_orders = _make_engine(sl_fills=(0.0, 0.0))
    position = _track(engine, current_size=0.1, original_size=0.2)
    engine.live_exit_handler._execute_full_exit = None

    with caplog.at_level(logging.CRITICAL):
        _finish_via_partials(engine, position, delta_fraction=0.1)

    assert calls == []
    assert close_orders == []
    engine.exchange_interface.cancel_order.assert_not_called()
    assert any("No exit-coordinator route wired" in r.message for r in caplog.records)
