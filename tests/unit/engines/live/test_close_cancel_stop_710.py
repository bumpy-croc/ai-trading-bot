"""#710: inventory-aware active close of a stop-protected margin position.

On margin a resting stop-loss order reserves the position's base asset, so a market
close submitted while it rests is rejected -2010 ("insufficient balance"). The close
path must:

* re-query the stop's filled quantity BEFORE cancelling — if it has ANY fill (or its
  state is unreadable), DEFER to the reconciler (held base != tracked size, so a
  full-size close would over-sell a long / over-buy a short); leave it resting;
* only cancel a provably clean (zero-fill) stop, then RE-CHECK the fill after the
  (terminal) cancel and defer if it raced a fill;
* close the full size only when the stop is confirmed clean both times;
* if the close then fails, re-protect immediately — but only after verifying the
  position is still actually held (avoid orphaning a stop on an ambiguous/already-
  executed close), sizing for prior partial exits, with bounded retry.
"""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.fast

from src.data_providers.exchange_interface import OrderSide
from src.engines.live.execution.exit_handler import LiveExitResult
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


def _make_live_engine(
    exit_result,
    *,
    cancel_returns=True,
    cancel_raises=False,
    sl_fills=(0.0, 0.0),
    held=True,
    borrowed=False,
    place_returns="sl-new",
):
    """A LiveTradingEngine in live mode with mocked exchange + exit handler.

    ``sl_fills`` is the sequence of filled quantities ``get_order`` reports across the
    pre-/post-cancel checks (repeats the last). ``held`` controls the held-inventory
    check; ``borrowed`` routes held via the short (borrowed) branch. Returns
    ``(engine, calls)`` recording the cancel/close/reprotect order.
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
    engine.exchange_interface = exchange
    engine.order_tracker = Mock()
    engine.performance_tracker.record_trade = Mock()
    engine._check_stop_loss_filled = Mock(return_value=(False, None))

    # get_order feeds _stop_loss_filled_quantity (pre/post cancel).
    fills = list(sl_fills)

    def _get_order(*_a, **_k):
        f = fills.pop(0) if len(fills) > 1 else fills[0]
        return _order(f)

    exchange.get_order.side_effect = _get_order

    # held-inventory check (_position_still_held).
    if held and borrowed:
        asset = {"free": "0", "locked": "0", "borrowed": "0.5", "netAsset": "-0.5"}
    elif held:
        asset = {"free": "1.0", "locked": "0", "borrowed": "0", "netAsset": "1.0"}
    else:
        asset = {"free": "0", "locked": "0", "borrowed": "0", "netAsset": "0"}
    exchange.get_margin_account_asset.return_value = asset

    calls: list[str] = []

    def _cancel(*_a, **_k):
        calls.append("cancel")
        if cancel_raises:
            raise RuntimeError("boom")
        return cancel_returns

    def _close(*_a, **_k):
        calls.append("close")
        return exit_result

    place_seq = list(place_returns) if isinstance(place_returns, list | tuple) else None

    def _place(*_a, **_k):
        calls.append("reprotect")
        if place_seq is not None:
            return place_seq.pop(0) if len(place_seq) > 1 else place_seq[0]
        return place_returns

    exchange.cancel_order.side_effect = _cancel
    exchange.place_stop_loss_order.side_effect = _place
    handler = Mock()
    handler.execute_exit.side_effect = _close
    engine.live_exit_handler = handler
    return engine, calls


def _track(
    engine,
    *,
    side=PositionSide.LONG,
    stop_loss_order_id="sl-1",
    stop_loss=95.0,
    quantity=0.5,
    current_size=0.25,
    original_size=0.25,
):
    position = Position(
        symbol="ETHUSDT",
        side=side,
        size=0.25,
        entry_price=100.0,
        entry_time=datetime(2025, 1, 1, tzinfo=UTC),
        order_id="order-1",
        original_size=original_size,
        current_size=current_size,
    )
    position.stop_loss_order_id = stop_loss_order_id
    position.stop_loss = stop_loss
    position.quantity = quantity
    engine.live_position_tracker.track_recovered_position(position, db_id=None)
    return position


def _exit(engine, position):
    engine._execute_exit(
        position=position,
        reason="signal-exit",
        limit_price=None,
        current_price=110.0,
        candle_high=None,
        candle_low=None,
        candle=None,
        skip_live_close=False,
    )


def _ok(price=110.0):
    return LiveExitResult(success=True, realized_pnl=2.0, exit_price=price)


def _fail(err="-2010"):
    return LiveExitResult(success=False, error=err)


# --- clean stop -> cancel then close --------------------------------------------


def test_clean_stop_cancelled_then_closed():
    engine, calls = _make_live_engine(_ok(), sl_fills=(0.0, 0.0))
    position = _track(engine)

    _exit(engine, position)

    assert calls == ["cancel", "close"]
    engine.exchange_interface.cancel_order.assert_called_once_with("sl-1", "ETHUSDT")
    engine.live_exit_handler.execute_exit.assert_called_once()
    engine.exchange_interface.place_stop_loss_order.assert_not_called()


# --- defer-on-fill (over-sell / over-buy prevention) ----------------------------


def test_pre_cancel_fill_defers_without_cancel_or_close():
    engine, calls = _make_live_engine(_ok(), sl_fills=(0.3,))
    position = _track(engine)

    _exit(engine, position)

    # Partial fill before cancel -> defer: do not cancel or close.
    engine.exchange_interface.cancel_order.assert_not_called()
    engine.live_exit_handler.execute_exit.assert_not_called()
    assert calls == []


def test_pre_cancel_unreadable_fill_defers():
    engine, calls = _make_live_engine(_ok(), sl_fills=(None,))
    position = _track(engine)

    _exit(engine, position)

    engine.exchange_interface.cancel_order.assert_not_called()
    engine.live_exit_handler.execute_exit.assert_not_called()


def test_post_cancel_race_fill_defers_close():
    # Clean before cancel, but a fill raced the cancel -> defer (no close).
    engine, calls = _make_live_engine(_ok(), sl_fills=(0.0, 0.4))
    position = _track(engine)

    _exit(engine, position)

    engine.exchange_interface.cancel_order.assert_called_once()
    engine.live_exit_handler.execute_exit.assert_not_called()
    assert calls == ["cancel"]


def test_unconfirmed_cancel_aborts_close():
    engine, calls = _make_live_engine(_ok(), sl_fills=(0.0,), cancel_returns=False)
    position = _track(engine)

    _exit(engine, position)

    engine.live_exit_handler.execute_exit.assert_not_called()
    assert "close" not in calls
    assert engine.live_position_tracker.has_position("order-1")


def test_cancel_order_raises_aborts_close():
    engine, calls = _make_live_engine(_ok(), sl_fills=(0.0,), cancel_raises=True)
    position = _track(engine)

    _exit(engine, position)

    engine.live_exit_handler.execute_exit.assert_not_called()


# --- re-protect after a failed close --------------------------------------------


def test_failed_close_reprotects_when_still_held():
    engine, calls = _make_live_engine(_fail(), sl_fills=(0.0, 0.0), held=True)
    position = _track(engine, quantity=0.5, current_size=0.5, original_size=0.5)

    _exit(engine, position)

    assert calls == ["cancel", "close", "reprotect"]
    engine.exchange_interface.place_stop_loss_order.assert_called_once()
    kw = engine.exchange_interface.place_stop_loss_order.call_args.kwargs
    assert kw["symbol"] == "ETHUSDT"
    assert kw["stop_price"] == 95.0
    assert kw["side"] == OrderSide.SELL
    assert kw["quantity"] == pytest.approx(0.5)


def test_failed_close_skips_reprotect_when_not_held():
    # Ambiguous close that actually executed -> no inventory -> do NOT orphan a stop.
    engine, calls = _make_live_engine(_fail(), sl_fills=(0.0, 0.0), held=False)
    position = _track(engine)

    _exit(engine, position)

    engine.exchange_interface.place_stop_loss_order.assert_not_called()
    assert "reprotect" not in calls


def test_reprotect_scales_quantity_for_partial_exit():
    # current/original = 0.5 -> protect half the entry quantity.
    engine, _calls = _make_live_engine(_fail(), sl_fills=(0.0, 0.0), held=True)
    position = _track(engine, quantity=0.8, current_size=0.5, original_size=1.0)

    _exit(engine, position)

    kw = engine.exchange_interface.place_stop_loss_order.call_args.kwargs
    assert kw["quantity"] == pytest.approx(0.8 * 0.5 / 1.0)


def test_short_reprotect_uses_buy_when_still_borrowed():
    engine, _calls = _make_live_engine(_fail(), sl_fills=(0.0, 0.0), held=True, borrowed=True)
    position = _track(
        engine, side=PositionSide.SHORT, quantity=0.5, current_size=0.5, original_size=0.5
    )

    _exit(engine, position)

    engine.exchange_interface.place_stop_loss_order.assert_called_once()
    kw = engine.exchange_interface.place_stop_loss_order.call_args.kwargs
    assert kw["side"] == OrderSide.BUY


def test_reprotect_retries_then_succeeds():
    engine, _calls = _make_live_engine(
        _fail(), sl_fills=(0.0, 0.0), held=True, place_returns=[None, "sl-2"]
    )
    position = _track(engine, quantity=0.5, current_size=0.5, original_size=0.5)

    _exit(engine, position)

    assert engine.exchange_interface.place_stop_loss_order.call_count == 2


# --- unchanged paths ------------------------------------------------------------


def test_no_resting_stop_closes_directly():
    engine, _calls = _make_live_engine(_ok())
    position = _track(engine, stop_loss_order_id=None)

    _exit(engine, position)

    engine.exchange_interface.cancel_order.assert_not_called()
    engine.exchange_interface.get_order.assert_not_called()
    engine.live_exit_handler.execute_exit.assert_called_once()


def test_filled_stop_path_does_not_cancel_or_reprotect():
    engine, _calls = _make_live_engine(_ok(price=95.0))
    engine._check_stop_loss_filled = Mock(return_value=(True, 95.0))
    engine.live_exit_handler.execute_filled_exit = Mock(return_value=_ok(price=95.0))
    position = _track(engine)

    _exit(engine, position)

    engine.live_exit_handler.execute_filled_exit.assert_called_once()
    engine.exchange_interface.cancel_order.assert_not_called()
    engine.exchange_interface.place_stop_loss_order.assert_not_called()
