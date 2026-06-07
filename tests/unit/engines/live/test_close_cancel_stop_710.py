"""#710: cancel the resting stop-loss BEFORE a market close.

On margin a resting stop-loss order reserves the position's base asset, so a
market close submitted while it rests is rejected by the exchange with -2010
("insufficient balance"). The close path must therefore:

* cancel the resting stop FIRST (freeing the base balance), then close;
* NOT submit the close if it cannot confirm the cancel (the order may have just
  filled — closing would over-sell — or still rest — closing would -2010);
* re-place the stop if the close then fails (never leave the position naked).

These tests exercise ``LiveTradingEngine._execute_exit`` in live mode with mocked
collaborators and assert the ordering / branching above.
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
def _mock_db(monkeypatch):
    """Use the in-memory MockDatabaseManager so construction needs no real DB."""
    monkeypatch.setattr("src.engines.live.trading_engine.DatabaseManager", MockDatabaseManager)


def _make_live_engine(exit_result, *, cancel_returns=True):
    """A LiveTradingEngine flipped to live mode with mocked exchange + exit handler.

    ``exit_result`` is the ``LiveExitResult`` the mocked market close returns.
    Returns ``(engine, calls)`` where ``calls`` records the order of
    cancel_order / execute_exit / place_stop_loss_order so tests can assert
    cancel-before-close. The mocks live on ``engine.exchange_interface`` and
    ``engine.live_exit_handler``.
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

    # Flip to live mode with mocked collaborators.
    engine.enable_live_trading = True
    exchange = Mock()
    exchange.is_margin_mode = True
    engine.exchange_interface = exchange
    engine.order_tracker = Mock()
    engine.performance_tracker.record_trade = Mock()

    # Force the market-close path (stop not already filled on the exchange).
    engine._check_stop_loss_filled = Mock(return_value=(False, None))

    calls: list[str] = []

    def _cancel(*_a, **_k):
        calls.append("cancel")
        return cancel_returns

    def _close(*_a, **_k):
        calls.append("close")
        return exit_result

    def _reprotect(*_a, **_k):
        calls.append("reprotect")
        return "sl-new"

    exchange.cancel_order.side_effect = _cancel
    exchange.place_stop_loss_order.side_effect = _reprotect
    handler = Mock()
    handler.execute_exit.side_effect = _close
    engine.live_exit_handler = handler

    return engine, calls


def _track_long(engine, *, stop_loss_order_id="sl-1", stop_loss=95.0, quantity=0.5):
    position = Position(
        symbol="ETHUSDT",
        side=PositionSide.LONG,
        size=0.25,
        entry_price=100.0,
        entry_time=datetime(2025, 1, 1, tzinfo=UTC),
        order_id="order-1",
        original_size=0.25,
        current_size=0.25,
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


def test_market_close_cancels_resting_stop_before_submitting():
    """The resting stop is cancelled BEFORE the market close is submitted."""
    engine, calls = _make_live_engine(
        LiveExitResult(success=True, realized_pnl=2.0, exit_price=110.0)
    )
    position = _track_long(engine, stop_loss_order_id="sl-1")

    _exit(engine, position)

    assert calls[:2] == ["cancel", "close"]
    engine.exchange_interface.cancel_order.assert_called_once_with("sl-1", "ETHUSDT")
    engine.live_exit_handler.execute_exit.assert_called_once()


def test_market_close_aborts_when_cancel_unconfirmed():
    """If the stop cancel cannot be confirmed, the close is NOT submitted."""
    engine, calls = _make_live_engine(
        LiveExitResult(success=True, realized_pnl=2.0, exit_price=110.0),
        cancel_returns=False,
    )
    position = _track_long(engine, stop_loss_order_id="sl-1")

    _exit(engine, position)

    # Close must not be attempted; the position stays tracked and protected.
    engine.live_exit_handler.execute_exit.assert_not_called()
    assert "close" not in calls
    assert engine.live_position_tracker.has_position("order-1")


def test_failed_close_after_cancel_reprotects():
    """A close that fails after a confirmed cancel re-places the stop-loss."""
    engine, calls = _make_live_engine(LiveExitResult(success=False, error="-2010"))
    position = _track_long(engine, stop_loss_order_id="sl-1", stop_loss=95.0, quantity=0.5)

    _exit(engine, position)

    assert calls == ["cancel", "close", "reprotect"]
    engine.exchange_interface.place_stop_loss_order.assert_called_once()
    kwargs = engine.exchange_interface.place_stop_loss_order.call_args.kwargs
    assert kwargs["symbol"] == "ETHUSDT"
    assert kwargs["stop_price"] == 95.0
    assert kwargs["quantity"] == 0.5
    assert kwargs["side"] == OrderSide.SELL  # SELL stop protects a long


def test_market_close_without_resting_stop_does_not_cancel():
    """No tracked stop → nothing to cancel; the close proceeds directly."""
    engine, _calls = _make_live_engine(
        LiveExitResult(success=True, realized_pnl=2.0, exit_price=110.0)
    )
    position = _track_long(engine, stop_loss_order_id=None)

    _exit(engine, position)

    engine.exchange_interface.cancel_order.assert_not_called()
    engine.live_exit_handler.execute_exit.assert_called_once()


def test_filled_stop_path_does_not_cancel_or_reprotect():
    """When the stop already filled, no market close / cancel / re-protect occurs."""
    engine, _calls = _make_live_engine(
        LiveExitResult(success=True, realized_pnl=2.0, exit_price=95.0)
    )
    engine._check_stop_loss_filled = Mock(return_value=(True, 95.0))
    engine.live_exit_handler.execute_filled_exit = Mock(
        return_value=LiveExitResult(success=True, realized_pnl=2.0, exit_price=95.0)
    )
    position = _track_long(engine, stop_loss_order_id="sl-1")

    _exit(engine, position)

    engine.live_exit_handler.execute_filled_exit.assert_called_once()
    engine.exchange_interface.cancel_order.assert_not_called()
    engine.exchange_interface.place_stop_loss_order.assert_not_called()
