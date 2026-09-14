"""#1167: a trailing-stop ratchet must move the resting exchange stop-loss
order, not just ``position.stop_loss`` in memory/DB.

Before this fix, ``LiveExitHandler.update_trailing_stops`` updated the
tracked stop price but never touched the exchange, so the resting order kept
protecting at its original (lower) price forever while the engine believed
protection sat at the new, higher price — the direct trigger of #1165's abort
storm (the engine's own ``_check_stop_loss`` re-fires every loop iteration
against a stop the exchange will never actually hit).
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

from src.data_providers.exchange_interface import OrderSide
from src.engines.live.execution.exit_handler import LiveExitHandler
from src.engines.live.execution.position_tracker import (
    LivePosition,
    LivePositionTracker,
    PositionSide,
)
from src.engines.live.execution.stop_loss_manager import LiveStopLossManager
from src.engines.shared.execution.execution_model import ExecutionModel
from src.engines.shared.execution.fill_policy import default_fill_policy
from src.position_management.trailing_stops import TrailingStopPolicy

pytestmark = pytest.mark.fast


def _spot_exchange(*, free: float = 0.02, new_order_id: str = "sl-new") -> Mock:
    """Exchange stand-in confirmed as holding the position's base asset (spot)."""
    exchange = Mock()
    exchange.is_margin_mode = False
    exchange.get_balance.return_value = SimpleNamespace(free=free, locked=0.0)
    exchange.get_open_orders_checked.return_value = []
    exchange.cancel_order.return_value = True
    exchange.place_stop_loss_order.return_value = new_order_id
    return exchange


def _long_position(**overrides) -> LivePosition:
    position = LivePosition(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_price=100.0,
        entry_time=datetime.now(UTC),
        size=0.02,
        stop_loss=95.0,
        stop_loss_order_id="sl-old",
        quantity=0.02,
        current_size=0.02,
        original_size=0.02,
        order_id="entry-1",
        trailing_stop_activated=True,
    )
    for key, value in overrides.items():
        setattr(position, key, value)
    return position


def _candles() -> pd.DataFrame:
    return pd.DataFrame({"close": [100.0, 110.0], "high": [100.0, 110.0], "low": [100.0, 110.0]})


def _build_exit_handler(
    exchange: Mock, *, bind_stop_loss_manager: bool = True
) -> tuple[LiveExitHandler, LivePositionTracker]:
    position_tracker = LivePositionTracker()
    execution_engine = MagicMock()
    exit_handler = LiveExitHandler(
        position_tracker=position_tracker,
        execution_engine=execution_engine,
        execution_model=ExecutionModel(default_fill_policy()),
        # activation_threshold=0 => trailing is active from the first favorable
        # tick; trailing_distance_pct=1% keeps the arithmetic simple.
        trailing_stop_policy=TrailingStopPolicy(
            activation_threshold=0.0,
            trailing_distance_pct=0.01,
        ),
    )
    if bind_stop_loss_manager:
        state = SimpleNamespace(
            enable_live_trading=True,
            exchange_interface=exchange,
            order_tracker=Mock(),
            live_position_tracker=position_tracker,
        )
        stop_loss_manager = LiveStopLossManager(engine_state=state, send_alert=Mock())
        exit_handler.bind_stop_loss_manager(stop_loss_manager)
    return exit_handler, position_tracker


class TestTrailingStopRatchetMovesExchangeOrder:
    def test_ratchet_cancels_old_stop_and_places_new_one_at_new_price(self):
        # Arrange: a position with an already-activated trailing stop resting
        # on the exchange at the OLD price ($95), and a favorable price move
        # that ratchets the trail up to $108.90 (110 - 1% * 110).
        exchange = _spot_exchange()
        exit_handler, position_tracker = _build_exit_handler(exchange)
        position_tracker.open_position(_long_position())

        # Act
        exit_handler.update_trailing_stops(_candles(), current_index=1, current_price=110.0)

        # Assert: the OLD resting order was cancelled...
        exchange.cancel_order.assert_called_once_with("sl-old", "BTCUSDT")
        # ...and a NEW one was placed at the ratcheted price.
        exchange.place_stop_loss_order.assert_called_once()
        call = exchange.place_stop_loss_order.call_args
        assert call.kwargs["symbol"] == "BTCUSDT"
        assert call.kwargs["side"] == OrderSide.SELL
        assert call.kwargs["stop_price"] == pytest.approx(108.9)
        # The tracked position now points at the new exchange order id.
        position = position_tracker.get_position("entry-1")
        assert position.stop_loss_order_id == "sl-new"
        assert position.stop_loss == pytest.approx(108.9)

    def test_without_a_bound_stop_loss_manager_only_memory_updates(self):
        # A handler built without bind_stop_loss_manager() (e.g. an older
        # construction path, or a test double) must not blow up — it just
        # falls back to the pre-#1167 memory/DB-only behavior.
        exchange = _spot_exchange()
        exit_handler, position_tracker = _build_exit_handler(exchange, bind_stop_loss_manager=False)
        position_tracker.open_position(_long_position())

        exit_handler.update_trailing_stops(_candles(), current_index=1, current_price=110.0)

        exchange.cancel_order.assert_not_called()
        exchange.place_stop_loss_order.assert_not_called()
        assert position_tracker.get_position("entry-1").stop_loss == pytest.approx(108.9)
