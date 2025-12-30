from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.engines.live.execution.position_tracker import (
    LivePosition,
    LivePositionTracker,
    PositionSide,
)


@pytest.mark.fast
def test_open_position_persists_executed_quantity(mock_db_manager) -> None:
    tracker = LivePositionTracker(db_manager=mock_db_manager)
    position = LivePosition(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=1.0,
        entry_price=100.0,
        entry_time=datetime.now(timezone.utc),
        entry_balance=999.0,
        quantity=10.0,
        order_id="order-123",
    )

    tracker.open_position(position, session_id=1, strategy_name="test")

    call_kwargs = mock_db_manager.log_position.call_args.kwargs
    assert call_kwargs["quantity"] == 10.0


@pytest.mark.fast
def test_recover_positions_restores_quantity(mock_db_manager) -> None:
    entry_time = datetime.now(timezone.utc)
    db_pos = SimpleNamespace(
        id=123,
        symbol="BTCUSDT",
        side="long",
        size=0.1,
        entry_price=50000.0,
        entry_time=entry_time,
        entry_balance=1000.0,
        stop_loss=48000.0,
        take_profit=55000.0,
        entry_order_id="order-456",
        original_size=0.1,
        current_size=0.1,
        partial_exits_taken=0,
        scale_ins_taken=0,
        trailing_stop_activated=False,
        trailing_stop_price=None,
        breakeven_triggered=False,
        quantity=0.02,
    )
    mock_db_manager.get_open_positions.return_value = [db_pos]

    tracker = LivePositionTracker(db_manager=mock_db_manager)
    recovered = tracker.recover_positions(session_id=1)

    assert recovered[0].quantity == 0.02
