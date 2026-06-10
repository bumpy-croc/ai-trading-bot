from __future__ import annotations

from datetime import UTC, datetime

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
        entry_time=datetime.now(UTC),
        entry_balance=999.0,
        quantity=10.0,
        order_id="order-123",
    )

    tracker.open_position(position, session_id=1, strategy_name="test")

    call_kwargs = mock_db_manager.log_position.call_args.kwargs
    assert call_kwargs["quantity"] == 10.0


def _db_position_row(**overrides) -> dict:
    """Dict row in the shape DatabaseManager.get_active_positions returns."""
    row = {
        "id": 123,
        "symbol": "BTCUSDT",
        "side": "LONG",  # DB enum .value is uppercase
        "size": 0.1,
        "entry_price": 50000.0,
        "entry_time": datetime.now(UTC),
        "entry_balance": 1000.0,
        "stop_loss": 48000.0,
        "take_profit": 55000.0,
        "entry_order_id": "order-456",
        "stop_loss_order_id": None,
        "client_order_id": "atb_abc123",
        "original_size": 0.1,
        "current_size": 0.1,
        "partial_exits_taken": 0,
        "scale_ins_taken": 0,
        "trailing_stop_activated": False,
        "trailing_stop_price": None,
        "breakeven_triggered": False,
        "quantity": 0.02,
    }
    row.update(overrides)
    return row


@pytest.mark.fast
def test_recover_positions_restores_quantity(mock_db_manager) -> None:
    """Regression for #764: recovery reads get_active_positions (the real
    DatabaseManager API — get_open_positions never existed) and hydrates the
    persisted quantity."""
    mock_db_manager.get_active_positions.return_value = [_db_position_row()]

    tracker = LivePositionTracker(db_manager=mock_db_manager)
    recovered = tracker.recover_positions(session_id=1)

    mock_db_manager.get_active_positions.assert_called_once_with(1)
    assert len(recovered) == 1
    assert recovered[0].quantity == 0.02


@pytest.mark.fast
def test_recover_positions_tracks_and_maps_fields(mock_db_manager) -> None:
    """Recovered positions are registered in the tracker with the entry order
    id as key, the uppercase DB side is normalized, and partial-op state and
    reconciliation ids are hydrated."""
    mock_db_manager.get_active_positions.return_value = [
        _db_position_row(
            original_size=0.2,
            current_size=0.1,
            partial_exits_taken=1,
            stop_loss_order_id="sl-789",
        )
    ]

    tracker = LivePositionTracker(db_manager=mock_db_manager)
    recovered = tracker.recover_positions(session_id=1)

    assert len(recovered) == 1
    position = recovered[0]
    assert position.side == PositionSide.LONG
    assert position.order_id == "order-456"
    assert position.exchange_order_id == "order-456"
    assert position.client_order_id == "atb_abc123"
    assert position.db_position_id == 123
    assert position.stop_loss_order_id == "sl-789"
    assert position.original_size == 0.2
    assert position.current_size == 0.1
    assert position.partial_exits_taken == 1
    # Registered in the tracker under the entry order id
    assert tracker.positions["order-456"] is position


@pytest.mark.fast
def test_recover_positions_falls_back_to_db_id_key(mock_db_manager) -> None:
    """Without an entry order id, the database row id becomes the tracker key
    (mirrors LiveTradingEngine._recover_active_positions)."""
    mock_db_manager.get_active_positions.return_value = [_db_position_row(entry_order_id=None)]

    tracker = LivePositionTracker(db_manager=mock_db_manager)
    recovered = tracker.recover_positions(session_id=1)

    assert len(recovered) == 1
    assert recovered[0].order_id == "123"
    assert "123" in tracker.positions


@pytest.mark.fast
def test_recover_positions_skips_corrupt_row_recovers_rest(mock_db_manager) -> None:
    """One corrupt row (invalid entry price) must not abort recovery of the
    remaining positions."""
    mock_db_manager.get_active_positions.return_value = [
        _db_position_row(id=1, entry_order_id="bad", entry_price=0.0),
        _db_position_row(id=2, entry_order_id="good"),
    ]

    tracker = LivePositionTracker(db_manager=mock_db_manager)
    recovered = tracker.recover_positions(session_id=1)

    assert len(recovered) == 1
    assert recovered[0].order_id == "good"
    assert "bad" not in tracker.positions
