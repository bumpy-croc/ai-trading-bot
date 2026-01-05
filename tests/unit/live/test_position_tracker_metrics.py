from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.engines.live.execution.position_tracker import (
    LivePosition,
    LivePositionTracker,
    PositionSide,
)


@pytest.mark.fast
def test_close_position_preserves_mfe_mae_metrics() -> None:
    """Close position returns MFE/MAE metrics before tracker clears them."""
    # Arrange
    tracker = LivePositionTracker()
    order_id = "close-metrics-1"
    position = LivePosition(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=0.1,
        entry_price=100.0,
        entry_time=datetime.now(UTC),
        entry_balance=1000.0,
        order_id=order_id,
    )
    tracker.open_position(position)

    tracker.mfe_mae_tracker.update_position_metrics(
        position_key=order_id,
        entry_price=position.entry_price,
        current_price=110.0,
        side=position.side.value,
        position_fraction=position.size,
        current_time=datetime.now(UTC),
    )

    # Act
    result = tracker.close_position(
        order_id=order_id,
        exit_price=105.0,
        exit_reason="test",
        basis_balance=1000.0,
    )

    # Assert
    assert result is not None
    assert result.mfe_mae_metrics is not None
    assert result.mfe_mae_metrics.mfe > 0.0
    assert result.mfe_mae_metrics.mfe_price == 110.0
    assert tracker.mfe_mae_tracker.get_position_metrics(order_id) is None
