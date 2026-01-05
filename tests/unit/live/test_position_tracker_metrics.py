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


@pytest.mark.fast
def test_close_position_invalid_price_retains_position() -> None:
    """Close with invalid price returns None but keeps position in tracker for retry."""
    # Arrange
    tracker = LivePositionTracker()
    order_id = "invalid-price-1"
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

    # Act - try to close with invalid price (NaN)
    result = tracker.close_position(
        order_id=order_id,
        exit_price=float("nan"),
        exit_reason="test",
        basis_balance=1000.0,
    )

    # Assert - close failed but position is still tracked
    assert result is None
    assert tracker.has_position(order_id), "Position should remain in tracker after validation failure"
    assert tracker.get_position(order_id) is not None


@pytest.mark.fast
def test_close_position_invalid_basis_balance_retains_position() -> None:
    """Close with invalid basis_balance (NaN) returns None but keeps position in tracker."""
    # Arrange
    tracker = LivePositionTracker()
    order_id = "invalid-basis-1"
    position = LivePosition(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=0.1,
        entry_price=100.0,
        entry_time=datetime.now(UTC),
        entry_balance=None,  # No entry balance, will use basis_balance
        order_id=order_id,
    )
    tracker.open_position(position)

    # Act - try to close with invalid basis_balance (NaN)
    result = tracker.close_position(
        order_id=order_id,
        exit_price=105.0,
        exit_reason="test",
        basis_balance=float("nan"),
    )

    # Assert - close failed but position is still tracked
    assert result is None
    assert tracker.has_position(order_id), "Position should remain in tracker after invalid basis"
    assert tracker.get_position(order_id) is not None


@pytest.mark.fast
def test_close_position_negative_basis_balance_retains_position() -> None:
    """Close with negative basis_balance returns None but keeps position in tracker."""
    # Arrange
    tracker = LivePositionTracker()
    order_id = "negative-basis-1"
    position = LivePosition(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=0.1,
        entry_price=100.0,
        entry_time=datetime.now(UTC),
        entry_balance=None,  # No entry balance, will use basis_balance
        order_id=order_id,
    )
    tracker.open_position(position)

    # Act - try to close with negative basis_balance
    result = tracker.close_position(
        order_id=order_id,
        exit_price=105.0,
        exit_reason="test",
        basis_balance=-100.0,
    )

    # Assert - close failed but position is still tracked
    assert result is None
    assert tracker.has_position(order_id), "Position should remain in tracker after negative basis"
    assert tracker.get_position(order_id) is not None
