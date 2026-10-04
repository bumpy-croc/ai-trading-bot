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
    # Raw price excursion: (110 - 100) / 100, independent of the 0.1 size.
    assert result.mfe_mae_metrics.mfe == pytest.approx(0.10)
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
    assert tracker.has_position(
        order_id
    ), "Position should remain in tracker after validation failure"
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


@pytest.mark.fast
def test_close_position_with_zero_size_succeeds() -> None:
    """Close with zero current_size (after partial exits) succeeds and removes position."""
    # Arrange
    tracker = LivePositionTracker()
    order_id = "zero-size-1"
    position = LivePosition(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=0.1,
        entry_price=100.0,
        entry_time=datetime.now(UTC),
        entry_balance=1000.0,
        order_id=order_id,
    )
    # Simulate position reduced to zero by partial exits
    position.current_size = 0.0
    tracker.open_position(position)

    # Act - close position with zero current_size
    result = tracker.close_position(
        order_id=order_id,
        exit_price=105.0,
        exit_reason="partial_exits_complete",
        basis_balance=1000.0,
    )

    # Assert - close succeeded, position removed from tracker
    assert result is not None, "Close should succeed even with zero current_size"
    assert result.realized_pnl == 0.0, "P&L should be zero for zero-sized position"
    assert not tracker.has_position(order_id), "Position should be removed from tracker"
    assert tracker.get_position(order_id) is None


@pytest.mark.fast
def test_total_unrealized_pnl_sums_open_positions() -> None:
    """total_unrealized_pnl is the mark-to-market sum feeding the circuit breaker."""
    tracker = LivePositionTracker()
    assert tracker.total_unrealized_pnl() == 0.0

    for order_id, side in (("cb-eq-1", PositionSide.LONG), ("cb-eq-2", PositionSide.SHORT)):
        tracker.open_position(
            LivePosition(
                symbol="BTCUSDT",
                side=side,
                size=0.1,
                entry_price=100.0,
                entry_time=datetime.now(UTC),
                entry_balance=1000.0,
                order_id=order_id,
            )
        )
    tracker.update_pnl(current_price=110.0, fallback_balance=1000.0)

    positions = tracker.positions
    expected = sum(p.unrealized_pnl for p in positions.values())
    # Long +10% and short -10% on a 0.1 fraction of 1000 cancel out exactly,
    # so also assert the per-position marks are non-zero (the sum is real).
    assert any(p.unrealized_pnl != 0.0 for p in positions.values())
    assert tracker.total_unrealized_pnl() == pytest.approx(expected)


def _open_long(tracker: LivePositionTracker, order_id: str = "mark-1") -> LivePosition:
    position = LivePosition(
        symbol="ETHUSDT",
        side=PositionSide.LONG,
        size=0.2,
        entry_price=100.0,
        entry_time=datetime.now(UTC),
        entry_balance=1000.0,
        order_id=order_id,
    )
    tracker.open_position(position)
    return position


@pytest.mark.fast
def test_mark_reading_is_fresh_after_a_valid_pricing_pass() -> None:
    tracker = LivePositionTracker()
    _open_long(tracker)

    tracker.update_pnl(current_price=90.0, fallback_balance=1000.0)
    reading = tracker.mark_reading(max_age_seconds=600)

    assert reading.fresh is True
    assert reading.unrealized_pnl == pytest.approx(tracker.total_unrealized_pnl())
    assert reading.unrealized_pnl < 0


@pytest.mark.fast
def test_mark_reading_not_fresh_before_any_pricing_pass() -> None:
    tracker = LivePositionTracker()
    _open_long(tracker)

    assert tracker.mark_reading(max_age_seconds=600).fresh is False


@pytest.mark.fast
def test_mark_reading_not_fresh_when_price_is_invalid() -> None:
    """update_pnl writes 0.0 for an unpriceable position; that must read as unknown."""
    tracker = LivePositionTracker()
    _open_long(tracker)

    tracker.update_pnl(current_price=float("nan"), fallback_balance=1000.0)

    reading = tracker.mark_reading(max_age_seconds=600)
    assert reading.fresh is False
    assert reading.unrealized_pnl == 0.0


@pytest.mark.fast
def test_mark_reading_not_fresh_when_the_mark_is_too_old(monkeypatch) -> None:
    import src.engines.live.execution.position_tracker as module

    tracker = LivePositionTracker()
    _open_long(tracker)
    clock = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    tracker.update_pnl(current_price=95.0, fallback_balance=1000.0)

    clock[0] += 599
    assert tracker.mark_reading(max_age_seconds=600).fresh is True
    clock[0] += 2
    assert tracker.mark_reading(max_age_seconds=600).fresh is False


@pytest.mark.fast
def test_mark_reading_recovered_position_is_unmarked_until_repriced() -> None:
    tracker = LivePositionTracker()
    tracker.update_pnl(current_price=100.0, fallback_balance=1000.0)  # empty book, fresh pass
    recovered = LivePosition(
        symbol="ETHUSDT",
        side=PositionSide.LONG,
        size=0.2,
        entry_price=100.0,
        entry_time=datetime.now(UTC),
        entry_balance=1000.0,
        order_id="recovered-1",
        unrealized_pnl=-5.0,
    )
    tracker.track_recovered_position(recovered, db_id=1)

    assert tracker.mark_reading(max_age_seconds=600).fresh is False

    tracker.update_pnl(current_price=90.0, fallback_balance=1000.0)
    assert tracker.mark_reading(max_age_seconds=600).fresh is True


@pytest.mark.fast
def test_mark_reading_with_no_positions_is_trivially_fresh_and_sequence_advances() -> None:
    tracker = LivePositionTracker()
    first = tracker.mark_reading(max_age_seconds=600)
    assert first.fresh is True and first.unrealized_pnl == 0.0

    tracker.update_pnl(current_price=100.0, fallback_balance=1000.0)
    assert tracker.mark_reading(max_age_seconds=600).sequence == first.sequence + 1
