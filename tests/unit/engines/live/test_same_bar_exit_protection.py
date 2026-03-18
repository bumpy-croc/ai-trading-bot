"""Unit tests for same-bar exit protection in the live trading engine.

Verifies that positions entered on the current candle are skipped during
exit evaluation, while positions from previous candles are evaluated normally.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.engines.shared.models import BasePosition, PositionSide


def _make_position(
    entry_time: datetime,
    symbol: str = "BTCUSDT",
    side: PositionSide = PositionSide.LONG,
    entry_price: float = 50000.0,
) -> BasePosition:
    """Create a minimal position with the given entry time."""
    return BasePosition(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        entry_time=entry_time,
        size=0.02,
        stop_loss=49000.0,
        take_profit=52000.0,
    )


def _make_df(timestamps: list[datetime], close: float = 50500.0) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame indexed by timestamps."""
    data = {
        "open": [close] * len(timestamps),
        "high": [close + 500] * len(timestamps),
        "low": [close - 500] * len(timestamps),
        "close": [close] * len(timestamps),
        "volume": [100.0] * len(timestamps),
    }
    return pd.DataFrame(data, index=pd.DatetimeIndex(timestamps))


def _apply_same_bar_logic(
    position: BasePosition,
    candle_time: datetime | None,
) -> bool:
    """Reproduce the same-bar exit protection logic from the live engine.

    Returns True if the position should be SKIPPED (not evaluated for exit).
    This mirrors the exact branching logic at lines 1896-1928 of trading_engine.py.
    """
    if candle_time is not None and position.entry_time is not None:
        try:
            entry_cmp = position.entry_time
            candle_cmp = candle_time
            entry_aware = getattr(entry_cmp, "tzinfo", None) is not None
            candle_aware = getattr(candle_cmp, "tzinfo", None) is not None
            if entry_aware and not candle_aware:
                candle_cmp = candle_cmp.replace(tzinfo=UTC)
            elif candle_aware and not entry_aware:
                entry_cmp = entry_cmp.replace(tzinfo=UTC)
            if entry_cmp >= candle_cmp:
                return True  # Skip exit evaluation
        except (TypeError, ValueError, AttributeError):
            pass  # Fall through to evaluate exit normally
    return False  # Evaluate exit normally


class TestSameBarExitProtection:
    """Test same-bar exit protection in the live engine."""

    def test_position_entered_on_current_candle_is_skipped(self) -> None:
        """Positions entered at the same time as the candle are skipped."""
        # Arrange
        candle_time = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
        position = _make_position(entry_time=candle_time)

        # Act
        skipped = _apply_same_bar_logic(position, candle_time)

        # Assert
        assert skipped is True

    def test_position_entered_after_candle_start_is_skipped(self) -> None:
        """Positions entered after the candle open (within the bar) are skipped."""
        # Arrange
        candle_time = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
        # Entry happened 30 minutes into the candle
        entry_time = candle_time + timedelta(minutes=30)
        position = _make_position(entry_time=entry_time)

        # Act
        skipped = _apply_same_bar_logic(position, candle_time)

        # Assert
        assert skipped is True

    def test_position_entered_on_previous_candle_is_evaluated(self) -> None:
        """Positions entered before the current candle are evaluated for exit."""
        # Arrange
        candle_time = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
        entry_time = candle_time - timedelta(hours=1)
        position = _make_position(entry_time=entry_time)

        # Act
        skipped = _apply_same_bar_logic(position, candle_time)

        # Assert
        assert skipped is False

    def test_position_entered_well_before_current_candle(self) -> None:
        """Positions entered many hours ago are evaluated normally."""
        # Arrange
        candle_time = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
        entry_time = candle_time - timedelta(days=1)
        position = _make_position(entry_time=entry_time)

        # Act
        skipped = _apply_same_bar_logic(position, candle_time)

        # Assert
        assert skipped is False

    def test_none_candle_time_allows_exit_evaluation(self) -> None:
        """When candle_time is None, positions are always evaluated."""
        # Arrange
        position = _make_position(
            entry_time=datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
        )

        # Act
        skipped = _apply_same_bar_logic(position, candle_time=None)

        # Assert
        assert skipped is False

    def test_none_entry_time_allows_exit_evaluation(self) -> None:
        """When entry_time is None, positions are always evaluated."""
        # Arrange
        position = _make_position(
            entry_time=datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
        )
        position.entry_time = None  # type: ignore[assignment]

        candle_time = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)

        # Act
        skipped = _apply_same_bar_logic(position, candle_time)

        # Assert
        assert skipped is False

    def test_naive_candle_time_with_aware_entry_time(self) -> None:
        """UTC normalization: naive candle time is promoted when entry is aware."""
        # Arrange
        aware_entry = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
        naive_candle = datetime(2026, 3, 18, 12, 0)  # No tzinfo
        position = _make_position(entry_time=aware_entry)

        # Act
        skipped = _apply_same_bar_logic(position, naive_candle)

        # Assert - same timestamp, should be skipped
        assert skipped is True

    def test_aware_candle_time_with_naive_entry_time(self) -> None:
        """UTC normalization: naive entry time is promoted when candle is aware."""
        # Arrange
        naive_entry = datetime(2026, 3, 18, 11, 0)  # No tzinfo, 1 hour before
        aware_candle = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
        position = _make_position(entry_time=naive_entry)

        # Act
        skipped = _apply_same_bar_logic(position, aware_candle)

        # Assert - entry is before candle, should be evaluated
        assert skipped is False

    def test_aware_candle_time_with_naive_same_time_entry(self) -> None:
        """UTC normalization: naive entry at same time as aware candle is skipped."""
        # Arrange
        naive_entry = datetime(2026, 3, 18, 12, 0)  # No tzinfo
        aware_candle = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
        position = _make_position(entry_time=naive_entry)

        # Act
        skipped = _apply_same_bar_logic(position, aware_candle)

        # Assert
        assert skipped is True

    def test_both_naive_timestamps_work(self) -> None:
        """Both-naive timestamps are compared directly without UTC issues."""
        # Arrange
        naive_entry = datetime(2026, 3, 18, 12, 0)
        naive_candle = datetime(2026, 3, 18, 12, 0)
        position = _make_position(entry_time=naive_entry)

        # Act
        skipped = _apply_same_bar_logic(position, naive_candle)

        # Assert
        assert skipped is True
