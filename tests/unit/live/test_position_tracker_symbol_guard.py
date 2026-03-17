"""Tests for LivePositionTracker.has_position_for_symbol.

Guards against opening duplicate positions on the same asset when
max_concurrent_positions > 1.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.engines.live.execution.position_tracker import (
    LivePosition,
    LivePositionTracker,
    PositionSide,
)


def _make_position(symbol: str, order_id: str) -> LivePosition:
    return LivePosition(
        symbol=symbol,
        side=PositionSide.SHORT,
        size=0.1,
        entry_price=50000.0,
        entry_time=datetime.now(UTC),
        order_id=order_id,
    )


@pytest.fixture
def tracker() -> LivePositionTracker:
    return LivePositionTracker()


@pytest.fixture
def mock_position() -> MagicMock:
    pos = MagicMock(spec=LivePosition)
    pos.symbol = "BTCUSDT"
    pos.order_id = "order_1"
    return pos


@pytest.mark.fast
def test_has_position_for_symbol_returns_true_when_position_exists(
    tracker: LivePositionTracker, mock_position: MagicMock
) -> None:
    """Tracker correctly detects an existing position on the symbol."""
    mock_position.symbol = "BTCUSDT"
    mock_position.order_id = "order_1"
    tracker._positions["order_1"] = mock_position
    assert tracker.has_position_for_symbol("BTCUSDT") is True


@pytest.mark.fast
def test_has_position_for_symbol_returns_false_when_no_position(
    tracker: LivePositionTracker,
) -> None:
    """Tracker correctly reports no position for an unknown symbol."""
    assert tracker.has_position_for_symbol("BTCUSDT") is False


@pytest.mark.fast
def test_has_position_for_symbol_does_not_collide_across_symbols(
    tracker: LivePositionTracker, mock_position: MagicMock
) -> None:
    """A position on BTCUSDT does not block ETHUSDT entry."""
    mock_position.symbol = "BTCUSDT"
    mock_position.order_id = "order_1"
    tracker._positions["order_1"] = mock_position
    assert tracker.has_position_for_symbol("ETHUSDT") is False


@pytest.mark.fast
def test_has_position_for_symbol_is_exact_match(tracker: LivePositionTracker) -> None:
    """Symbol matching is exact — lowercase does not match uppercase position."""
    position = _make_position("BTCUSDT", "order_1")
    tracker._positions["order_1"] = position
    assert tracker.has_position_for_symbol("btcusdt") is False
    assert tracker.has_position_for_symbol("BTCUSDT") is True


@pytest.mark.fast
def test_has_position_for_symbol_detects_multiple_positions(
    tracker: LivePositionTracker,
) -> None:
    """Returns True even when the target symbol is not the first position."""
    tracker._positions["order_1"] = _make_position("ETHUSDT", "order_1")
    tracker._positions["order_2"] = _make_position("BTCUSDT", "order_2")
    assert tracker.has_position_for_symbol("BTCUSDT") is True
