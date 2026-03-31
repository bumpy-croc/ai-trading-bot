"""Tests for MarginInterestTracker service."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.engines.live.margin_interest_tracker import MarginInterestTracker


@pytest.fixture
def mock_exchange() -> MagicMock:
    """Create a mock exchange with get_margin_interest_history method."""
    return MagicMock()


@pytest.fixture
def tracker(mock_exchange: MagicMock) -> MarginInterestTracker:
    """Create a MarginInterestTracker with mocked exchange."""
    return MarginInterestTracker(exchange=mock_exchange)


class TestGetPositionInterestCost:
    """Tests for get_position_interest_cost method."""

    def test_returns_summed_interest_from_records(
        self, tracker: MarginInterestTracker, mock_exchange: MagicMock
    ) -> None:
        """Sum interest field (as string) from all returned records."""
        mock_exchange.get_margin_interest_history.return_value = [
            {"interest": "0.00012345", "asset": "BTC"},
            {"interest": "0.00034567", "asset": "BTC"},
        ]
        entry_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)

        result = tracker.get_position_interest_cost("BTC", entry_time)

        assert math.isclose(result, 0.00046912, rel_tol=1e-9)

    def test_returns_zero_when_no_records(
        self, tracker: MarginInterestTracker, mock_exchange: MagicMock
    ) -> None:
        """Return 0.0 when exchange returns empty list."""
        mock_exchange.get_margin_interest_history.return_value = []
        entry_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)

        result = tracker.get_position_interest_cost("BTC", entry_time)

        assert result == 0.0

    def test_returns_zero_on_exchange_error(
        self, tracker: MarginInterestTracker, mock_exchange: MagicMock
    ) -> None:
        """Return 0.0 when exchange raises an exception."""
        mock_exchange.get_margin_interest_history.side_effect = Exception(
            "API error"
        )
        entry_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)

        result = tracker.get_position_interest_cost("BTC", entry_time)

        assert result == 0.0

    def test_skips_non_finite_interest_values(
        self, tracker: MarginInterestTracker, mock_exchange: MagicMock
    ) -> None:
        """Skip records with non-finite interest (inf, nan)."""
        mock_exchange.get_margin_interest_history.return_value = [
            {"interest": "0.001", "asset": "BTC"},
            {"interest": "inf", "asset": "BTC"},
            {"interest": "nan", "asset": "BTC"},
            {"interest": "0.002", "asset": "BTC"},
        ]
        entry_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)

        result = tracker.get_position_interest_cost("BTC", entry_time)

        assert math.isclose(result, 0.003, rel_tol=1e-9)

    def test_converts_entry_time_to_milliseconds(
        self, tracker: MarginInterestTracker, mock_exchange: MagicMock
    ) -> None:
        """Pass entry_time as milliseconds timestamp to exchange API."""
        mock_exchange.get_margin_interest_history.return_value = []
        # 2025-01-15T10:00:00Z = 1736935200 seconds = 1736935200000 ms
        entry_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)

        tracker.get_position_interest_cost("BTC", entry_time)

        mock_exchange.get_margin_interest_history.assert_called_once_with(
            asset="BTC", start_time=1736935200000
        )

    def test_returns_zero_when_exchange_returns_none(
        self, tracker: MarginInterestTracker, mock_exchange: MagicMock
    ) -> None:
        """Return 0.0 when exchange returns None instead of a list."""
        mock_exchange.get_margin_interest_history.return_value = None
        entry_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)

        result = tracker.get_position_interest_cost("BTC", entry_time)

        assert result == 0.0

    def test_skips_records_with_invalid_interest_string(
        self, tracker: MarginInterestTracker, mock_exchange: MagicMock
    ) -> None:
        """Skip records where interest cannot be converted to float."""
        mock_exchange.get_margin_interest_history.return_value = [
            {"interest": "0.001", "asset": "BTC"},
            {"interest": "not_a_number", "asset": "BTC"},
            {"interest": "0.002", "asset": "BTC"},
        ]
        entry_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)

        result = tracker.get_position_interest_cost("BTC", entry_time)

        assert math.isclose(result, 0.003, rel_tol=1e-9)


class TestIsMarginPosition:
    """Tests for is_margin_position method."""

    def test_short_is_margin_position(
        self, tracker: MarginInterestTracker
    ) -> None:
        """SHORT positions borrow and incur interest."""
        assert tracker.is_margin_position("SHORT") is True

    def test_long_is_not_margin_position(
        self, tracker: MarginInterestTracker
    ) -> None:
        """LONG positions do not borrow."""
        assert tracker.is_margin_position("LONG") is False

    def test_other_side_is_not_margin_position(
        self, tracker: MarginInterestTracker
    ) -> None:
        """Arbitrary strings are not margin positions."""
        assert tracker.is_margin_position("BUY") is False
        assert tracker.is_margin_position("") is False
