"""Session management tests for DatabaseManager."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, call, patch

import pytest

pytestmark = pytest.mark.unit


class TestSessionManagement:
    """Test session management methods"""

    def test_create_trading_session(self, mock_postgresql_db):
        """Test creating a trading session"""
        mock_session_obj = Mock()
        mock_session_obj.id = 123
        mock_postgresql_db._mock_session.add.return_value = None

        with patch("database.manager.TradingSession") as mock_trading_session_class:
            mock_trading_session_class.return_value = mock_session_obj

            session_id = mock_postgresql_db.create_trading_session(
                strategy_name="TestStrategy",
                symbol="BTCUSDT",
                timeframe="1h",
                mode="PAPER",
                initial_balance=10000.0,
            )

            assert session_id == 123
            mock_postgresql_db._mock_session.add.assert_called()
            mock_postgresql_db._mock_session.commit.assert_called()

    def test_end_trading_session(self, mock_postgresql_db):
        """Test ending a trading session"""
        mock_session_obj = Mock()
        mock_session_obj.id = 123
        mock_session_obj.session_name = "test_session"
        mock_session_obj.start_time = datetime.now(UTC)

        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_session_obj
        mock_query.filter_by.return_value.all.return_value = []
        mock_postgresql_db._mock_session.query.return_value = mock_query

        mock_postgresql_db._current_session_id = 123
        mock_postgresql_db.end_trading_session()

        mock_postgresql_db._mock_session.commit.assert_called()


class TestGetLastSessionId:
    """Tests for DatabaseManager.get_last_session_id() — the clean-restart fallback."""

    def _build_query_chain(self, mock_session: Mock, result: Mock | None) -> Mock:
        """Wire up a mock query chain ending in .first() -> result."""
        mock_query = Mock()
        mock_filter = Mock()
        mock_order = Mock()

        mock_session.query.return_value = mock_query
        # Each .filter() call returns the same mock so chaining works.
        mock_query.filter.return_value = mock_filter
        mock_filter.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_order
        mock_order.first.return_value = result
        return mock_query

    @pytest.mark.fast
    def test_returns_id_of_inactive_session(self, mock_postgresql_db):
        """A recently-closed (is_active=False) session's ID is returned."""
        inactive = Mock()
        inactive.id = 99
        inactive.is_active = False
        self._build_query_chain(mock_postgresql_db._mock_session, inactive)

        result = mock_postgresql_db.get_last_session_id(within_hours=24)

        assert result == 99

    @pytest.mark.fast
    def test_returns_none_when_no_session_in_window(self, mock_postgresql_db):
        """None is returned when no session exists within the time window."""
        self._build_query_chain(mock_postgresql_db._mock_session, None)

        result = mock_postgresql_db.get_last_session_id(within_hours=24)

        assert result is None

    @pytest.mark.fast
    def test_excludes_active_sessions(self, mock_postgresql_db):
        """get_last_session_id must never return an is_active=True session.

        Returning an active session would let a second engine instance share the
        same session row as a concurrently-running engine, corrupting trade
        attribution and balance accounting. The ~TradingSession.is_active filter
        in the query prevents this. We verify by confirming filter() receives two
        conditions (the time cutoff AND ~is_active), not just one.
        """
        mock_query = Mock()
        mock_filter = Mock()
        mock_order = Mock()
        mock_postgresql_db._mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_order
        mock_order.first.return_value = None

        mock_postgresql_db.get_last_session_id(within_hours=24)

        # filter() must be called with exactly 2 conditions: cutoff AND ~is_active.
        assert mock_query.filter.call_count == 1
        filter_args = mock_query.filter.call_args[0]
        assert len(filter_args) == 2, (
            "filter() should receive 2 conditions (cutoff + ~is_active). "
            f"Got {len(filter_args)}: {filter_args}"
        )
