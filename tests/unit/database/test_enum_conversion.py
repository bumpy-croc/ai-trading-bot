"""Enum conversion tests for DatabaseManager."""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.unit


class TestEnumConversion:
    """Test enum string conversion"""

    def test_trade_side_string_conversion(self, mock_postgresql_db):
        """Test converting side string to enum"""
        mock_trade_obj = Mock()
        mock_trade_obj.id = 456

        with patch("database.manager.Trade") as mock_trade_class:
            mock_trade_class.return_value = mock_trade_obj

            trade_id = mock_postgresql_db.log_trade(
                symbol="BTCUSDT",
                side="LONG",
                entry_price=45000.0,
                exit_price=46000.0,
                size=0.1,
                entry_time=datetime.now(UTC),
                exit_time=datetime.now(UTC),
                pnl=100.0,
                exit_reason="Take profit",
                strategy_name="TestStrategy",
            )

            assert trade_id == 456

    def test_event_type_string_conversion(self, mock_postgresql_db):
        """Test converting event type string to enum"""
        mock_event_obj = Mock()
        mock_event_obj.id = 101

        with patch("database.manager.SystemEvent") as mock_event_class:
            mock_event_class.return_value = mock_event_obj

            event_id = mock_postgresql_db.log_event(
                event_type="TEST",
                message="Test event message",
            )

            assert event_id == 101
