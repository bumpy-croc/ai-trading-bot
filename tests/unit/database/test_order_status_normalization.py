"""Order status normalization tests for DatabaseManager."""

from unittest.mock import Mock

import pytest

from src.database.models import OrderStatus

pytestmark = pytest.mark.unit


class TestOrderStatusNormalization:
    """Tests for order status normalization in DatabaseManager.update_order_status"""

    def test_update_order_status_normalizes_values(self, mock_postgresql_db):
        mock_position = Mock()
        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_position
        mock_postgresql_db._mock_session.query.return_value = mock_query

        assert mock_postgresql_db.update_order_status(1, "open") is True
        assert mock_position.status.value == OrderStatus.OPEN.value

        assert mock_postgresql_db.update_order_status(1, "Partially_Filled") is True
        assert mock_position.status.value == OrderStatus.FILLED.value

        assert mock_postgresql_db.update_order_status(1, "rejected") is True
        assert mock_position.status.value == OrderStatus.FAILED.value

        assert mock_postgresql_db.update_order_status(1, "expired") is True
        assert mock_position.status.value == OrderStatus.CANCELLED.value

    def test_update_order_status_invalid_value(self, mock_postgresql_db):
        mock_position = Mock()
        mock_query = Mock()
        mock_query.filter_by.return_value.first.return_value = mock_position
        mock_postgresql_db._mock_session.query.return_value = mock_query

        assert mock_postgresql_db.update_order_status(1, "unknown_status") is False
