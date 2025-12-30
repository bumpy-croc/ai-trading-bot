"""
Unit tests for order status management methods in DatabaseManager.

Tests the new methods for handling order status transitions and validation.
"""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest

from src.database.manager import DatabaseManager
from src.database.models import OrderStatus


class TestOrderStatusMethods:
    """Test order status management methods."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager for unit testing."""
        return Mock(spec=DatabaseManager)

    def test_update_order_status_new_success(self):
        """Test successful updating of order status using new Order table."""
        db_manager = DatabaseManager()

        # * Mock the database session and order
        mock_order = Mock()
        mock_order.status = OrderStatus.PENDING
        mock_order.filled_quantity = None
        mock_order.filled_price = None

        with patch.object(db_manager, "get_session") as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_order

            # * Test the method
            result = db_manager.update_order_status_new(
                order_id=1, status=OrderStatus.FILLED, filled_quantity=0.001, filled_price=50100.0
            )

            # * Verify success
            assert result is True
            assert mock_order.status == OrderStatus.FILLED
            assert float(mock_order.filled_quantity) == 0.001  # Convert Decimal to float
            assert float(mock_order.filled_price) == 50100.0  # Convert Decimal to float
            mock_session.commit.assert_called_once()

    def test_update_order_status_new_not_found(self):
        """Test updating order when not found."""
        db_manager = DatabaseManager()

        with patch.object(db_manager, "get_session") as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter_by.return_value.first.return_value = None

            result = db_manager.update_order_status_new(order_id=999, status=OrderStatus.FILLED)

            assert result is False

    def test_update_order_status_new_already_filled(self):
        """Test updating order that's already filled (should still succeed)."""
        db_manager = DatabaseManager()

        mock_order = Mock()
        mock_order.status = OrderStatus.FILLED  # Already filled
        mock_order.filled_at = datetime.now(UTC)  # Already has filled timestamp

        with patch.object(db_manager, "get_session") as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_order

            result = db_manager.update_order_status_new(order_id=1, status=OrderStatus.FILLED)

            # The method doesn't prevent updating already filled orders
            assert result is True

    def test_validate_position_status_consistency(self):
        """Test position status consistency validation."""
        db_manager = DatabaseManager()

        # * Mock query results for current implementation
        # orphaned_positions (OPEN positions with missing data)
        # total_open (all OPEN positions)
        # total_closed (all CLOSED positions)
        mock_results = [
            (3,),  # orphaned_open (OPEN positions with NULL/0 data)
            (8,),  # total_open
            (12,),  # total_closed
        ]

        with patch.object(db_manager, "get_session") as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value.__enter__.return_value = mock_session

            # * Set up execute calls for the 3 queries in current implementation
            mock_session.execute.side_effect = [
                Mock(fetchone=lambda: mock_results[0]),  # orphaned_positions
                Mock(fetchone=lambda: mock_results[1]),  # total_open
                Mock(fetchone=lambda: mock_results[2]),  # total_closed
            ]

            result = db_manager.validate_position_status_consistency()

            expected = {
                "orphaned_open": 3,
                "total_open": 8,
                "total_closed": 12,
            }
            assert result == expected

    def test_fix_position_status_inconsistencies(self):
        """Test fixing position status inconsistencies."""
        db_manager = DatabaseManager()

        with patch.object(db_manager, "get_session") as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value.__enter__.return_value = mock_session

            # * Mock update result for orphaned positions fix
            mock_result_orphaned = Mock(rowcount=5)
            mock_session.execute.return_value = mock_result_orphaned

            result = db_manager.fix_position_status_inconsistencies()

            expected = {
                "orphaned_to_closed": 5,
            }
            assert result == expected
            mock_session.commit.assert_called_once()

    def test_get_pending_orders_method_exists(self):
        """Test that get_pending_orders method exists and is callable."""
        db_manager = DatabaseManager()

        # * Verify method exists
        assert hasattr(db_manager, "get_pending_orders")
        assert callable(db_manager.get_pending_orders)

        # * Verify old method name doesn't exist
        assert not hasattr(db_manager, "get_open_orders")

    def test_update_order_status_with_logging(self):
        """Test enhanced update_order_status with improved logging."""
        db_manager = DatabaseManager()

        mock_position = Mock()
        mock_position.status = OrderStatus.PENDING

        with (
            patch.object(db_manager, "get_session") as mock_get_session,
            patch.object(db_manager, "_normalize_order_status") as mock_normalize,
        ):

            mock_session = Mock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter_by.return_value.first.return_value = (
                mock_position
            )
            mock_normalize.return_value = OrderStatus.FILLED

            result = db_manager.update_order_status(1, "FILLED")

            assert result is True
            assert mock_position.status == OrderStatus.FILLED
            mock_normalize.assert_called_once_with("FILLED")
            mock_session.commit.assert_called_once()

    @pytest.mark.parametrize(
        "status,expected",
        [
            ("PENDING", OrderStatus.PENDING),
            ("OPEN", OrderStatus.OPEN),
            ("FILLED", OrderStatus.FILLED),
            ("CANCELLED", OrderStatus.CANCELLED),
            ("FAILED", OrderStatus.FAILED),
        ],
    )
    def test_order_status_normalization(self, status, expected):
        """Test order status normalization for various inputs."""
        db_manager = DatabaseManager()

        # * Test string normalization
        result = db_manager._normalize_order_status(status)
        assert result == expected

        # * Test enum input (should pass through)
        result = db_manager._normalize_order_status(expected)
        assert result == expected

    def test_invalid_order_status_normalization(self):
        """Test handling of invalid order status."""
        db_manager = DatabaseManager()

        with pytest.raises(ValueError):
            db_manager._normalize_order_status("INVALID_STATUS")

    def test_update_order_status_new_partial_parameters(self):
        """Test update_order_status_new with only some parameters provided."""
        db_manager = DatabaseManager()

        mock_order = Mock()
        mock_order.status = OrderStatus.PENDING
        mock_order.filled_quantity = None
        mock_order.filled_price = None

        with patch.object(db_manager, "get_session") as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_order

            # * Test with only filled_price
            result = db_manager.update_order_status_new(
                order_id=1, status=OrderStatus.FILLED, filled_price=50200.0
            )

            assert result is True
            assert mock_order.status == OrderStatus.FILLED
            assert float(mock_order.filled_price) == 50200.0
            assert mock_order.filled_quantity is None  # Unchanged

            # * Reset for next test
            mock_order.status = OrderStatus.PENDING
            mock_order.filled_quantity = None

            # * Test with only filled_quantity
            result = db_manager.update_order_status_new(
                order_id=1, status=OrderStatus.FILLED, filled_quantity=0.002
            )

            assert result is True
            assert mock_order.status == OrderStatus.FILLED
            assert float(mock_order.filled_price) == 50200.0  # Unchanged from previous
            assert float(mock_order.filled_quantity) == 0.002  # Updated
