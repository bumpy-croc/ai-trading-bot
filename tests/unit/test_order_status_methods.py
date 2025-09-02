"""
Unit tests for order status management methods in DatabaseManager.

Tests the new methods for handling order status transitions and validation.
"""

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

    def test_fill_pending_order_success(self):
        """Test successful filling of a pending order."""
        db_manager = DatabaseManager()
        
        # * Mock the database session and position
        mock_position = Mock()
        mock_position.status = OrderStatus.PENDING
        mock_position.entry_price = 50000.0
        mock_position.quantity = 0.001
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_position
            
            # * Test the method
            result = db_manager.fill_pending_order(
                order_id="test_order",
                filled_price=50100.0,
                filled_quantity=0.0015
            )
            
            # * Verify success
            assert result is True
            assert mock_position.status == OrderStatus.OPEN
            assert mock_position.entry_price == 50100.0
            assert mock_position.quantity == 0.0015
            mock_session.commit.assert_called_once()

    def test_fill_pending_order_not_found(self):
        """Test filling order when position not found."""
        db_manager = DatabaseManager()
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter_by.return_value.first.return_value = None
            
            result = db_manager.fill_pending_order("nonexistent_order")
            
            assert result is False

    def test_fill_pending_order_wrong_status(self):
        """Test filling order that's not in PENDING status."""
        db_manager = DatabaseManager()
        
        mock_position = Mock()
        mock_position.status = OrderStatus.OPEN  # ! Already filled
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_position
            
            result = db_manager.fill_pending_order("already_filled_order")
            
            assert result is False

    def test_validate_position_status_consistency(self):
        """Test position status consistency validation."""
        db_manager = DatabaseManager()
        
        # * Mock query results
        mock_results = [
            (2,),  # inconsistent_pending
            (1,),  # orphaned_positions  
            (5,),  # total_pending
            (10,), # total_open
        ]
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            
            # * Set up multiple execute calls
            mock_session.execute.side_effect = [
                Mock(fetchone=lambda: mock_results[0]),
                Mock(fetchone=lambda: mock_results[1]), 
                Mock(fetchone=lambda: mock_results[2]),
                Mock(fetchone=lambda: mock_results[3]),
            ]
            
            result = db_manager.validate_position_status_consistency()
            
            expected = {
                "inconsistent_pending": 2,
                "orphaned_open": 1,
                "total_pending": 5,
                "total_open": 10,
            }
            assert result == expected

    def test_fix_position_status_inconsistencies(self):
        """Test fixing position status inconsistencies."""
        db_manager = DatabaseManager()
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            
            # * Mock update results
            mock_result_pending = Mock(rowcount=3)
            mock_result_orphaned = Mock(rowcount=1)
            mock_session.execute.side_effect = [mock_result_pending, mock_result_orphaned]
            
            result = db_manager.fix_position_status_inconsistencies()
            
            expected = {
                "pending_to_open": 3,
                "orphaned_to_failed": 1,
            }
            assert result == expected
            mock_session.commit.assert_called_once()

    def test_get_pending_orders_method_exists(self):
        """Test that get_pending_orders method exists and is callable."""
        db_manager = DatabaseManager()
        
        # * Verify method exists
        assert hasattr(db_manager, 'get_pending_orders')
        assert callable(db_manager.get_pending_orders)
        
        # * Verify old method name doesn't exist
        assert not hasattr(db_manager, 'get_open_orders')

    def test_update_order_status_with_logging(self):
        """Test enhanced update_order_status with improved logging."""
        db_manager = DatabaseManager()
        
        mock_position = Mock()
        mock_position.status = OrderStatus.PENDING
        
        with patch.object(db_manager, 'get_session') as mock_get_session, \
             patch.object(db_manager, '_normalize_order_status') as mock_normalize:
            
            mock_session = Mock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_position
            mock_normalize.return_value = OrderStatus.FILLED
            
            result = db_manager.update_order_status(1, "FILLED")
            
            assert result is True
            assert mock_position.status == OrderStatus.FILLED
            mock_normalize.assert_called_once_with("FILLED")
            mock_session.commit.assert_called_once()

    @pytest.mark.parametrize("status,expected", [
        ("PENDING", OrderStatus.PENDING),
        ("OPEN", OrderStatus.OPEN), 
        ("FILLED", OrderStatus.FILLED),
        ("CANCELLED", OrderStatus.CANCELLED),
        ("FAILED", OrderStatus.FAILED),
    ])
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

    def test_fill_pending_order_partial_parameters(self):
        """Test fill_pending_order with only some parameters provided."""
        db_manager = DatabaseManager()
        
        mock_position = Mock()
        mock_position.status = OrderStatus.PENDING
        mock_position.entry_price = 50000.0
        mock_position.quantity = 0.001
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_position
            
            # * Test with only filled_price
            result = db_manager.fill_pending_order(
                order_id="test_order",
                filled_price=50200.0
            )
            
            assert result is True
            assert mock_position.status == OrderStatus.OPEN
            assert mock_position.entry_price == 50200.0
            assert mock_position.quantity == 0.001  # Unchanged
            
            # * Reset for next test
            mock_position.status = OrderStatus.PENDING
            
            # * Test with only filled_quantity  
            result = db_manager.fill_pending_order(
                order_id="test_order",
                filled_quantity=0.002
            )
            
            assert result is True
            assert mock_position.status == OrderStatus.OPEN
            assert mock_position.entry_price == 50200.0  # Unchanged from previous
            assert mock_position.quantity == 0.002  # Updated
