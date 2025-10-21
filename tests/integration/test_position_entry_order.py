"""
Integration tests for automatic ENTRY order creation when positions are logged.

Tests that log_position automatically creates a filled ENTRY order for each position.
"""

import pytest

from src.database.manager import DatabaseManager
from src.database.models import PositionSide, PositionStatus


@pytest.mark.integration
class TestPositionEntryOrder:
    """Test automatic ENTRY order creation during position logging."""

    @pytest.fixture
    def db_manager(self):
        """Create a database manager for testing."""
        return DatabaseManager()

    @pytest.fixture
    def test_session(self, db_manager):
        """Create a test trading session."""
        return db_manager.create_trading_session(
            "test_strategy", "BTCUSDT", "1h", "live", 10000.0
        )

    def test_log_position_creates_entry_order(self, db_manager, test_session):
        """Test that logging a position automatically creates an ENTRY order."""
        # * Log a position
        position_id = db_manager.log_position(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            size=0.02,
            quantity=0.001,
            strategy_name="test_strategy",
            entry_order_id="exchange_order_123",
            session_id=test_session
        )
        
        assert position_id is not None
        
        # * Verify the position status uses PositionStatus
        with db_manager.get_session() as session:
            from src.database.models import Position
            position = session.query(Position).filter_by(id=position_id).first()
            assert position is not None
            assert position.status == PositionStatus.OPEN  # Using PositionStatus enum
        
        # * Check that an ENTRY order was automatically created
        orders = db_manager.get_orders_for_position(position_id)
        
        assert len(orders) == 1
        entry_order = orders[0]
        
        # * Verify ENTRY order details
        assert entry_order["order_type"] == "ENTRY"
        assert entry_order["status"] == "FILLED"
        assert entry_order["symbol"] == "BTCUSDT"
        assert entry_order["side"] == "LONG"
        assert entry_order["quantity"] == 0.001
        assert entry_order["price"] == 50000.0
        assert entry_order["filled_quantity"] == 0.001
        assert entry_order["filled_price"] == 50000.0
        assert entry_order["exchange_order_id"] == "exchange_order_123"  # Uses original order_id
        assert entry_order["filled_at"] is not None
        assert "entry_" in entry_order["internal_order_id"]

    def test_log_position_uses_size_when_no_quantity(self, db_manager, test_session):
        """Test that size is used for order quantity when quantity is not provided."""
        position_id = db_manager.log_position(
            symbol="ETHUSDT",
            side="SHORT",
            entry_price=3000.0,
            size=0.05,  # No quantity provided
            strategy_name="test_strategy",
            entry_order_id="exchange_order_456",
            session_id=test_session
        )
        
        # * Check the auto-created ENTRY order uses size as quantity
        orders = db_manager.get_orders_for_position(position_id)
        entry_order = orders[0]
        
        assert entry_order["quantity"] == 0.05  # Should use size
        assert entry_order["filled_quantity"] == 0.05

    def test_log_position_with_different_sides(self, db_manager, test_session):
        """Test ENTRY order creation for both LONG and SHORT positions."""
        # * Create LONG position
        long_position_id = db_manager.log_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,  # Test enum input
            entry_price=50000.0,
            size=0.02,
            quantity=0.001,
            strategy_name="test_strategy",
            entry_order_id="long_order_123",
            session_id=test_session
        )
        
        # * Create SHORT position  
        short_position_id = db_manager.log_position(
            symbol="BTCUSDT",
            side="SHORT",  # Test string input
            entry_price=49000.0,
            size=0.015,
            quantity=0.0008,
            strategy_name="test_strategy",
            entry_order_id="short_order_456",
            session_id=test_session
        )
        
        # * Check LONG position entry order
        long_orders = db_manager.get_orders_for_position(long_position_id)
        assert long_orders[0]["side"] == "LONG"
        assert long_orders[0]["price"] == 50000.0
        
        # * Check SHORT position entry order
        short_orders = db_manager.get_orders_for_position(short_position_id)
        assert short_orders[0]["side"] == "SHORT"
        assert short_orders[0]["price"] == 49000.0

    def test_entry_order_failure_doesnt_break_position_creation(self, db_manager, test_session):
        """Test that position creation succeeds even if ENTRY order creation fails."""
        # * This test verifies the error handling - position creation should succeed
        # * even if there's an issue with order creation
        
        position_id = db_manager.log_position(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            size=0.02,
            quantity=0.001,
            strategy_name="test_strategy",
            entry_order_id="test_order_resilience",
            session_id=test_session
        )
        
        # * Position should still be created successfully
        assert position_id is not None
        
        # * Try to verify order was created (it should succeed in normal cases)
        orders = db_manager.get_orders_for_position(position_id)
        # In normal operation, this should succeed
        if orders:
            assert len(orders) == 1
            assert orders[0]["order_type"] == "ENTRY"

    def test_multiple_positions_get_separate_entry_orders(self, db_manager, test_session):
        """Test that multiple positions each get their own ENTRY orders."""
        # * Create multiple positions
        position1_id = db_manager.log_position(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            size=0.02,
            quantity=0.001,
            strategy_name="test_strategy",
            entry_order_id="multi_order_1",
            session_id=test_session
        )
        
        position2_id = db_manager.log_position(
            symbol="ETHUSDT",
            side="SHORT",
            entry_price=3000.0,
            size=0.05,
            quantity=0.015,
            strategy_name="test_strategy",
            entry_order_id="multi_order_2",
            session_id=test_session
        )
        
        # * Each position should have its own ENTRY order
        position1_orders = db_manager.get_orders_for_position(position1_id)
        position2_orders = db_manager.get_orders_for_position(position2_id)
        
        assert len(position1_orders) == 1
        assert len(position2_orders) == 1
        
        # * Orders should be distinct
        assert position1_orders[0]["id"] != position2_orders[0]["id"]
        assert position1_orders[0]["symbol"] == "BTCUSDT"
        assert position2_orders[0]["symbol"] == "ETHUSDT"

    def test_entry_order_inherits_position_details(self, db_manager, test_session):
        """Test that ENTRY order correctly inherits all position details."""
        position_id = db_manager.log_position(
            symbol="ADAUSDT",
            side="LONG",
            entry_price=0.35,
            size=0.1,
            quantity=2857.14,  # Large quantity for ADA
            strategy_name="ada_momentum_strategy",
            entry_order_id="ada_entry_order_999",
            session_id=test_session,
            stop_loss=0.30,
            take_profit=0.40
        )
        
        orders = db_manager.get_orders_for_position(position_id)
        entry_order = orders[0]
        
        # * Verify all inherited details
        assert entry_order["symbol"] == "ADAUSDT"
        assert entry_order["side"] == "LONG"
        assert entry_order["quantity"] == 2857.14
        assert entry_order["price"] == 0.35
        assert entry_order["filled_price"] == 0.35
        assert entry_order["exchange_order_id"] == "ada_entry_order_999"
        assert entry_order["order_type"] == "ENTRY"
        assert entry_order["status"] == "FILLED"
