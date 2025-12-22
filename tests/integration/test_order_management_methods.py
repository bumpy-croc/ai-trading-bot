"""
Integration tests for new Order management methods in DatabaseManager.

Tests the Phase 2 methods: create_order, update_order_status_new, 
get_orders_for_position, get_pending_orders_new.
"""

import pytest

from src.database.manager import DatabaseManager
from src.database.models import OrderStatus, OrderType, PositionSide


@pytest.mark.integration
class TestOrderManagementMethods:
    """Test new Order management methods in DatabaseManager."""

    @pytest.fixture
    def db_manager(self):
        """Create a database manager for testing."""
        return DatabaseManager()

    @pytest.fixture
    def test_position(self, db_manager):
        """Create a test position for order testing."""
        import random
        import time

        session_id = db_manager.create_trading_session(
            "test_strategy", "BTCUSDT", "1h", "live", 10000.0
        )

        # * Generate unique entry_order_id to avoid unique constraint violations
        unique_id = f"test_position_{int(time.time() * 1000000)}_{random.randint(1000, 9999)}"

        position_id = db_manager.log_position(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            size=0.02,
            strategy_name="test_strategy",
            entry_order_id=unique_id,
            session_id=session_id,
        )

        return {"position_id": position_id, "session_id": session_id}

    def test_create_order_basic(self, db_manager, test_position):
        """Test basic order creation."""
        order_id = db_manager.create_order(
            position_id=test_position["position_id"],
            order_type=OrderType.ENTRY,
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.001,
            strategy_name="test_strategy",
            session_id=test_position["session_id"],
            price=50000.0,
        )

        assert order_id is not None
        assert isinstance(order_id, int)
        assert order_id > 0

    def test_create_order_with_string_enums(self, db_manager, test_position):
        """Test order creation with string enum values."""
        order_id = db_manager.create_order(
            position_id=test_position["position_id"],
            order_type="PARTIAL_EXIT",  # String instead of enum
            symbol="BTCUSDT",
            side="SHORT",  # String instead of enum
            quantity=0.0005,
            strategy_name="test_strategy",
            session_id=test_position["session_id"],
            target_level=1,
            size_fraction=0.5,
        )

        assert order_id is not None

        # * Verify the order was created correctly
        orders = db_manager.get_orders_for_position(test_position["position_id"])
        created_order = next(o for o in orders if o["id"] == order_id)
        assert created_order["order_type"] == "PARTIAL_EXIT"
        assert created_order["side"] == "SHORT"  # Enum values are uppercase in database
        assert created_order["target_level"] == 1
        assert created_order["size_fraction"] == 0.5

    def test_create_order_auto_internal_id(self, db_manager, test_position):
        """Test that internal order ID is auto-generated when not provided."""
        order_id = db_manager.create_order(
            position_id=test_position["position_id"],
            order_type=OrderType.SCALE_IN,
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.0005,
            strategy_name="test_strategy",
            session_id=test_position["session_id"],
        )

        orders = db_manager.get_orders_for_position(test_position["position_id"])
        created_order = next(o for o in orders if o["id"] == order_id)

        # * Should have generated internal order ID
        assert created_order["internal_order_id"] is not None
        assert "scale_in" in created_order["internal_order_id"]
        assert str(test_position["position_id"]) in created_order["internal_order_id"]

    def test_update_order_status_new(self, db_manager, test_position):
        """Test updating order status with execution details."""
        # * Create an order
        order_id = db_manager.create_order(
            position_id=test_position["position_id"],
            order_type=OrderType.ENTRY,
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.001,
            strategy_name="test_strategy",
            session_id=test_position["session_id"],
        )

        # * Update to filled with execution details
        success = db_manager.update_order_status_new(
            order_id=order_id,
            status=OrderStatus.FILLED,
            filled_quantity=0.001,
            filled_price=50100.0,
            exchange_order_id="binance_67890",
            commission=0.5,
        )

        assert success is True

        # * Verify the update
        orders = db_manager.get_orders_for_position(test_position["position_id"])
        updated_order = next(o for o in orders if o["id"] == order_id)

        assert updated_order["status"] == "FILLED"
        assert updated_order["filled_quantity"] == 0.001
        assert updated_order["filled_price"] == 50100.0
        assert updated_order["exchange_order_id"] == "binance_67890"
        assert updated_order["commission"] == 0.5
        assert updated_order["filled_at"] is not None

    def test_update_order_status_cancelled(self, db_manager, test_position):
        """Test updating order status to cancelled."""
        # * Create an order
        order_id = db_manager.create_order(
            position_id=test_position["position_id"],
            order_type=OrderType.PARTIAL_EXIT,
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            quantity=0.0005,
            strategy_name="test_strategy",
            session_id=test_position["session_id"],
        )

        # * Cancel the order
        success = db_manager.update_order_status_new(
            order_id=order_id, status="CANCELLED"  # Test string enum conversion
        )

        assert success is True

        # * Verify cancellation
        orders = db_manager.get_orders_for_position(test_position["position_id"])
        cancelled_order = next(o for o in orders if o["id"] == order_id)

        assert cancelled_order["status"] == "CANCELLED"
        assert cancelled_order["cancelled_at"] is not None

    def test_get_orders_for_position(self, db_manager, test_position):
        """Test retrieving all orders for a position."""
        # * Create multiple orders for the same position
        db_manager.create_order(
            position_id=test_position["position_id"],
            order_type=OrderType.ENTRY,
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.001,
            strategy_name="test_strategy",
            session_id=test_position["session_id"],
        )

        db_manager.create_order(
            position_id=test_position["position_id"],
            order_type=OrderType.PARTIAL_EXIT,
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            quantity=0.0005,
            strategy_name="test_strategy",
            session_id=test_position["session_id"],
            target_level=1,
        )

        # * Retrieve all orders
        orders = db_manager.get_orders_for_position(test_position["position_id"])

        # * Note: log_position automatically creates 1 ENTRY order, plus 2 created here = 3 total
        assert len(orders) == 3

        # * Check order types
        order_types = [order["order_type"] for order in orders]
        assert "ENTRY" in order_types
        assert "PARTIAL_EXIT" in order_types

        # * Orders should be sorted by creation time
        assert orders[0]["created_at"] <= orders[1]["created_at"] <= orders[2]["created_at"]

    def test_get_pending_orders_new(self, db_manager, test_position):
        """Test retrieving pending orders for a session."""
        # * Create multiple orders with different statuses
        pending_order = db_manager.create_order(
            position_id=test_position["position_id"],
            order_type=OrderType.ENTRY,
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.001,
            strategy_name="test_strategy",
            session_id=test_position["session_id"],
        )

        filled_order = db_manager.create_order(
            position_id=test_position["position_id"],
            order_type=OrderType.PARTIAL_EXIT,
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            quantity=0.0005,
            strategy_name="test_strategy",
            session_id=test_position["session_id"],
        )

        # * Update one to filled
        db_manager.update_order_status_new(filled_order, OrderStatus.FILLED)

        # * Get pending orders
        pending_orders = db_manager.get_pending_orders_new(test_position["session_id"])

        # * Should only return the pending order
        assert len(pending_orders) == 1
        assert pending_orders[0]["id"] == pending_order
        assert pending_orders[0]["order_type"] == "ENTRY"

    def test_update_nonexistent_order(self, db_manager):
        """Test updating a non-existent order."""
        success = db_manager.update_order_status_new(
            order_id=99999, status=OrderStatus.FILLED  # Non-existent ID
        )

        assert success is False

    def test_order_creation_duplicate_internal_id_handling(self, db_manager, test_position):
        """Test that duplicate internal order IDs are handled gracefully."""
        # * Create order with specific internal ID
        internal_id = "test_duplicate_id_123"

        order1_id = db_manager.create_order(
            position_id=test_position["position_id"],
            order_type=OrderType.ENTRY,
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.001,
            strategy_name="test_strategy",
            session_id=test_position["session_id"],
            internal_order_id=internal_id,
        )

        # * Try to create another order with same internal ID
        order2_id = db_manager.create_order(
            position_id=test_position["position_id"],
            order_type=OrderType.PARTIAL_EXIT,
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            quantity=0.0005,
            strategy_name="test_strategy",
            session_id=test_position["session_id"],
            internal_order_id=internal_id,  # Same ID
        )

        # * Both should succeed with different internal IDs
        assert order1_id != order2_id

        orders = db_manager.get_orders_for_position(test_position["position_id"])
        # * Note: log_position automatically creates 1 ENTRY order, plus 2 created here = 3 total
        assert len(orders) == 3

        # * Filter out the auto-created ENTRY order to focus on the manually created ones
        manual_orders = [o for o in orders if o["id"] in [order1_id, order2_id]]

        # * One should have original ID, other should have modified ID
        internal_ids = [order["internal_order_id"] for order in manual_orders]
        assert internal_id in internal_ids
        assert any(id != internal_id for id in internal_ids)
