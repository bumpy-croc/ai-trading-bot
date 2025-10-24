"""
Unit tests for Order table and PositionStatus enum.

Tests the Order table, PositionStatus enum, and OrderType enum.
"""

from datetime import datetime

from src.database.models import (
    Order,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    PositionStatus,
)


class TestOrderModels:
    """Test the Order table and enums."""

    def test_position_status_enum_values(self):
        """Test PositionStatus enum has correct values."""
        assert PositionStatus.OPEN.value == "OPEN"
        assert PositionStatus.CLOSED.value == "CLOSED"
        assert len(PositionStatus) == 2

    def test_order_type_enum_values(self):
        """Test OrderType enum has correct values."""
        assert OrderType.ENTRY.value == "ENTRY"
        assert OrderType.PARTIAL_EXIT.value == "PARTIAL_EXIT"
        assert OrderType.SCALE_IN.value == "SCALE_IN"
        assert OrderType.FULL_EXIT.value == "FULL_EXIT"
        assert len(OrderType) == 4

    def test_order_model_attributes(self):
        """Test Order model has all required attributes."""
        order = Order(
            position_id=1,
            order_type=OrderType.ENTRY,
            status=OrderStatus.PENDING,
            exchange_order_id="binance_123",
            internal_order_id="internal_456",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.001,
            price=50000.0,
            strategy_name="test_strategy",
        )

        # * Test required fields
        assert order.position_id == 1
        assert order.order_type == OrderType.ENTRY
        assert order.status == OrderStatus.PENDING
        assert order.exchange_order_id == "binance_123"
        assert order.internal_order_id == "internal_456"
        assert order.symbol == "BTCUSDT"
        assert order.side == PositionSide.LONG
        assert order.quantity == 0.001
        assert order.price == 50000.0
        assert order.strategy_name == "test_strategy"

        # * Test optional fields (defaults would be set by database)
        assert order.filled_price is None
        assert order.filled_at is None
        assert order.cancelled_at is None
        # Note: filled_quantity and commission defaults are set by SQLAlchemy when persisting

    def test_order_model_relationships(self):
        """Test Order model can be created with position relationship."""
        # * This would be tested in integration tests with actual database
        # * Here we just verify the model structure
        order = Order(
            position_id=1,
            order_type=OrderType.ENTRY,
            status=OrderStatus.PENDING,
            internal_order_id="test_order_1",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.001,
            strategy_name="test_strategy",
        )

        # * Verify foreign key relationship
        assert hasattr(order, "position_id")
        assert order.position_id == 1

    def test_position_orders_relationship(self):
        """Test Position model has orders relationship."""
        # * Verify the relationship exists in the model
        assert hasattr(Position, "orders")

        # * This would be tested in integration tests with actual database session
        # * Here we just verify the model structure

    def test_order_partial_operation_fields(self):
        """Test Order model has fields for partial operations."""
        order = Order(
            position_id=1,
            order_type=OrderType.PARTIAL_EXIT,
            status=OrderStatus.FILLED,
            internal_order_id="partial_exit_1",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.0005,
            strategy_name="test_strategy",
            target_level=1,  # First partial exit level
            size_fraction=0.25,  # 25% of original position
        )

        assert order.target_level == 1
        assert order.size_fraction == 0.25
        assert order.order_type == OrderType.PARTIAL_EXIT

    def test_order_timestamps(self):
        """Test Order model timestamp fields."""
        order = Order(
            position_id=1,
            order_type=OrderType.ENTRY,
            status=OrderStatus.FILLED,
            internal_order_id="timestamp_test",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.001,
            strategy_name="test_strategy",
            filled_at=datetime(2025, 1, 11, 12, 0, 0),
        )

        assert order.filled_at == datetime(2025, 1, 11, 12, 0, 0)
        assert order.cancelled_at is None
        # created_at and last_update would be set by database defaults

    def test_order_enum_combinations(self):
        """Test valid combinations of OrderType and OrderStatus."""
        # * Entry order flow
        entry_order = Order(
            position_id=1,
            order_type=OrderType.ENTRY,
            status=OrderStatus.PENDING,
            internal_order_id="entry_pending",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.001,
            strategy_name="test_strategy",
        )
        assert entry_order.order_type == OrderType.ENTRY
        assert entry_order.status == OrderStatus.PENDING

        # * Partial exit flow
        partial_order = Order(
            position_id=1,
            order_type=OrderType.PARTIAL_EXIT,
            status=OrderStatus.FILLED,
            internal_order_id="partial_filled",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.0005,
            strategy_name="test_strategy",
        )
        assert partial_order.order_type == OrderType.PARTIAL_EXIT
        assert partial_order.status == OrderStatus.FILLED
