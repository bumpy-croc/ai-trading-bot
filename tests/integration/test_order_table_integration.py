"""
Integration tests for the Order table with the database.

Tests that the Order table works correctly with the database and Position relationships.
"""

from datetime import UTC, datetime

import pytest

from src.database.manager import DatabaseManager
from src.database.models import (
    Order,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
)


@pytest.mark.integration
class TestOrderTableIntegration:
    """Test Order table integration with database."""

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

    def test_create_order_in_database(self, db_manager, test_position):
        """Test creating an Order record in the database."""
        with db_manager.get_session() as session:
            order = Order(
                position_id=test_position["position_id"],
                order_type=OrderType.ENTRY,
                status=OrderStatus.PENDING,
                exchange_order_id="binance_12345",
                internal_order_id="internal_67890",
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                quantity=0.001,
                price=50000.0,
                strategy_name="test_strategy",
                session_id=test_position["session_id"],
            )

            session.add(order)
            session.commit()

            # * Verify the order was created
            assert order.id is not None
            assert order.position_id == test_position["position_id"]
            assert order.order_type == OrderType.ENTRY
            assert order.status == OrderStatus.PENDING
            assert order.exchange_order_id == "binance_12345"
            assert order.internal_order_id == "internal_67890"

    def test_position_orders_relationship(self, db_manager, test_position):
        """Test the relationship between Position and Orders."""
        with db_manager.get_session() as session:
            # * Get the position
            position = session.query(Position).filter_by(id=test_position["position_id"]).first()
            assert position is not None

            # * Create multiple orders for this position
            entry_order = Order(
                position_id=test_position["position_id"],
                order_type=OrderType.ENTRY,
                status=OrderStatus.FILLED,
                internal_order_id="entry_order_1",
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                quantity=0.001,
                strategy_name="test_strategy",
                session_id=test_position["session_id"],
            )

            partial_exit_order = Order(
                position_id=test_position["position_id"],
                order_type=OrderType.PARTIAL_EXIT,
                status=OrderStatus.PENDING,
                internal_order_id="partial_exit_1",
                symbol="BTCUSDT",
                side=PositionSide.SHORT,  # Opposite side for exit
                quantity=0.0005,
                strategy_name="test_strategy",
                session_id=test_position["session_id"],
                target_level=1,
                size_fraction=0.5,
            )

            session.add(entry_order)
            session.add(partial_exit_order)
            session.commit()

            # * Test the relationship
            session.refresh(position)
            # * Note: log_position automatically creates 1 ENTRY order, plus 2 created here = 3 total
            assert len(position.orders) == 3

            # * Check order types
            order_types = [order.order_type for order in position.orders]
            assert OrderType.ENTRY in order_types
            assert OrderType.PARTIAL_EXIT in order_types

    def test_order_status_transitions(self, db_manager, test_position):
        """Test updating order statuses."""
        with db_manager.get_session() as session:
            order = Order(
                position_id=test_position["position_id"],
                order_type=OrderType.ENTRY,
                status=OrderStatus.PENDING,
                internal_order_id="status_test_order",
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                quantity=0.001,
                strategy_name="test_strategy",
                session_id=test_position["session_id"],
            )

            session.add(order)
            session.commit()

            # * Update to filled
            order.status = OrderStatus.FILLED
            order.filled_quantity = 0.001
            order.filled_price = 50100.0
            order.filled_at = datetime.now(UTC)
            session.commit()

            # * Verify the update
            session.refresh(order)
            assert order.status == OrderStatus.FILLED
            assert float(order.filled_quantity) == 0.001
            assert float(order.filled_price) == 50100.0
            assert order.filled_at is not None

    def test_order_partial_operation_fields(self, db_manager, test_position):
        """Test partial operation specific fields."""
        with db_manager.get_session() as session:
            partial_order = Order(
                position_id=test_position["position_id"],
                order_type=OrderType.PARTIAL_EXIT,
                status=OrderStatus.FILLED,
                internal_order_id="partial_test_order",
                symbol="BTCUSDT",
                side=PositionSide.SHORT,
                quantity=0.0003,
                strategy_name="test_strategy",
                session_id=test_position["session_id"],
                target_level=2,  # Second partial exit level
                size_fraction=0.3,  # 30% of original position
            )

            session.add(partial_order)
            session.commit()

            # * Verify partial operation fields
            assert partial_order.target_level == 2
            assert float(partial_order.size_fraction) == 0.3
            assert partial_order.order_type == OrderType.PARTIAL_EXIT

    def test_order_unique_constraints(self, db_manager, test_position):
        """Test that unique constraints work correctly."""
        with db_manager.get_session() as session:
            order1 = Order(
                position_id=test_position["position_id"],
                order_type=OrderType.ENTRY,
                status=OrderStatus.PENDING,
                internal_order_id="unique_test_order",
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                quantity=0.001,
                strategy_name="test_strategy",
                session_id=test_position["session_id"],
            )

            session.add(order1)
            session.commit()

            # * Try to create another order with same internal_order_id and session_id
            order2 = Order(
                position_id=test_position["position_id"],
                order_type=OrderType.ENTRY,
                status=OrderStatus.PENDING,
                internal_order_id="unique_test_order",  # Same as order1
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                quantity=0.001,
                strategy_name="test_strategy",
                session_id=test_position["session_id"],  # Same as order1
            )

            session.add(order2)

            # * This should raise an integrity error due to unique constraint
            from sqlalchemy.exc import IntegrityError

            with pytest.raises(IntegrityError):
                session.commit()
