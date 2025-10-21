"""
Integration tests for order lifecycle and status transitions.

Tests the complete flow from order creation to position management,
ensuring proper status transitions and data consistency.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from src.data_providers.exchange_interface import Order, OrderSide
from src.data_providers.exchange_interface import OrderStatus as ExchangeOrderStatus
from src.data_providers.exchange_interface import OrderType as ExchangeOrderType
from src.database.manager import DatabaseManager
from src.database.models import OrderStatus, OrderType, PositionSide, PositionStatus
from src.live.account_sync import AccountSynchronizer


@pytest.mark.integration
class TestOrderLifecycle:
    """Test complete order lifecycle from creation to closure."""

    @pytest.fixture
    def mock_exchange(self):
        """Create a mock exchange provider"""
        exchange = Mock()
        exchange.get_account_balance.return_value = {"USD": 10000.0}
        exchange.get_open_positions.return_value = []
        exchange.get_open_orders.return_value = []
        return exchange

    @pytest.fixture
    def db_manager(self):
        """Create a test database manager"""
        return DatabaseManager()  # Uses test database from conftest

    @pytest.fixture
    def synchronizer(self, mock_exchange, db_manager):
        """Create an AccountSynchronizer instance for testing"""
        return AccountSynchronizer(mock_exchange, db_manager, session_id=1)

    def test_order_creation_and_fill_lifecycle(self, db_manager):
        """Test the complete lifecycle of an order from creation to fill."""
        # * Phase 1: Create a pending order using direct database manipulation
        session_id = db_manager.create_trading_session("test_strategy", "BTCUSDT", "1h", "live", 10000.0)
        
        # * Create position with OPEN status (positions are active when created)
        with db_manager.get_session() as session:
            from src.database.models import Position
            position = Position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                status=PositionStatus.OPEN,  # * Positions start as OPEN
                entry_price=50000.0,
                quantity=0.001,
                size=0.02,
                strategy_name="test_strategy",
                session_id=session_id,
                entry_time=datetime.utcnow()
            )
            session.add(position)
            session.commit()
            position_id = position.id

        # * Create a PENDING entry order for this position
        db_manager.create_order(
            position_id=position_id,
            order_type=OrderType.ENTRY,
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.001,
            strategy_name="test_strategy",
            session_id=session_id,
            price=50000.0,
            exchange_order_id="test_order_123"
        )

        # * Verify order is created as PENDING
        pending_orders = db_manager.get_pending_orders_new(session_id)
        assert len(pending_orders) == 1
        assert pending_orders[0]["status"] == "PENDING"
        assert pending_orders[0]["order_id"] == "test_order_123"
        
        # * Verify position is active (created positions start as OPEN)
        active_positions = db_manager.get_active_positions(session_id)
        assert len(active_positions) == 1
        
        # * Phase 2: Fill the entry order (PENDING -> FILLED)
        orders = db_manager.get_orders_for_position(position_id)
        entry_order_id = orders[0]["id"]

        success = db_manager.update_order_status_new(
            order_id=entry_order_id,
            status=OrderStatus.FILLED,
            filled_price=50100.0,  # Slight slippage
            filled_quantity=0.001
        )
        assert success is True

        # * Verify order is no longer pending
        pending_orders = db_manager.get_pending_orders_new(session_id)
        assert len(pending_orders) == 0

        # * Verify position is still active (entry order filling doesn't change position status)
        active_positions = db_manager.get_active_positions(session_id)
        assert len(active_positions) == 1
        assert active_positions[0]["id"] == position_id
        assert float(active_positions[0]["entry_price"]) == 50000.0  # Original price, not filled price
        assert float(active_positions[0]["quantity"]) == 0.001  # Updated quantity
        
        # * Phase 3: Close the position
        success = db_manager.close_position(
            position_id=position_id,
            exit_price=51000.0,
            pnl=900.0  # (51000 - 50100) * 0.001 = 0.9
        )
        assert success is True
        
        # * Verify no active positions
        active_positions = db_manager.get_active_positions(session_id)
        assert len(active_positions) == 0

    def test_order_cancellation_lifecycle(self, db_manager):
        """Test order cancellation before fill."""
        session_id = db_manager.create_trading_session("test_strategy", "BTCUSDT", "1h", "live", 10000.0)

        # * Create position and pending order
        position_id = db_manager.log_position(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            size=0.02,
            quantity=0.001,
            strategy_name="test_strategy",
            entry_order_id="test_order_456",
            session_id=session_id
        )

        # * Create a pending exit order
        order_id = db_manager.create_order(
            position_id=position_id,
            order_type=OrderType.PARTIAL_EXIT,
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            quantity=0.0005,
            strategy_name="test_strategy",
            session_id=session_id,
            price=51000.0
        )

        # * Cancel the order
        success = db_manager.update_order_status_new(order_id, OrderStatus.CANCELLED)
        assert success is True
        
        # * Verify no pending orders
        pending_orders = db_manager.get_pending_orders_new(session_id)
        assert len(pending_orders) == 0

        # * Verify position is still active (only the exit order was cancelled)
        active_positions = db_manager.get_active_positions(session_id)
        assert len(active_positions) == 1

    def test_account_sync_order_fill(self, synchronizer, db_manager):
        """Test order fill through account synchronization."""
        # * Create a trading session first
        session_id = db_manager.create_trading_session("test_strategy", "BTCUSDT", "1h", "live", 10000.0)

        # Update synchronizer to use the actual session
        synchronizer.session_id = session_id

        # Create position first
        position_id = db_manager.log_position(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            size=0.02,
            quantity=0.001,
            strategy_name="test_strategy",
            entry_order_id="sync_test_order",
            session_id=session_id
        )

        # Create a pending order for this position
        db_manager.create_order(
            position_id=position_id,
            order_type=OrderType.PARTIAL_EXIT,
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            quantity=0.0005,
            strategy_name="test_strategy",
            session_id=session_id,
            exchange_order_id="sync_test_exit_order",
            price=51000.0
        )
        
        # * Mock exchange returning filled order
        filled_order = Order(
            order_id="sync_test_exit_order",
            symbol="BTCUSDT",
            side=OrderSide.SELL,  # Use OrderSide enum
            order_type=ExchangeOrderType.LIMIT,  # Use exchange OrderType enum
            quantity=0.001,
            price=51000.0,
            status=ExchangeOrderStatus.FILLED,
            filled_quantity=0.0005,
            average_price=51000.0,
            commission=0.0,
            commission_asset="USDT",
            create_time=datetime.utcnow(),
            update_time=datetime.utcnow()
        )
        
        synchronizer.exchange.get_open_orders.return_value = [filled_order]
        
        # * Run synchronization
        result = synchronizer._sync_orders([filled_order])
        
        # * Verify sync was successful
        assert result["synced"] is True
        
        # * Verify position is still active (exit order was partial, so position should remain open)
        active_positions = db_manager.get_active_positions(session_id)
        assert len(active_positions) == 1
        assert float(active_positions[0]["entry_price"]) == 50000.0  # Entry price should remain unchanged

    def test_status_consistency_validation(self, db_manager):
        """Test that positions maintain consistent status (OPEN/CLOSED only)."""
        session_id = db_manager.create_trading_session("test_strategy", "BTCUSDT", "1h", "live", 10000.0)

        # * Create a normal position (should be OPEN)
        with db_manager.get_session() as session:
            from src.database.models import Position
            position = Position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                status=PositionStatus.OPEN,  # Only valid statuses now
                entry_price=50000.0,
                quantity=0.001,
                size=0.02,
                strategy_name="test_strategy",
                session_id=session_id,
                entry_time=datetime.utcnow()
            )
            session.add(position)
            session.commit()
            position_id = position.id

        # * Verify position has valid status
        with db_manager.get_session() as session:
            position = session.query(Position).filter_by(id=position_id).first()
            assert position.status in [PositionStatus.OPEN, PositionStatus.CLOSED]

        # * Test closing the position
        success = db_manager.close_position(position_id, exit_price=51000.0, pnl=1000.0)
        assert success is True

        # * Verify position is now closed
        with db_manager.get_session() as session:
            position = session.query(Position).filter_by(id=position_id).first()
            assert position.status == PositionStatus.CLOSED

        # * Verify no active positions (position was closed)
        active_positions = db_manager.get_active_positions(session_id)
        assert len(active_positions) == 0

    def test_orphaned_position_handling(self, db_manager):
        """Test that positions with valid data are properly handled."""
        session_id = db_manager.create_trading_session("test_strategy", "BTCUSDT", "1h", "live", 10000.0)

        # * Create a normal position (positions now always have valid data)
        with db_manager.get_session() as session:
            from src.database.models import Position
            position = Position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                status=PositionStatus.OPEN,
                entry_price=50000.0,  # Valid entry price
                quantity=0.001,  # Valid quantity
                size=0.02,
                strategy_name="test_strategy",
                session_id=session_id,
                entry_time=datetime.utcnow()
            )
            session.add(position)
            session.commit()

        # * Validate status consistency for this session only
        # Count positions for this session
        with db_manager.get_session() as session:
            from src.database.models import Position
            session_positions = session.query(Position).filter_by(session_id=session_id).all()
            open_positions = [p for p in session_positions if p.status.value == 'OPEN']
            assert len(open_positions) == 1

            # Check that our position is not orphaned
            our_position = open_positions[0]
            assert our_position.entry_price is not None
            assert our_position.quantity is not None
            assert our_position.quantity > 0

        # * Verify position is active
        active_positions = db_manager.get_active_positions(session_id)
        assert len(active_positions) == 1

    def test_method_name_consistency(self, db_manager):
        """Test that new order-based methods work correctly."""
        session_id = db_manager.create_trading_session("test_strategy", "BTCUSDT", "1h", "live", 10000.0)

        # * Test get_pending_orders_new method
        pending_orders = db_manager.get_pending_orders_new(session_id)
        assert isinstance(pending_orders, list)

        # * Create a position and pending order
        position_id = db_manager.log_position(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            size=0.02,
            quantity=0.001,
            strategy_name="test_strategy",
            entry_order_id="test_method_consistency",
            session_id=session_id
        )

        # * Create a pending order for this position
        db_manager.create_order(
            position_id=position_id,
            order_type=OrderType.PARTIAL_EXIT,
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            quantity=0.0005,
            strategy_name="test_strategy",
            session_id=session_id,
            price=51000.0
        )

        # * Verify method returns the pending order
        pending_orders = db_manager.get_pending_orders_new(session_id)
        assert len(pending_orders) == 1
        assert pending_orders[0]["status"] == "PENDING"

        # * Verify get_active_positions returns the position (entry order was auto-filled)
        active_positions = db_manager.get_active_positions(session_id)
        assert len(active_positions) == 1
