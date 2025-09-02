"""
Integration tests for order lifecycle and status transitions.

Tests the complete flow from order creation to position management,
ensuring proper status transitions and data consistency.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from src.data_providers.exchange_interface import OrderStatus as ExchangeOrderStatus
from src.database.manager import DatabaseManager
from src.database.models import OrderStatus, PositionSide
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
        
        # * Create position directly with PENDING status
        with db_manager.get_session() as session:
            from src.database.models import Position
            pending_position = Position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                status=OrderStatus.PENDING,  # * Start as PENDING
                entry_price=50000.0,
                quantity=0.001,
                size=0.02,
                strategy_name="test_strategy",
                session_id=session_id,
                order_id="test_order_123",
                entry_time=datetime.utcnow()
            )
            session.add(pending_position)
            session.commit()
            position_id = pending_position.id
        
        # * Verify position is created as PENDING
        pending_orders = db_manager.get_pending_orders(session_id)
        assert len(pending_orders) == 1
        assert pending_orders[0]["status"] == "PENDING"
        assert pending_orders[0]["order_id"] == "test_order_123"
        
        # * Verify no active positions yet
        active_positions = db_manager.get_active_positions(session_id)
        assert len(active_positions) == 0
        
        # * Phase 2: Fill the order (PENDING -> OPEN)
        success = db_manager.fill_pending_order(
            order_id="test_order_123",
            filled_price=50100.0,  # Slight slippage
            filled_quantity=0.001
        )
        assert success is True
        
        # * Verify order is no longer pending
        pending_orders = db_manager.get_pending_orders(session_id)
        assert len(pending_orders) == 0
        
        # * Verify position is now active
        active_positions = db_manager.get_active_positions(session_id)
        assert len(active_positions) == 1
        assert active_positions[0]["id"] == position_id
        assert float(active_positions[0]["entry_price"]) == 50100.0  # Updated price
        assert float(active_positions[0]["quantity"]) == 0.001
        
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
        
        # * Create pending order directly
        with db_manager.get_session() as session:
            from src.database.models import Position
            pending_position = Position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                status=OrderStatus.PENDING,
                entry_price=50000.0,
                quantity=0.001,
                size=0.02,
                strategy_name="test_strategy",
                session_id=session_id,
                order_id="test_order_456",
                entry_time=datetime.utcnow()
            )
            session.add(pending_position)
            session.commit()
            position_id = pending_position.id
        
        # * Cancel the order
        success = db_manager.update_order_status(position_id, "CANCELLED")
        assert success is True
        
        # * Verify no pending orders
        pending_orders = db_manager.get_pending_orders(session_id)
        assert len(pending_orders) == 0
        
        # * Verify no active positions
        active_positions = db_manager.get_active_positions(session_id)
        assert len(active_positions) == 0

    def test_account_sync_order_fill(self, synchronizer, db_manager):
        """Test order fill through account synchronization."""
        # * Create a pending order in database
        session_id = synchronizer.session_id
        
        with db_manager.get_session() as session:
            from src.database.models import Position
            pending_position = Position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                status=OrderStatus.PENDING,
                entry_price=50000.0,
                quantity=0.001,
                size=0.02,
                strategy_name="test_strategy",
                session_id=session_id,
                order_id="sync_test_order",
                entry_time=datetime.utcnow()
            )
            session.add(pending_position)
            session.commit()
        
        # * Mock exchange returning filled order
        from src.data_providers.exchange_interface import Order
        filled_order = Order(
            order_id="sync_test_order",
            symbol="BTCUSDT",
            order_type="LIMIT",
            side=PositionSide.LONG,
            quantity=0.001,
            price=50000.0,
            status=ExchangeOrderStatus.FILLED,
            created_time=datetime.utcnow(),
            update_time=datetime.utcnow(),
            filled_quantity=0.001,
            average_price=50050.0
        )
        
        synchronizer.exchange.get_open_orders.return_value = [filled_order]
        
        # * Run synchronization
        result = synchronizer._sync_orders([filled_order])
        
        # * Verify sync was successful
        assert result["synced"] is True
        
        # * Verify position is now active
        active_positions = db_manager.get_active_positions(session_id)
        assert len(active_positions) == 1
        assert float(active_positions[0]["entry_price"]) == 50050.0  # Average price from exchange

    def test_status_consistency_validation(self, db_manager):
        """Test position status consistency validation."""
        session_id = db_manager.create_trading_session("test_strategy", "BTCUSDT", "1h", "live", 10000.0)
        
        # * Create an inconsistent position (PENDING with filled data)
        with db_manager.get_session() as session:
            from src.database.models import Position
            inconsistent_position = Position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                status=OrderStatus.PENDING,  # ! Should be OPEN
                entry_price=50000.0,  # ! Has filled data
                quantity=0.001,  # ! Has filled data
                size=0.02,
                strategy_name="test_strategy",
                session_id=session_id,
                order_id="inconsistent_order",
                entry_time=datetime.utcnow()
            )
            session.add(inconsistent_position)
            session.commit()
            position_id = inconsistent_position.id
        
        # * Validate status consistency
        validation = db_manager.validate_position_status_consistency()
        assert validation["inconsistent_pending"] == 1
        assert validation["total_pending"] == 1
        assert validation["total_open"] == 0
        
        # * Fix the inconsistency
        fixes = db_manager.fix_position_status_inconsistencies()
        assert fixes["pending_to_open"] == 1
        
        # * Verify fix worked
        validation_after = db_manager.validate_position_status_consistency()
        assert validation_after["inconsistent_pending"] == 0
        assert validation_after["total_pending"] == 0
        assert validation_after["total_open"] == 1
        
        # * Verify position is now active
        active_positions = db_manager.get_active_positions(session_id)
        assert len(active_positions) == 1
        assert active_positions[0]["id"] == position_id

    def test_orphaned_position_handling(self, db_manager):
        """Test handling of orphaned positions (OPEN without data)."""
        session_id = db_manager.create_trading_session("test_strategy", "BTCUSDT", "1h", "live", 10000.0)
        
        # * Create an orphaned position (OPEN without proper data)
        with db_manager.get_session() as session:
            from src.database.models import Position
            orphaned_position = Position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                status=OrderStatus.OPEN,  # ! Says OPEN
                entry_price=None,  # ! But missing data
                quantity=None,  # ! Missing data
                size=0.02,
                strategy_name="test_strategy",
                session_id=session_id,
                order_id="orphaned_order",
                entry_time=datetime.utcnow()
            )
            session.add(orphaned_position)
            session.commit()
        
        # * Validate and fix
        validation = db_manager.validate_position_status_consistency()
        assert validation["orphaned_open"] == 1
        
        fixes = db_manager.fix_position_status_inconsistencies()
        assert fixes["orphaned_to_failed"] == 1
        
        # * Verify no active positions after fix
        active_positions = db_manager.get_active_positions(session_id)
        assert len(active_positions) == 0

    def test_method_name_consistency(self, db_manager):
        """Test that renamed methods work correctly."""
        session_id = db_manager.create_trading_session("test_strategy", "BTCUSDT", "1h", "live", 10000.0)
        
        # * Test get_pending_orders method
        pending_orders = db_manager.get_pending_orders(session_id)
        assert isinstance(pending_orders, list)
        
        # * Create a pending order directly
        with db_manager.get_session() as session:
            from src.database.models import Position
            pending_position = Position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                status=OrderStatus.PENDING,
                entry_price=50000.0,
                quantity=0.001,
                size=0.02,
                strategy_name="test_strategy",
                session_id=session_id,
                order_id="naming_test_order",
                entry_time=datetime.utcnow()
            )
            session.add(pending_position)
            session.commit()
        
        # * Verify method returns the pending order
        pending_orders = db_manager.get_pending_orders(session_id)
        assert len(pending_orders) == 1
        assert pending_orders[0]["status"] == "PENDING"
        
        # * Verify get_active_positions returns empty (since order not filled)
        active_positions = db_manager.get_active_positions(session_id)
        assert len(active_positions) == 0
