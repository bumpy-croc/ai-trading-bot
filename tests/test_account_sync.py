"""
Tests for Account Synchronization functionality

This module tests the account synchronization service that ensures data integrity
between the exchange and the bot's database.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

from src.live.account_sync import AccountSynchronizer, SyncResult
from src.data_providers.exchange_interface import (
    AccountBalance, Position, Order, Trade,
    OrderSide, OrderType, OrderStatus as ExchangeOrderStatus
)
from src.database.models import PositionSide, TradeSource


class TestSyncResult:
    """Test the SyncResult dataclass"""
    
    def test_sync_result_creation(self):
        """Test creating a SyncResult instance"""
        data = {"test": "data"}
        timestamp = datetime.utcnow()
        
        result = SyncResult(
            success=True,
            message="Test message",
            data=data,
            timestamp=timestamp
        )
        
        assert result.success is True
        assert result.message == "Test message"
        assert result.data == data
        assert result.timestamp == timestamp


class TestAccountSynchronizer:
    """Test the AccountSynchronizer class"""
    
    @pytest.fixture
    def mock_exchange(self):
        """Create a mock exchange interface"""
        exchange = Mock()
        exchange.sync_account_data.return_value = {
            'sync_successful': True,
            'balances': [],
            'positions': [],
            'open_orders': []
        }
        return exchange
    
    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager"""
        db_manager = Mock()
        db_manager.get_current_balance.return_value = 10000.0
        db_manager.get_active_positions.return_value = []
        db_manager.get_open_orders.return_value = []
        db_manager.get_trades_by_symbol_and_date.return_value = []
        return db_manager
    
    @pytest.fixture
    def synchronizer(self, mock_exchange, mock_db_manager):
        """Create an AccountSynchronizer instance for testing"""
        return AccountSynchronizer(
            exchange=mock_exchange,
            db_manager=mock_db_manager,
            session_id=1
        )
    
    def test_initialization(self, mock_exchange, mock_db_manager):
        """Test AccountSynchronizer initialization"""
        sync = AccountSynchronizer(
            exchange=mock_exchange,
            db_manager=mock_db_manager,
            session_id=1
        )
        
        assert sync.exchange == mock_exchange
        assert sync.db_manager == mock_db_manager
        assert sync.session_id == 1
        assert sync.last_sync_time is None
    
    def test_sync_account_data_success(self, synchronizer, mock_exchange):
        """Test successful account synchronization"""
        # Setup mock exchange data
        mock_exchange.sync_account_data.return_value = {
            'sync_successful': True,
            'balances': [
                AccountBalance(
                    asset='USDT',
                    free=10000.0,
                    locked=0.0,
                    total=10000.0,
                    last_updated=datetime.utcnow()
                )
            ],
            'positions': [],
            'open_orders': []
        }
        
        result = synchronizer.sync_account_data()
        
        assert result.success is True
        assert "completed" in result.message.lower()
        assert 'balance_sync' in result.data
        assert 'position_sync' in result.data
        assert 'order_sync' in result.data
    
    def test_sync_account_data_exchange_failure(self, synchronizer, mock_exchange):
        """Test account synchronization when exchange fails"""
        mock_exchange.sync_account_data.return_value = {
            'sync_successful': False,
            'error': 'API rate limit exceeded'
        }
        
        result = synchronizer.sync_account_data()
        
        assert result.success is False
        assert "failed" in result.message.lower()
        assert "rate limit" in result.message
    
    def test_sync_account_data_too_frequent(self, synchronizer, mock_exchange):
        """Test that sync is skipped if too recent"""
        # Set last sync time to 2 minutes ago
        synchronizer.last_sync_time = datetime.utcnow() - timedelta(minutes=2)
        
        result = synchronizer.sync_account_data()
        
        assert result.success is True
        assert "skipped" in result.message.lower()
        mock_exchange.sync_account_data.assert_not_called()
    
    def test_sync_account_data_force_sync(self, synchronizer, mock_exchange):
        """Test that force sync bypasses frequency check"""
        # Set last sync time to 2 minutes ago
        synchronizer.last_sync_time = datetime.utcnow() - timedelta(minutes=2)
        
        result = synchronizer.sync_account_data(force=True)
        
        assert result.success is True
        assert "completed" in result.message.lower()
        mock_exchange.sync_account_data.assert_called_once()
    
    def test_sync_balances_no_discrepancy(self, synchronizer, mock_db_manager):
        """Test balance sync when no discrepancy exists"""
        balances = [
            AccountBalance(
                asset='USDT',
                free=10000.0,
                locked=0.0,
                total=10000.0,
                last_updated=datetime.utcnow()
            )
        ]
        
        result = synchronizer._sync_balances(balances)
        
        assert result['synced'] is True
        assert result['corrected'] is False
        assert result['balance'] == 10000.0
    
    def test_sync_balances_with_discrepancy(self, synchronizer, mock_db_manager):
        """Test balance sync when discrepancy exists"""
        balances = [
            AccountBalance(
                asset='USDT',
                free=11000.0,
                locked=0.0,
                total=11000.0,
                last_updated=datetime.utcnow()
            )
        ]
        
        result = synchronizer._sync_balances(balances)
        
        assert result['synced'] is True
        assert result['corrected'] is True
        assert result['old_balance'] == 10000.0
        assert result['new_balance'] == 11000.0
        assert result['difference'] == 1000.0
        assert result['difference_percent'] == 10.0
        
        # Verify database was updated
        mock_db_manager.update_balance.assert_called_once()
    
    def test_sync_balances_no_usdt(self, synchronizer):
        """Test balance sync when no USDT balance found"""
        balances = [
            AccountBalance(
                asset='BTC',
                free=1.0,
                locked=0.0,
                total=1.0,
                last_updated=datetime.utcnow()
            )
        ]
        
        result = synchronizer._sync_balances(balances)
        
        assert result['synced'] is False
        assert 'No USDT balance found' in result['error']
    
    def test_sync_positions_no_positions(self, synchronizer, mock_db_manager):
        """Test position sync when no positions exist"""
        positions = []
        
        result = synchronizer._sync_positions(positions)
        
        assert result['synced'] is True
        assert result['total_exchange_positions'] == 0
        assert result['total_db_positions'] == 0
    
    def test_sync_positions_new_position(self, synchronizer, mock_db_manager):
        """Test position sync when new position found on exchange"""
        positions = [
            Position(
                symbol='BTCUSDT',
                side='long',
                size=0.1,
                entry_price=50000.0,
                current_price=51000.0,
                unrealized_pnl=100.0,
                margin_type='isolated',
                leverage=10.0,
                order_id='test_order_123',
                open_time=datetime.utcnow(),
                last_update_time=datetime.utcnow()
            )
        ]
        
        mock_db_manager.log_position.return_value = 1
        
        result = synchronizer._sync_positions(positions)
        
        assert result['synced'] is True
        assert result['new_positions'] == 1
        assert result['total_exchange_positions'] == 1
        
        # Verify position was logged
        mock_db_manager.log_position.assert_called_once()
    
    def test_sync_positions_position_update(self, synchronizer, mock_db_manager):
        """Test position sync when position size changes"""
        positions = [
            Position(
                symbol='BTCUSDT',
                side='long',
                size=0.15,  # Changed from 0.1
                entry_price=50000.0,
                current_price=51000.0,
                unrealized_pnl=150.0,
                margin_type='isolated',
                leverage=10.0,
                order_id='test_order_123',
                open_time=datetime.utcnow(),
                last_update_time=datetime.utcnow()
            )
        ]
        
        # Mock existing position in database
        mock_db_manager.get_active_positions.return_value = [
            {
                'id': 1,
                'symbol': 'BTCUSDT',
                'side': 'long',
                'size': 0.1
            }
        ]
        
        result = synchronizer._sync_positions(positions)
        
        assert result['synced'] is True
        assert result['synced_positions'] == 1
        
        # Verify position was updated
        mock_db_manager.update_position.assert_called_once()
    
    def test_sync_positions_position_closed(self, synchronizer, mock_db_manager):
        """Test position sync when position closed on exchange"""
        positions = []  # No positions on exchange
        
        # Mock existing position in database
        mock_db_manager.get_active_positions.return_value = [
            {
                'id': 1,
                'symbol': 'BTCUSDT',
                'side': 'long',
                'size': 0.1
            }
        ]
        
        result = synchronizer._sync_positions(positions)
        
        assert result['synced'] is True
        assert result['closed_positions'] == 1
        
        # Verify position was closed
        mock_db_manager.close_position.assert_called_once_with(1)
    
    def test_sync_orders_no_orders(self, synchronizer, mock_db_manager):
        """Test order sync when no orders exist"""
        orders = []
        
        result = synchronizer._sync_orders(orders)
        
        assert result['synced'] is True
        assert result['total_exchange_orders'] == 0
        assert result['total_db_orders'] == 0
    
    def test_sync_orders_new_order(self, synchronizer, mock_db_manager):
        """Test order sync when new order found on exchange"""
        orders = [
            Order(
                order_id='test_order_123',
                symbol='BTCUSDT',
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.1,
                price=50000.0,
                status=ExchangeOrderStatus.PENDING,
                filled_quantity=0.0,
                average_price=None,
                commission=0.0,
                commission_asset='USDT',
                create_time=datetime.utcnow(),
                update_time=datetime.utcnow()
            )
        ]
        
        result = synchronizer._sync_orders(orders)
        
        assert result['synced'] is True
        assert result['new_orders'] == 1
        assert result['total_exchange_orders'] == 1
    
    def test_sync_orders_status_update(self, synchronizer, mock_db_manager):
        """Test order sync when order status changes"""
        orders = [
            Order(
                order_id='test_order_123',
                symbol='BTCUSDT',
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.1,
                price=50000.0,
                status=ExchangeOrderStatus.FILLED,  # Changed from PENDING
                filled_quantity=0.1,
                average_price=50000.0,
                commission=0.0,
                commission_asset='USDT',
                create_time=datetime.utcnow(),
                update_time=datetime.utcnow()
            )
        ]
        
        # Mock existing order in database
        mock_db_manager.get_open_orders.return_value = [
            {
                'id': 1,
                'order_id': 'test_order_123',
                'symbol': 'BTCUSDT'
            }
        ]
        
        result = synchronizer._sync_orders(orders)
        
        assert result['synced'] is True
        assert result['synced_orders'] == 1
        
        # Verify order status was updated
        mock_db_manager.update_order_status.assert_called_once()
    
    def test_sync_orders_order_cancelled(self, synchronizer, mock_db_manager):
        """Test order sync when order cancelled on exchange"""
        orders = []  # No orders on exchange
        
        # Mock existing order in database
        mock_db_manager.get_open_orders.return_value = [
            {
                'id': 1,
                'order_id': 'test_order_123',
                'symbol': 'BTCUSDT'
            }
        ]
        
        result = synchronizer._sync_orders(orders)
        
        assert result['synced'] is True
        assert result['cancelled_orders'] == 1
        
        # Verify order was marked as cancelled
        mock_db_manager.update_order_status.assert_called_once_with(
            1, ExchangeOrderStatus.CANCELLED.value
        )
    
    def test_recover_missing_trades_no_missing(self, synchronizer, mock_exchange, mock_db_manager):
        """Test trade recovery when no missing trades"""
        # Mock exchange trades
        mock_exchange.get_recent_trades.return_value = [
            Trade(
                trade_id='trade_123',
                order_id='order_123',
                symbol='BTCUSDT',
                side=OrderSide.BUY,
                quantity=0.1,
                price=50000.0,
                commission=0.0,
                commission_asset='USDT',
                time=datetime.utcnow()
            )
        ]
        
        # Mock database trades (same trade exists)
        mock_db_manager.get_trades_by_symbol_and_date.return_value = [
            {'trade_id': 'trade_123'}
        ]
        
        result = synchronizer.recover_missing_trades('BTCUSDT', days_back=7)
        
        assert result['recovered'] is True
        assert result['missing_trades'] == 0
        assert result['recovered_trades'] == 0
    
    def test_recover_missing_trades_with_missing(self, synchronizer, mock_exchange, mock_db_manager):
        """Test trade recovery when missing trades found"""
        # Mock exchange trades
        mock_exchange.get_recent_trades.return_value = [
            Trade(
                trade_id='trade_123',
                order_id='order_123',
                symbol='BTCUSDT',
                side=OrderSide.BUY,
                quantity=0.1,
                price=50000.0,
                commission=0.0,
                commission_asset='USDT',
                time=datetime.utcnow()
            )
        ]
        
        # Mock database trades (no trades exist)
        mock_db_manager.get_trades_by_symbol_and_date.return_value = []
        mock_db_manager.log_trade.return_value = 1
        
        result = synchronizer.recover_missing_trades('BTCUSDT', days_back=7)
        
        assert result['recovered'] is True
        assert result['missing_trades'] == 1
        assert result['recovered_trades'] == 1
        
        # Verify trade was logged
        mock_db_manager.log_trade.assert_called_once()
    
    def test_emergency_sync_success(self, synchronizer, mock_exchange):
        """Test emergency sync functionality"""
        # Setup mock exchange data
        mock_exchange.sync_account_data.return_value = {
            'sync_successful': True,
            'balances': [
                AccountBalance(
                    asset='USDT',
                    free=10000.0,
                    locked=0.0,
                    total=10000.0,
                    last_updated=datetime.utcnow()
                )
            ],
            'positions': [],
            'open_orders': []
        }
        
        # Mock trade recovery
        with patch.object(synchronizer, 'recover_missing_trades') as mock_recover:
            mock_recover.return_value = {
                'recovered': True,
                'missing_trades': 0,
                'recovered_trades': 0
            }
            
            result = synchronizer.emergency_sync()
        
        assert result.success is True
        assert 'emergency_trade_recovery' in result.data
        assert len(result.data['emergency_trade_recovery']) == 4  # 4 common symbols
    
    def test_emergency_sync_failure(self, synchronizer, mock_exchange):
        """Test emergency sync when initial sync fails"""
        mock_exchange.sync_account_data.return_value = {
            'sync_successful': False,
            'error': 'API error'
        }
        
        result = synchronizer.emergency_sync()
        
        assert result.success is False
        assert 'error' in result.message.lower()
    
    def test_sync_balances_exception(self, synchronizer):
        """Test balance sync exception handling"""
        with patch.object(synchronizer.db_manager, 'get_current_balance', side_effect=Exception("DB error")):
            result = synchronizer._sync_balances([])
        
        assert result['synced'] is False
        assert 'error' in result
        assert 'DB error' in result['error']
    
    def test_sync_positions_exception(self, synchronizer):
        """Test position sync exception handling"""
        with patch.object(synchronizer.db_manager, 'get_active_positions', side_effect=Exception("DB error")):
            result = synchronizer._sync_positions([])
        
        assert result['synced'] is False
        assert 'error' in result
        assert 'DB error' in result['error']
    
    def test_sync_orders_exception(self, synchronizer):
        """Test order sync exception handling"""
        with patch.object(synchronizer.db_manager, 'get_open_orders', side_effect=Exception("DB error")):
            result = synchronizer._sync_orders([])
        
        assert result['synced'] is False
        assert 'error' in result
        assert 'DB error' in result['error']
    
    def test_recover_missing_trades_exception(self, synchronizer, mock_exchange):
        """Test trade recovery exception handling"""
        mock_exchange.get_recent_trades.side_effect = Exception("API error")
        
        result = synchronizer.recover_missing_trades('BTCUSDT', days_back=7)
        
        assert result['recovered'] is False
        assert 'error' in result
        assert 'API error' in result['error']
    
    def test_sync_account_data_exception(self, synchronizer, mock_exchange):
        """Test account sync exception handling"""
        mock_exchange.sync_account_data.side_effect = Exception("Exchange error")
        
        result = synchronizer.sync_account_data()
        
        assert result.success is False
        assert 'error' in result.message.lower()
        assert 'Exchange error' in result.message


class TestAccountSynchronizerIntegration:
    """Integration tests for AccountSynchronizer"""
    
    @pytest.fixture
    def real_synchronizer(self):
        """Create a real AccountSynchronizer with mocked dependencies"""
        exchange = Mock()
        db_manager = Mock()
        
        # Setup realistic mock data
        exchange.sync_account_data.return_value = {
            'sync_successful': True,
            'balances': [
                AccountBalance(
                    asset='USDT',
                    free=10000.0,
                    locked=100.0,
                    total=10100.0,
                    last_updated=datetime.utcnow()
                ),
                AccountBalance(
                    asset='BTC',
                    free=0.5,
                    locked=0.0,
                    total=0.5,
                    last_updated=datetime.utcnow()
                )
            ],
            'positions': [
                Position(
                    symbol='BTCUSDT',
                    side='long',
                    size=0.1,
                    entry_price=50000.0,
                    current_price=51000.0,
                    unrealized_pnl=100.0,
                    margin_type='isolated',
                    leverage=10.0,
                    order_id='order_123',
                    open_time=datetime.utcnow(),
                    last_update_time=datetime.utcnow()
                )
            ],
            'open_orders': [
                Order(
                    order_id='order_456',
                    symbol='ETHUSDT',
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=1.0,
                    price=3000.0,
                    status=ExchangeOrderStatus.PENDING,
                    filled_quantity=0.0,
                    average_price=None,
                    commission=0.0,
                    commission_asset='USDT',
                    create_time=datetime.utcnow(),
                    update_time=datetime.utcnow()
                )
            ]
        }
        
        db_manager.get_current_balance.return_value = 10000.0
        db_manager.get_active_positions.return_value = []
        db_manager.get_open_orders.return_value = []
        db_manager.log_position.return_value = 1
        db_manager.log_trade.return_value = 1
        
        return AccountSynchronizer(exchange, db_manager, session_id=1)
    
    def test_full_sync_integration(self, real_synchronizer):
        """Test a complete synchronization cycle"""
        result = real_synchronizer.sync_account_data()
        
        assert result.success is True
        assert result.data['balance_sync']['corrected'] is True
        assert result.data['position_sync']['new_positions'] == 1
        assert result.data['order_sync']['new_orders'] == 1
    
    def test_emergency_sync_integration(self, real_synchronizer):
        """Test emergency sync with trade recovery"""
        with patch.object(real_synchronizer, 'recover_missing_trades') as mock_recover:
            mock_recover.return_value = {
                'recovered': True,
                'missing_trades': 2,
                'recovered_trades': 2
            }
            
            result = real_synchronizer.emergency_sync()
        
        assert result.success is True
        assert 'emergency_trade_recovery' in result.data
        
        # Verify trade recovery was called for all common symbols
        assert mock_recover.call_count == 4  # BTCUSDT, ETHUSDT, SOLUSDT, AVAXUSDT


if __name__ == '__main__':
    pytest.main([__file__])