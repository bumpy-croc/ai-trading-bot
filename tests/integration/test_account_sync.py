"""
Optimized Tests for Account Synchronization functionality

This module tests the account synchronization service with focused, non-redundant tests
that cover all critical functionality without unnecessary repetition.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.coinbase_provider import CoinbaseProvider
from src.data_providers.exchange_interface import (
    AccountBalance,
    Order,
    OrderSide,
    OrderType,
    Position,
    Trade,
)
from src.data_providers.exchange_interface import OrderStatus as ExchangeOrderStatus
from src.live.account_sync import AccountSynchronizer

pytestmark = pytest.mark.integration


@pytest.fixture
def binance_account_balance_fixture():
    """Binance-style account balance fixture (USDT)"""
    return {
        "asset": "USDT",
        "walletBalance": "10000.00",
        "unrealizedProfit": "0.00",
        "marginBalance": "10000.00",
        "maintMargin": "0.00",
        "initialMargin": "0.00",
        "positionInitialMargin": "0.00",
        "openOrderInitialMargin": "0.00",
        "crossWalletBalance": "10000.00",
        "crossUnPnl": "0.00",
        "availableBalance": "10000.00",
        "maxWithdrawAmount": "10000.00",
        "updateTime": 1625474304765,
    }


@pytest.fixture
def binance_position_fixture():
    """Binance-style position fixture (BTCUSDT long)"""
    return {
        "symbol": "BTCUSDT",
        "positionSide": "LONG",
        "positionAmt": "0.100",
        "unrealizedProfit": "100.00",
        "isolatedMargin": "0.00",
        "notional": "5100.00",
        "isolatedWallet": "0.00",
        "initialMargin": "0.00",
        "maintMargin": "0.00",
        "updateTime": 1625474304765,
    }


@pytest.fixture
def binance_order_fixture():
    """Binance-style open order fixture (ETHUSDT limit sell)"""
    return {
        "orderId": 456,
        "symbol": "ETHUSDT",
        "side": "SELL",
        "type": "LIMIT",
        "quantity": "1.0",
        "price": "3000.00",
        "status": "NEW",
        "filledQuantity": "0.0",
        "avgPrice": None,
        "commission": "0.0",
        "commissionAsset": "USDT",
        "createTime": 1625474304765,
        "updateTime": 1625474304765,
    }


class TestAccountSynchronizer:
    """Test the AccountSynchronizer class with optimized test coverage"""

    @pytest.fixture
    def mock_exchange(self):
        """Create a mock exchange interface"""
        exchange = Mock()
        exchange.sync_account_data.return_value = {
            "sync_successful": True,
            "balances": [],
            "positions": [],
            "open_orders": [],
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
        db_manager.log_position.return_value = 1
        db_manager.log_trade.return_value = 1
        db_manager.update_balance.return_value = True
        db_manager.update_position.return_value = True
        db_manager.update_order_status.return_value = True
        db_manager.close_position.return_value = True
        return db_manager

    @pytest.fixture
    def synchronizer(self, mock_exchange, mock_db_manager):
        """Create an AccountSynchronizer instance for testing"""
        return AccountSynchronizer(exchange=mock_exchange, db_manager=mock_db_manager, session_id=1)

    def test_sync_account_data_success(self, synchronizer, mock_exchange):
        """Test successful account synchronization"""
        # Setup mock exchange data
        mock_exchange.sync_account_data.return_value = {
            "sync_successful": True,
            "balances": [
                AccountBalance(
                    asset="USDT",
                    free=10000.0,
                    locked=0.0,
                    total=10000.0,
                    last_updated=datetime.utcnow(),
                )
            ],
            "positions": [],
            "open_orders": [],
        }

        result = synchronizer.sync_account_data()

        assert result.success is True
        assert "completed" in result.message.lower()
        assert "balance_sync" in result.data
        assert "position_sync" in result.data
        assert "order_sync" in result.data

    def test_sync_account_data_exchange_failure(self, synchronizer, mock_exchange):
        """Test account synchronization when exchange fails"""
        mock_exchange.sync_account_data.return_value = {
            "sync_successful": False,
            "error": "API rate limit exceeded",
        }

        result = synchronizer.sync_account_data()

        assert result.success is False
        assert "failed" in result.message.lower()
        assert "rate limit" in result.message

    def test_sync_account_data_frequency_control(self, synchronizer, mock_exchange):
        """Test sync frequency control and force sync"""
        # Test that sync is skipped if too recent
        synchronizer.last_sync_time = datetime.utcnow() - timedelta(minutes=2)

        result = synchronizer.sync_account_data()
        assert result.success is True
        assert "skipped" in result.message.lower()
        mock_exchange.sync_account_data.assert_not_called()

        # Test that force sync bypasses frequency check
        result = synchronizer.sync_account_data(force=True)
        assert result.success is True
        assert "completed" in result.message.lower()
        mock_exchange.sync_account_data.assert_called_once()

    def test_sync_balances_comprehensive(self, synchronizer, mock_db_manager):
        """Test balance synchronization with all scenarios"""
        # Test no discrepancy
        balances = [
            AccountBalance(
                asset="USDT",
                free=10000.0,
                locked=0.0,
                total=10000.0,
                last_updated=datetime.utcnow(),
            )
        ]

        result = synchronizer._sync_balances(balances)
        assert result["synced"] is True
        assert result["corrected"] is False
        assert result["balance"] == 10000.0

        # Test with discrepancy
        balances[0].free = 11000.0
        balances[0].total = 11000.0

        result = synchronizer._sync_balances(balances)
        assert result["synced"] is True
        assert result["corrected"] is True
        assert result["old_balance"] == 10000.0
        assert result["new_balance"] == 11000.0
        assert result["difference"] == 1000.0
        assert result["difference_percent"] == 10.0
        mock_db_manager.update_balance.assert_called_once()

        # Test no USDT balance
        balances = [
            AccountBalance(
                asset="BTC", free=1.0, locked=0.0, total=1.0, last_updated=datetime.utcnow()
            )
        ]

        result = synchronizer._sync_balances(balances)
        assert result["synced"] is False
        assert "No USDT balance found" in result["error"]

    def test_sync_positions_comprehensive(self, synchronizer, mock_db_manager):
        """Test position synchronization with all scenarios"""
        # Test no positions
        result = synchronizer._sync_positions([])
        assert result["synced"] is True
        assert result["total_exchange_positions"] == 0
        assert result["total_db_positions"] == 0

        # Test new position
        positions = [
            Position(
                symbol="BTCUSDT",
                side="long",
                size=0.1,
                entry_price=50000.0,
                current_price=51000.0,
                unrealized_pnl=100.0,
                margin_type="isolated",
                leverage=10.0,
                order_id="test_order_123",
                open_time=datetime.utcnow(),
                last_update_time=datetime.utcnow(),
            )
        ]

        result = synchronizer._sync_positions(positions)
        assert result["synced"] is True
        assert result["new_positions"] == 1
        assert result["total_exchange_positions"] == 1
        mock_db_manager.log_position.assert_called_once()

        # Test position update
        mock_db_manager.get_active_positions.return_value = [
            {"id": 1, "symbol": "BTCUSDT", "side": "long", "size": 0.1}
        ]
        positions[0].size = 0.15

        result = synchronizer._sync_positions(positions)
        assert result["synced"] is True
        assert result["synced_positions"] == 1
        mock_db_manager.update_position.assert_called_once()

        # Test position closed
        result = synchronizer._sync_positions([])
        assert result["synced"] is True
        assert result["closed_positions"] == 1
        mock_db_manager.close_position.assert_called_once_with(1)

    def test_sync_orders_comprehensive(self, synchronizer, mock_db_manager):
        """Test order synchronization with all scenarios"""
        # Test no orders
        result = synchronizer._sync_orders([])
        assert result["synced"] is True
        assert result["total_exchange_orders"] == 0
        assert result["total_db_orders"] == 0

        # Test new order
        orders = [
            Order(
                order_id="test_order_123",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.1,
                price=50000.0,
                status=ExchangeOrderStatus.PENDING,
                filled_quantity=0.0,
                average_price=None,
                commission=0.0,
                commission_asset="USDT",
                create_time=datetime.utcnow(),
                update_time=datetime.utcnow(),
            )
        ]

        result = synchronizer._sync_orders(orders)
        assert result["synced"] is True
        assert result["new_orders"] == 1
        assert result["total_exchange_orders"] == 1

        # Test order status update
        mock_db_manager.get_open_orders.return_value = [
            {"id": 1, "order_id": "test_order_123", "symbol": "BTCUSDT"}
        ]
        orders[0].status = ExchangeOrderStatus.FILLED
        orders[0].filled_quantity = 0.1
        orders[0].average_price = 50000.0

        result = synchronizer._sync_orders(orders)
        assert result["synced"] is True
        assert result["synced_orders"] == 1
        mock_db_manager.update_order_status.assert_called_once()

        # Test order cancelled
        result = synchronizer._sync_orders([])
        assert result["synced"] is True
        assert result["cancelled_orders"] == 1
        mock_db_manager.update_order_status.assert_called_with(
            1, ExchangeOrderStatus.CANCELLED.value
        )

    def test_recover_missing_trades_comprehensive(
        self, synchronizer, mock_exchange, mock_db_manager
    ):
        """Test trade recovery with all scenarios"""
        # Test no missing trades
        mock_exchange.get_recent_trades.return_value = [
            Trade(
                trade_id="trade_123",
                order_id="order_123",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                quantity=0.1,
                price=50000.0,
                commission=0.0,
                commission_asset="USDT",
                time=datetime.utcnow(),
            )
        ]

        mock_db_manager.get_trades_by_symbol_and_date.return_value = [{"trade_id": "trade_123"}]

        result = synchronizer.recover_missing_trades("BTCUSDT", days_back=7)
        assert result["recovered"] is True
        assert result["missing_trades"] == 0
        assert result["recovered_trades"] == 0

        # Test missing trades
        mock_db_manager.get_trades_by_symbol_and_date.return_value = []

        result = synchronizer.recover_missing_trades("BTCUSDT", days_back=7)
        assert result["recovered"] is True
        assert result["missing_trades"] == 1
        assert result["recovered_trades"] == 1
        mock_db_manager.log_trade.assert_called_once()

    def test_emergency_sync_success(self, synchronizer, mock_exchange):
        """Test emergency sync functionality"""
        mock_exchange.sync_account_data.return_value = {
            "sync_successful": True,
            "balances": [
                AccountBalance(
                    asset="USDT",
                    free=10000.0,
                    locked=0.0,
                    total=10000.0,
                    last_updated=datetime.utcnow(),
                )
            ],
            "positions": [],
            "open_orders": [],
        }

        with patch.object(synchronizer, "recover_missing_trades") as mock_recover:
            mock_recover.return_value = {
                "recovered": True,
                "missing_trades": 0,
                "recovered_trades": 0,
            }

            result = synchronizer.emergency_sync()

        assert result.success is True
        assert "emergency_trade_recovery" in result.data
        assert len(result.data["emergency_trade_recovery"]) == 4  # 4 common symbols

    def test_emergency_sync_failure(self, synchronizer, mock_exchange):
        """Test emergency sync when initial sync fails"""
        mock_exchange.sync_account_data.return_value = {
            "sync_successful": False,
            "error": "API error",
        }

        result = synchronizer.emergency_sync()

        assert result.success is False
        assert "error" in result.message.lower()

    def test_exception_handling(self, synchronizer, mock_exchange):
        """Test exception handling for all sync operations"""
        # Test balance sync exception
        with patch.object(
            synchronizer.db_manager, "get_current_balance", side_effect=Exception("DB error")
        ):
            result = synchronizer._sync_balances([])
            assert result["synced"] is False
            assert "error" in result
            assert "DB error" in result["error"]

        # Test position sync exception
        with patch.object(
            synchronizer.db_manager, "get_active_positions", side_effect=Exception("DB error")
        ):
            result = synchronizer._sync_positions([])
            assert result["synced"] is False
            assert "error" in result
            assert "DB error" in result["error"]

        # Test order sync exception
        with patch.object(
            synchronizer.db_manager, "get_open_orders", side_effect=Exception("DB error")
        ):
            result = synchronizer._sync_orders([])
            assert result["synced"] is False
            assert "error" in result
            assert "DB error" in result["error"]

        # Test trade recovery exception
        mock_exchange.get_recent_trades.side_effect = Exception("API error")
        result = synchronizer.recover_missing_trades("BTCUSDT", days_back=7)
        assert result["recovered"] is False
        assert "error" in result
        assert "API error" in result["error"]

        # Test account sync exception
        mock_exchange.sync_account_data.side_effect = Exception("Exchange error")
        result = synchronizer.sync_account_data()
        assert result.success is False
        assert "error" in result.message.lower()
        assert "Exchange error" in result.message


class TestAccountSynchronizerIntegration:
    """Integration tests for AccountSynchronizer"""

    @pytest.fixture(params=["binance", "coinbase"])
    def real_synchronizer(self, request):
        """Create a real AccountSynchronizer with mocked dependencies for both providers"""
        provider = request.param
        if provider == "binance":
            exchange = BinanceProvider(api_key="test", api_secret="test", testnet=True)
        else:
            exchange = CoinbaseProvider(api_key="test", api_secret="test", testnet=True)
        db_manager = Mock()
        # Setup realistic mock data
        exchange.sync_account_data = Mock(
            return_value={
                "sync_successful": True,
                "balances": [
                    AccountBalance(
                        asset="USDT",
                        free=10200.0,
                        locked=0.0,
                        total=10200.0,
                        last_updated=datetime.utcnow(),
                    ),
                    AccountBalance(
                        asset="BTC", free=0.5, locked=0.0, total=0.5, last_updated=datetime.utcnow()
                    ),
                ],
                "positions": [
                    Position(
                        symbol="BTCUSDT",
                        side="long",
                        size=0.1,
                        entry_price=50000.0,
                        current_price=51000.0,
                        unrealized_pnl=100.0,
                        margin_type="isolated",
                        leverage=10.0,
                        order_id="order_123",
                        open_time=datetime.utcnow(),
                        last_update_time=datetime.utcnow(),
                    )
                ],
                "open_orders": [
                    Order(
                        order_id="order_456",
                        symbol="ETHUSDT",
                        side=OrderSide.SELL,
                        order_type=OrderType.LIMIT,
                        quantity=1.0,
                        price=3000.0,
                        status=ExchangeOrderStatus.PENDING,
                        filled_quantity=0.0,
                        average_price=None,
                        commission=0.0,
                        commission_asset="USDT",
                        create_time=datetime.utcnow(),
                        update_time=datetime.utcnow(),
                    )
                ],
            }
        )
        db_manager.get_current_balance.return_value = 10000.0
        db_manager.get_active_positions.return_value = []
        db_manager.get_open_orders.return_value = []
        db_manager.log_position.return_value = 1
        db_manager.log_trade.return_value = 1
        return AccountSynchronizer(exchange, db_manager, session_id=1)

    def test_full_sync_integration(self, real_synchronizer):
        """Test a complete synchronization cycle for both providers"""
        result = real_synchronizer.sync_account_data()
        assert result.success is True
        assert result.data["balance_sync"]["corrected"] is True
        assert result.data["position_sync"]["new_positions"] == 1
        assert result.data["order_sync"]["new_orders"] == 1

    def test_emergency_sync_integration(self, real_synchronizer):
        """Test emergency sync with trade recovery for both providers"""
        with patch.object(real_synchronizer, "recover_missing_trades") as mock_recover:
            mock_recover.return_value = {
                "recovered": True,
                "missing_trades": 2,
                "recovered_trades": 2,
            }
            result = real_synchronizer.emergency_sync()
        assert result.success is True
        assert "emergency_trade_recovery" in result.data
        assert mock_recover.call_count == 4  # BTCUSDT, ETHUSDT, SOLUSDT, AVAXUSDT


if __name__ == "__main__":
    pytest.main([__file__])
