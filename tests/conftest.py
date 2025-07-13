"""
Test configuration and fixtures for the trading bot test suite.

This file contains shared fixtures and configuration that can be used
across all test files.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import test dependencies
try:
    from src.data_providers.exchange_interface import (
        AccountBalance, Position, Order, Trade,
        OrderSide, OrderType, OrderStatus as ExchangeOrderStatus
    )
    from src.database.models import PositionSide, TradeSource
except ImportError as e:
    print(f"Warning: Could not import test dependencies: {e}")


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_account_balance():
    """Create a sample account balance for testing."""
    return AccountBalance(
        asset='USDT',
        free=10000.0,
        locked=100.0,
        total=10100.0,
        last_updated=datetime.utcnow()
    )


@pytest.fixture(scope="session")
def sample_position():
    """Create a sample position for testing."""
    return Position(
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


@pytest.fixture(scope="session")
def sample_order():
    """Create a sample order for testing."""
    return Order(
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


@pytest.fixture(scope="session")
def sample_trade():
    """Create a sample trade for testing."""
    return Trade(
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


@pytest.fixture
def mock_exchange():
    """Create a mock exchange interface for testing."""
    exchange = Mock()
    
    # Setup default return values
    exchange.sync_account_data.return_value = {
        'sync_successful': True,
        'balances': [],
        'positions': [],
        'open_orders': []
    }
    
    exchange.get_recent_trades.return_value = []
    exchange.get_balances.return_value = []
    exchange.get_positions.return_value = []
    exchange.get_open_orders.return_value = []
    
    return exchange


@pytest.fixture
def mock_db_manager():
    """Create a mock database manager for testing."""
    db_manager = Mock()
    
    # Setup default return values
    db_manager.get_current_balance.return_value = 10000.0
    db_manager.get_active_positions.return_value = []
    db_manager.get_open_orders.return_value = []
    db_manager.get_trades_by_symbol_and_date.return_value = []
    
    # Setup method return values
    db_manager.log_position.return_value = 1
    db_manager.log_trade.return_value = 1
    db_manager.update_balance.return_value = True
    db_manager.update_position.return_value = True
    db_manager.update_order_status.return_value = True
    db_manager.close_position.return_value = True
    
    return db_manager


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture(scope="session")
def test_session_id():
    """Get a test session ID."""
    return 1


@pytest.fixture
def sample_sync_data():
    """Create sample synchronization data for testing."""
    return {
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


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "live_trading: mark test as requiring live trading"
    )
    config.addinivalue_line(
        "markers", "risk_management: mark test as risk management related"
    )


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add unit marker to tests that don't have integration in the name
        if "integration" not in item.nodeid.lower():
            item.add_marker(pytest.mark.unit)
        else:
            item.add_marker(pytest.mark.integration)


# Test reporting
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom summary information."""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    # Count test results
    passed = len(terminalreporter.stats.get('passed', []))
    failed = len(terminalreporter.stats.get('failed', []))
    skipped = len(terminalreporter.stats.get('skipped', []))
    errors = len(terminalreporter.stats.get('error', []))
    
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"Total: {passed + failed + skipped + errors}")
    
    if exitstatus == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    print("="*80)