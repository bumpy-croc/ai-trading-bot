"""
Pytest configuration and shared fixtures for the trading bot test suite.

This file contains fixtures that are used across multiple test modules,
especially for setting up mock data, test environments, and common objects.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import tempfile
import os
from pathlib import Path
import sys
import subprocess

# Add project root and src directory to PYTHONPATH for test imports
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
for _p in (str(_PROJECT_ROOT), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import core components for fixture creation
from data_providers.data_provider import DataProvider
from data_providers.binance_data_provider import BinanceDataProvider
from risk.risk_manager import RiskManager, RiskParameters
from strategies.base import BaseStrategy
from strategies.adaptive import AdaptiveStrategy

# Import account sync dependencies
try:
    from data_providers.exchange_interface import (
        AccountBalance, Position, Order, Trade,
        OrderSide, OrderType, OrderStatus as ExchangeOrderStatus
    )
    from database.models import PositionSide, TradeSource
except ImportError as e:
    print(f"Warning: Could not import account sync dependencies: {e}")

# ---------- Database setup for tests ----------
# Spin up a temporary PostgreSQL instance via Testcontainers so that modules that
# require a valid DATABASE_URL can import successfully.  This happens at import
# time, so we do it at module import as well.

_POSTGRES_CONTAINER = None
try:
    from testcontainers.postgres import PostgresContainer  # type: ignore
    _POSTGRES_CONTAINER = PostgresContainer("postgres:15-alpine")
    _POSTGRES_CONTAINER.start()
    os.environ["DATABASE_URL"] = _POSTGRES_CONTAINER.get_connection_url()
except Exception as _e:  # pragma: no cover -- fallback if Docker not available
    # Provide a dummy URL so the code can still import, but mark that Postgres is
    # unavailable.  Tests that actually require DB connectivity should handle the
    # failure or be skipped.
    os.environ.setdefault(
        "DATABASE_URL",
        "postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot_test"
    )
    _POSTGRES_CONTAINER = None

    # --- Ensure the fallback local test database exists (for developers without Docker) ---
    try:
        from sqlalchemy.engine.url import make_url
        from sqlalchemy import create_engine, text

        _raw_url = os.environ["DATABASE_URL"]
        _url_obj = make_url(_raw_url)

        # Attempt connection; if database missing, create it.
        try:
            _test_engine = create_engine(_raw_url, isolation_level="AUTOCOMMIT")
            with _test_engine.connect() as _conn:
                pass  # connection successful -> DB exists
        except Exception:
            # Connect to default 'postgres' database to create target DB
            _default_db_url = _url_obj.set(database="postgres")
            _admin_engine = create_engine(_default_db_url, isolation_level="AUTOCOMMIT")
            with _admin_engine.connect() as _conn:
                _db_name = _url_obj.database
                _conn.execute(text(f"CREATE DATABASE {_db_name};"))
            # Retry connection
            _test_engine = create_engine(_raw_url, isolation_level="AUTOCOMMIT")
            with _test_engine.connect():
                pass
    except Exception:
        # If we can't ensure db creation, tests that need it will fail/skipped.
        pass


def pytest_sessionstart(session):
    """Ensure the database is set up and cleared before running tests."""
    print("\n[pytest] Running local setup script to reset database...")
    result = subprocess.run([
        sys.executable, "scripts/setup_local_development.py", "--reset-db", "--no-interactive"
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        pytest.exit("Database setup/reset failed before tests.")


def pytest_sessionfinish(session, exitstatus):  # noqa: D401
    """Cleanup the Postgres container after the entire test session."""
    global _POSTGRES_CONTAINER
    if _POSTGRES_CONTAINER is not None:
        try:
            _POSTGRES_CONTAINER.stop()
        except Exception:
            pass  # We tried, nothing else to do


@pytest.fixture
def sample_ohlcv_data():
    """Generate realistic OHLCV data for testing"""
    np.random.seed(42)  # For reproducible tests
    
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    
    # Generate realistic price data with some trends and volatility
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, 100)  # 2% volatility
    
    closes = [base_price]
    for change in price_changes[1:]:
        closes.append(closes[-1] * (1 + change))
    
    # Generate OHLC from closes
    data = []
    for i, close in enumerate(closes):
        volatility = abs(np.random.normal(0, 0.01))  # Daily volatility
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        open_price = closes[i-1] if i > 0 else close
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


@pytest.fixture
def mock_data_provider():
    """Create a mock data provider for testing"""
    mock_provider = Mock(spec=DataProvider)
    
    # Setup default return values
    mock_provider.get_historical_data.return_value = pd.DataFrame({
        'open': [50000, 50100, 50200],
        'high': [50200, 50300, 50400],
        'low': [49800, 49900, 50000],
        'close': [50100, 50200, 50300],
        'volume': [1000, 1100, 1200]
    }, index=pd.date_range('2024-01-01', periods=3, freq='1H'))
    
    mock_provider.get_live_data.return_value = pd.DataFrame({
        'open': [50300],
        'high': [50400],
        'low': [50200],
        'close': [50350],
        'volume': [1150]
    }, index=[datetime.now()])
    
    return mock_provider


@pytest.fixture(scope="session")
def btcusdt_1h_2023_2024():
    """Load cached BTCUSDT 1-hour candles for 2023-01-01 → 2024-12-31.

    The data must be generated with ``scripts/download_binance_data.py`` and
    committed to the repository (preferably via Git LFS) under
    ``tests/data``.  If the file is missing, tests that depend on this
    fixture will be skipped automatically.
    """
    from pathlib import Path
    path = Path(__file__).parent / "data" / "BTCUSDT_1h_2023-01-01_2024-12-31.feather"
    if not path.exists():
        pytest.skip("Cached Binance data file not found")
    import pandas as pd
    df = pd.read_feather(path)
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture
def risk_parameters():
    """Standard risk parameters for testing"""
    return RiskParameters(
        base_risk_per_trade=0.02,
        max_risk_per_trade=0.03,
        max_position_size=0.25,
        max_daily_risk=0.06,
        max_drawdown=0.20
    )


@pytest.fixture
def mock_strategy():
    """Create a mock strategy for testing"""
    mock_strategy = Mock(spec=BaseStrategy)
    mock_strategy.name = "TestStrategy"
    mock_strategy.trading_pair = "BTCUSDT"
    
    # Setup default behaviors
    mock_strategy.calculate_indicators.return_value = pd.DataFrame({
        'open': [50000, 50100],
        'close': [50100, 50200],
        'rsi': [45, 55],
        'atr': [500, 510]
    })
    
    mock_strategy.check_entry_conditions.return_value = True
    mock_strategy.check_exit_conditions.return_value = False
    mock_strategy.calculate_position_size.return_value = 0.1
    mock_strategy.calculate_stop_loss.return_value = 49500
    mock_strategy.get_parameters.return_value = {"test": "params"}
    
    return mock_strategy


@pytest.fixture
def real_adaptive_strategy():
    """Create a real adaptive strategy instance for integration tests"""
    return AdaptiveStrategy()


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_model_file(temp_directory):
    """Create a mock ONNX model file for testing"""
    model_path = temp_directory / "test_model.onnx"
    
    # Create a dummy file (in real tests, you might want to create a valid ONNX model)
    with open(model_path, 'wb') as f:
        f.write(b"mock_onnx_model_data")
    
    return model_path


@pytest.fixture
def sample_trade_data():
    """Sample trade data for testing"""
    return {
        'symbol': 'BTCUSDT',
        'side': 'long',
        'entry_price': 50000,
        'exit_price': 51000,
        'size': 0.1,
        'entry_time': datetime(2024, 1, 1, 10, 0),
        'exit_time': datetime(2024, 1, 1, 11, 0),
        'pnl': 100.0
    }


@pytest.fixture
def sample_positions():
    """Sample position data for testing"""
    return [
        {
            'symbol': 'BTCUSDT',
            'side': 'long',
            'size': 0.1,
            'entry_price': 50000,
            'entry_time': datetime.now() - timedelta(hours=1),
            'stop_loss': 49000,
            'take_profit': 52000
        },
        {
            'symbol': 'ETHUSDT',
            'side': 'long',
            'size': 0.15,
            'entry_price': 3000,
            'entry_time': datetime.now() - timedelta(hours=2),
            'stop_loss': 2900,
            'take_profit': 3200
        }
    ]


@pytest.fixture
def market_conditions():
    """Different market condition scenarios for testing"""
    return {
        'bull_market': {
            'trend': 'up',
            'volatility': 'low',
            'volume': 'high'
        },
        'bear_market': {
            'trend': 'down',
            'volatility': 'high',
            'volume': 'low'
        },
        'sideways_market': {
            'trend': 'flat',
            'volatility': 'medium',
            'volume': 'medium'
        },
        'volatile_market': {
            'trend': 'mixed',
            'volatility': 'very_high',
            'volume': 'high'
        }
    }


# ---------- Account Synchronization Fixtures ----------

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

@pytest.fixture
def mock_sentiment_provider():
    """Mock sentiment provider for testing"""
    mock_provider = Mock()
    mock_provider.get_live_sentiment.return_value = {
        'sentiment_primary': 0.1,
        'sentiment_momentum': 0.05,
        'sentiment_volatility': 0.3,
        'sentiment_extreme_positive': 0,
        'sentiment_extreme_negative': 0,
        'sentiment_ma_3': 0.08,
        'sentiment_ma_7': 0.12,
        'sentiment_ma_14': 0.15,
        'sentiment_confidence': 0.8,
        'sentiment_freshness': 1
    }
    # Create a proper DataFrame with datetime index for sentiment data
    sentiment_df = pd.DataFrame({
        'sentiment_primary': [0.1, 0.2, -0.1, 0.0, 0.3],
        'sentiment_momentum': [0.05, 0.1, -0.05, 0.0, 0.15],
        'sentiment_volatility': [0.3, 0.25, 0.4, 0.35, 0.2]
    })
    sentiment_df.index = pd.date_range('2024-01-01', periods=5, freq='D')
    mock_provider.get_historical_sentiment.return_value = sentiment_df
    
    # Mock the aggregate_sentiment method
    aggregated_sentiment = pd.DataFrame({
        'sentiment_score': [0.1, 0.2, -0.1, 0.0, 0.3]
    })
    aggregated_sentiment.index = pd.date_range('2024-01-01', periods=5, freq='D')
    mock_provider.aggregate_sentiment.return_value = aggregated_sentiment
    
    return mock_provider


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


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests"""
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)  # Suppress logs during tests


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    # Marker registration is now handled declaratively in pytest.ini to avoid duplication.
    # Add any runtime configuration changes here if needed.
    pass


# Test categories for easy selection
pytest_plugins = [
    "tests.test_live_trading",
    "tests.test_risk_management", 
    "tests.test_strategies",
    "tests.test_data_providers"
]