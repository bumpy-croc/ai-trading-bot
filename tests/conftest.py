"""
Pytest configuration and shared fixtures for the trading bot test suite.

This file contains fixtures that are used across multiple test modules,
especially for setting up mock data, test environments, and common objects.
"""

import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Import core components for fixture creation
from src.data_providers.data_provider import DataProvider
from src.risk.risk_manager import RiskParameters
from src.strategies.base import BaseStrategy

# Import account sync dependencies
try:
    from src.data_providers.exchange_interface import (
        AccountBalance,
        Order,
        OrderSide,
        OrderType,
        Position,
        Trade,
    )
    from src.data_providers.exchange_interface import OrderStatus as ExchangeOrderStatus
except ImportError as e:
    print(f"Warning: Could not import account sync dependencies: {e}")

# ---------- Database setup for tests ----------
# Avoid import-time heavy setup. Use a session-scoped autouse fixture to configure
# an in-memory DB for unit tests, or start a Postgres container / use external DB
# for integration runs when ENABLE_INTEGRATION_TESTS=1.


def _is_integration_enabled() -> bool:
    return os.getenv("ENABLE_INTEGRATION_TESTS", "0") == "1"


@pytest.fixture(scope="session", autouse=True)
def maybe_setup_database():
    """Configure test database per run mode.

    - Unit/default: ensure lightweight in-memory SQLite via DATABASE_URL default.
    - Integration: if DATABASE_URL is already set (e.g., CI Postgres service), use it;
      otherwise start a Postgres testcontainer, export its URL, and stop it at teardown.
      Also run schema reset before tests.
    """
    if not _is_integration_enabled():
        os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
        yield
        return

    started_container = None
    if not os.getenv("DATABASE_URL"):
        try:
            from testcontainers.postgres import PostgresContainer  # type: ignore

            print(
                f"\n[Database Setup] Starting PostgreSQL container at {datetime.now().strftime('%H:%M:%S')}"
            )
            container = PostgresContainer("postgres:15-alpine")
            container.start()
            os.environ["DATABASE_URL"] = container.get_connection_url()
            started_container = container
            print("[Database Setup] ✅ Postgres container ready")
        except Exception as exc:  # pragma: no cover
            pytest.exit(f"Failed to start Postgres container for integration tests: {exc}")

    # Reset DB schema/content before tests
    print("\n[pytest] Running database reset before integration tests...")
    db_reset_start = time.time()
    result = subprocess.run(
        [sys.executable, "scripts/setup_local_development.py", "--reset-db", "--no-interactive"],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        pytest.exit("Database setup/reset failed before tests.")
    else:
        print(f"[pytest] ✅ Database reset completed in {time.time() - db_reset_start:.2f} seconds")

    try:
        yield
    finally:
        if started_container is not None:
            try:
                started_container.stop()
            except Exception:
                pass


def pytest_collection_modifyitems(config, items):  # noqa: D401
    """Skip integration tests unless explicitly enabled via env.

    This prevents accidental DB/network usage on local/unit runs.
    """
    if _is_integration_enabled():
        return
    skip_integration = pytest.mark.skip(
        reason="integration tests disabled; set ENABLE_INTEGRATION_TESTS=1"
    )
    for item in items:
        if any(marker.name == "integration" for marker in item.iter_markers()):
            item.add_marker(skip_integration)


@pytest.fixture
def sample_ohlcv_data():
    """Generate realistic OHLCV data for testing"""
    np.random.seed(42)  # For reproducible tests

    dates = pd.date_range("2024-01-01", periods=100, freq="1h")

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
        open_price = closes[i - 1] if i > 0 else close
        volume = np.random.uniform(1000, 10000)

        data.append(
            {
                "timestamp": dates[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture
def mock_data_provider():
    """Create a mock data provider for testing"""
    mock_provider = Mock(spec=DataProvider)

    # Setup default return values
    mock_provider.get_historical_data.return_value = pd.DataFrame(
        {
            "open": [50000, 50100, 50200],
            "high": [50200, 50300, 50400],
            "low": [49800, 49900, 50000],
            "close": [50100, 50200, 50300],
            "volume": [1000, 1100, 1200],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1h"),
    )

    mock_provider.get_live_data.return_value = pd.DataFrame(
        {"open": [50300], "high": [50400], "low": [50200], "close": [50350], "volume": [1150]},
        index=[datetime.now()],
    )

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
        max_drawdown=0.20,
    )


@pytest.fixture
def mock_strategy():
    """Create a mock strategy for testing"""
    mock_strategy = Mock(spec=BaseStrategy)
    mock_strategy.name = "TestStrategy"
    mock_strategy.trading_pair = "BTCUSDT"

    # Setup default behaviors
    mock_strategy.calculate_indicators.return_value = pd.DataFrame(
        {"open": [50000, 50100], "close": [50100, 50200], "rsi": [45, 55], "atr": [500, 510]}
    )

    mock_strategy.check_entry_conditions.return_value = True
    mock_strategy.check_exit_conditions.return_value = False
    mock_strategy.calculate_position_size.return_value = 0.1
    mock_strategy.calculate_stop_loss.return_value = 49500
    mock_strategy.get_parameters.return_value = {"test": "params"}

    return mock_strategy


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
    with open(model_path, "wb") as f:
        f.write(b"mock_onnx_model_data")

    return model_path


@pytest.fixture
def sample_trade_data():
    """Sample trade data for testing"""
    return {
        "symbol": "BTCUSDT",
        "side": "long",
        "entry_price": 50000,
        "exit_price": 51000,
        "size": 0.1,
        "entry_time": datetime(2024, 1, 1, 10, 0),
        "exit_time": datetime(2024, 1, 1, 11, 0),
        "pnl": 100.0,
    }


@pytest.fixture
def sample_positions():
    """Sample position data for testing"""
    return [
        {
            "symbol": "BTCUSDT",
            "side": "long",
            "size": 0.1,
            "entry_price": 50000,
            "entry_time": datetime.now() - timedelta(hours=1),
            "stop_loss": 49000,
            "take_profit": 52000,
        },
        {
            "symbol": "ETHUSDT",
            "side": "long",
            "size": 0.15,
            "entry_price": 3000,
            "entry_time": datetime.now() - timedelta(hours=2),
            "stop_loss": 2900,
            "take_profit": 3200,
        },
    ]


@pytest.fixture
def market_conditions():
    """Different market condition scenarios for testing"""
    return {
        "bull_market": {"trend": "up", "volatility": "low", "volume": "high"},
        "bear_market": {"trend": "down", "volatility": "high", "volume": "low"},
        "sideways_market": {"trend": "flat", "volatility": "medium", "volume": "medium"},
        "volatile_market": {"trend": "mixed", "volatility": "very_high", "volume": "high"},
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
        asset="USDT", free=10000.0, locked=100.0, total=10100.0, last_updated=datetime.utcnow()
    )


@pytest.fixture(scope="session")
def sample_position():
    """Create a sample position for testing."""
    return Position(
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


@pytest.fixture(scope="session")
def sample_order():
    """Create a sample order for testing."""
    return Order(
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


@pytest.fixture(scope="session")
def sample_trade():
    """Create a sample trade for testing."""
    return Trade(
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


@pytest.fixture
def mock_exchange():
    """Create a mock exchange interface for testing."""
    exchange = Mock()

    # Setup default return values
    exchange.sync_account_data.return_value = {
        "sync_successful": True,
        "balances": [],
        "positions": [],
        "open_orders": [],
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
def mock_database_manager():
    """Create a mock database manager for unit tests"""
    from tests.mocks import MockDatabaseManager

    return MockDatabaseManager()


@pytest.fixture
def fast_db():
    """Alias for mock_database_manager - use this in unit tests"""
    from tests.mocks import MockDatabaseManager

    return MockDatabaseManager()


@pytest.fixture
def mock_sentiment_provider():
    """Mock sentiment provider for testing"""
    mock_provider = Mock()
    mock_provider.get_live_sentiment.return_value = {
        "sentiment_primary": 0.1,
        "sentiment_momentum": 0.05,
        "sentiment_volatility": 0.3,
        "sentiment_extreme_positive": 0,
        "sentiment_extreme_negative": 0,
        "sentiment_ma_3": 0.08,
        "sentiment_ma_7": 0.12,
        "sentiment_ma_14": 0.15,
        "sentiment_confidence": 0.8,
        "sentiment_freshness": 1,
    }
    # Create a proper DataFrame with datetime index for sentiment data
    sentiment_df = pd.DataFrame(
        {
            "sentiment_primary": [0.1, 0.2, -0.1, 0.0, 0.3],
            "sentiment_momentum": [0.05, 0.1, -0.05, 0.0, 0.15],
            "sentiment_volatility": [0.3, 0.25, 0.4, 0.35, 0.2],
        }
    )
    sentiment_df.index = pd.date_range("2024-01-01", periods=5, freq="D")
    mock_provider.get_historical_sentiment.return_value = sentiment_df

    # Mock the aggregate_sentiment method
    aggregated_sentiment = pd.DataFrame({"sentiment_score": [0.1, 0.2, -0.1, 0.0, 0.3]})
    aggregated_sentiment.index = pd.date_range("2024-01-01", periods=5, freq="D")
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
        "sync_successful": True,
        "balances": [
            AccountBalance(
                asset="USDT",
                free=10000.0,
                locked=100.0,
                total=10100.0,
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
