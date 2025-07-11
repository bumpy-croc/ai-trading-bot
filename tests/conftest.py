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


@pytest.fixture
def mock_sentiment_provider():
    """Create a mock sentiment provider for testing"""
    mock_provider = Mock()
    
    # Setup default return values
    mock_provider.get_historical_sentiment.return_value = pd.DataFrame({
        'sentiment_score': [0.1, 0.2, -0.1, 0.3, 0.0],
        'volume': [100, 150, 80, 200, 120]
    }, index=pd.date_range('2024-01-01', periods=5, freq='1H'))
    
    mock_provider.aggregate_sentiment.return_value = pd.DataFrame({
        'sentiment_score': [0.15, 0.1, 0.2],
        'volume': [125, 140, 160]
    }, index=pd.date_range('2024-01-01', periods=3, freq='1H'))
    
    return mock_provider