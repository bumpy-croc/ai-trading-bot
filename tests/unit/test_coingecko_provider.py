"""Unit tests for CoinGecko data provider."""

import pytest
from datetime import UTC, datetime, timedelta
import pandas as pd

from src.data_providers.coingecko_provider import CoinGeckoProvider


@pytest.fixture
def provider():
    """Create a CoinGecko provider instance."""
    return CoinGeckoProvider()


@pytest.mark.integration
def test_get_current_price(provider):
    """Test fetching current price."""
    price = provider.get_current_price("BTC-USD")

    assert price > 0
    assert isinstance(price, float)
    # BTC price should be reasonable (between $10k and $200k)
    assert 10000 < price < 200000


@pytest.mark.integration
def test_symbol_mapping(provider):
    """Test symbol conversion to CoinGecko coin IDs."""
    test_cases = {
        "BTC-USD": "bitcoin",
        "BTCUSDT": "bitcoin",
        "BTC": "bitcoin",
        "ETH-USD": "ethereum",
        "ETHUSDT": "ethereum",
        "SOL-USD": "solana",
    }

    for symbol, expected_id in test_cases.items():
        coin_id = provider._convert_symbol(symbol)
        assert coin_id == expected_id, f"Expected {symbol} → {expected_id}, got {coin_id}"


@pytest.mark.integration
def test_get_historical_data(provider):
    """Test fetching historical OHLCV data."""
    start = datetime.now(UTC) - timedelta(days=7)
    end = datetime.now(UTC)

    df = provider.get_historical_data(
        symbol="BTC-USD",
        timeframe="4h",
        start=start,
        end=end
    )

    # Validate DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # Check columns
    expected_columns = ["open", "high", "low", "close", "volume"]
    assert all(col in df.columns for col in expected_columns)

    # Check index
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None  # Should be timezone-aware

    # Validate data types
    for col in expected_columns:
        assert pd.api.types.is_numeric_dtype(df[col])

    # Validate OHLC relationships
    assert (df["high"] >= df["low"]).all()
    assert (df["open"] >= df["low"]).all()
    assert (df["open"] <= df["high"]).all()
    assert (df["close"] >= df["low"]).all()
    assert (df["close"] <= df["high"]).all()

    # No NaN values
    assert not df.isnull().any().any()


@pytest.mark.integration
def test_data_persistence(provider):
    """Test that data is persisted in provider.data attribute."""
    start = datetime.now(UTC) - timedelta(days=3)
    end = datetime.now(UTC)

    df = provider.get_historical_data(
        symbol="ETH-USD",
        timeframe="4h",
        start=start,
        end=end
    )

    # provider.data should be set after fetching
    assert provider.data is not None
    assert not provider.data.empty
    assert len(provider.data) == len(df)

    # Should be same DataFrame
    pd.testing.assert_frame_equal(provider.data, df)


def test_close_connection(provider):
    """Test closing the provider connection."""
    # Should not raise any exceptions
    provider.close()

    # Session should be closed
    assert provider._session is None
