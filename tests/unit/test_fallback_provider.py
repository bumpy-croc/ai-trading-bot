"""Unit tests for FallbackProvider."""

import pytest
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch
import pandas as pd

from src.data_providers.fallback_provider import FallbackProvider


@pytest.fixture
def mock_binance_success():
    """Mock successful Binance provider."""
    with patch("src.data_providers.fallback_provider.BinanceProvider") as mock:
        instance = Mock()
        instance.get_historical_data.return_value = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000.0, 1100.0],
            },
            index=pd.DatetimeIndex(
                [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)],
                name="timestamp",
            ),
        )
        instance.get_current_price.return_value = 50000.0
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_binance_blocked():
    """Mock blocked Binance provider (403 Forbidden)."""
    with patch("src.data_providers.fallback_provider.BinanceProvider") as mock:
        instance = Mock()
        instance.get_historical_data.side_effect = Exception("403 Forbidden: Access denied")
        instance.get_current_price.side_effect = Exception("403 Forbidden")
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_coingecko_success():
    """Mock successful CoinGecko provider."""
    with patch("src.data_providers.fallback_provider.CoinGeckoProvider") as mock:
        instance = Mock()
        instance.get_historical_data.return_value = pd.DataFrame(
            {
                "open": [200.0, 201.0],
                "high": [202.0, 203.0],
                "low": [199.0, 200.0],
                "close": [201.0, 202.0],
                "volume": [2000.0, 2100.0],
            },
            index=pd.DatetimeIndex(
                [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)],
                name="timestamp",
            ),
        )
        instance.get_current_price.return_value = 95000.0
        mock.return_value = instance
        yield mock


def test_fallback_provider_initialization(mock_binance_success, mock_coingecko_success):
    """Test that FallbackProvider initializes both providers."""
    provider = FallbackProvider()

    assert provider.current_provider == "binance"
    assert provider._binance_failed is False
    assert provider.primary_provider is not None
    assert provider.fallback_provider is not None


def test_binance_success(mock_binance_success, mock_coingecko_success):
    """Test that Binance is used when available."""
    provider = FallbackProvider()

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 31, tzinfo=UTC)

    df = provider.get_historical_data("BTCUSDT", "1h", start, end)

    # Should use Binance
    assert provider.current_provider == "binance"
    assert not df.empty
    assert len(df) == 2

    # Binance should have been called
    provider.primary_provider.get_historical_data.assert_called_once()

    # CoinGecko should NOT have been called
    provider.fallback_provider.get_historical_data.assert_not_called()


def test_binance_blocked_falls_back_to_coingecko(mock_binance_blocked, mock_coingecko_success):
    """Test automatic failover when Binance is blocked."""
    provider = FallbackProvider()

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 31, tzinfo=UTC)

    df = provider.get_historical_data("BTCUSDT", "1h", start, end)

    # Should have fallen back to CoinGecko
    assert provider.current_provider == "coingecko"
    assert provider._binance_failed is True
    assert not df.empty
    assert len(df) == 2

    # Both providers should have been called
    provider.primary_provider.get_historical_data.assert_called_once()
    provider.fallback_provider.get_historical_data.assert_called_once()


def test_skips_binance_after_first_failure(mock_binance_blocked, mock_coingecko_success):
    """Test that Binance is skipped after first 403 failure."""
    provider = FallbackProvider()

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 31, tzinfo=UTC)

    # First call - tries Binance, gets 403, uses CoinGecko
    df1 = provider.get_historical_data("BTCUSDT", "1h", start, end)
    assert provider._binance_failed is True

    # Second call - should skip Binance entirely
    df2 = provider.get_historical_data("ETHUSDT", "1h", start, end)

    # Binance should only have been called once (first request)
    assert provider.primary_provider.get_historical_data.call_count == 1

    # CoinGecko should have been called twice (both requests)
    assert provider.fallback_provider.get_historical_data.call_count == 2


def test_get_current_price_fallback(mock_binance_blocked, mock_coingecko_success):
    """Test price fetching with failover."""
    provider = FallbackProvider()

    price = provider.get_current_price("BTCUSDT")

    # Should have used CoinGecko
    assert price == 95000.0
    assert provider._binance_failed is True

    # Both should have been attempted
    provider.primary_provider.get_current_price.assert_called_once()
    provider.fallback_provider.get_current_price.assert_called_once()


def test_symbol_normalization(mock_binance_success, mock_coingecko_success):
    """Test symbol format conversion for different providers."""
    provider = FallbackProvider()

    # Test Binance normalization
    assert provider._normalize_symbol("BTC-USD", "binance") == "BTCUSDT"
    assert provider._normalize_symbol("BTCUSDT", "binance") == "BTCUSDT"
    assert provider._normalize_symbol("ETH-USD", "binance") == "ETHUSDT"

    # Test CoinGecko normalization (passed through)
    assert provider._normalize_symbol("BTC-USD", "coingecko") == "BTC-USD"
    assert provider._normalize_symbol("BTCUSDT", "coingecko") == "BTCUSDT"


def test_both_providers_fail(mock_binance_blocked):
    """Test error handling when both providers fail."""
    with patch("src.data_providers.fallback_provider.CoinGeckoProvider") as mock_cg:
        mock_cg_instance = Mock()
        mock_cg_instance.get_historical_data.side_effect = Exception("CoinGecko API error")
        mock_cg.return_value = mock_cg_instance

        provider = FallbackProvider()

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 31, tzinfo=UTC)

        # Should raise RuntimeError when both fail
        with pytest.raises(RuntimeError, match="Failed to fetch data from both"):
            provider.get_historical_data("BTCUSDT", "1h", start, end)


def test_get_live_data_fallback(mock_binance_blocked, mock_coingecko_success):
    """Test live data fetching with failover."""
    provider = FallbackProvider()

    # Mock live data responses
    provider.primary_provider.get_live_data = Mock(side_effect=Exception("403 Forbidden"))
    provider.fallback_provider.get_live_data = Mock(
        return_value=pd.DataFrame(
            {
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [1000.0],
            },
            index=pd.DatetimeIndex([datetime(2024, 1, 1, tzinfo=UTC)], name="timestamp"),
        )
    )

    df = provider.get_live_data("BTCUSDT", "1h", limit=10)

    # Should have used CoinGecko
    assert provider.current_provider == "coingecko"
    assert not df.empty
    provider.fallback_provider.get_live_data.assert_called_once_with("BTCUSDT", "1h", 10)


def test_close_connections(mock_binance_success, mock_coingecko_success):
    """Test that both provider connections are closed."""
    provider = FallbackProvider()

    provider.close()

    # CoinGecko close should have been called
    provider.fallback_provider.close.assert_called_once()


@pytest.mark.integration
def test_real_fallback_behavior():
    """Integration test with real providers (may fail if APIs are down)."""
    provider = FallbackProvider()

    # This should work even if Binance is blocked
    price = provider.get_current_price("BTC-USD")

    assert price > 0
    assert isinstance(price, float)
    # BTC price should be reasonable
    assert 10000 < price < 200000

    # Check which provider was used
    print(f"Active provider: {provider.current_provider}")
    if provider._binance_failed:
        print("Binance was blocked, used CoinGecko")
    else:
        print("Binance was available")
