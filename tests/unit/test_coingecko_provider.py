"""Unit tests for CoinGecko data provider."""

import pytest
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch
import pandas as pd
import requests

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
@pytest.mark.skip(reason="CoinGecko /ohlc endpoint returns 400 - free tier may not support this endpoint")
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


# ==============================================================================
# Unit Tests with Mocked Responses
# ==============================================================================


@pytest.mark.fast
class TestCoinGeckoProviderMocked:
    """Unit tests for CoinGecko provider with mocked HTTP responses."""

    @pytest.fixture
    def mock_provider(self):
        """Create a CoinGecko provider instance with mocked session."""
        with patch("src.data_providers.coingecko_provider.get_config") as mock_config:
            mock_config.return_value.get.return_value = 30  # Default timeout
            provider = CoinGeckoProvider()
            yield provider
            provider.close()

    def test_invalid_symbol_returns_empty_dataframe(self, mock_provider):
        """Test that invalid symbols return empty DataFrame."""
        # Arrange
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)

        # Mock 404 response for invalid coin ID
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Coin not found")

        with patch.object(mock_provider._session, "get", return_value=mock_response):
            # Act & Assert - should raise HTTPError which is caught by retry decorator
            with pytest.raises(requests.HTTPError):
                mock_provider.get_historical_data("INVALID-SYMBOL", "4h", start, end)

    def test_network_error_raises_after_retries(self, mock_provider):
        """Test that network errors raise after exhausting retries."""
        # Arrange - use recent dates so CoinGecko market_chart path is taken
        start = datetime.now(UTC) - timedelta(days=10)
        end = datetime.now(UTC) - timedelta(days=9)

        # Mock connection error
        with (
            patch.object(
                mock_provider._session, "get",
                side_effect=requests.ConnectionError("Network unreachable"),
            ),
            patch("time.sleep"),
        ):
            # Act & Assert - should raise after retries
            with pytest.raises(requests.ConnectionError):
                mock_provider.get_historical_data("BTC-USD", "4h", start, end)

    def test_timeout_error_raises_after_retries(self, mock_provider):
        """Test that timeout errors raise after exhausting retries."""
        # Arrange - use recent dates so CoinGecko market_chart path is taken
        start = datetime.now(UTC) - timedelta(days=10)
        end = datetime.now(UTC) - timedelta(days=9)

        # Mock timeout error
        with (
            patch.object(
                mock_provider._session, "get",
                side_effect=requests.Timeout("Request timed out"),
            ),
            patch("time.sleep"),
        ):
            # Act & Assert
            with pytest.raises(requests.Timeout):
                mock_provider.get_historical_data("BTC-USD", "4h", start, end)

    def test_rate_limit_error_retries(self, mock_provider):
        """Test that 429 rate limit errors trigger retry logic."""
        # Arrange - use recent dates so CoinGecko market_chart path is taken
        start = datetime.now(UTC) - timedelta(days=10)
        end = datetime.now(UTC) - timedelta(days=9)

        # Mock 429 response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")

        with (
            patch.object(mock_provider._session, "get", return_value=mock_response),
            patch("time.sleep"),
        ):
            # Act & Assert - should raise after retries
            with pytest.raises(requests.HTTPError) as exc_info:
                mock_provider.get_historical_data("BTC-USD", "4h", start, end)

            assert "429" in str(exc_info.value)

    def test_malformed_ohlc_response_returns_empty_dataframe(self, mock_provider):
        """Test that malformed OHLC responses are handled gracefully."""
        # Arrange
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)

        # Mock malformed OHLC response (missing required fields)
        mock_ohlc_response = Mock()
        mock_ohlc_response.status_code = 200
        mock_ohlc_response.raise_for_status = Mock()
        mock_ohlc_response.json.return_value = [
            [1704067200000, 42000],  # Missing high, low, close
            "invalid_row",  # Not a list
            [1704153600000],  # Too few elements
        ]

        mock_volume_response = Mock()
        mock_volume_response.status_code = 200
        mock_volume_response.raise_for_status = Mock()
        mock_volume_response.json.return_value = {"total_volumes": []}

        # Mock session to return malformed responses
        with patch.object(
            mock_provider._session,
            "get",
            side_effect=[mock_ohlc_response, mock_volume_response],
        ):
            # Act
            result = mock_provider.get_historical_data("BTC-USD", "4h", start, end)

            # Assert - should return empty DataFrame due to all rows being invalid
            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_empty_ohlc_response_returns_empty_dataframe(self, mock_provider):
        """Test that empty OHLC responses return empty DataFrame."""
        # Arrange
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)

        # Mock empty OHLC response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = []

        with patch.object(mock_provider._session, "get", return_value=mock_response):
            # Act
            result = mock_provider.get_historical_data("BTC-USD", "4h", start, end)

            # Assert
            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_valid_ohlc_response_with_volume(self, mock_provider):
        """Test that valid market_chart responses are processed correctly."""
        # Arrange - use recent dates so CoinGecko market_chart path is taken
        # Align to midnight UTC to get clean daily candle boundaries
        now = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        start = now - timedelta(days=5)
        end = now - timedelta(days=3)

        # Build hourly price points for exactly 2 days starting at midnight
        base_ts = int(start.timestamp() * 1000)
        prices = []
        volumes = []
        for i in range(48):
            ts = base_ts + i * 3600 * 1000  # Hourly intervals
            price = 42000.0 + (i % 24) * 50  # Varies within each day
            vol = 1000000.0 + i * 10000
            prices.append([ts, price])
            volumes.append([ts, vol])

        # Mock market_chart response (single API call returns prices + volumes)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "prices": prices,
            "total_volumes": volumes,
        }

        with patch.object(mock_provider._session, "get", return_value=mock_response):
            # Act
            result = mock_provider.get_historical_data("BTC-USD", "1d", start, end)

            # Assert
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert all(col in result.columns for col in ["open", "high", "low", "close", "volume"])
            # First candle open should be the first price point
            assert result["open"].iloc[0] == 42000.0

    def test_invalid_volume_data_filtered(self, mock_provider):
        """Test that invalid volume values (NaN, negative) are cleaned."""
        # Arrange - use recent dates so CoinGecko market_chart path is taken
        # Align to midnight UTC to get clean daily candle boundaries
        now = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        start = now - timedelta(days=5)
        end = now - timedelta(days=4)

        # Build hourly price points for 1 day starting at midnight
        base_ts = int(start.timestamp() * 1000)
        prices = []
        for i in range(24):
            ts = base_ts + i * 3600 * 1000
            prices.append([ts, 42000.0 + i * 10])

        # Volume data with invalid values (NaN, negative) at matching timestamps
        volumes = []
        for i in range(24):
            ts = base_ts + i * 3600 * 1000
            if i < 8:
                volumes.append([ts, float("nan")])
            elif i < 16:
                volumes.append([ts, -1000.0])
            else:
                volumes.append([ts, 0.0])

        # Mock market_chart response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "prices": prices,
            "total_volumes": volumes,
        }

        with patch.object(mock_provider._session, "get", return_value=mock_response):
            # Act
            result = mock_provider.get_historical_data("BTC-USD", "1d", start, end)

            # Assert - should have 1 candle with volume cleaned
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1
            # NaN volumes are filled with 0, negative volumes are set to 0
            assert result["volume"].iloc[0] >= 0

    def test_get_current_price_with_mock(self, mock_provider):
        """Test get_current_price with mocked response."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"bitcoin": {"usd": 42000.0}}

        with patch.object(mock_provider._session, "get", return_value=mock_response):
            # Act
            price = mock_provider.get_current_price("BTC-USD")

            # Assert
            assert price == 42000.0
            assert isinstance(price, float)

    def test_symbol_conversion(self, mock_provider):
        """Test symbol conversion to CoinGecko IDs."""
        # Act & Assert
        assert mock_provider._convert_symbol("BTC-USD") == "bitcoin"
        assert mock_provider._convert_symbol("BTCUSDT") == "bitcoin"
        assert mock_provider._convert_symbol("BTC") == "bitcoin"
        assert mock_provider._convert_symbol("ETH-USD") == "ethereum"
        assert mock_provider._convert_symbol("ETHUSDT") == "ethereum"
        assert mock_provider._convert_symbol("SOL-USD") == "solana"

    def test_unknown_symbol_returns_lowercase(self, mock_provider):
        """Test that unknown symbols are returned as lowercase."""
        # Act
        result = mock_provider._convert_symbol("UNKNOWN-COIN")

        # Assert - should warn and return lowercase
        assert result == "unknown-coin"

    def test_rate_limiting_free_tier(self, mock_provider):
        """Test that free tier provider enforces rate limiting."""
        # Arrange - provider without API key (free tier)
        assert mock_provider.api_key is None

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"bitcoin": {"usd": 42000.0}}

        # Act - make two requests and measure time
        import time

        with patch.object(mock_provider._session, "get", return_value=mock_response):
            start_time = time.time()
            mock_provider._request("/simple/price", {"ids": "bitcoin", "vs_currencies": "usd"})
            mock_provider._request("/simple/price", {"ids": "bitcoin", "vs_currencies": "usd"})
            elapsed = time.time() - start_time

            # Assert - should have waited at least RATE_LIMIT_DELAY_SECONDS
            assert elapsed >= mock_provider.RATE_LIMIT_DELAY_SECONDS

    def test_close_handles_none_session(self):
        """Test that close() handles None session gracefully."""
        # Arrange
        with patch("src.data_providers.coingecko_provider.get_config") as mock_config:
            mock_config.return_value.get.return_value = 30
            provider = CoinGeckoProvider()
            provider._session = None

            # Act & Assert - should not raise
            provider.close()
            assert provider._session is None

    def test_unsupported_symbol_beyond_365_days_raises_value_error(self, mock_provider):
        """Test that a symbol with no Binance mapping fails fast for multi-year requests.

        Regression guard for the P1 Codex review finding: silently capping
        a >365-day request to 365 days produces misleading backtest results.
        """
        # Arrange - request more than 365 days for a symbol not in BINANCE_SYMBOL_MAPPING
        start = datetime.now(UTC) - timedelta(days=800)
        end = datetime.now(UTC)

        # Patch BINANCE_SYMBOL_MAPPING so 'bitcoin' has no entry
        with patch.dict(mock_provider.BINANCE_SYMBOL_MAPPING, {}, clear=True):
            # Act & Assert
            with pytest.raises(ValueError, match="no Binance archive mapping"):
                mock_provider.get_historical_data("BTC-USD", "1h", start, end)

    def test_unsupported_symbol_within_365_days_succeeds(self, mock_provider):
        """Test that a symbol with no Binance mapping works fine within 365 days."""
        # Arrange - request within 365 days for an unmapped symbol
        now = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        start = now - timedelta(days=10)
        end = now - timedelta(days=8)

        base_ts = int(start.timestamp() * 1000)
        prices = [[base_ts + i * 3600 * 1000, 42000.0 + i] for i in range(48)]
        volumes = [[base_ts + i * 3600 * 1000, 1_000_000.0] for i in range(48)]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"prices": prices, "total_volumes": volumes}

        with (
            patch.dict(mock_provider.BINANCE_SYMBOL_MAPPING, {}, clear=True),
            patch.object(mock_provider._session, "get", return_value=mock_response),
        ):
            # Act - should NOT raise because it's within 365 days
            result = mock_provider.get_historical_data("BTC-USD", "1d", start, end)

            # Assert - data is returned
            assert isinstance(result, pd.DataFrame)
            assert not result.empty

    def test_binance_coingecko_stitch_uses_candle_interval_not_full_day(
        self, mock_provider
    ):
        """Test that the Binance→CoinGecko stitch advances by one candle interval.

        Regression guard for the P2 Codex review finding: using timedelta(days=1)
        creates an intraday data gap for 1h/4h timeframes.
        """
        # Arrange - set up a historical request (>365 days) for a mapped symbol
        now = datetime.now(UTC)
        start = now - timedelta(days=800)
        end = now

        # Last Binance archive candle at 2024-11-30 18:00 UTC (an evening timestamp,
        # not midnight – specifically to expose the +1 day vs +1 hour difference)
        last_archive_ts = datetime(2024, 11, 30, 18, 0, 0, tzinfo=UTC)

        # Build a minimal archive DataFrame whose last candle is at 18:00
        archive_df = pd.DataFrame(
            {
                "open": [40000.0],
                "high": [41000.0],
                "low": [39000.0],
                "close": [40500.0],
                "volume": [100.0],
            },
            index=pd.DatetimeIndex([last_archive_ts], name="timestamp"),
        )

        # CoinGecko backfill mock – captures the `start` parameter it receives
        captured_cg_starts: list[datetime] = []

        def fake_fetch_coingecko(
            coin_id: str, cg_start: datetime, cg_end: datetime, timeframe: str
        ) -> pd.DataFrame:
            captured_cg_starts.append(cg_start)
            return pd.DataFrame()  # Return empty so concat logic is skipped

        with (
            patch.object(
                mock_provider,
                "_fetch_via_binance_archive",
                return_value=archive_df,
            ),
            patch.object(
                mock_provider,
                "_fetch_via_coingecko_market_chart",
                side_effect=fake_fetch_coingecko,
            ),
        ):
            mock_provider.get_historical_data("BTC-USD", "1h", start, end)

        # Assert: CoinGecko was asked to start at last_archive_ts + 1 hour (19:00),
        # NOT at last_archive_ts + 1 day (2024-12-01 18:00)
        assert len(captured_cg_starts) == 1
        expected_cg_start = last_archive_ts + timedelta(hours=1)
        assert captured_cg_starts[0] == expected_cg_start, (
            f"Expected CoinGecko backfill to start at {expected_cg_start} (+1h), "
            f"but got {captured_cg_starts[0]}"
        )
