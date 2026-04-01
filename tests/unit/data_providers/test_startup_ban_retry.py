"""Tests for ban-aware startup retry in BinanceProvider._initialize_client()."""

from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.unit

# Credentials must be >= 20 chars to pass validation
FAKE_KEY = "A" * 64
FAKE_SECRET = "B" * 64

try:
    from src.data_providers.binance_provider import (
        BinanceProvider,
        _parse_ban_expiry,
    )

    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    BinanceProvider = Mock
    _parse_ban_expiry = None


def _make_ban_exception(code: int, ban_until_ms: int) -> Exception:
    """Create a mock BinanceAPIException with the given code and ban timestamp."""
    exc = Exception(
        f"APIError(code={code}): Way too much request weight used; "
        f"IP banned until {ban_until_ms}. Please use WebSocket Streams."
    )
    exc.code = code
    return exc


def _make_generic_exception() -> Exception:
    """Create a non-ban exception (e.g. network error)."""
    exc = Exception("Connection refused")
    # No .code attribute — not a Binance API error
    return exc


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestHandleStartupBan:
    """Tests for BinanceProvider._handle_startup_ban static method."""

    def test_returns_wait_time_for_ban_error(self):
        """IP ban with parseable expiry returns wait time + 5s buffer."""
        # Ban expires 60s from now
        now_ms = 1_775_000_000_000
        ban_until_ms = now_ms + 60_000
        exc = _make_ban_exception(-1003, ban_until_ms)

        with patch(
            "src.data_providers.binance_provider._parse_ban_expiry", return_value=60.0
        ):
            result = BinanceProvider._handle_startup_ban(
                exc, attempt=0, max_retries=3, max_wait=600
            )

        assert result == 65.0  # 60s + 5s buffer

    def test_returns_none_for_non_ban_error(self):
        """Non-ban errors (no .code or wrong code) are not retryable."""
        exc = _make_generic_exception()
        result = BinanceProvider._handle_startup_ban(
            exc, attempt=0, max_retries=3, max_wait=600
        )
        assert result is None

    def test_returns_none_when_retries_exhausted(self):
        """Returns None when attempt >= max_retries."""
        exc = _make_ban_exception(-1003, 1_775_000_060_000)

        with patch(
            "src.data_providers.binance_provider._parse_ban_expiry", return_value=60.0
        ):
            result = BinanceProvider._handle_startup_ban(
                exc, attempt=3, max_retries=3, max_wait=600
            )

        assert result is None

    def test_returns_none_when_ban_exceeds_max_wait(self):
        """Returns None when ban wait exceeds max_wait threshold."""
        exc = _make_ban_exception(-1003, 1_775_001_000_000)

        with patch(
            "src.data_providers.binance_provider._parse_ban_expiry", return_value=700.0
        ):
            result = BinanceProvider._handle_startup_ban(
                exc, attempt=0, max_retries=3, max_wait=600
            )

        assert result is None

    def test_uses_default_wait_when_expiry_not_parseable(self):
        """Falls back to 30s + 5s buffer when ban timestamp can't be parsed."""
        exc = _make_ban_exception(-1003, 0)

        with patch(
            "src.data_providers.binance_provider._parse_ban_expiry", return_value=None
        ):
            result = BinanceProvider._handle_startup_ban(
                exc, attempt=0, max_retries=3, max_wait=600
            )

        assert result == 35.0  # 30s default + 5s buffer

    def test_handles_minus_1015_too_many_orders(self):
        """-1015 (too many orders) is also retryable."""
        exc = _make_ban_exception(-1015, 1_775_000_060_000)

        with patch(
            "src.data_providers.binance_provider._parse_ban_expiry", return_value=60.0
        ):
            result = BinanceProvider._handle_startup_ban(
                exc, attempt=0, max_retries=3, max_wait=600
            )

        assert result == 65.0

    def test_handles_zero_remaining_ban_time(self):
        """When ban already expired (0s remaining), use default wait."""
        exc = _make_ban_exception(-1003, 1_775_000_000_000)

        with patch(
            "src.data_providers.binance_provider._parse_ban_expiry", return_value=0.0
        ):
            result = BinanceProvider._handle_startup_ban(
                exc, attempt=0, max_retries=3, max_wait=600
            )

        assert result == 35.0  # 30s default + 5s buffer


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestInitializeClientBanRetry:
    """Integration tests for the full _initialize_client retry loop."""

    @patch("src.data_providers.binance_provider.get_binance_api_endpoint", return_value="binance")
    @patch("src.data_providers.binance_provider.is_us_location", return_value=False)
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_succeeds_after_ban_expires(
        self, mock_config, mock_client_class, _mock_us, _mock_endpoint
    ):
        """Client init succeeds on retry after ban lifts."""
        mock_config_obj = Mock()
        mock_config_obj.get.side_effect = lambda key, default=None: {
            "BINANCE_ACCOUNT_TYPE": "margin",
            "TRADING_MODE": "live",
        }.get(key, default)
        mock_config_obj.get_required.side_effect = lambda key: FAKE_KEY if "KEY" in key else FAKE_SECRET
        mock_config.return_value = mock_config_obj

        ban_exc = _make_ban_exception(-1003, 1_775_000_010_000)

        mock_client = Mock()
        mock_client.get_server_time.return_value = {"serverTime": 1234}
        mock_client.get_margin_account.return_value = {
            "tradeEnabled": True,
            "borrowEnabled": True,
            "userAssets": [{"asset": "USDT", "free": "100", "locked": "0", "netAsset": "100"}],
        }

        # First call raises ban, second succeeds
        mock_client_class.side_effect = [ban_exc, mock_client]

        with patch("time.sleep") as mock_sleep, patch(
            "src.data_providers.binance_provider._parse_ban_expiry", return_value=5.0
        ):
            provider = BinanceProvider(FAKE_KEY, FAKE_SECRET)

        mock_sleep.assert_called_once_with(10.0)  # 5s ban + 5s buffer
        assert provider._client == mock_client

    @patch("src.data_providers.binance_provider.get_binance_api_endpoint", return_value="binance")
    @patch("src.data_providers.binance_provider.is_us_location", return_value=False)
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_fails_after_max_retries_in_margin_mode(
        self, mock_config, mock_client_class, _mock_us, _mock_endpoint
    ):
        """Raises RuntimeError after exhausting retries in live margin mode."""
        mock_config_obj = Mock()
        mock_config_obj.get.side_effect = lambda key, default=None: {
            "BINANCE_ACCOUNT_TYPE": "margin",
            "TRADING_MODE": "live",
        }.get(key, default)
        mock_config_obj.get_required.side_effect = lambda key: FAKE_KEY if "KEY" in key else FAKE_SECRET
        mock_config.return_value = mock_config_obj

        ban_exc = _make_ban_exception(-1003, 1_775_000_010_000)
        mock_client_class.side_effect = ban_exc  # Always fails

        with patch("time.sleep"), patch(
            "src.data_providers.binance_provider._parse_ban_expiry", return_value=5.0
        ), pytest.raises(RuntimeError, match="FATAL"):
            BinanceProvider(FAKE_KEY, FAKE_SECRET)

    @patch("src.data_providers.binance_provider.get_binance_api_endpoint", return_value="binance")
    @patch("src.data_providers.binance_provider.is_us_location", return_value=False)
    @patch("src.data_providers.binance_provider.Client")
    @patch("src.data_providers.binance_provider.get_config")
    def test_non_ban_error_does_not_retry(
        self, mock_config, mock_client_class, _mock_us, _mock_endpoint
    ):
        """Non-ban errors fall through immediately without retrying."""
        mock_config_obj = Mock()
        mock_config_obj.get.side_effect = lambda key, default=None: {
            "BINANCE_ACCOUNT_TYPE": "spot",
            "TRADING_MODE": "paper",
        }.get(key, default)
        mock_config_obj.get_required.side_effect = lambda key: FAKE_KEY if "KEY" in key else FAKE_SECRET
        mock_config.return_value = mock_config_obj

        mock_client_class.side_effect = ConnectionError("Network unreachable")

        with patch("time.sleep") as mock_sleep:
            provider = BinanceProvider(FAKE_KEY, FAKE_SECRET)

        mock_sleep.assert_not_called()
        # Should fall back to offline stub in paper/spot mode
        assert provider._client is not None
