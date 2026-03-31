"""Tests for BinanceProvider.get_margin_interest_history() method."""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.fast


@pytest.fixture
def margin_provider():
    """Create a BinanceProvider instance in margin mode with a mocked client."""
    with patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", True):
        from src.data_providers.binance_provider import BinanceProvider

        provider = BinanceProvider.__new__(BinanceProvider)
        provider._use_margin = True
        provider._client = MagicMock()
        yield provider


@pytest.fixture
def spot_provider():
    """Create a BinanceProvider instance NOT in margin mode."""
    with patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", True):
        from src.data_providers.binance_provider import BinanceProvider

        provider = BinanceProvider.__new__(BinanceProvider)
        provider._use_margin = False
        provider._client = MagicMock()
        yield provider


class TestGetMarginInterestHistory:
    """Tests for get_margin_interest_history method."""

    def test_returns_data_in_margin_mode(self, margin_provider):
        """Should return interest history when in margin mode."""
        expected = [
            {
                "txId": 1,
                "interestAccuredTime": 1672531200000,
                "asset": "BTC",
                "interest": "0.00012345",
                "interestRate": "0.0001",
                "principal": "1.5",
                "type": "ON_BORROW",
            }
        ]
        margin_provider._client.get_margin_interest_history.return_value = expected

        result = margin_provider.get_margin_interest_history(asset="BTC")

        assert result == expected

    def test_returns_empty_list_when_not_margin_mode(self, spot_provider):
        """Should return empty list when not in margin mode."""
        result = spot_provider.get_margin_interest_history(asset="BTC")

        assert result == []
        spot_provider._client.get_margin_interest_history.assert_not_called()

    def test_returns_empty_list_when_binance_unavailable(self):
        """Should return empty list when BINANCE_AVAILABLE is False."""
        with patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", False):
            from src.data_providers.binance_provider import BinanceProvider

            provider = BinanceProvider.__new__(BinanceProvider)
            provider._use_margin = True
            provider._client = MagicMock()

            result = provider.get_margin_interest_history(asset="BTC")

            assert result == []

    def test_returns_empty_list_when_no_client(self):
        """Should return empty list when client is None."""
        with patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", True):
            from src.data_providers.binance_provider import BinanceProvider

            provider = BinanceProvider.__new__(BinanceProvider)
            provider._use_margin = True
            provider._client = None

            result = provider.get_margin_interest_history(asset="BTC")

            assert result == []

    def test_returns_empty_list_on_api_error(self, margin_provider):
        """Should return empty list and log warning on API error."""
        margin_provider._client.get_margin_interest_history.side_effect = Exception(
            "API timeout"
        )

        result = margin_provider.get_margin_interest_history(asset="BTC")

        assert result == []

    def test_passes_correct_params_all_specified(self, margin_provider):
        """Should pass asset, startTime, endTime to client when all provided."""
        margin_provider._client.get_margin_interest_history.return_value = []

        margin_provider.get_margin_interest_history(
            asset="ETH", start_time=1000, end_time=2000
        )

        margin_provider._client.get_margin_interest_history.assert_called_once_with(
            asset="ETH", size=100, startTime=1000, endTime=2000
        )

    def test_filters_none_params(self, margin_provider):
        """Should not pass startTime/endTime when they are None."""
        margin_provider._client.get_margin_interest_history.return_value = []

        margin_provider.get_margin_interest_history(asset="BTC")

        margin_provider._client.get_margin_interest_history.assert_called_once_with(
            asset="BTC", size=100
        )

    def test_filters_only_end_time_none(self, margin_provider):
        """Should pass startTime but not endTime when only end_time is None."""
        margin_provider._client.get_margin_interest_history.return_value = []

        margin_provider.get_margin_interest_history(asset="BTC", start_time=5000)

        margin_provider._client.get_margin_interest_history.assert_called_once_with(
            asset="BTC", size=100, startTime=5000
        )


class TestOfflineClientStub:
    """Test that the offline client stub returns empty list."""

    def test_offline_stub_returns_empty_list(self):
        """Offline client stub for get_margin_interest_history should return []."""
        with patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", False):
            from src.data_providers.binance_provider import BinanceProvider

            provider = BinanceProvider.__new__(BinanceProvider)
            offline_client = provider._create_offline_client()
            result = offline_client.get_margin_interest_history(asset="BTC")
            assert result == []
