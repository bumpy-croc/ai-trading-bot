"""Unit tests for provider_factory module."""

from unittest.mock import Mock, patch

import pytest

from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.coinbase_provider import CoinbaseProvider
from src.data_providers.coingecko_provider import CoinGeckoProvider
from src.data_providers.fallback_provider import FallbackProvider
from src.data_providers.provider_factory import create_data_provider, get_default_provider


@pytest.mark.fast
class TestCreateDataProvider:
    """Test create_data_provider function."""

    @patch("src.data_providers.provider_factory.get_config")
    def test_creates_fallback_provider_with_auto_type(self, mock_get_config):
        # Arrange
        mock_config = Mock()
        mock_config.get.return_value = None
        mock_get_config.return_value = mock_config

        # Act
        provider = create_data_provider(provider_type="auto")

        # Assert
        assert isinstance(provider, FallbackProvider)
        provider.close()

    @patch("src.data_providers.provider_factory.get_config")
    def test_creates_fallback_provider_with_fallback_type(self, mock_get_config):
        # Arrange
        mock_config = Mock()
        mock_config.get.return_value = None
        mock_get_config.return_value = mock_config

        # Act
        provider = create_data_provider(provider_type="fallback")

        # Assert
        assert isinstance(provider, FallbackProvider)
        provider.close()

    @patch("src.data_providers.provider_factory.get_config")
    def test_creates_binance_provider(self, mock_get_config):
        # Arrange
        mock_config = Mock()
        mock_config.get.return_value = None
        mock_get_config.return_value = mock_config

        # Act
        provider = create_data_provider(provider_type="binance")

        # Assert
        assert isinstance(provider, BinanceProvider)
        # BinanceProvider uses ccxt and doesn't have close() method

    @patch("src.data_providers.provider_factory.get_config")
    def test_creates_coinbase_provider(self, mock_get_config):
        # Arrange
        mock_config = Mock()
        mock_config.get.return_value = None
        mock_get_config.return_value = mock_config

        # Act
        provider = create_data_provider(provider_type="coinbase")

        # Assert
        assert isinstance(provider, CoinbaseProvider)
        provider.close()

    @patch("src.data_providers.provider_factory.get_config")
    def test_creates_coingecko_provider(self, mock_get_config):
        # Arrange
        mock_config = Mock()
        mock_config.get.return_value = None
        mock_get_config.return_value = mock_config

        # Act
        provider = create_data_provider(provider_type="coingecko")

        # Assert
        assert isinstance(provider, CoinGeckoProvider)
        provider.close()

    @patch("src.data_providers.provider_factory.get_config")
    def test_loads_binance_credentials_from_config(self, mock_get_config):
        # Arrange
        mock_config = Mock()
        mock_config.get.side_effect = lambda key: {
            "BINANCE_API_KEY": "config_key",
            "BINANCE_API_SECRET": "config_secret",
        }.get(key)
        mock_get_config.return_value = mock_config

        # Act
        with patch("src.data_providers.provider_factory.BinanceProvider") as mock_binance:
            mock_binance.return_value = Mock()
            create_data_provider(provider_type="binance")

            # Assert
            mock_binance.assert_called_once_with(
                api_key="config_key",
                api_secret="config_secret",
                testnet=False,
            )

    @patch("src.data_providers.provider_factory.get_config")
    def test_uses_provided_credentials_over_config(self, mock_get_config):
        # Arrange
        mock_config = Mock()
        mock_config.get.side_effect = lambda key: {
            "BINANCE_API_KEY": "config_key",
            "BINANCE_API_SECRET": "config_secret",
        }.get(key)
        mock_get_config.return_value = mock_config

        # Act
        with patch("src.data_providers.provider_factory.BinanceProvider") as mock_binance:
            mock_binance.return_value = Mock()
            create_data_provider(
                provider_type="binance",
                binance_api_key="provided_key",
                binance_api_secret="provided_secret",
            )

            # Assert - provided credentials should override config
            mock_binance.assert_called_once_with(
                api_key="provided_key",
                api_secret="provided_secret",
                testnet=False,
            )

    @patch("src.data_providers.provider_factory.get_config")
    def test_loads_coinbase_credentials_from_config(self, mock_get_config):
        # Arrange
        mock_config = Mock()
        mock_config.get.side_effect = lambda key: {
            "COINBASE_API_KEY": "config_key",
            "COINBASE_API_SECRET": "config_secret",
            "COINBASE_API_PASSPHRASE": "config_passphrase",
        }.get(key)
        mock_get_config.return_value = mock_config

        # Act
        with patch("src.data_providers.provider_factory.CoinbaseProvider") as mock_coinbase:
            mock_coinbase.return_value = Mock()
            create_data_provider(provider_type="coinbase")

            # Assert
            mock_coinbase.assert_called_once_with(
                api_key="config_key",
                api_secret="config_secret",
                passphrase="config_passphrase",
                testnet=False,
            )

    @patch("src.data_providers.provider_factory.get_config")
    def test_loads_coingecko_credentials_from_config(self, mock_get_config):
        # Arrange
        mock_config = Mock()
        mock_config.get.side_effect = lambda key: {
            "COINGECKO_API_KEY": "config_key",
        }.get(key)
        mock_get_config.return_value = mock_config

        # Act
        with patch("src.data_providers.provider_factory.CoinGeckoProvider") as mock_coingecko:
            mock_coingecko.return_value = Mock()
            create_data_provider(provider_type="coingecko")

            # Assert
            mock_coingecko.assert_called_once_with(api_key="config_key")

    @patch("src.data_providers.provider_factory.get_config")
    def test_passes_testnet_flag(self, mock_get_config):
        # Arrange
        mock_config = Mock()
        mock_config.get.return_value = None
        mock_get_config.return_value = mock_config

        # Act
        with patch("src.data_providers.provider_factory.BinanceProvider") as mock_binance:
            mock_binance.return_value = Mock()
            create_data_provider(provider_type="binance", testnet=True)

            # Assert
            mock_binance.assert_called_once_with(
                api_key=None,
                api_secret=None,
                testnet=True,
            )

    @patch("src.data_providers.provider_factory.get_config")
    def test_raises_error_for_invalid_provider_type(self, mock_get_config):
        # Arrange
        mock_config = Mock()
        mock_config.get.return_value = None
        mock_get_config.return_value = mock_config

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            create_data_provider(provider_type="invalid_type")  # type: ignore

        assert "Unknown provider type: invalid_type" in str(exc_info.value)
        assert "auto, fallback, binance, coinbase, coingecko" in str(exc_info.value)


@pytest.mark.fast
class TestGetDefaultProvider:
    """Test get_default_provider function."""

    @patch("src.data_providers.provider_factory.get_config")
    def test_returns_fallback_provider(self, mock_get_config):
        # Arrange
        mock_config = Mock()
        mock_config.get.return_value = None
        mock_get_config.return_value = mock_config

        # Act
        provider = get_default_provider()

        # Assert
        assert isinstance(provider, FallbackProvider)
        provider.close()

    @patch("src.data_providers.provider_factory.get_config")
    def test_passes_testnet_flag(self, mock_get_config):
        # Arrange
        mock_config = Mock()
        mock_config.get.return_value = None
        mock_get_config.return_value = mock_config

        # Act
        with patch("src.data_providers.provider_factory.FallbackProvider") as mock_fallback:
            mock_fallback.return_value = Mock()
            get_default_provider(testnet=True)

            # Assert - testnet flag should be passed through
            call_kwargs = mock_fallback.call_args[1]
            assert call_kwargs["testnet"] is True
