"""
Data Provider Factory

Creates data providers with automatic failover support.
"""

import logging
from typing import Literal

from src.config import get_config

from .binance_provider import BinanceProvider
from .coinbase_provider import CoinbaseProvider
from .coingecko_provider import CoinGeckoProvider
from .data_provider import DataProvider
from .fallback_provider import FallbackProvider

logger = logging.getLogger(__name__)

ProviderType = Literal["auto", "fallback", "binance", "coinbase", "coingecko"]


def create_data_provider(
    provider_type: ProviderType = "auto",
    binance_api_key: str | None = None,
    binance_api_secret: str | None = None,
    coinbase_api_key: str | None = None,
    coinbase_api_secret: str | None = None,
    coinbase_passphrase: str | None = None,
    coingecko_api_key: str | None = None,
    testnet: bool = False,
) -> DataProvider:
    """
    Create a data provider with automatic failover support.

    Args:
        provider_type: Type of provider to create:
            - "auto" (default): Use FallbackProvider (Binance → CoinGecko)
            - "fallback": Explicitly use FallbackProvider
            - "binance": Use only Binance (will fail if blocked)
            - "coinbase": Use only Coinbase
            - "coingecko": Use only CoinGecko
        binance_api_key: Optional Binance API key
        binance_api_secret: Optional Binance API secret
        coinbase_api_key: Optional Coinbase API key
        coinbase_api_secret: Optional Coinbase API secret
        coinbase_passphrase: Optional Coinbase passphrase
        coingecko_api_key: Optional CoinGecko API key
        testnet: Whether to use testnet

    Returns:
        DataProvider instance

    Examples:
        # Default: automatic failover from Binance to CoinGecko
        provider = create_data_provider()

        # Explicitly use only CoinGecko
        provider = create_data_provider(provider_type="coingecko")

        # Use Binance with credentials
        provider = create_data_provider(
            provider_type="binance",
            binance_api_key="...",
            binance_api_secret="..."
        )
    """
    config = get_config()

    # Get credentials from config if not provided
    if binance_api_key is None:
        binance_api_key = config.get("BINANCE_API_KEY")
    if binance_api_secret is None:
        binance_api_secret = config.get("BINANCE_API_SECRET")

    if coinbase_api_key is None:
        coinbase_api_key = config.get("COINBASE_API_KEY")
    if coinbase_api_secret is None:
        coinbase_api_secret = config.get("COINBASE_API_SECRET")
    if coinbase_passphrase is None:
        coinbase_passphrase = config.get("COINBASE_API_PASSPHRASE")

    if coingecko_api_key is None:
        coingecko_api_key = config.get("COINGECKO_API_KEY")

    # Create provider based on type
    if provider_type == "auto" or provider_type == "fallback":
        logger.info("Creating FallbackProvider (Binance → CoinGecko)")
        return FallbackProvider(
            binance_api_key=binance_api_key,
            binance_api_secret=binance_api_secret,
            coingecko_api_key=coingecko_api_key,
            testnet=testnet,
        )

    elif provider_type == "binance":
        logger.info("Creating BinanceProvider")
        return BinanceProvider(
            api_key=binance_api_key,
            api_secret=binance_api_secret,
            testnet=testnet,
        )

    elif provider_type == "coinbase":
        logger.info("Creating CoinbaseProvider")
        return CoinbaseProvider(
            api_key=coinbase_api_key,
            api_secret=coinbase_api_secret,
            passphrase=coinbase_passphrase,
            testnet=testnet,
        )

    elif provider_type == "coingecko":
        logger.info("Creating CoinGeckoProvider")
        return CoinGeckoProvider(api_key=coingecko_api_key)

    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Must be one of: auto, fallback, binance, coinbase, coingecko"
        )


# Convenience function for backward compatibility
def get_default_provider(testnet: bool = False) -> DataProvider:
    """
    Get the default data provider (FallbackProvider with Binance → CoinGecko failover).

    This is a convenience function for backward compatibility. New code should use
    create_data_provider() directly.

    Args:
        testnet: Whether to use testnet

    Returns:
        FallbackProvider instance
    """
    return create_data_provider(provider_type="auto", testnet=testnet)
