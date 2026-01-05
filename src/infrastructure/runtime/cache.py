"""
Cache utilities for handling offline environments and TTL management.
"""

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Extended TTL for offline environments (10 years)
OFFLINE_CACHE_TTL_HOURS = 87600  # 10 * 365 * 24 hours


class DataProviderProtocol(Protocol):
    """Protocol for data providers that may have an offline client."""

    _client: Any


def _is_client_offline(client: Any) -> bool:
    """Check if a client is an offline stub.

    Uses multiple detection strategies for robustness:
    1. Check for 'is_offline' attribute (preferred marker)
    2. Fall back to class name matching (for backwards compatibility)
    """
    # Preferred: Check for explicit offline marker attribute
    if hasattr(client, "is_offline"):
        return bool(client.is_offline)

    # Fallback: Check class name (for backwards compatibility)
    # This is maintained for older versions that don't have the marker
    return client.__class__.__name__ == "_OfflineClient"


def get_cache_ttl_for_provider(provider: DataProviderProtocol, default_ttl_hours: int = 24) -> int:
    """Determine appropriate cache TTL based on provider state.

    In offline environments where the provider has fallen back to an offline stub,
    use an extended TTL to ensure preloaded cache data remains valid.

    Args:
        provider: Data provider instance (e.g., BinanceProvider).
        default_ttl_hours: Default TTL to use for online environments (must be positive).

    Returns:
        Appropriate cache TTL in hours.

    Raises:
        ValueError: If default_ttl_hours is not positive.
    """
    if default_ttl_hours <= 0:
        raise ValueError(f"default_ttl_hours must be positive, got {default_ttl_hours}")

    # Check if we're in offline mode (provider using offline stub)
    if hasattr(provider, "_client") and provider._client is not None:
        if _is_client_offline(provider._client):
            logger.info(
                "Detected offline environment - using extended cache TTL for preloaded data"
            )
            return OFFLINE_CACHE_TTL_HOURS

    return default_ttl_hours


def is_provider_offline(provider: DataProviderProtocol) -> bool:
    """Check if a provider is operating in offline mode.

    Args:
        provider: Data provider instance.

    Returns:
        True if provider is using offline stub, False otherwise.
    """
    if hasattr(provider, "_client") and provider._client is not None:
        return _is_client_offline(provider._client)
    return False
