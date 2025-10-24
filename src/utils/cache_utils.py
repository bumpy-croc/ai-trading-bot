"""
Cache utilities for handling offline environments and TTL management.
"""

import logging

logger = logging.getLogger(__name__)

# Extended TTL for offline environments (10 years)
OFFLINE_CACHE_TTL_HOURS = 87600  # 10 * 365 * 24 hours


def get_cache_ttl_for_provider(provider, default_ttl_hours: int = 24) -> int:
    """
    Determine appropriate cache TTL based on provider state.

    In offline environments where the provider has fallen back to an offline stub,
    use an extended TTL to ensure preloaded cache data remains valid.

    Args:
        provider: Data provider instance (e.g., BinanceProvider)
        default_ttl_hours: Default TTL to use for online environments

    Returns:
        int: Appropriate cache TTL in hours
    """
    # Check if we're in offline mode (provider using offline stub)
    if hasattr(provider, "_client") and provider._client is not None:
        # Check if the client is the offline stub
        client_class_name = provider._client.__class__.__name__
        if client_class_name == "_OfflineClient":
            logger.info(
                "Detected offline environment - using extended cache TTL for preloaded data"
            )
            return OFFLINE_CACHE_TTL_HOURS

    return default_ttl_hours


def is_provider_offline(provider) -> bool:
    """
    Check if a provider is operating in offline mode.

    Args:
        provider: Data provider instance

    Returns:
        bool: True if provider is using offline stub, False otherwise
    """
    if hasattr(provider, "_client") and provider._client is not None:
        client_class_name = provider._client.__class__.__name__
        return client_class_name == "_OfflineClient"
    return False
