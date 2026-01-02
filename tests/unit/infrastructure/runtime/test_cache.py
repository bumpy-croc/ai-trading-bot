"""Tests for infrastructure.runtime.cache module."""

from unittest.mock import MagicMock

import pytest

from src.infrastructure.runtime.cache import (
    OFFLINE_CACHE_TTL_HOURS,
    get_cache_ttl_for_provider,
    is_provider_offline,
)


class TestOfflineCacheTTL:
    """Tests for the OFFLINE_CACHE_TTL_HOURS constant."""

    def test_offline_ttl_is_ten_years(self):
        """Test that offline TTL is approximately 10 years in hours."""
        expected_hours = 10 * 365 * 24  # 87600 hours
        assert OFFLINE_CACHE_TTL_HOURS == expected_hours


class TestGetCacheTTLForProvider:
    """Tests for get_cache_ttl_for_provider function."""

    def test_returns_default_for_online_provider(self):
        """Test default TTL returned for online provider."""
        provider = MagicMock()
        provider._client = MagicMock()
        provider._client.__class__.__name__ = "BinanceClient"

        result = get_cache_ttl_for_provider(provider, default_ttl_hours=24)
        assert result == 24

    def test_returns_extended_for_offline_provider(self):
        """Test extended TTL returned for offline provider."""
        provider = MagicMock()
        provider._client = MagicMock()
        provider._client.__class__.__name__ = "_OfflineClient"

        result = get_cache_ttl_for_provider(provider, default_ttl_hours=24)
        assert result == OFFLINE_CACHE_TTL_HOURS

    def test_returns_default_when_client_is_none(self):
        """Test default TTL when provider._client is None."""
        provider = MagicMock()
        provider._client = None

        result = get_cache_ttl_for_provider(provider, default_ttl_hours=48)
        assert result == 48

    def test_returns_default_when_no_client_attr(self):
        """Test default TTL when provider has no _client attribute."""
        provider = MagicMock(spec=[])  # No _client attribute

        result = get_cache_ttl_for_provider(provider, default_ttl_hours=12)
        assert result == 12

    def test_custom_default_ttl(self):
        """Test that custom default TTL is respected."""
        provider = MagicMock()
        provider._client = MagicMock()
        provider._client.__class__.__name__ = "OnlineClient"

        result = get_cache_ttl_for_provider(provider, default_ttl_hours=100)
        assert result == 100

    def test_default_default_ttl_is_24(self):
        """Test that default TTL defaults to 24 hours."""
        provider = MagicMock()
        provider._client = MagicMock()
        provider._client.__class__.__name__ = "OnlineClient"

        result = get_cache_ttl_for_provider(provider)
        assert result == 24


class TestIsProviderOffline:
    """Tests for is_provider_offline function."""

    def test_returns_true_for_offline_client(self):
        """Test True returned for _OfflineClient."""
        provider = MagicMock()
        provider._client = MagicMock()
        provider._client.__class__.__name__ = "_OfflineClient"

        assert is_provider_offline(provider) is True

    def test_returns_false_for_online_client(self):
        """Test False returned for regular client."""
        provider = MagicMock()
        provider._client = MagicMock()
        provider._client.__class__.__name__ = "BinanceClient"

        assert is_provider_offline(provider) is False

    def test_returns_false_when_client_is_none(self):
        """Test False returned when client is None."""
        provider = MagicMock()
        provider._client = None

        assert is_provider_offline(provider) is False

    def test_returns_false_when_no_client_attr(self):
        """Test False returned when no _client attribute."""
        provider = MagicMock(spec=[])  # No _client attribute

        assert is_provider_offline(provider) is False

    def test_case_sensitive_client_name(self):
        """Test that client name matching is case-sensitive."""
        provider = MagicMock()
        provider._client = MagicMock()
        provider._client.__class__.__name__ = "_offlineclient"  # lowercase

        assert is_provider_offline(provider) is False

    def test_partial_name_not_matched(self):
        """Test that partial name matches don't count."""
        provider = MagicMock()
        provider._client = MagicMock()
        provider._client.__class__.__name__ = "_OfflineClientWrapper"

        assert is_provider_offline(provider) is False


@pytest.mark.fast
class TestCacheIntegration:
    """Integration tests for cache utilities."""

    def test_offline_and_online_providers_differentiated(self):
        """Test that offline and online providers get different TTLs."""
        online_provider = MagicMock()
        online_provider._client = MagicMock()
        online_provider._client.__class__.__name__ = "BinanceClient"

        offline_provider = MagicMock()
        offline_provider._client = MagicMock()
        offline_provider._client.__class__.__name__ = "_OfflineClient"

        online_ttl = get_cache_ttl_for_provider(online_provider)
        offline_ttl = get_cache_ttl_for_provider(offline_provider)

        assert offline_ttl > online_ttl
        assert offline_ttl == OFFLINE_CACHE_TTL_HOURS
        assert online_ttl == 24

    def test_is_offline_consistent_with_ttl(self):
        """Test that is_provider_offline is consistent with get_cache_ttl_for_provider."""
        provider = MagicMock()
        provider._client = MagicMock()
        provider._client.__class__.__name__ = "_OfflineClient"

        is_offline = is_provider_offline(provider)
        ttl = get_cache_ttl_for_provider(provider)

        assert is_offline is True
        assert ttl == OFFLINE_CACHE_TTL_HOURS
