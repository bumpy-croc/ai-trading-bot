"""Unit tests for the CacheEntry dataclass used by feature caching."""

import time

import pandas as pd

from src.prediction.utils.caching import CacheEntry


class TestCacheEntry:
    """Verify CacheEntry lifecycle helpers."""

    def test_cache_entry_creation(self):
        """Entries should store provided metadata and validate initially."""

        test_data = pd.DataFrame({"a": [1, 2, 3]})
        entry = CacheEntry(
            data=test_data,
            timestamp=time.time(),
            ttl=300,
            data_hash="test_hash",
            quick_hash="quick_test_hash",
        )

        assert not entry.data.empty
        assert entry.ttl == 300
        assert entry.data_hash == "test_hash"
        assert entry.quick_hash == "quick_test_hash"
        assert entry.is_valid()

    def test_cache_entry_expiration(self):
        """Expired entries should be flagged invalid."""

        test_data = pd.DataFrame({"a": [1, 2, 3]})

        entry = CacheEntry(
            data=test_data,
            timestamp=time.time() - 10,
            ttl=5,
            data_hash="test_hash",
            quick_hash="quick_test_hash",
        )

        assert entry.is_expired()
        assert not entry.is_valid()

    def test_cache_entry_not_expired(self):
        """Fresh entries should report valid status."""

        test_data = pd.DataFrame({"a": [1, 2, 3]})

        entry = CacheEntry(
            data=test_data,
            timestamp=time.time(),
            ttl=3600,
            data_hash="test_hash",
            quick_hash="quick_test_hash",
        )

        assert not entry.is_expired()
        assert entry.is_valid()
