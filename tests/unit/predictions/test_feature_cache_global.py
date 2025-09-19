"""Tests for the module-level feature cache helpers."""

import pandas as pd

from src.prediction.utils.caching import (
    clear_global_feature_cache,
    get_global_feature_cache,
)


class TestGlobalFeatureCache:
    """Ensure the singleton cache behaves as expected across calls."""

    def teardown_method(self) -> None:  # type: ignore[override]
        clear_global_feature_cache()

    def test_get_global_cache_singleton(self):
        cache1 = get_global_feature_cache()
        cache2 = get_global_feature_cache()

        assert cache1 is cache2

    def test_clear_global_cache(self):
        cache = get_global_feature_cache()
        sample_data = pd.DataFrame({"a": [1, 2, 3]})

        cache.set(sample_data, "test", {}, sample_data)
        assert cache.has(sample_data, "test", {})

        clear_global_feature_cache()

        assert not cache.has(sample_data, "test", {})

    def test_global_cache_persistence_across_calls(self):
        sample_data = pd.DataFrame({"a": [1, 2, 3]})

        cache1 = get_global_feature_cache()
        cache1.set(sample_data, "test", {}, sample_data)

        cache2 = get_global_feature_cache()
        assert cache2.has(sample_data, "test", {})
