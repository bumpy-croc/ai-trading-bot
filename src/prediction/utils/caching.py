"""
Feature Caching Utilities

This module provides caching functionality for feature extraction to improve
performance by avoiding redundant calculations.
"""

import pandas as pd
import numpy as np
import hashlib
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from src.config.constants import DEFAULT_FEATURE_CACHE_TTL


@dataclass
class CacheEntry:
    """Represents a cache entry with data and metadata."""
    data: pd.DataFrame
    timestamp: float
    ttl: int
    data_hash: str
    quick_hash: str  # Added quick hash for two-tier approach
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.timestamp > self.ttl
    
    def is_valid(self) -> bool:
        """Check if the cache entry is valid (not expired)."""
        return not self.is_expired()


class FeatureCache:
    """
    Simple in-memory cache for feature extraction results.
    
    This cache uses a two-tier hashing approach to optimize performance:
    1. Quick hash: Based on DataFrame shape and sample values (fast)
    2. Full hash: Complete content hash when quick hash matches (accurate)
    """
    
    def __init__(self, default_ttl: int = DEFAULT_FEATURE_CACHE_TTL):
        """
        Initialize the feature cache.
        
        Args:
            default_ttl: Default time-to-live for cache entries in seconds
        """
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._quick_hash_cache: Dict[str, str] = {}  # Quick hash to full hash mapping
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0,
            'quick_hash_matches': 0,  # Track quick hash performance
            'full_hash_verifications': 0  # Track full hash usage
        }
    
    def _generate_quick_hash(self, data: pd.DataFrame) -> str:
        """
        Generate a quick hash based on DataFrame shape and sample values.
        
        This is much faster than full content hashing and provides a good
        first-level filter for cache lookups.
        
        Args:
            data: DataFrame to hash
            
        Returns:
            Quick hash string
        """
        try:
            # Use shape, dtypes, and sample values for quick identification
            shape_str = f"{data.shape[0]}_{data.shape[1]}"
            dtypes_str = str(sorted(data.dtypes.astype(str).items()))
            
            # Sample first and last few values for quick comparison
            sample_size = min(3, len(data))
            if len(data) > 0:
                first_values = data.iloc[:sample_size].values.flatten()
                last_values = data.iloc[-sample_size:].values.flatten()
                sample_str = f"{first_values.tobytes().hex()}_{last_values.tobytes().hex()}"
            else:
                sample_str = "empty"
            
            # Combine shape, dtypes, and samples for quick hash
            quick_content = f"{shape_str}_{dtypes_str}_{sample_str}"
            return hashlib.md5(quick_content.encode()).hexdigest()
        except Exception:
            # Fallback to simple shape-based hash
            return hashlib.md5(f"{data.shape}_{data.dtypes.astype(str)}".encode()).hexdigest()
    
    def _generate_full_data_hash(self, data: pd.DataFrame) -> str:
        """
        Generate a complete hash of the DataFrame content for cache key generation.
        
        This is more expensive but provides accurate content identification.
        
        Args:
            data: DataFrame to hash
            
        Returns:
            Full hash string representing the data content
        """
        try:
            # Compute hash for the DataFrame content using pandas utility
            content_hash = pd.util.hash_pandas_object(data, index=True).values
            
            # Combine with index and column hashes for uniqueness
            index_hash = pd.util.hash_pandas_object(data.index).values
            columns_hash = pd.util.hash_pandas_object(data.columns).values
            
            # Concatenate all hashes and generate a final hash
            combined_hash = np.concatenate([content_hash, index_hash, columns_hash])
            return hashlib.sha256(combined_hash.tobytes()).hexdigest()
        except pd.errors.PandasError:
            # Fallback to a less efficient but robust method
            return hashlib.sha256(pd.util.hash_pandas_object(data, index=True).values.tobytes()).hexdigest()
    
    def _generate_cache_key(self, data: pd.DataFrame, extractor_name: str, config: Dict[str, Any], use_quick_hash: bool = True) -> str:
        """
        Generate a cache key for the given data, extractor, and configuration.
        
        Args:
            data: Input data
            extractor_name: Name of the feature extractor
            config: Extractor configuration
            use_quick_hash: Whether to use quick hash for key generation
            
        Returns:
            Cache key string
        """
        if use_quick_hash:
            data_hash = self._generate_quick_hash(data)
        else:
            data_hash = self._generate_full_data_hash(data)
            
        config_str = str(sorted(config.items()))
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"{extractor_name}_{data_hash}_{config_hash}"
    
    def _find_by_quick_hash(self, data: pd.DataFrame, extractor_name: str, config: Dict[str, Any]) -> Optional[Tuple[str, CacheEntry]]:
        """
        Find cache entry using quick hash first, then verify with full hash.
        
        Args:
            data: Input data
            extractor_name: Name of the feature extractor
            config: Extractor configuration
            
        Returns:
            Tuple of (cache_key, cache_entry) if found, None otherwise
        """
        quick_key = self._generate_cache_key(data, extractor_name, config, use_quick_hash=True)
        
        # Check if quick hash exists in cache
        if quick_key not in self._cache:
            return None
        
        entry = self._cache[quick_key]
        
        # Verify with full hash to ensure accuracy
        full_hash = self._generate_full_data_hash(data)
        if entry.data_hash != full_hash:
            # Hash mismatch - remove incorrect entry
            del self._cache[quick_key]
            self._stats['evictions'] += 1
            return None
        
        self._stats['quick_hash_matches'] += 1
        return quick_key, entry
    
    def get(self, data: pd.DataFrame, extractor_name: str, config: Dict[str, Any], copy: bool = True) -> Optional[pd.DataFrame]:
        """
        Get cached feature extraction result using two-tier hashing.
        
        Args:
            data: Input data used for feature extraction
            extractor_name: Name of the feature extractor
            config: Extractor configuration
            copy: Whether to return a copy of the cached data (default: True)
            
        Returns:
            Cached DataFrame if available and valid, None otherwise
        """
        # Try quick hash first for performance
        result = self._find_by_quick_hash(data, extractor_name, config)
        if result is None:
            self._stats['misses'] += 1
            return None
        
        cache_key, entry = result
        
        # Check TTL using entry's own TTL value
        if not entry.is_valid():
            del self._cache[cache_key]
            self._stats['evictions'] += 1
            self._stats['misses'] += 1
            return None
        
        self._stats['hits'] += 1
        return entry.data.copy() if copy else entry.data
    
    def set(self, data: pd.DataFrame, extractor_name: str, config: Dict[str, Any], 
            result: pd.DataFrame, ttl: Optional[int] = None) -> None:
        """
        Cache feature extraction result using two-tier hashing.
        
        Args:
            data: Input data used for feature extraction
            extractor_name: Name of the feature extractor
            config: Extractor configuration
            result: Feature extraction result to cache
            ttl: Time-to-live for this entry (uses default if None)
        """
        quick_key = self._generate_cache_key(data, extractor_name, config, use_quick_hash=True)
        full_hash = self._generate_full_data_hash(data)
        quick_hash = self._generate_quick_hash(data)
        ttl = ttl or self.default_ttl
        
        entry = CacheEntry(
            data=result.copy(),
            timestamp=time.time(),
            ttl=ttl,
            data_hash=full_hash,
            quick_hash=quick_hash
        )
        
        self._cache[quick_key] = entry
        self._stats['sets'] += 1
    
    def has(self, data: pd.DataFrame, extractor_name: str, config: Dict[str, Any]) -> bool:
        """
        Check if cached result exists and is valid using two-tier hashing.
        
        Args:
            data: Input data
            extractor_name: Name of the feature extractor
            config: Extractor configuration
            
        Returns:
            True if valid cached result exists, False otherwise
        """
        return self.get(data, extractor_name, config, copy=False) is not None
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._quick_hash_cache.clear()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0,
            'quick_hash_matches': 0,
            'full_hash_verifications': 0
        }
    
    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of entries removed
        """
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            self._stats['evictions'] += 1
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics including two-tier hashing performance.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0.0
        
        # Calculate quick hash efficiency
        quick_hash_efficiency = (
            self._stats['quick_hash_matches'] / total_requests 
            if total_requests > 0 else 0.0
        )
        
        return {
            'total_entries': len(self._cache),
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'quick_hash_efficiency': quick_hash_efficiency,
            **self._stats
        }
    
    def get_size_info(self) -> Dict[str, Any]:
        """
        Get information about cache size and memory usage.
        
        Returns:
            Dictionary with size information
        """
        total_memory = 0
        entry_sizes = []
        
        for entry in self._cache.values():
            # Rough estimate of DataFrame memory usage
            entry_size = entry.data.memory_usage(deep=True).sum()
            entry_sizes.append(entry_size)
            total_memory += entry_size
        
        return {
            'total_entries': len(self._cache),
            'total_memory_bytes': total_memory,
            'average_entry_size_bytes': np.mean(entry_sizes) if entry_sizes else 0,
            'largest_entry_size_bytes': max(entry_sizes) if entry_sizes else 0
        }


# Global feature cache instance
_global_feature_cache: Optional[FeatureCache] = None


def get_global_feature_cache() -> FeatureCache:
    """
    Get the global feature cache instance.
    
    Returns:
        Global FeatureCache instance
    """
    global _global_feature_cache
    if _global_feature_cache is None:
        _global_feature_cache = FeatureCache()
    return _global_feature_cache


def clear_global_feature_cache() -> None:
    """Clear the global feature cache."""
    global _global_feature_cache
    if _global_feature_cache is not None:
        _global_feature_cache.clear()
