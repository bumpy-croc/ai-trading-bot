"""
Caching utilities for model predictions and features.
"""

import time
from typing import Any, Optional, Dict, Tuple
from functools import wraps


class ModelCache:
    """Simple cache for model predictions"""
    
    def __init__(self, ttl: int = 600):
        """
        Initialize cache with time-to-live setting.
        
        Args:
            ttl: Time-to-live in seconds
        """
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Cache value with timestamp"""
        self.cache[key] = (value, time.time())
    
    def clear(self) -> None:
        """Clear all cached values"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)


def cache_prediction(ttl: int = 600):
    """Decorator to cache model predictions"""
    def decorator(func):
        cache = ModelCache(ttl)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Run function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            
            return result
        
        # Expose cache for testing/debugging
        wrapper._cache = cache
        return wrapper
    return decorator