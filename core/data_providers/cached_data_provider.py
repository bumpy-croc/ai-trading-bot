import os
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import logging
from .data_provider import DataProvider

logger = logging.getLogger(__name__)

class CachedDataProvider(DataProvider):
    """
    A wrapper around any DataProvider that caches fetched data to disk.
    Subsequent requests for the same data will be served from cache.
    """
    
    def __init__(self, data_provider: DataProvider, cache_dir: str = "data/cache", cache_ttl_hours: int = 24):
        """
        Initialize the cached data provider.
        
        Args:
            data_provider: The underlying data provider to wrap
            cache_dir: Directory to store cache files
            cache_ttl_hours: Cache time-to-live in hours
        """
        super().__init__()
        self.data_provider = data_provider
        self.cache_dir = cache_dir
        self.cache_ttl_hours = cache_ttl_hours
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
    def _generate_cache_key(self, symbol: str, timeframe: str, start: datetime, end: Optional[datetime] = None) -> str:
        """
        Generate a unique cache key for the data request.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Data timeframe
            start: Start datetime
            end: End datetime
            
        Returns:
            Cache key string
        """
        # Create a string representation of the request
        request_str = f"{symbol}_{timeframe}_{start.isoformat()}"
        if end:
            # For specific end dates, use exact timestamp
            request_str += f"_{end.isoformat()}"
        else:
            # For current time requests, round to nearest hour to improve cache hit rate
            current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
            request_str += f"_{current_hour.isoformat()}"
        
        # Generate hash for consistent filename
        return hashlib.md5(request_str.encode()).hexdigest()
        
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the full path for a cache file."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
    def _is_cache_valid(self, cache_path: str) -> bool:
        """
        Check if the cache file exists and is not expired.
        
        Args:
            cache_path: Path to the cache file
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not os.path.exists(cache_path):
            return False
            
        # Check if cache is expired
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        current_time = datetime.now()
        age_hours = (current_time - file_time).total_seconds() / 3600
        
        return age_hours < self.cache_ttl_hours
        
    def _load_from_cache(self, cache_path: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache file.
        
        Args:
            cache_path: Path to the cache file
            
        Returns:
            DataFrame if successful, None otherwise
        """
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded data from cache: {cache_path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
            return None
            
    def _save_to_cache(self, cache_path: str, data: pd.DataFrame):
        """
        Save data to cache file.
        
        Args:
            cache_path: Path to the cache file
            data: DataFrame to cache
        """
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved data to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_path}: {e}")
            
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch historical data with caching.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            start: Start datetime
            end: End datetime (optional, defaults to current time)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Generate cache key and path
        cache_key = self._generate_cache_key(symbol, timeframe, start, end)
        cache_path = self._get_cache_path(cache_key)
        
        # Check if valid cache exists
        if self._is_cache_valid(cache_path):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                self.data = cached_data
                return cached_data
        
        # Fetch data from underlying provider
        logger.info(f"Cache miss or expired, fetching data from provider for {symbol} {timeframe}")
        data = self.data_provider.get_historical_data(symbol, timeframe, start, end)
        
        # Cache the data
        if not data.empty:
            self._save_to_cache(cache_path, data)
            self.data = data
        
        return data
        
    def get_live_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch current market data (not cached).
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        # Live data is not cached as it changes frequently
        return self.data_provider.get_live_data(symbol, timeframe, limit)
        
    def update_live_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Update the latest market data (not cached).
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            
        Returns:
            DataFrame with updated OHLCV data
        """
        # Live data updates are not cached
        return self.data_provider.update_live_data(symbol, timeframe)
        
    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """
        Clear cache files.
        
        Args:
            symbol: If specified, only clear cache for this symbol
            timeframe: If specified, only clear cache for this timeframe
        """
        if not os.path.exists(self.cache_dir):
            return
            
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    # Load the cached data to check its metadata
                    with open(file_path, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    # Check if we should delete this cache file
                    should_delete = True
                    if symbol and hasattr(cached_data, 'symbol') and cached_data.symbol != symbol:
                        should_delete = False
                    if timeframe and hasattr(cached_data, 'timeframe') and cached_data.timeframe != timeframe:
                        should_delete = False
                        
                    if should_delete:
                        os.remove(file_path)
                        logger.info(f"Cleared cache file: {filename}")
                        
                except Exception as e:
                    logger.warning(f"Failed to process cache file {filename}: {e}")
                    
    def get_cache_info(self) -> dict:
        """
        Get information about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        if not os.path.exists(self.cache_dir):
            return {'total_files': 0, 'total_size_mb': 0, 'oldest_file': None, 'newest_file': None}
            
        files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in files)
        
        if not files:
            return {'total_files': 0, 'total_size_mb': 0, 'oldest_file': None, 'newest_file': None}
            
        file_times = []
        for f in files:
            file_path = os.path.join(self.cache_dir, f)
            file_times.append((f, os.path.getmtime(file_path)))
            
        file_times.sort(key=lambda x: x[1])
        
        return {
            'total_files': len(files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'oldest_file': datetime.fromtimestamp(file_times[0][1]).isoformat(),
            'newest_file': datetime.fromtimestamp(file_times[-1][1]).isoformat()
        } 