import os
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import pandas as pd
import logging
from .data_provider import DataProvider

logger = logging.getLogger(__name__)

class CachedDataProvider(DataProvider):
    """
    A wrapper around any DataProvider that caches fetched data to disk using year-based caching.
    Each year of data is cached separately, allowing efficient reuse for overlapping date ranges.
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
        self.provider = data_provider  # Add this alias for backward compatibility
        self.cache_dir = cache_dir
        self.cache_ttl_hours = cache_ttl_hours
        self.cache = {}  # Add cache attribute for backward compatibility
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
    def _generate_year_cache_key(self, symbol: str, timeframe: str, year: int) -> str:
        """
        Generate a cache key for a specific year of data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Data timeframe
            year: Year for the data
            
        Returns:
            Cache key string
        """
        request_str = f"{symbol}_{timeframe}_{year}"
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
            logger.debug(f"Saved data to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_path}: {e}")
            
    def _get_year_ranges(self, start: datetime, end: datetime) -> List[Tuple[int, datetime, datetime]]:
        """
        Split a date range into year-based chunks.
        
        Args:
            start: Start datetime
            end: End datetime
            
        Returns:
            List of tuples (year, year_start, year_end)
        """
        ranges = []
        current = start
        
        while current < end:
            year = current.year
            year_start = max(current, datetime(year, 1, 1))
            year_end = min(end, datetime(year + 1, 1, 1) - timedelta(seconds=1))
            
            ranges.append((year, year_start, year_end))
            current = datetime(year + 1, 1, 1)
            
        return ranges
        
    def _load_year_data(self, symbol: str, timeframe: str, year: int, 
                       year_start: datetime, year_end: datetime) -> Optional[pd.DataFrame]:
        """
        Load data for a specific year, either from cache or by fetching.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Data timeframe
            year: Year to load
            year_start: Start of the year range needed
            year_end: End of the year range needed
            
        Returns:
            DataFrame for the year or None if failed
        """
        cache_key = self._generate_year_cache_key(symbol, timeframe, year)
        cache_path = self._get_cache_path(cache_key)
        
        # Try to load from cache first
        if self._is_cache_valid(cache_path):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                logger.debug(f"Loaded {year} data from cache for {symbol} {timeframe}")
                # Filter to the exact range needed
                if hasattr(cached_data.index, 'to_pydatetime'):
                    mask = (cached_data.index >= year_start) & (cached_data.index <= year_end)
                    return cached_data[mask]
                return cached_data
        
        # Cache miss - need to fetch this year's data
        logger.info(f"Fetching {year} data for {symbol} {timeframe}")
        
        # Fetch the entire year to maximize cache efficiency
        fetch_start = datetime(year, 1, 1)
        fetch_end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
        
        try:
            year_data = self.data_provider.get_historical_data(symbol, timeframe, fetch_start, fetch_end)
            
            if not year_data.empty:
                # Cache the entire year's data
                self._save_to_cache(cache_path, year_data)
                
                # Return only the requested range
                if hasattr(year_data.index, 'to_pydatetime'):
                    mask = (year_data.index >= year_start) & (year_data.index <= year_end)
                    return year_data[mask]
                return year_data
            else:
                logger.warning(f"No data returned for {symbol} {timeframe} in {year}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch {year} data for {symbol} {timeframe}: {e}")
            return None
            
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch historical data with year-based caching.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            start: Start datetime
            end: End datetime (optional, defaults to current time)
            
        Returns:
            DataFrame with OHLCV data
        """
        if end is None:
            end = datetime.now()
            
        # Get year ranges for the requested period
        year_ranges = self._get_year_ranges(start, end)
        
        if not year_ranges:
            logger.warning(f"No valid year ranges for {start} to {end}")
            return pd.DataFrame()
            
        # Load data for each year
        year_dataframes = []
        cached_years = []
        missing_years = []
        
        for year, year_start, year_end in year_ranges:
            cache_key = self._generate_year_cache_key(symbol, timeframe, year)
            cache_path = self._get_cache_path(cache_key)
            
            if self._is_cache_valid(cache_path):
                cached_years.append(year)
            else:
                missing_years.append(year)
                
            year_data = self._load_year_data(symbol, timeframe, year, year_start, year_end)
            if year_data is not None and not year_data.empty:
                year_dataframes.append(year_data)
                
        # Log cache efficiency
        if cached_years:
            logger.info(f"Cache hit for years: {cached_years}")
        if missing_years:
            logger.info(f"Cache miss for years: {missing_years}")
            
        # Combine all year data
        if year_dataframes:
            combined_data = pd.concat(year_dataframes, ignore_index=False)
            combined_data = combined_data.sort_index()
            
            # Final filter to exact requested range
            if hasattr(combined_data.index, 'to_pydatetime'):
                mask = (combined_data.index >= start) & (combined_data.index <= end)
                combined_data = combined_data[mask]
                
            self.data = combined_data
            logger.info(f"Combined data: {len(combined_data)} candles from {len(year_ranges)} years")
            return combined_data
        else:
            logger.warning(f"No data available for {symbol} {timeframe} from {start} to {end}")
            empty_df = pd.DataFrame()
            self.data = empty_df
            return empty_df
        
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
        
    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None, year: Optional[int] = None):
        """
        Clear cache files.
        
        Args:
            symbol: If specified, only clear cache for this symbol
            timeframe: If specified, only clear cache for this timeframe
            year: If specified, only clear cache for this year
        """
        if not os.path.exists(self.cache_dir):
            return
            
        cleared_count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                file_path = os.path.join(self.cache_dir, filename)
                should_delete = True
                
                # If filters are specified, check if this file matches
                if symbol or timeframe or year:
                    # For year-based caching, we can't easily determine the contents
                    # without loading the file, so we'll use a simpler approach
                    # and just delete files that match the hash pattern
                    if symbol and timeframe and year:
                        expected_key = self._generate_year_cache_key(symbol, timeframe, year)
                        expected_filename = f"{expected_key}.pkl"
                        should_delete = (filename == expected_filename)
                    else:
                        # For partial matches, we'll need to be more conservative
                        should_delete = True
                        
                if should_delete:
                    try:
                        os.remove(file_path)
                        cleared_count += 1
                        logger.info(f"Cleared cache file: {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to remove cache file {filename}: {e}")
                        
        logger.info(f"Cleared {cleared_count} cache files")
                    
    def get_cache_info(self) -> dict:
        """
        Get information about the cache with year-based details.
        
        Returns:
            Dictionary with cache statistics
        """
        if not os.path.exists(self.cache_dir):
            return {'total_files': 0, 'total_size_mb': 0, 'oldest_file': None, 'newest_file': None, 'years_cached': []}
            
        files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in files)
        
        if not files:
            return {'total_files': 0, 'total_size_mb': 0, 'oldest_file': None, 'newest_file': None, 'years_cached': []}
            
        file_times = []
        for f in files:
            file_path = os.path.join(self.cache_dir, f)
            file_times.append((f, os.path.getmtime(file_path)))
            
        file_times.sort(key=lambda x: x[1])
        
        return {
            'total_files': len(files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'oldest_file': datetime.fromtimestamp(file_times[0][1]).isoformat(),
            'newest_file': datetime.fromtimestamp(file_times[-1][1]).isoformat(),
            'cache_strategy': 'year-based'
        } 