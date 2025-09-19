import hashlib
import logging
import os
import pickle  # nosec B403: used for trusted local cache serialization
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.config.paths import get_cache_dir
from src.data_providers.data_provider import DataProvider

logger = logging.getLogger(__name__)


class CachedDataProvider(DataProvider):
    """
    A wrapper around any DataProvider that caches fetched data to disk using year-based caching.
    Each year of data is cached separately, allowing efficient reuse for overlapping date ranges.
    """

    def __init__(
        self,
        data_provider: DataProvider,
        cache_dir: Optional[str] = None,
        cache_ttl_hours: int = 24,
    ):
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
        self.cache_ttl_hours = cache_ttl_hours
        self.cache: dict[str, pd.DataFrame] = {}

        # Try to use the provided cache directory or get the default one
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            try:
                # Try to use the default cache directory
                self.cache_dir = str(get_cache_dir())
            except (PermissionError, OSError):
                # Fallback to a temporary directory if we can't access the default location
                import tempfile
                self.cache_dir = os.path.join(tempfile.gettempdir(), "ai_trading_bot_cache")
                logger.warning(f"Could not access default cache directory, using fallback: {self.cache_dir}")

        # Create cache directory if it doesn't exist
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except (PermissionError, OSError) as e:
            logger.warning(f"Could not create cache directory {self.cache_dir}: {e}")
            # Use a temporary directory as final fallback
            import tempfile
            self.cache_dir = os.path.join(tempfile.gettempdir(), "ai_trading_bot_cache_fallback")
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                logger.info(f"Using fallback cache directory: {self.cache_dir}")
            except Exception as fallback_error:
                logger.error(f"Could not create any cache directory: {fallback_error}")
                # Disable caching by setting cache_dir to None
                self.cache_dir = None

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
        return hashlib.sha256(request_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Optional[str]:
        """Get the full path for a cache file."""
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def _is_cache_valid(self, cache_path: str, year: Optional[int] = None) -> bool:
        """
        Check if the cache file exists and is not expired.

        Args:
            cache_path: Path to the cache file
            year: Optional year context for the cache entry. If provided and the
                year is strictly in the past (i.e., less than the current
                calendar year), the cache is considered valid regardless of TTL,
                since historical data is immutable.

        Returns:
            True if cache is valid, False otherwise
        """
        if not os.path.exists(cache_path):
            return False

        # For fully historical years, treat cache as permanently valid
        if year is not None:
            current_year = datetime.now().year
            if year < current_year:
                return True

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
            with open(cache_path, "rb") as f:
                data = pickle.load(
                    f
                )  # nosec B301: loading only trusted, locally-created cache files
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
            return None

    def _save_to_cache(self, cache_path: Optional[str], data: pd.DataFrame):
        """
        Save data to cache file.

        Args:
            cache_path: Path to the cache file (None if caching is disabled)
            data: DataFrame to cache
        """
        if cache_path is None:
            return
            
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)  # nosec B301: writing trusted, locally-used cache files
            logger.debug(f"Saved data to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_path}: {e}")

    def _get_year_ranges(
        self, start: datetime, end: datetime
    ) -> list[tuple[int, datetime, datetime]]:
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

        while current <= end:
            year = current.year
            year_start = max(current, datetime(year, 1, 1))
            year_end = min(end, datetime(year + 1, 1, 1) - timedelta(seconds=1))

            ranges.append((year, year_start, year_end))
            current = datetime(year + 1, 1, 1)

        return ranges

    def _load_year_data(
        self, symbol: str, timeframe: str, year: int, year_start: datetime, year_end: datetime
    ) -> Optional[pd.DataFrame]:
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
        # Check if we're trying to fetch future data
        current_year = datetime.now().year
        if year > current_year:
            logger.warning(
                f"Cannot fetch data for future year {year} (current year: {current_year})"
            )
            return None

        cache_key = self._generate_year_cache_key(symbol, timeframe, year)
        cache_path = self._get_cache_path(cache_key)

        # Try to load from cache first
        if cache_path and self._is_cache_valid(cache_path, year):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                logger.debug(f"Loaded {year} data from cache for {symbol} {timeframe}")
                # Filter to the exact range needed
                if hasattr(cached_data.index, "to_pydatetime"):
                    mask = (cached_data.index >= year_start) & (cached_data.index <= year_end)
                    return cached_data[mask]
                return cached_data

        # Cache miss - need to fetch this year's data
        logger.info(f"Fetching {year} data for {symbol} {timeframe}")

        # Fetch the entire year to maximize cache efficiency
        fetch_start = datetime(year, 1, 1)
        fetch_end = datetime(year + 1, 1, 1) - timedelta(seconds=1)

        # Don't fetch beyond current time
        current_time = datetime.now()
        if fetch_end > current_time:
            fetch_end = current_time

        try:
            year_data = self.data_provider.get_historical_data(
                symbol, timeframe, fetch_start, fetch_end
            )

            if not year_data.empty:
                # Cache the entire year's data
                self._save_to_cache(cache_path, year_data)

                # Return only the requested range
                if hasattr(year_data.index, "to_pydatetime"):
                    mask = (year_data.index >= year_start) & (year_data.index <= year_end)
                    filtered_data = year_data[mask]
                    logger.info(f"Returning {len(filtered_data)} candles for {symbol} {timeframe} in {year} (range: {year_start} to {year_end})")
                    return filtered_data
                return year_data
            else:
                # Check if this is expected (future data) or an actual error
                current_time = datetime.now()
                if fetch_end > current_time - timedelta(hours=1):  # Within last hour
                    logger.info(f"No recent data available for {symbol} {timeframe} in {year} (fetch_end: {fetch_end}, current_time: {current_time})")
                else:
                    logger.warning(f"No data returned for {symbol} {timeframe} in {year} (fetch range: {fetch_start} to {fetch_end})")
                return None

        except Exception as e:
            logger.error(f"Failed to fetch {year} data for {symbol} {timeframe}: {e}")
            return None

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
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
        # Detect if caller swapped parameters (timeframe passed as datetime)
        if isinstance(timeframe, datetime):
            # Shift parameters
            end = start if end is not None else datetime.now()
            start = timeframe
            timeframe = "1d"  # default to daily
        if end is None:
            end = datetime.now()

        # Don't fetch future data
        current_time = datetime.now()
        if end > current_time:
            logger.info(f"Adjusting end time from {end} to {current_time} to avoid future data")
            end = current_time

        if start is None:
            logger.warning("Start time not provided; returning empty DataFrame")
            return pd.DataFrame()

        if start > current_time:
            logger.warning(f"Start time {start} is in the future, returning empty DataFrame")
            return pd.DataFrame()

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

            if cache_path and self._is_cache_valid(cache_path, year):
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
            if hasattr(combined_data.index, "to_pydatetime"):
                mask = (combined_data.index >= start) & (combined_data.index <= end)
                combined_data = combined_data[mask]

            self.data = combined_data
            logger.info(
                f"Combined data: {len(combined_data)} candles from {len(year_ranges)} years"
            )
            return combined_data
        else:
            logger.warning(f"No data available for {symbol} {timeframe} from {start} to {end}")
            empty_df = pd.DataFrame()
            self.data = empty_df
            return empty_df

    def get_live_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
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

    def clear_cache(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        year: Optional[int] = None,
    ):
        """
        Clear cache files.

        Args:
            symbol: If specified, only clear cache for this symbol
            timeframe: If specified, only clear cache for this timeframe
            year: If specified, only clear cache for this year
        """
        if self.cache_dir is None:
            return

        if not os.path.exists(self.cache_dir):
            return

        cleared_count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".pkl"):
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
                        should_delete = filename == expected_filename
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
        if self.cache_dir is None:
            return {
                "total_files": 0,
                "total_size_mb": 0,
                "oldest_file": None,
                "newest_file": None,
                "years_cached": [],
            }

        if not os.path.exists(self.cache_dir):
            return {
                "total_files": 0,
                "total_size_mb": 0,
                "oldest_file": None,
                "newest_file": None,
                "years_cached": [],
            }

        files = [f for f in os.listdir(self.cache_dir) if f.endswith(".pkl")]
        total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in files)

        if not files:
            return {
                "total_files": 0,
                "total_size_mb": 0,
                "oldest_file": None,
                "newest_file": None,
                "years_cached": [],
            }

        file_times = []
        for f in files:
            file_path = os.path.join(self.cache_dir, f)
            file_times.append((f, os.path.getmtime(file_path)))

        file_times.sort(key=lambda x: x[1])

        return {
            "total_files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "oldest_file": datetime.fromtimestamp(file_times[0][1]).isoformat(),
            "newest_file": datetime.fromtimestamp(file_times[-1][1]).isoformat(),
            "cache_strategy": "year-based",
        }

    def get_current_price(self, symbol: str) -> float:
        """Get current price by delegating to underlying provider"""
        if hasattr(self.data_provider, "get_current_price"):
            return self.data_provider.get_current_price(symbol)
        raise AttributeError("Underlying data provider does not implement get_current_price")
