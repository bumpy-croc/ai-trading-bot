from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from typing import Optional

class DataProvider(ABC):
    """
    Abstract base class for data providers.
    Implementations should handle different data sources (Binance, CSV, etc.)
    """
    
    def __init__(self):
        self.data = None
        
    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch historical market data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            start: Start datetime
            end: End datetime (optional, defaults to current time)
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
        
    @abstractmethod
    def get_live_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch current market data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
        
    @abstractmethod
    def update_live_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Update the latest market data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            
        Returns:
            DataFrame with updated OHLCV data
        """
        pass 