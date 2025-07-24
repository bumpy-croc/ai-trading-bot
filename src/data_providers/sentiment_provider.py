from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class SentimentDataProvider(ABC):
    """Abstract base class for sentiment data providers"""
    
    def __init__(self):
        self.data = None
        
    @abstractmethod
    def get_historical_sentiment(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch historical sentiment data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            start: Start datetime
            end: End datetime (optional, defaults to current time)
            
        Returns:
            DataFrame with sentiment scores and timestamps
        """
        pass
        
    def calculate_sentiment_score(self, sentiment_data: List[Dict]) -> float:
        """
        Calculate a normalized sentiment score from raw sentiment data.
        Override this method in concrete implementations.
        
        Args:
            sentiment_data: List of sentiment data points
            
        Returns:
            Normalized sentiment score between -1 and 1
        """
        pass
        
    def aggregate_sentiment(self, df: pd.DataFrame, window: str = '1h') -> pd.DataFrame:
        """
        Aggregate sentiment data to match the timeframe of price data.
        
        Args:
            df: DataFrame with sentiment data
            window: Aggregation window (e.g., '1h', '4h', '1d')
            
        Returns:
            DataFrame with aggregated sentiment scores
        """
        if df.empty:
            return df
            
        # Resample and calculate mean sentiment
        aggregated = df.resample(window).agg({
            'sentiment_score': 'mean',
            'volume': 'sum'  # If volume of sentiment data is available
        }).ffill()
        
        return aggregated 