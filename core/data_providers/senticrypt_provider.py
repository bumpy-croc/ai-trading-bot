import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging
from .sentiment_provider import SentimentDataProvider

logger = logging.getLogger(__name__)

class SentiCryptProvider(SentimentDataProvider):
    """
    SentiCrypt sentiment data provider using their API or local CSV data.
    Provides normalized sentiment scores correlated with Bitcoin price data.
    """
    
    def __init__(self, csv_path: Optional[str] = None, api_url: str = "https://api.senticrypt.com/v2/all.json"):
        super().__init__()
        self.csv_path = csv_path
        self.api_url = api_url
        self.data = None
        self._load_data()
        
    def _load_data(self):
        """Load sentiment data from CSV file or API"""
        try:
            if self.csv_path and os.path.exists(self.csv_path):
                logger.info(f"Loading sentiment data from CSV: {self.csv_path}")
                self.data = pd.read_csv(self.csv_path)
            else:
                logger.info(f"Fetching sentiment data from API: {self.api_url}")
                response = requests.get(self.api_url, timeout=30)
                response.raise_for_status()
                self.data = pd.DataFrame(response.json())
                
            # Process the data
            self._process_data()
            
        except Exception as e:
            logger.error(f"Failed to load sentiment data: {e}")
            self.data = pd.DataFrame()
    
    def _process_data(self):
        """Process and normalize the sentiment data"""
        if self.data.empty:
            return
            
        # Convert date to datetime
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index('date', inplace=True)
        self.data.sort_index(inplace=True)
        
        # Create normalized sentiment features
        self._create_sentiment_features()
        
        logger.info(f"Processed {len(self.data)} sentiment records from {self.data.index.min()} to {self.data.index.max()}")
    
    def _create_sentiment_features(self):
        """Create normalized sentiment features from raw scores"""
        # Normalize individual scores to [-1, 1] range
        for col in ['score1', 'score2', 'score3']:
            if col in self.data.columns:
                # Use tanh to normalize to [-1, 1] range, preserving extreme values
                self.data[f'{col}_norm'] = np.tanh(self.data[col])
        
        # Create composite sentiment score
        if all(col in self.data.columns for col in ['score1', 'score2', 'score3']):
            # Weighted average of the three scores (you can adjust weights based on importance)
            weights = [0.4, 0.3, 0.3]  # score1 gets higher weight
            self.data['sentiment_composite'] = (
                weights[0] * self.data['score1_norm'] + 
                weights[1] * self.data['score2_norm'] + 
                weights[2] * self.data['score3_norm']
            )
        
        # Use the mean score as primary sentiment if available
        if 'mean' in self.data.columns:
            self.data['sentiment_primary'] = np.tanh(self.data['mean'])
        else:
            self.data['sentiment_primary'] = self.data.get('sentiment_composite', 0)
        
        # Create sentiment momentum (rate of change)
        self.data['sentiment_momentum'] = self.data['sentiment_primary'].pct_change().fillna(0)
        
        # Create sentiment volatility (rolling standard deviation)
        self.data['sentiment_volatility'] = self.data['sentiment_primary'].rolling(window=7).std().fillna(0)
        
        # Create sentiment extremes (binary flags for very positive/negative sentiment)
        sentiment_std = self.data['sentiment_primary'].std()
        sentiment_mean = self.data['sentiment_primary'].mean()
        
        self.data['sentiment_extreme_positive'] = (
            self.data['sentiment_primary'] > sentiment_mean + 2 * sentiment_std
        ).astype(int)
        
        self.data['sentiment_extreme_negative'] = (
            self.data['sentiment_primary'] < sentiment_mean - 2 * sentiment_std
        ).astype(int)
        
        # Create sentiment moving averages
        for window in [3, 7, 14]:
            self.data[f'sentiment_ma_{window}'] = self.data['sentiment_primary'].rolling(window=window).mean()
    
    def get_historical_sentiment(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical sentiment data for the specified period.
        
        Args:
            symbol: Trading pair symbol (currently only supports BTC-related pairs)
            start: Start datetime
            end: End datetime (optional, defaults to current time)
            
        Returns:
            DataFrame with sentiment features aligned to the date range
        """
        if self.data.empty:
            logger.warning("No sentiment data available")
            return pd.DataFrame()
        
        # SentiCrypt data is BTC-focused, so we'll use it for all BTC pairs
        if 'BTC' not in symbol.upper():
            logger.warning(f"SentiCrypt data is BTC-focused, using for {symbol} with caution")
        
        # Filter data by date range
        if end is None:
            end = datetime.now()
            
        mask = (self.data.index >= start) & (self.data.index <= end)
        filtered_data = self.data.loc[mask].copy()
        
        if filtered_data.empty:
            logger.warning(f"No sentiment data found for period {start} to {end}")
            return pd.DataFrame()
        
        # Select relevant columns for ML model
        sentiment_columns = [
            'sentiment_primary',
            'sentiment_momentum', 
            'sentiment_volatility',
            'sentiment_extreme_positive',
            'sentiment_extreme_negative',
            'sentiment_ma_3',
            'sentiment_ma_7',
            'sentiment_ma_14'
        ]
        
        # Add raw scores if needed
        for col in ['score1_norm', 'score2_norm', 'score3_norm']:
            if col in filtered_data.columns:
                sentiment_columns.append(col)
        
        result = filtered_data[sentiment_columns].copy()
        
        # Forward fill missing values
        result = result.fillna(method='ffill').fillna(0)
        
        logger.info(f"Retrieved {len(result)} sentiment records for {symbol} from {start} to {end}")
        return result
    
    def calculate_sentiment_score(self, sentiment_data: List[Dict]) -> float:
        """
        Calculate a single normalized sentiment score from raw sentiment data.
        
        Args:
            sentiment_data: List of sentiment data points
            
        Returns:
            Normalized sentiment score between -1 and 1
        """
        if not sentiment_data:
            return 0.0
        
        # Extract mean scores and calculate average
        scores = []
        for item in sentiment_data:
            if 'mean' in item:
                scores.append(item['mean'])
            elif all(key in item for key in ['score1', 'score2', 'score3']):
                # Calculate weighted average if individual scores are available
                composite = 0.4 * item['score1'] + 0.3 * item['score2'] + 0.3 * item['score3']
                scores.append(composite)
        
        if not scores:
            return 0.0
        
        # Return normalized average
        avg_score = np.mean(scores)
        return np.tanh(avg_score)  # Normalize to [-1, 1]
    
    def get_sentiment_for_date(self, date: datetime) -> Dict[str, float]:
        """
        Get sentiment features for a specific date.
        
        Args:
            date: The date to get sentiment for
            
        Returns:
            Dictionary of sentiment features
        """
        if self.data.empty:
            return {}
        
        # Find the closest date
        closest_date = self.data.index[self.data.index.get_indexer([date], method='nearest')[0]]
        
        if abs((closest_date - date).days) > 7:  # If more than 7 days away, return zeros
            return {col: 0.0 for col in self.data.columns if col.startswith('sentiment_')}
        
        row = self.data.loc[closest_date]
        return {col: row[col] for col in self.data.columns if col.startswith('sentiment_')}
    
    def resample_to_timeframe(self, timeframe: str = '1h') -> pd.DataFrame:
        """
        Resample sentiment data to match price data timeframe.
        
        Args:
            timeframe: Target timeframe (e.g., '1h', '4h', '1d')
            
        Returns:
            Resampled DataFrame
        """
        if self.data.empty:
            return pd.DataFrame()
        
        # Define aggregation methods for different columns
        agg_methods = {}
        
        # Sentiment scores: use mean
        for col in self.data.columns:
            if col.startswith('sentiment_'):
                if 'extreme' in col:
                    agg_methods[col] = 'max'  # Use max for extreme flags
                else:
                    agg_methods[col] = 'mean'
            elif col in ['score1_norm', 'score2_norm', 'score3_norm']:
                agg_methods[col] = 'mean'
            elif col in ['count', 'volume']:
                agg_methods[col] = 'sum'
            elif col in ['price']:
                agg_methods[col] = 'last'  # Use last price in period
            else:
                agg_methods[col] = 'mean'
        
        # Resample the data
        resampled = self.data.resample(timeframe).agg(agg_methods)
        
        # Forward fill missing values
        resampled = resampled.fillna(method='ffill')
        
        return resampled 