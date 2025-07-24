import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import logging
from .sentiment_provider import SentimentDataProvider
from src.utils.symbol_factory import SymbolFactory

logger = logging.getLogger(__name__)

class AugmentoProvider(SentimentDataProvider):
    """
    Augmento API provider for cryptocurrency sentiment analysis
    
    Provides sentiment data from Twitter, Reddit, and Bitcointalk
    with 93 different topic and sentiment categories.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Augmento API provider
        
        Args:
            api_key: Optional API key for authenticated requests (gets real-time data)
                    Without key, data is limited to 30+ days old
        """
        self.base_url = "https://api.augmento.ai/v0.1"
        self.api_key = api_key
        self.headers = {}
        
        if api_key:
            self.headers["Api-Key"] = api_key
            
        # Cache for topics mapping
        self._topics_cache = None
        self._coins_cache = None
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests (300/5min = conservative)
        
    def _make_request(self, endpoint: str, params: Dict = None) -> requests.Response:
        """Make rate-limited API request"""
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url, params=params, headers=self.headers, 
                              timeout=30)
        
        self.last_request_time = time.time()
        
        if response.status_code == 200:
            return response
        elif response.status_code == 429:
            logger.warning("Rate limit exceeded, waiting 60 seconds...")
            time.sleep(60)
            return self._make_request(endpoint, params)  # Retry
        else:
            response.raise_for_status()
            
    def get_topics(self) -> Dict[str, str]:
        """Get mapping of topic IDs to descriptions"""
        if self._topics_cache is None:
            response = self._make_request("topics")
            self._topics_cache = response.json()
        return self._topics_cache
    
    def get_available_coins(self) -> List[str]:
        """Get list of available coins"""
        if self._coins_cache is None:
            response = self._make_request("coins")
            self._coins_cache = response.json()
        return self._coins_cache
    
    def _map_symbol_to_coin(self, symbol: str) -> str:
        # Normalize symbol to Coinbase style for mapping
        normalized = SymbolFactory.to_exchange_symbol(symbol, 'coinbase')
        symbol_map = {
            'BTC-USD': 'bitcoin',
            'ETH-USD': 'ethereum',
            'SOL-USD': 'solana',
            'ADA-USD': 'cardano',
            'DOT-USD': 'polkadot',
            'LINK-USD': 'chainlink',
            'MATIC-USD': 'polygon',
            'AVAX-USD': 'avalanche'
        }
        coin = symbol_map.get(normalized)
        if not coin:
            # Try to extract base currency and convert to lowercase
            base = normalized.split('-')[0].lower()
            available_coins = self.get_available_coins()
            if base in available_coins:
                coin = base
            else:
                logger.warning(f"Could not map symbol {symbol} to Augmento coin")
                return 'bitcoin'  # Default fallback
        return coin
    
    def get_historical_sentiment(
        self, 
        symbol: str,
        start: datetime,
        end: datetime,
        source: str = "twitter",
        bin_size: str = "24H"
    ) -> pd.DataFrame:
        """
        Get historical sentiment data
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            start: Start datetime
            end: End datetime  
            source: Data source ('twitter', 'reddit', 'bitcointalk')
            bin_size: Time bin size ('1H' or '24H')
        """
        coin = self._map_symbol_to_coin(symbol)
        
        # Format dates for API
        start_str = start.strftime('%Y-%m-%dT00:00:00Z')
        end_str = end.strftime('%Y-%m-%dT23:59:59Z')
        
        params = {
            'source': source,
            'coin': coin,
            'bin_size': bin_size,
            'start_datetime': start_str,
            'end_datetime': end_str,
            'start_ptr': 0,
            'count_ptr': 1000  # Max allowed
        }
        
        try:
            response = self._make_request("events/aggregated", params)
            data = response.json()
            
            if not data:
                logger.warning(f"No sentiment data returned for {symbol} ({coin})")
                return pd.DataFrame()
            
            return self._process_sentiment_data(data, symbol, source)
            
        except Exception as e:
            logger.error(f"Error fetching sentiment data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _process_sentiment_data(self, raw_data: List[Dict], symbol: str, source: str) -> pd.DataFrame:
        """Process raw API response into structured DataFrame"""
        if not raw_data:
            return pd.DataFrame()
        
        topics = self.get_topics()
        
        # Convert to DataFrame
        records = []
        for entry in raw_data:
            timestamp = pd.to_datetime(entry['datetime'])
            counts = entry['counts']
            
            # Create record with individual topic counts
            record = {'timestamp': timestamp}
            
            # Add individual topic sentiment scores
            for topic_id, count in enumerate(counts):
                if str(topic_id) in topics:
                    topic_name = topics[str(topic_id)].lower().replace('/', '_').replace(' ', '_')
                    record[f'sentiment_{source}_{topic_name}'] = count
            
            # Calculate aggregate sentiment scores
            sentiment_indices = self._get_sentiment_topic_indices(topics)
            
            positive_score = sum(counts[i] for i in sentiment_indices['positive'])
            negative_score = sum(counts[i] for i in sentiment_indices['negative'])
            neutral_score = sum(counts[i] for i in sentiment_indices['neutral'])
            
            total_mentions = sum(counts)
            
            if total_mentions > 0:
                # Normalize to [-1, 1] scale
                record[f'sentiment_{source}_positive_ratio'] = positive_score / total_mentions
                record[f'sentiment_{source}_negative_ratio'] = negative_score / total_mentions
                record[f'sentiment_{source}_neutral_ratio'] = neutral_score / total_mentions
                
                # Overall sentiment score
                record[f'sentiment_{source}_overall'] = (positive_score - negative_score) / total_mentions
                record[f'sentiment_{source}_volume'] = total_mentions
            else:
                record[f'sentiment_{source}_positive_ratio'] = 0
                record[f'sentiment_{source}_negative_ratio'] = 0  
                record[f'sentiment_{source}_neutral_ratio'] = 0
                record[f'sentiment_{source}_overall'] = 0
                record[f'sentiment_{source}_volume'] = 0
            
            records.append(record)
        
        df = pd.DataFrame(records)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def _get_sentiment_topic_indices(self, topics: Dict[str, str]) -> Dict[str, List[int]]:
        """Categorize topic indices by sentiment type"""
        positive_keywords = ['optimistic', 'positive', 'bullish', 'good', 'great', 'moon']
        negative_keywords = ['pessimistic', 'doubtful', 'negative', 'bearish', 'bad', 'crash', 'hack']
        
        sentiment_indices = {
            'positive': [],
            'negative': [], 
            'neutral': []
        }
        
        for topic_id, topic_name in topics.items():
            topic_lower = topic_name.lower()
            topic_idx = int(topic_id)
            
            if any(keyword in topic_lower for keyword in positive_keywords):
                sentiment_indices['positive'].append(topic_idx)
            elif any(keyword in topic_lower for keyword in negative_keywords):
                sentiment_indices['negative'].append(topic_idx)
            else:
                sentiment_indices['neutral'].append(topic_idx)
        
        return sentiment_indices
    
    def get_multi_source_sentiment(
        self,
        symbol: str,
        start: datetime, 
        end: datetime,
        sources: List[str] = None
    ) -> pd.DataFrame:
        """
        Get sentiment data from multiple sources and combine
        
        Args:
            symbol: Trading symbol
            start: Start datetime
            end: End datetime
            sources: List of sources ['twitter', 'reddit', 'bitcointalk']
        """
        if sources is None:
            sources = ['twitter', 'reddit']  # Most reliable sources
        
        combined_df = pd.DataFrame()
        
        for source in sources:
            logger.info(f"Fetching {source} sentiment for {symbol}...")
            source_df = self.get_historical_sentiment(symbol, start, end, source)
            
            if not source_df.empty:
                if combined_df.empty:
                    combined_df = source_df
                else:
                    combined_df = combined_df.join(source_df, how='outer')
        
        # Fill missing values and calculate combined metrics
        combined_df = combined_df.fillna(0)
        
        if not combined_df.empty:
            # Calculate cross-source sentiment averages
            overall_cols = [col for col in combined_df.columns if col.endswith('_overall')]
            if overall_cols:
                combined_df['sentiment_combined_overall'] = combined_df[overall_cols].mean(axis=1)
            
            volume_cols = [col for col in combined_df.columns if col.endswith('_volume')]
            if volume_cols:
                combined_df['sentiment_combined_volume'] = combined_df[volume_cols].sum(axis=1)
                
        return combined_df
    
    def get_current_datetime(self) -> datetime:
        """Get current server datetime from API"""
        try:
            response = self._make_request("datetime")
            data = response.json()
            return pd.to_datetime(data['datetime'])
        except Exception as e:
            logger.error(f"Error getting server datetime: {e}")
            return datetime.utcnow() 