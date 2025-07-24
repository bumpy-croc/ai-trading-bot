import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging
from .sentiment_provider import SentimentDataProvider
from config.paths import resolve_data_path
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

class CryptoCompareSentimentProvider(SentimentDataProvider):
    """Implementation of SentimentDataProvider using CryptoCompare's news API"""
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv('CRYPTO_COMPARE_API_KEY')
        if not self.api_key:
            raise ValueError("CRYPTO_COMPARE_API_KEY not found in environment variables")
            
        self.base_url = "https://min-api.cryptocompare.com/data/v2"
        self.stop_words = set(stopwords.words('english'))
        
    def get_historical_sentiment(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch historical news data from CryptoCompare"""
        try:
            # Convert symbol to base currency (e.g., BTCUSDT -> BTC)
            base_currency = symbol[:3] if len(symbol) > 3 else symbol
            print(f"Fetching sentiment data for {base_currency} from {start} to {end or datetime.now()}")
            
            # Define cache file path for each year
            cache_file = resolve_data_path(f"sentiment_cache_{base_currency}_{start.year}.csv")
            
            # Check if cache file exists
            if os.path.exists(cache_file):
                print(f"Loading sentiment data from cache: {cache_file}")
                df = pd.read_csv(cache_file, index_col='timestamp', parse_dates=True)
                # Filter by date range
                mask = (df.index >= start) & (df.index <= (end or datetime.now()))
                df = df[mask]
                return df
            
            # Initialize list to store all news data
            all_news_data = []
            current_timestamp = start
            
            # Fetch news data in chunks to handle API limits
            while current_timestamp < (end or datetime.now()):
                # Calculate end timestamp for this chunk (max 7 days per request)
                chunk_end = min(current_timestamp + timedelta(days=7), end or datetime.now())
                print(f"Fetching chunk from {current_timestamp} to {chunk_end}")
                
                # Fetch news data
                url = f"{self.base_url}/news/?lang=EN&api_key={self.api_key}"
                response = requests.get(url)
                response.raise_for_status()
                
                # Process news data
                news_data = response.json()['Data']
                print(f"Fetched {len(news_data)} news items")
                
                for news in news_data:
                    # Calculate sentiment score for each news item
                    sentiment_score = self.calculate_sentiment_score([{
                        'title': news['title'],
                        'body': news['body']
                    }])
                    
                    # Calculate engagement score
                    engagement_score = self.calculate_engagement_score(news)
                    
                    all_news_data.append({
                        'timestamp': datetime.fromtimestamp(news['published_on']),
                        'sentiment_score': sentiment_score,
                        'engagement_score': engagement_score,
                        'title': news['title'],
                        'source': news['source'],
                        'categories': news.get('categories', ''),
                        'url': news.get('url', '')
                    })
                
                # Move to next chunk
                current_timestamp = chunk_end
                
                # Add delay to respect API rate limits
                time.sleep(1)
            
            # Convert to DataFrame
            df = pd.DataFrame(all_news_data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
                
                # Filter by date range
                mask = (df.index >= start) & (df.index <= (end or datetime.now()))
                df = df[mask]
                print(f"Final DataFrame shape after filtering: {df.shape}")
                
                # Calculate weighted sentiment score
                df['weighted_sentiment'] = df['sentiment_score'] * df['engagement_score']
                
                # Save to cache file
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                df.to_csv(cache_file)
                print(f"Saved sentiment data to cache: {cache_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching sentiment data: {str(e)}")
            return pd.DataFrame()
            
    def calculate_sentiment_score(self, sentiment_data: List[Dict]) -> float:
        """Calculate sentiment score using TextBlob and custom weighting"""
        if not sentiment_data:
            return 0.0
            
        total_score = 0.0
        total_weight = 0.0
        
        for item in sentiment_data:
            # Combine title and body for analysis
            text = f"{item['title']} {item.get('body', '')}"
            
            # Tokenize and remove stopwords
            tokens = word_tokenize(text.lower())
            tokens = [t for t in tokens if t not in self.stop_words]
            
            # Calculate sentiment using TextBlob
            analysis = TextBlob(text)
            
            # Weight title more heavily than body
            title_weight = 0.7
            body_weight = 0.3
            
            # Calculate weighted score
            score = (
                analysis.sentiment.polarity * title_weight +
                analysis.sentiment.polarity * body_weight
            )
            
            # Add to total with weight
            total_score += score
            total_weight += 1.0
            
        return total_score / total_weight if total_weight > 0 else 0.0
        
    def calculate_engagement_score(self, news: Dict) -> float:
        """Calculate engagement score based on news metadata"""
        score = 1.0  # Base score
        
        # Weight by source reliability
        reliable_sources = {
            'coindesk': 1.2,
            'cointelegraph': 1.2,
            'bloomberg': 1.3,
            'reuters': 1.3,
            'wsj': 1.3
        }
        
        source = news.get('source', '').lower()
        if source in reliable_sources:
            score *= reliable_sources[source]
            
        # Weight by categories
        important_categories = {
            'Trading': 1.2,
            'Market': 1.2,
            'Technology': 1.1,
            'Mining': 1.1
        }
        
        categories = news.get('categories', '').split('|')
        for category in categories:
            if category in important_categories:
                score *= important_categories[category]
                
        return min(score, 2.0)  # Cap at 2.0
        
    def aggregate_sentiment(self, df: pd.DataFrame, window: str = '1h') -> pd.DataFrame:
        """Aggregate sentiment data to match the timeframe of price data"""
        if df.empty:
            return df
            
        # Resample and calculate weighted metrics
        aggregated = df.resample(window).agg({
            'sentiment_score': 'mean',
            'engagement_score': 'mean',
            'weighted_sentiment': 'mean'
        }).ffill()
        
        # Calculate sentiment momentum
        aggregated['sentiment_momentum'] = aggregated['weighted_sentiment'].diff()
        
        # Calculate sentiment volatility
        aggregated['sentiment_volatility'] = aggregated['weighted_sentiment'].rolling(window=24).std()
        
        return aggregated 