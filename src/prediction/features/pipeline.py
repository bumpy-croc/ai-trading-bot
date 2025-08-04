"""
Feature pipeline for extracting and processing market data features.

This module provides a FeaturePipeline class that extracts technical indicators
and other features from OHLCV data for use in ML predictions.
"""
import time
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import hashlib

from indicators.technical import (
    calculate_moving_averages, calculate_rsi, calculate_atr,
    calculate_bollinger_bands, calculate_macd, calculate_ema
)


class FeaturePipeline:
    """
    Feature pipeline for extracting and processing market data features.
    
    This class handles feature extraction from OHLCV data, including technical
    indicators and optional sentiment data. It includes caching capabilities
    to improve performance.
    """
    
    def __init__(self, 
                 enable_sentiment: bool = False,
                 enable_market_microstructure: bool = False,
                 cache_ttl: int = 300):
        """
        Initialize feature pipeline.
        
        Args:
            enable_sentiment: Whether to include sentiment features (MVP: disabled)
            enable_market_microstructure: Whether to include market microstructure features (MVP: disabled)
            cache_ttl: Cache time-to-live in seconds
        """
        self.enable_sentiment = enable_sentiment
        self.enable_market_microstructure = enable_market_microstructure
        self.cache_ttl = cache_ttl
        
        # Feature cache: {hash: (features, timestamp)}
        self._cache: Dict[str, tuple] = {}
        
        # Feature extraction statistics
        self._extraction_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._last_extraction_time = 0.0
    
    def transform(self, data: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
        """
        Transform OHLCV data into features for ML prediction.
        
        Args:
            data: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            use_cache: Whether to use feature caching
            
        Returns:
            DataFrame with extracted features
            
        Raises:
            ValueError: If required columns are missing or data is insufficient
        """
        start_time = time.time()
        
        try:
            # Validate input data
            self._validate_input_data(data)
            
            # Check cache if enabled
            if use_cache:
                cache_key = self._generate_cache_key(data)
                cached_features = self._get_cached_features(cache_key)
                if cached_features is not None:
                    self._cache_hits += 1
                    return cached_features
                self._cache_misses += 1
            
            # Extract features
            features = self._extract_features(data)
            
            # Cache the results if enabled
            if use_cache:
                self._cache_features(cache_key, features)
            
            # Update statistics
            self._extraction_count += 1
            self._last_extraction_time = time.time() - start_time
            
            return features
            
        except Exception as e:
            # Update statistics for failed extractions
            self._extraction_count += 1
            self._last_extraction_time = time.time() - start_time
            raise e
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data has required columns and sufficient length."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(data) < 50:  # Minimum for technical indicators
            raise ValueError(f"Insufficient data: {len(data)} rows, minimum 50 required")
        
        # Check for invalid values
        if data[required_columns].isnull().any().any():
            raise ValueError("Data contains null values")
        
        if (data[required_columns] < 0).any().any():
            raise ValueError("Data contains negative values")
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract all features from the input data."""
        # Start with the original data
        features = data.copy()
        
        # Extract technical indicators
        features = self._extract_technical_features(features)
        
        # Extract sentiment features (if enabled)
        if self.enable_sentiment:
            features = self._extract_sentiment_features(features)
        
        # Extract market microstructure features (if enabled)
        if self.enable_market_microstructure:
            features = self._extract_microstructure_features(features)
        
        # Normalize features for ML consumption
        features = self._normalize_features(features)
        
        return features
    
    def _extract_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract technical indicator features."""
        # Start with a copy of the data
        data_with_indicators = data.copy()
        
        # Calculate technical indicators using individual functions
        data_with_indicators = calculate_moving_averages(data_with_indicators, [5, 10, 20])
        data_with_indicators['rsi'] = calculate_rsi(data_with_indicators, period=14)
        data_with_indicators = calculate_atr(data_with_indicators, period=14)
        data_with_indicators = calculate_bollinger_bands(data_with_indicators, period=20, std_dev=2.0)
        data_with_indicators = calculate_macd(data_with_indicators)
        
        # Calculate EMAs
        data_with_indicators['ema_12'] = calculate_ema(data_with_indicators['close'], period=12)
        data_with_indicators['ema_26'] = calculate_ema(data_with_indicators['close'], period=26)
        
        # Rename moving averages to match expected format
        if 'ma_5' in data_with_indicators.columns:
            data_with_indicators['sma_5'] = data_with_indicators['ma_5']
        if 'ma_10' in data_with_indicators.columns:
            data_with_indicators['sma_10'] = data_with_indicators['ma_10']
        if 'ma_20' in data_with_indicators.columns:
            data_with_indicators['sma_20'] = data_with_indicators['ma_20']
        
        # Add MACD signal
        if 'macd_signal' not in data_with_indicators.columns and 'macd' in data_with_indicators.columns:
            data_with_indicators['macd_signal'] = data_with_indicators['macd'].ewm(span=9).mean()
        
        # Add Stochastic oscillator (simple implementation)
        window = 14
        data_with_indicators['stoch_k'] = ((data_with_indicators['close'] - data_with_indicators['low'].rolling(window).min()) / 
                                          (data_with_indicators['high'].rolling(window).max() - data_with_indicators['low'].rolling(window).min())) * 100
        data_with_indicators['stoch_d'] = data_with_indicators['stoch_k'].rolling(3).mean()
        
        # Add price-based features
        data_with_indicators['price_change'] = data_with_indicators['close'].pct_change()
        data_with_indicators['price_volatility'] = data_with_indicators['price_change'].rolling(window=20).std()
        data_with_indicators['volume_change'] = data_with_indicators['volume'].pct_change()
        
        # Add lag features (previous values)
        for lag in [1, 2, 3]:
            data_with_indicators[f'close_lag_{lag}'] = data_with_indicators['close'].shift(lag)
            data_with_indicators[f'volume_lag_{lag}'] = data_with_indicators['volume'].shift(lag)
        
        # Add rolling statistics
        for window in [5, 10, 20]:
            data_with_indicators[f'close_ma_{window}'] = data_with_indicators['close'].rolling(window=window).mean()
            data_with_indicators[f'volume_ma_{window}'] = data_with_indicators['volume'].rolling(window=window).mean()
        
        return data_with_indicators
    
    def _extract_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract sentiment features (placeholder for MVP)."""
        # For MVP, sentiment is disabled, but we add placeholder columns
        # This will be implemented in later phases
        data['sentiment_primary'] = 0.0
        data['sentiment_confidence'] = 0.5
        data['sentiment_momentum'] = 0.0
        
        return data
    
    def _extract_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract market microstructure features (placeholder for MVP)."""
        # For MVP, market microstructure is disabled, but we add placeholder columns
        # This will be implemented in later phases
        data['bid_ask_spread'] = 0.0
        data['order_flow_imbalance'] = 0.0
        data['trade_intensity'] = 0.0
        
        return data
    
    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize features for ML consumption."""
        # Create a copy to avoid modifying the original
        normalized_data = data.copy()
        
        # Price normalization (min-max scaling for basic features)
        price_features = ['open', 'high', 'low', 'close']
        for feature in price_features:
            if feature in normalized_data.columns:
                feature_values = normalized_data[feature].dropna()
                if len(feature_values) > 0:
                    min_val = feature_values.min()
                    max_val = feature_values.max()
                    if max_val != min_val:
                        normalized_data[f'{feature}_normalized'] = (
                            (normalized_data[feature] - min_val) / (max_val - min_val)
                        )
                    else:
                        normalized_data[f'{feature}_normalized'] = 0.5
        
        # Volume normalization (log transform)
        if 'volume' in normalized_data.columns:
            normalized_data['volume_normalized'] = np.log1p(normalized_data['volume'])
            # Min-max scale the log volume
            log_vol = normalized_data['volume_normalized'].dropna()
            if len(log_vol) > 0:
                min_vol = log_vol.min()
                max_vol = log_vol.max()
                if max_vol != min_vol:
                    normalized_data['volume_normalized'] = (
                        (normalized_data['volume_normalized'] - min_vol) / (max_vol - min_vol)
                    )
                else:
                    normalized_data['volume_normalized'] = 0.5
        
        # Fill NaN values with 0 (or forward fill for indicators)
        normalized_data = normalized_data.ffill().fillna(0)
        
        return normalized_data
    
    def _generate_cache_key(self, data: pd.DataFrame) -> str:
        """Generate a cache key for the input data."""
        # Create a hash based on the data content and configuration
        data_hash = hashlib.md5()
        
        # Include data content
        data_hash.update(str(data.values.tobytes()).encode())
        
        # Include configuration
        config_str = f"{self.enable_sentiment}_{self.enable_market_microstructure}"
        data_hash.update(config_str.encode())
        
        return data_hash.hexdigest()
    
    def _get_cached_features(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get features from cache if available and not expired."""
        if cache_key not in self._cache:
            return None
        
        features, timestamp = self._cache[cache_key]
        
        # Check if cache is expired
        if time.time() - timestamp > self.cache_ttl:
            del self._cache[cache_key]
            return None
        
        return features.copy()
    
    def _cache_features(self, cache_key: str, features: pd.DataFrame) -> None:
        """Cache the extracted features."""
        self._cache[cache_key] = (features.copy(), time.time())
        
        # Clean up expired cache entries (simple cleanup)
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp > self.cache_ttl
        ]
        for key in expired_keys:
            del self._cache[key]
    
    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self._cache.clear()
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_size': len(self._cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'extraction_count': self._extraction_count,
            'last_extraction_time': self._last_extraction_time
        }
    
    def get_feature_names(self, sample_data: Optional[pd.DataFrame] = None) -> List[str]:
        """Get the names of features that would be extracted."""
        if sample_data is None:
            # Return basic feature names
            features = ['open_normalized', 'high_normalized', 'low_normalized', 
                       'close_normalized', 'volume_normalized']
            
            # Add technical indicator names (from TechnicalIndicators.calculate_all)
            technical_features = [
                'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26', 'macd', 'macd_signal',
                'rsi', 'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'stoch_k', 'stoch_d'
            ]
            features.extend(technical_features)
            
            # Add derived features
            derived_features = [
                'price_change', 'price_volatility', 'volume_change',
                'close_lag_1', 'close_lag_2', 'close_lag_3',
                'volume_lag_1', 'volume_lag_2', 'volume_lag_3',
                'close_ma_5', 'close_ma_10', 'close_ma_20',
                'volume_ma_5', 'volume_ma_10', 'volume_ma_20'
            ]
            features.extend(derived_features)
            
            if self.enable_sentiment:
                features.extend(['sentiment_primary', 'sentiment_confidence', 'sentiment_momentum'])
            
            if self.enable_market_microstructure:
                features.extend(['bid_ask_spread', 'order_flow_imbalance', 'trade_intensity'])
            
            return features
        else:
            # Extract features from sample data and return column names
            sample_features = self.transform(sample_data, use_cache=False)
            return list(sample_features.columns)