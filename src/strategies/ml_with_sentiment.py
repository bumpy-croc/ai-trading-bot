"""
ML Premium Strategy

This is the advanced machine learning strategy that combines price data with sentiment analysis
for enhanced prediction accuracy. It uses the unified prediction engine with sentiment-aware
configuration.

Key Features:
- Multi-modal predictions using unified prediction engine with sentiment support
- Real-time sentiment integration via SentiCrypt API (for logging and analysis)
- Graceful degradation when sentiment data is unavailable
- Advanced feature engineering with sentiment confidence scoring
- Live trading optimization with sentiment freshness tracking
- Automatic fallback when prediction engine is unavailable

Sentiment Features (for analysis and logging):
- Primary sentiment, momentum, volatility
- Extreme positive/negative sentiment detection
- Moving averages (3, 7, 14 days) of sentiment
- Sentiment confidence weighting

Ideal for:
- Maximum prediction accuracy in live trading
- Markets where sentiment significantly impacts price
- Advanced trading systems with reliable data feeds
- Sophisticated risk management scenarios
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timedelta
from src.strategies.base import BaseStrategy
from src.data_providers.senticrypt_provider import SentiCryptProvider

class MlWithSentiment(BaseStrategy):
    def __init__(self, name="MlWithSentiment", prediction_engine=None, 
                 use_sentiment=True, sentiment_csv_path=None, **kwargs):
        super().__init__(name, prediction_engine=prediction_engine)
        
        # Set strategy-specific trading pair
        self.trading_pair = 'BTCUSDT'  # Default, can be overridden
        
        # Configuration
        self.use_sentiment = use_sentiment
        
        # Initialize sentiment provider if needed (for analysis and logging)
        self.sentiment_provider = None
        if self.use_sentiment:
            self._initialize_sentiment_provider(sentiment_csv_path)
        
        # Trading parameters
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        
        # Feature tracking
        self.feature_names = []
        self.sentiment_features = []
    
    def _initialize_sentiment_provider(self, csv_path=None):
        """Initialize sentiment data provider for analysis and logging"""
        try:
            if csv_path is None:
                # Use default path
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                csv_path = os.path.join(project_root, 'data', 'senticrypt_sentiment_data.csv')
            
            # Initialize with live mode for real-time trading
            self.sentiment_provider = SentiCryptProvider(
                csv_path=csv_path, 
                live_mode=True,  # Enable live API calls
                cache_duration_minutes=15  # Refresh every 15 minutes
            )
            print(f"Initialized sentiment provider with {len(self.sentiment_provider.data)} records")
            
        except Exception as e:
            print(f"Warning: Could not initialize sentiment provider: {e}")
            self.sentiment_provider = None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators including sentiment features for analysis"""
        df = df.copy()
        
        # Add sentiment data if enabled (for analysis and logging)
        if self.use_sentiment and self.sentiment_provider is not None:
            df = self._add_sentiment_features(df)
        
        # Add placeholder columns for backward compatibility
        df['ml_prediction'] = np.nan
        df['prediction_confidence'] = np.nan
        df['sentiment_freshness'] = np.nan  # Track sentiment data age
        
        return df
    
    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment features to the dataframe for analysis and logging"""
        
        # Always initialize the sentiment data availability flag
        sentiment_data_available = False
        
        try:
            # Get sentiment data for the date range
            start_date = df.index.min()
            end_date = df.index.max()
            
            # Check if we're dealing with recent data (live trading scenario)
            current_time = pd.Timestamp.now()
            is_recent_data = (end_date - current_time).total_seconds() > -3600  # Within last hour
            
            if is_recent_data and hasattr(self.sentiment_provider, 'get_live_sentiment'):
                print("🔴 LIVE TRADING MODE: Attempting to use real-time sentiment data")
                
                try:
                    # Get historical sentiment for the bulk of the data
                    historical_sentiment = self.sentiment_provider.get_historical_sentiment(
                        symbol=self.trading_pair,
                        start=start_date,
                        end=end_date - pd.Timedelta(hours=2)  # Get historical up to 2 hours ago
                    )
                    
                    # Get live sentiment for the most recent data points
                    live_sentiment = self.sentiment_provider.get_live_sentiment()
                    
                    # Apply historical sentiment to older data
                    if not historical_sentiment.empty:
                        df = df.join(historical_sentiment, how='left')
                        sentiment_data_available = True
                    
                    # Apply live sentiment to recent data points
                    if live_sentiment:
                        recent_mask = df.index >= (end_date - pd.Timedelta(hours=1))
                        for feature, value in live_sentiment.items():
                            if feature not in df.columns:
                                df[feature] = 0.0
                            df.loc[recent_mask, feature] = value
                        sentiment_data_available = True
                        print(f"✅ Applied live sentiment to {recent_mask.sum()} recent data points")
                    
                except Exception as e:
                    print(f"⚠️ Live sentiment retrieval failed: {e}")
                    sentiment_data_available = False
                
            else:
                print("📊 BACKTEST MODE: Using historical sentiment data")
                
                try:
                    # Standard historical sentiment loading
                    sentiment_df = self.sentiment_provider.get_historical_sentiment(
                        symbol=self.trading_pair,
                        start=start_date,
                        end=end_date
                    )
                    
                    if not sentiment_df.empty:
                        # Check if sentiment data is too old/stale
                        latest_sentiment_date = sentiment_df.index.max()
                        data_age_days = (end_date - latest_sentiment_date).days
                        
                        if data_age_days > 30:  # More than 30 days old
                            print(f"⚠️ Sentiment data is {data_age_days} days old - may be stale")
                        
                        # Resample sentiment to match price data frequency
                        if len(df) > len(sentiment_df):
                            # Upsample sentiment data
                            sentiment_resampled = sentiment_df.resample('1h').ffill()
                        else:
                            sentiment_resampled = sentiment_df
                        
                        # Merge with price data
                        df = df.join(sentiment_resampled, how='left')
                        sentiment_data_available = True
                        print(f"✅ Applied historical sentiment for {len(sentiment_df)} data points")
                    else:
                        print("⚠️ No sentiment data found for this period")
                        
                except Exception as e:
                    print(f"⚠️ Historical sentiment retrieval failed: {e}")
                    sentiment_data_available = False
            
            # Ensure all required sentiment features exist with appropriate fallback values
            self._ensure_sentiment_features(df, sentiment_data_available)
                
        except Exception as e:
            print(f"⚠️ Sentiment feature processing failed: {e}")
            # Fallback to neutral sentiment values
            self._ensure_sentiment_features(df, False)
        
        return df
    
    def _ensure_sentiment_features(self, df: pd.DataFrame, data_available: bool):
        """Ensure all required sentiment features exist with appropriate fallback values"""
        
        # Define expected sentiment features
        expected_features = [
            'sentiment_primary', 'sentiment_momentum', 'sentiment_volatility',
            'sentiment_extreme_positive', 'sentiment_extreme_negative',
            'sentiment_ma_3', 'sentiment_ma_7', 'sentiment_ma_14', 'sentiment_confidence'
        ]
        
        if data_available:
            print("🔄 Processing available sentiment data...")
            
            # Forward fill existing sentiment columns
            sentiment_cols = [col for col in df.columns if col.startswith('sentiment_')]
            if sentiment_cols:
                df[sentiment_cols] = df[sentiment_cols].ffill()
            
            # Calculate derived sentiment features if base sentiment exists
            if 'sentiment_primary' in df.columns:
                # Calculate sentiment_confidence based on data availability and freshness
                df['sentiment_confidence'] = 0.8  # Default confidence for historical data
                
                # Check if we have recent/fresh data
                current_time = pd.Timestamp.now()
                recent_mask = df.index >= (current_time - pd.Timedelta(hours=2))
                if recent_mask.any():
                    df.loc[recent_mask, 'sentiment_confidence'] = 0.9  # Higher confidence for recent data
                
            # Ensure all expected features exist
            for feature in expected_features:
                if feature not in df.columns:
                    print(f"⚠️ Missing expected sentiment feature: {feature}, using neutral value")
                    df[feature] = self._get_neutral_sentiment_value(feature)
                else:
                    # Fill any remaining NaN with appropriate neutral values
                    neutral_value = self._get_neutral_sentiment_value(feature)
                    df[feature] = df[feature].fillna(neutral_value)
            
            # Mark sentiment freshness
            current_time = pd.Timestamp.now()
            df['sentiment_freshness'] = 0  # Default to historical
            
            # Check for recent data and mark as fresh
            recent_mask = df.index >= (current_time - pd.Timedelta(hours=2))
            if recent_mask.any():
                df.loc[recent_mask, 'sentiment_freshness'] = 1
                
        else:
            print("🔄 FALLBACK MODE: Using neutral sentiment values for all features")
            
            # No sentiment data available - use neutral/model-friendly values
            for feature in expected_features:
                neutral_value = self._get_neutral_sentiment_value(feature)
                df[feature] = neutral_value
                print(f"   {feature}: {neutral_value}")
                
            df['sentiment_freshness'] = -1  # No real sentiment data
            
        print(f"✅ Ensured {len(expected_features)} sentiment features are available")
    
    def _get_neutral_sentiment_value(self, feature_name: str) -> float:
        """Get appropriate neutral value for a sentiment feature"""
        
        # Map feature names to appropriate neutral values
        if 'primary' in feature_name.lower():
            return 0.5  # Neutral sentiment (assuming 0-1 scale)
        elif 'momentum' in feature_name.lower():
            return 0.0  # No momentum
        elif 'volatility' in feature_name.lower():
            return 0.3  # Low-moderate volatility
        elif 'extreme_positive' in feature_name.lower():
            return 0.0  # No extreme positive sentiment
        elif 'extreme_negative' in feature_name.lower():
            return 0.0  # No extreme negative sentiment
        elif 'ma_' in feature_name.lower():
            return 0.5  # Neutral moving average
        elif 'confidence' in feature_name.lower():
            return 0.7  # Moderate confidence
        else:
            return 0.0  # Default neutral value
    
    def _is_sentiment_data_fresh(self, df: pd.DataFrame) -> bool:
        """Check if sentiment data is reasonably fresh"""
        if 'sentiment_freshness' not in df.columns:
            return False
            
        # Consider data fresh if any recent points have fresh data
        recent_freshness = df['sentiment_freshness'].tail(10).max()
        return recent_freshness > 0

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check if conditions are met for entering a position"""
        if index < 1 or index >= len(df):
            return False
        
        # Get prediction from engine
        prediction = self.get_prediction(df, index)
        
        if prediction.get('error'):
            # Log the prediction error
            self.log_execution(
                signal_type='entry',
                action_taken='no_action',
                price=df['close'].iloc[index],
                reasons=[f'prediction_error: {prediction["error"]}'],
                additional_context={'prediction_available': False}
            )
            return False
        
        current_price = df['close'].iloc[index]
        predicted_price = prediction['price']
        confidence = prediction['confidence']
        
        if predicted_price is None:
            # Log missing prediction
            self.log_execution(
                signal_type='entry',
                action_taken='no_action',
                price=current_price,
                reasons=['missing_ml_prediction'],
                additional_context={'prediction_available': False}
            )
            return False
        
        # Get sentiment freshness (higher weight for fresh sentiment)
        sentiment_freshness = df.get('sentiment_freshness', pd.Series([0] * len(df))).iloc[index]
        freshness_boost = 1.1 if sentiment_freshness > 0 else 1.0  # 10% boost for live sentiment
        
        # Entry condition: prediction suggests price increase with sufficient confidence
        confidence_threshold = 0.3 / freshness_boost  # More lenient with fresh sentiment
        
        predicted_return = (predicted_price - current_price) / current_price
        
        # Entry signal using prediction engine
        entry_signal = (
            prediction['direction'] == 1 and 
            confidence > confidence_threshold
        )
        
        # Log detailed decision process
        sentiment_data = {}
        if 'sentiment_primary' in df.columns:
            sentiment_data = {
                'sentiment_primary': df['sentiment_primary'].iloc[index],
                'sentiment_confidence': df.get('sentiment_confidence', pd.Series([0])).iloc[index],
                'sentiment_freshness': sentiment_freshness
            }
        
        ml_predictions = {
            'raw_prediction': predicted_price,
            'current_price': current_price,
            'predicted_return': predicted_return,
            'confidence': confidence,
            'freshness_boost': freshness_boost,
            'model_name': prediction['model_name']
        }
        
        # Determine entry signal with sentiment boost
        reasons = [
            f'predicted_return_{predicted_return:.4f}',
            f'confidence_{confidence:.4f}_vs_threshold_{confidence_threshold:.4f}',
            f'freshness_boost_{freshness_boost:.2f}',
            'entry_signal_met' if entry_signal else 'entry_signal_not_met'
        ]
        
        # Add sentiment-specific reasons
        if sentiment_freshness > 0:
            reasons.append('fresh_sentiment_available')
            reasons.append(f'sentiment_primary_{sentiment_data.get("sentiment_primary", 0):.3f}')
        else:
            reasons.append('historical_sentiment_only')
        
        self.log_execution(
            signal_type='entry',
            action_taken='entry_signal' if entry_signal else 'no_action',
            price=current_price,
            signal_strength=predicted_return if entry_signal else 0.0,
            confidence_score=confidence,
            sentiment_data=sentiment_data if sentiment_data else None,
            ml_predictions=ml_predictions,
            reasons=reasons,
            additional_context={
                'model_type': 'ml_with_sentiment',
                'use_sentiment': self.use_sentiment,
                'prediction_available': True,
                'inference_time': prediction.get('inference_time', 0.0)
            }
        )
        
        if entry_signal and sentiment_freshness > 0:
            print(f"🚀 LIVE SENTIMENT ENTRY: Fresh sentiment boosted confidence!")
            print(f"   Predicted return: {predicted_return:.3f}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Sentiment primary: {df.get('sentiment_primary', pd.Series([0])).iloc[index]:.3f}")
        
        return entry_signal

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Check if conditions are met for exiting a position"""
        if index < 1 or index >= len(df):
            return False
        
        current_price = df['close'].iloc[index]
        returns = (current_price - entry_price) / entry_price
        
        # Basic stop loss and take profit
        hit_stop_loss = returns <= -self.stop_loss_pct
        hit_take_profit = returns >= self.take_profit_pct
        
        # Additional exit condition: if prediction turns negative with high confidence
        prediction = self.get_prediction(df, index)
        prediction_negative = False
        
        if not prediction.get('error') and prediction['price'] is not None:
            predicted_return = (prediction['price'] - current_price) / current_price
            prediction_negative = (
                prediction['direction'] == -1 and 
                predicted_return < -0.01 and  # Exit if predicting >1% drop
                prediction['confidence'] > 0.6
            )
        
        return hit_stop_loss or hit_take_profit or prediction_negative

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate position size based on prediction confidence and sentiment freshness"""
        if index >= len(df) or balance <= 0:
            return 0.0
        
        base_position_size = balance * 0.10  # Base 10% of balance
        
        # Get prediction confidence
        prediction = self.get_prediction(df, index)
        
        if prediction.get('error') or prediction['price'] is None:
            return 0.0
        
        confidence = prediction['confidence']
        
        # Adjust based on confidence
        confidence_multiplier = 0.5 + confidence  # 0.5x to 1.5x scaling
        
        # Additional boost for fresh sentiment
        sentiment_freshness = df.get('sentiment_freshness', pd.Series([0] * len(df))).iloc[index]
        freshness_multiplier = 1.1 if sentiment_freshness > 0 else 1.0
        
        final_multiplier = confidence_multiplier * freshness_multiplier
        return base_position_size * final_multiplier

    def get_parameters(self) -> dict:
        """Get strategy parameters"""
        params = {
            'name': self.name,
            'use_sentiment': self.use_sentiment,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'trading_pair': self.trading_pair,
            'prediction_engine_available': self.prediction_engine is not None,
            'sentiment_provider_available': self.sentiment_provider is not None
        }
        
        return params

    def calculate_stop_loss(self, df, index, price, side: str = 'long') -> float:
        """Calculate stop loss price"""
        if side == 'long':
            return price * (1 - self.stop_loss_pct)
        else:  # short
            return price * (1 + self.stop_loss_pct) 