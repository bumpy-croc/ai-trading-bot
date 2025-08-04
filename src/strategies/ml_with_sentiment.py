"""
ML Premium Strategy

This is the advanced machine learning strategy that combines price data with sentiment analysis
for enhanced prediction accuracy. It uses sophisticated feature engineering and can gracefully
handle missing or stale sentiment data.

Key Features:
- Multi-modal predictions using prediction engine with sentiment support
- Real-time sentiment integration via SentiCrypt API
- Graceful degradation when sentiment data is unavailable
- Advanced feature engineering with sentiment confidence scoring
- Live trading optimization with sentiment freshness tracking
- Automatic fallback to price-only mode when needed

Sentiment Features:
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
    """ML strategy with sentiment using prediction engine"""
    
    def __init__(self, name="MlWithSentiment", 
                 use_sentiment=True, 
                 sentiment_csv_path=None, 
                 **kwargs):
        super().__init__(name, **kwargs)
        
        # Set strategy-specific trading pair - will be determined by model
        self.trading_pair = kwargs.get('symbol', 'BTCUSDT')
        
        # Strategy configuration
        self.use_sentiment = use_sentiment
        self.sentiment_weight = 0.3  # Weight for combining ML and sentiment signals
        
        # Initialize sentiment provider if needed
        self.sentiment_provider = None
        if self.use_sentiment:
            self._initialize_sentiment_provider(sentiment_csv_path)
        
        # Trading parameters
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        
        # ML prediction thresholds
        self.min_confidence_threshold = 0.6  # Minimum confidence for entry
        
        # Feature tracking
        self.feature_names = []
        self.sentiment_features = []
    
    def _init_strategy_params(self, **kwargs):
        """Initialize strategy-specific parameters"""
        self.stop_loss_pct = kwargs.get('stop_loss_pct', 0.02)
        self.take_profit_pct = kwargs.get('take_profit_pct', 0.04)
        self.min_confidence_threshold = kwargs.get('min_confidence_threshold', 0.6)
        self.sentiment_weight = kwargs.get('sentiment_weight', 0.3)
    
    def _initialize_sentiment_provider(self, sentiment_csv_path=None):
        """Initialize sentiment data provider"""
        try:
            if sentiment_csv_path and os.path.exists(sentiment_csv_path):
                # CSV-based provider
                self.sentiment_provider = SentiCryptProvider(csv_file_path=sentiment_csv_path)
                print(f"✅ Initialized CSV sentiment provider: {sentiment_csv_path}")
            else:
                # API-based provider
                self.sentiment_provider = SentiCryptProvider()
                print("✅ Initialized API sentiment provider")
                
        except Exception as e:
            print(f"⚠️ Failed to initialize sentiment provider: {e}")
            self.sentiment_provider = None
            self.use_sentiment = False

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators and add sentiment features"""
        df = df.copy()
        
        # Process sentiment features if enabled
        if self.use_sentiment and self.sentiment_provider:
            df = self._add_sentiment_features(df)
        else:
            # Ensure sentiment features exist with neutral values
            self._ensure_sentiment_features(df, False)
        
        return df

    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment analysis features to the dataframe"""
        try:
            # Determine if we're in live trading or backtesting mode
            current_time = pd.Timestamp.now()
            latest_data_time = df.index.max()
            is_live_trading = (current_time - latest_data_time) < pd.Timedelta(hours=1)
            
            sentiment_data_available = False
            
            # Get date range for sentiment data
            start_date = df.index.min()
            end_date = df.index.max()
            
            if is_live_trading and hasattr(self.sentiment_provider, 'get_live_sentiment'):
                print("🔴 LIVE TRADING MODE: Attempting to fetch live sentiment...")
                
                try:
                    # Get historical sentiment for the period
                    historical_sentiment = self.sentiment_provider.get_historical_sentiment(
                        symbol=self.trading_pair,
                        start=start_date,
                        end=end_date - pd.Timedelta(hours=1)  # Don't overlap with live data
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
        
        # Define expected sentiment features (from model training)
        expected_features = [
            'sentiment_primary', 'sentiment_momentum', 'sentiment_volatility',
            'sentiment_extreme_positive', 'sentiment_extreme_negative',
            'sentiment_ma_3', 'sentiment_ma_7', 'sentiment_ma_14', 'sentiment_confidence'
        ]
        
        # Use model metadata features if available
        if hasattr(self, 'sentiment_features') and self.sentiment_features:
            expected_features = self.sentiment_features
        
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
                
                # Calculate moving averages
                if 'sentiment_ma_3' not in df.columns:
                    df['sentiment_ma_3'] = df['sentiment_primary'].rolling(window=3, min_periods=1).mean()
                if 'sentiment_ma_7' not in df.columns:
                    df['sentiment_ma_7'] = df['sentiment_primary'].rolling(window=7, min_periods=1).mean()
                if 'sentiment_ma_14' not in df.columns:
                    df['sentiment_ma_14'] = df['sentiment_primary'].rolling(window=14, min_periods=1).mean()
                
                # Calculate extreme sentiment flags
                if 'sentiment_extreme_positive' not in df.columns:
                    df['sentiment_extreme_positive'] = (df['sentiment_primary'] > 0.7).astype(int)
                if 'sentiment_extreme_negative' not in df.columns:
                    df['sentiment_extreme_negative'] = (df['sentiment_primary'] < -0.7).astype(int)
                
                print(f"✅ Processed sentiment features. Confidence range: {df['sentiment_confidence'].min():.2f} - {df['sentiment_confidence'].max():.2f}")
            
        else:
            print("🔄 Using neutral sentiment fallback values...")
            
        # Ensure all expected features exist
        for feature in expected_features:
            if feature not in df.columns:
                if 'confidence' in feature:
                    df[feature] = 0.5  # Medium confidence fallback
                elif 'extreme' in feature:
                    df[feature] = 0  # No extreme sentiment
                else:
                    df[feature] = 0.0  # Neutral sentiment
        
        # Fill any remaining NaN values
        sentiment_cols = [col for col in df.columns if col.startswith('sentiment_')]
        if sentiment_cols:
            df[sentiment_cols] = df[sentiment_cols].fillna(0.0)
            
        print(f"📊 Final sentiment features: {sentiment_cols}")

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check if we should enter a position using prediction engine"""
        if index < 1 or index >= len(df):
            return False
        
        # Get prediction from engine
        prediction = self.get_prediction(df, index)
        
        if prediction.get('error'):
            # Log the error
            self.log_execution(
                signal_type='entry',
                action_taken='no_action',
                price=df['close'].iloc[index],
                reasons=[f'prediction_error: {prediction["error"]}'],
                additional_context={'prediction_available': False}
            )
            return False
        
        # Get sentiment score
        sentiment_score = self._get_sentiment_score(df, index)
        
        # Combine ML prediction with sentiment
        combined_confidence = self._combine_confidence(
            prediction['confidence'], 
            sentiment_score
        )
        
        # Use combined signal for entry decision
        should_enter = (prediction['direction'] == 1 and 
                       combined_confidence > self.min_confidence_threshold)
        
        # Log the decision process
        ml_predictions = {
            'raw_prediction': prediction['price'],
            'current_price': df['close'].iloc[index],
            'confidence': prediction['confidence'],
            'direction': prediction['direction']
        }
        
        sentiment_data = {
            'sentiment_score': sentiment_score,
            'combined_confidence': combined_confidence,
            'sentiment_available': self.use_sentiment and sentiment_score != 0.0
        }
        
        reasons = [
            f'ml_confidence_{prediction["confidence"]:.4f}',
            f'sentiment_score_{sentiment_score:.4f}',
            f'combined_confidence_{combined_confidence:.4f}',
            f'direction_{prediction["direction"]}',
            'entry_signal_met' if should_enter else 'entry_signal_not_met'
        ]
        
        self.log_execution(
            signal_type='entry',
            action_taken='entry_signal' if should_enter else 'no_action',
            price=df['close'].iloc[index],
            signal_strength=combined_confidence if should_enter else 0.0,
            confidence_score=combined_confidence,
            ml_predictions=ml_predictions,
            sentiment_data=sentiment_data,
            reasons=reasons,
            additional_context={
                'model_type': 'ml_with_sentiment',
                'model_name': prediction['model_name'],
                'sentiment_weight': self.sentiment_weight,
                'prediction_available': True
            }
        )
        
        return should_enter

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Check if we should exit a position using prediction engine"""
        if index < 1 or index >= len(df):
            return False
        
        current_price = df['close'].iloc[index]
        returns = (current_price - entry_price) / entry_price
        
        # Basic stop loss and take profit
        hit_stop_loss = returns <= -self.stop_loss_pct
        hit_take_profit = returns >= self.take_profit_pct
        
        # Get prediction from engine for ML-based exit
        prediction = self.get_prediction(df, index)
        ml_exit_signal = False
        
        if not prediction.get('error'):
            # Get sentiment score
            sentiment_score = self._get_sentiment_score(df, index)
            
            # Combine ML prediction with sentiment
            combined_confidence = self._combine_confidence(
                prediction['confidence'], 
                sentiment_score
            )
            
            # Exit if combined signal suggests significant price drop
            ml_exit_signal = (prediction['direction'] == -1 and 
                            combined_confidence > self.min_confidence_threshold)
        
        should_exit = hit_stop_loss or hit_take_profit or ml_exit_signal
        
        # Determine exit reason
        exit_reason = []
        if hit_stop_loss:
            exit_reason.append('stop_loss')
        if hit_take_profit:
            exit_reason.append('take_profit')
        if ml_exit_signal:
            exit_reason.append('ml_sentiment_exit_signal')
        
        # Log exit decision
        if should_exit:
            sentiment_data = {
                'sentiment_score': self._get_sentiment_score(df, index) if not prediction.get('error') else 0.0,
                'combined_confidence': combined_confidence if not prediction.get('error') else 0.0
            }
            
            self.log_execution(
                signal_type='exit',
                action_taken='exit_signal',
                price=current_price,
                signal_strength=prediction.get('confidence', 0.0),
                confidence_score=prediction.get('confidence', 0.0),
                ml_predictions=prediction if not prediction.get('error') else None,
                sentiment_data=sentiment_data,
                reasons=exit_reason,
                additional_context={
                    'returns': returns,
                    'entry_price': entry_price,
                    'exit_type': ','.join(exit_reason)
                }
            )
        
        return should_exit

    def _get_sentiment_score(self, df: pd.DataFrame, index: int) -> float:
        """Get sentiment score for current data point"""
        # Use existing sentiment data if available
        if 'sentiment_primary' in df.columns and index < len(df):
            return df.iloc[index]['sentiment_primary']
        
        # Fallback to neutral sentiment
        return 0.0
    
    def _combine_confidence(self, ml_confidence: float, sentiment_score: float) -> float:
        """Combine ML confidence with sentiment score"""
        # Weighted combination
        sentiment_confidence = abs(sentiment_score)  # Convert to 0-1 scale
        combined = (1 - self.sentiment_weight) * ml_confidence + self.sentiment_weight * sentiment_confidence
        return min(1.0, combined)

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate position size based on combined ML and sentiment confidence"""
        if index >= len(df) or balance <= 0:
            return 0.0
        
        # Get prediction for position sizing
        prediction = self.get_prediction(df, index)
        
        if prediction.get('error'):
            return 0.0
        
        # Get sentiment score and combine with ML confidence
        sentiment_score = self._get_sentiment_score(df, index)
        combined_confidence = self._combine_confidence(
            prediction['confidence'], 
            sentiment_score
        )
        
        if combined_confidence < self.min_confidence_threshold:
            return 0.0
        
        # Base position size with combined confidence scaling
        base_position_size = 0.1  # 10% of balance
        min_position_size = 0.05  # 5% minimum
        max_position_size = 0.2   # 20% maximum
        
        # Scale position size based on combined confidence
        confidence_multiplier = combined_confidence / self.min_confidence_threshold
        dynamic_size = base_position_size * confidence_multiplier
        
        # Apply bounds
        final_size = max(min_position_size, min(max_position_size, dynamic_size))
        
        return final_size * balance

    def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = 'long') -> float:
        """Calculate stop loss price"""
        # Handle both string and enum inputs for backward compatibility
        side_str = side.value if hasattr(side, 'value') else str(side)
        
        if side_str == 'long':
            return price * (1 - self.stop_loss_pct)
        else:  # short
            return price * (1 + self.stop_loss_pct)

    def get_parameters(self) -> dict:
        """Return strategy parameters for logging"""
        return {
            'name': self.name,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'min_confidence_threshold': self.min_confidence_threshold,
            'sentiment_weight': self.sentiment_weight,
            'use_sentiment': self.use_sentiment,
            'symbol': self.symbol,
            'timeframe': self.timeframe
        } 