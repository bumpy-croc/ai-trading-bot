"""
ML Premium Strategy

This is the advanced machine learning strategy that combines price data with sentiment analysis
for enhanced prediction accuracy. It uses sophisticated feature engineering and can gracefully
handle missing or stale sentiment data.

Key Features:
- Multi-modal predictions using price + sentiment data (14 features total)
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
import onnx
import onnxruntime as ort
import os
import json
from datetime import datetime, timedelta
from strategies.base import BaseStrategy
from data_providers.senticrypt_provider import SentiCryptProvider

class MlWithSentiment(BaseStrategy):
    def __init__(self, name="MlWithSentiment", model_path=None, sequence_length=120, 
                 use_sentiment=True, sentiment_csv_path=None):
        super().__init__(name)
        
        # Set strategy-specific trading pair - will be determined by model
        self.trading_pair = 'BTCUSDT'  # Default, can be overridden
        
        # Model configuration
        self.sequence_length = sequence_length
        self.use_sentiment = use_sentiment
        
        # Determine model path
        if model_path is None:
            model_suffix = "sentiment" if use_sentiment else "price"
            model_path = f"ml/{self.trading_pair.lower()}_{model_suffix}.onnx"
        
        self.model_path = model_path
        
        # Load model and metadata
        self._load_model()
        self._load_metadata()
        
        # Initialize sentiment provider if needed
        self.sentiment_provider = None
        if self.use_sentiment:
            self._initialize_sentiment_provider(sentiment_csv_path)
        
        # Trading parameters
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        
        # Feature tracking
        self.feature_names = []
        self.sentiment_features = []
        
    def _load_model(self):
        """Load the ONNX model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.ort_session = ort.InferenceSession(self.model_path)
            self.input_name = self.ort_session.get_inputs()[0].name
            
            # Get input shape to determine expected features
            input_shape = self.ort_session.get_inputs()[0].shape
            self.expected_features = input_shape[2] if len(input_shape) > 2 else 1
            
            print(f"Loaded model: {self.model_path}")
            print(f"Expected input shape: {input_shape}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _load_metadata(self):
        """Load model metadata if available"""
        metadata_path = self.model_path.replace('.onnx', '_metadata.json')
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                # Extract feature information
                if 'feature_names' in self.metadata:
                    self.feature_names = self.metadata['feature_names']
                    self.sentiment_features = [f for f in self.feature_names if f.startswith('sentiment_')]
                
                # Update trading pair from metadata
                if 'symbol' in self.metadata:
                    self.trading_pair = self.metadata['symbol']
                
                print(f"Loaded metadata: {len(self.feature_names)} features")
                if self.sentiment_features:
                    print(f"Sentiment features: {len(self.sentiment_features)}")
                
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
                self.metadata = {}
        else:
            print(f"No metadata file found: {metadata_path}")
            self.metadata = {}
    
    def _initialize_sentiment_provider(self, csv_path=None):
        """Initialize sentiment data provider"""
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
            self.use_sentiment = False

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators including sentiment features"""
        df = df.copy()
        
        # Add sentiment data if enabled
        if self.use_sentiment and self.sentiment_provider is not None:
            df = self._add_sentiment_features(df)
        
        # Normalize price features (same as training)
        price_features = ['close', 'volume', 'high', 'low', 'open']
        for feature in price_features:
            if feature in df.columns:
                # Simple min-max normalization within the sequence window
                df[f'{feature}_normalized'] = df[feature].rolling(
                    window=self.sequence_length, min_periods=1
                ).apply(
                    lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) != np.min(x) else 0.5,
                    raw=True
                )
        
        # Prepare predictions column
        df['ml_prediction'] = np.nan
        df['prediction_confidence'] = np.nan
        df['sentiment_freshness'] = np.nan  # Track sentiment data age
        
        # Generate predictions
        self._generate_predictions(df)
        
        return df
    
    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment features to the dataframe with graceful missing data handling"""
        
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
                            sentiment_resampled = sentiment_df.resample('1h').fillna(method='ffill')
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
                df[sentiment_cols] = df[sentiment_cols].fillna(method='ffill')
            
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
    
    def _generate_predictions(self, df: pd.DataFrame):
        """Generate ML predictions for the dataframe"""
        try:
            # Calculate denormalization parameters from data
            price_min = df['close'].min()
            price_max = df['close'].max()
            price_range = price_max - price_min
            
            # Use exact feature names from model training (excluding metadata fields)
            if self.feature_names:
                features_to_use = self.feature_names.copy()
                # Remove metadata fields that shouldn't be model features
                metadata_fields = ['sentiment_freshness']
                features_to_use = [f for f in features_to_use if f not in metadata_fields]
            else:
                # Fallback to basic features
                features_to_use = ['close', 'volume', 'high', 'low', 'open']
                if self.use_sentiment:
                    # Only include actual sentiment features, not metadata
                    sentiment_features = [
                        'sentiment_primary', 'sentiment_momentum', 'sentiment_volatility',
                        'sentiment_extreme_positive', 'sentiment_extreme_negative',
                        'sentiment_ma_3', 'sentiment_ma_7', 'sentiment_ma_14', 'sentiment_confidence'
                    ]
                    features_to_use.extend(sentiment_features)
            
            # Ensure all required features are present
            available_features = []
            missing_features = 0
            for feature in features_to_use:
                if feature in df.columns:
                    available_features.append(feature)
                else:
                    # Create missing feature with neutral values
                    if feature.startswith('sentiment_'):
                        df[feature] = self._get_neutral_sentiment_value(feature)
                    else:
                        df[feature] = 0.0
                    available_features.append(feature)
                    missing_features += 1
            
            # Generate predictions for each valid window
            predictions_generated = 0
            prediction_errors = 0
            
            for i in range(self.sequence_length, len(df)):
                try:
                    # Extract feature window
                    window_data = df[available_features].iloc[i-self.sequence_length:i].values
                    
                    # Check for NaN values in the input
                    if np.isnan(window_data).any():
                        prediction_errors += 1
                        continue
                    
                    # Reshape for model input and ensure float32 type
                    input_window = window_data.astype(np.float32)
                    input_window = np.expand_dims(input_window, axis=0)  # Add batch dimension
                    
                    # Ensure correct number of features
                    if input_window.shape[2] != self.expected_features:
                        if input_window.shape[2] < self.expected_features:
                            # Pad with zeros
                            padding = np.zeros((1, self.sequence_length, 
                                              self.expected_features - input_window.shape[2]), dtype=np.float32)
                            input_window = np.concatenate([input_window, padding], axis=2)
                        else:
                            # Truncate
                            input_window = input_window[:, :, :self.expected_features]
                    
                    # Ensure input is float32
                    input_window = input_window.astype(np.float32)
                    
                    # Make prediction
                    output = self.ort_session.run(None, {self.input_name: input_window})
                    prediction = output[0][0][0]
                    
                    # Denormalize prediction from MinMaxScaler(0, 1) back to price
                    denormalized_pred = prediction * price_range + price_min
                    
                    # Skip invalid predictions
                    if np.isnan(denormalized_pred) or np.isinf(denormalized_pred):
                        prediction_errors += 1
                        continue
                    
                    df.iloc[i, df.columns.get_loc('ml_prediction')] = denormalized_pred
                    
                    # Calculate confidence (simple measure based on prediction vs current price)
                    current_price = df['close'].iloc[i-1]
                    confidence = 1.0 - abs(denormalized_pred - current_price) / current_price
                    df.iloc[i, df.columns.get_loc('prediction_confidence')] = max(0, min(1, confidence))
                    
                    predictions_generated += 1
                    
                except Exception as e:
                    prediction_errors += 1
                    continue
            
            # Report final results
            if predictions_generated > 0 or prediction_errors > 0:
                print(f"Generated {predictions_generated} ML predictions ({prediction_errors} errors)")
                    
        except Exception as e:
            print(f"Error generating predictions: {e}")

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check if conditions are met for entering a position"""
        if index < 1 or index >= len(df):
            return False
        
        # Check if we have a valid prediction
        if pd.isna(df['ml_prediction'].iloc[index]):
            # Log the missing prediction
            self.log_execution(
                signal_type='entry',
                action_taken='no_action',
                price=df['close'].iloc[index],
                reasons=['missing_ml_prediction'],
                additional_context={'prediction_available': False}
            )
            return False
        
        prediction = df['ml_prediction'].iloc[index]
        current_price = df['close'].iloc[index]
        confidence = df.get('prediction_confidence', pd.Series([0.5] * len(df))).iloc[index]
        
        # Get sentiment freshness (higher weight for fresh sentiment)
        sentiment_freshness = df.get('sentiment_freshness', pd.Series([0] * len(df))).iloc[index]
        freshness_boost = 1.1 if sentiment_freshness > 0 else 1.0  # 10% boost for live sentiment
        
        # Entry condition: prediction is higher than current price with sufficient confidence
        price_increase_threshold = 0.002 / freshness_boost  # More lenient: 0.2% vs 0.5%
        confidence_threshold = 0.3 / freshness_boost  # More lenient: 30% vs 60%
        
        predicted_return = (prediction - current_price) / current_price
        
        # Log detailed decision process
        sentiment_data = {}
        if 'sentiment_primary' in df.columns:
            sentiment_data = {
                'sentiment_primary': df['sentiment_primary'].iloc[index],
                'sentiment_confidence': df.get('sentiment_confidence', pd.Series([0])).iloc[index],
                'sentiment_freshness': sentiment_freshness
            }
        
        ml_predictions = {
            'raw_prediction': prediction,
            'current_price': current_price,
            'predicted_return': predicted_return,
            'confidence': confidence,
            'freshness_boost': freshness_boost
        }
        
        # Determine entry signal
        entry_signal = (predicted_return > price_increase_threshold and 
                       confidence > confidence_threshold)
        
        # Log the decision process
        reasons = [
            f'predicted_return_{predicted_return:.4f}_vs_threshold_{price_increase_threshold:.4f}',
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
                'sequence_length': self.sequence_length,
                'use_sentiment': self.use_sentiment,
                'prediction_available': True
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
        
        # Additional exit condition: if prediction turns negative
        if not pd.isna(df['ml_prediction'].iloc[index]):
            prediction = df['ml_prediction'].iloc[index]
            predicted_return = (prediction - current_price) / current_price
            prediction_negative = predicted_return < -0.01  # Exit if predicting >1% drop
        else:
            prediction_negative = False
        
        return hit_stop_loss or hit_take_profit or prediction_negative

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate position size based on prediction confidence"""
        if index >= len(df) or balance <= 0:
            return 0.0
        
        base_position_size = balance * 0.10  # Base 10% of balance
        
        # Adjust based on confidence if available
        if 'prediction_confidence' in df.columns and not pd.isna(df['prediction_confidence'].iloc[index]):
            confidence = df['prediction_confidence'].iloc[index]
            # Scale position size by confidence (0.5x to 1.5x)
            confidence_multiplier = 0.5 + confidence
            return base_position_size * confidence_multiplier
        
        return base_position_size

    def get_parameters(self) -> dict:
        """Get strategy parameters"""
        params = {
            'name': self.name,
            'model_path': self.model_path,
            'sequence_length': self.sequence_length,
            'use_sentiment': self.use_sentiment,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'trading_pair': self.trading_pair
        }
        
        if hasattr(self, 'metadata'):
            params['model_metadata'] = self.metadata
        
        return params

    def calculate_stop_loss(self, df, index, price, side: str = 'long') -> float:
        """Calculate stop loss price"""
        if side == 'long':
            return price * (1 - self.stop_loss_pct)
        else:  # short
            return price * (1 + self.stop_loss_pct) 