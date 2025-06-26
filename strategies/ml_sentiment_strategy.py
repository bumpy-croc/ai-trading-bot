import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
import os
import json
from datetime import datetime, timedelta
from strategies.base import BaseStrategy
from core.data_providers.senticrypt_provider import SentiCryptProvider

class MlSentimentStrategy(BaseStrategy):
    def __init__(self, name="MlSentimentStrategy", model_path=None, sequence_length=120, 
                 use_sentiment=True, sentiment_csv_path=None):
        super().__init__(name)
        
        # Set strategy-specific trading pair - will be determined by model
        self.trading_pair = 'BTCUSDT'  # Default, can be overridden
        
        # Model configuration
        self.sequence_length = sequence_length
        self.use_sentiment = use_sentiment
        
        # Determine model path
        if model_path is None:
            model_suffix = "sentiment" if use_sentiment else "price_only"
            model_path = f"ml/model_{self.trading_pair.lower()}_{model_suffix}.onnx"
        
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
                if 'features' in self.metadata:
                    self.feature_names = self.metadata['features']
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
            
            self.sentiment_provider = SentiCryptProvider(csv_path=csv_path)
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
        
        # Generate predictions
        self._generate_predictions(df)
        
        return df
    
    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment features to the dataframe"""
        try:
            # Get sentiment data for the date range
            start_date = df.index.min()
            end_date = df.index.max()
            
            sentiment_df = self.sentiment_provider.get_historical_sentiment(
                symbol=self.trading_pair,
                start=start_date,
                end=end_date
            )
            
            if not sentiment_df.empty:
                # Resample sentiment to match price data frequency
                if len(df) > len(sentiment_df):
                    # Upsample sentiment data
                    sentiment_resampled = sentiment_df.resample('1h').fillna(method='ffill')
                else:
                    sentiment_resampled = sentiment_df
                
                # Merge with price data
                df = df.join(sentiment_resampled, how='left')
                
                # Forward fill sentiment data
                sentiment_cols = [col for col in df.columns if col.startswith('sentiment_')]
                df[sentiment_cols] = df[sentiment_cols].fillna(method='ffill').fillna(0)
                
        except Exception as e:
            print(f"Warning: Could not add sentiment features: {e}")
            # Add dummy sentiment features if expected by model
            if self.sentiment_features:
                for feature in self.sentiment_features:
                    df[feature] = 0.0
        
        return df
    
    def _generate_predictions(self, df: pd.DataFrame):
        """Generate ML predictions for the dataframe"""
        try:
            # Determine which features to use based on model training
            if self.feature_names:
                features_to_use = self.feature_names.copy()
                # Replace original features with normalized versions for price features
                price_features = ['close', 'volume', 'high', 'low', 'open']
                for i, feature in enumerate(features_to_use):
                    if feature in price_features and f'{feature}_normalized' in df.columns:
                        features_to_use[i] = f'{feature}_normalized'
            else:
                # Fallback to basic features
                features_to_use = ['close_normalized']
                if self.use_sentiment:
                    sentiment_cols = [col for col in df.columns if col.startswith('sentiment_')]
                    features_to_use.extend(sentiment_cols)
            
            # Ensure all required features are present
            available_features = []
            for feature in features_to_use:
                if feature in df.columns:
                    available_features.append(feature)
                else:
                    print(f"Warning: Feature {feature} not found, using 0")
                    df[feature] = 0.0
                    available_features.append(feature)
            
            # Generate predictions for each valid window
            for i in range(self.sequence_length, len(df)):
                try:
                    # Extract feature window
                    window_data = df[available_features].iloc[i-self.sequence_length:i].values
                    
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
                    
                    # Store prediction (denormalized if needed)
                    if 'close_normalized' in available_features:
                        # Denormalize prediction
                        close_window = df['close'].iloc[i-self.sequence_length:i].values
                        min_close = np.min(close_window)
                        max_close = np.max(close_window)
                        if max_close != min_close:
                            denormalized_pred = prediction * (max_close - min_close) + min_close
                        else:
                            denormalized_pred = df['close'].iloc[i-1]
                    else:
                        denormalized_pred = prediction
                    
                    df.iloc[i, df.columns.get_loc('ml_prediction')] = denormalized_pred
                    
                    # Calculate confidence (simple measure based on prediction vs current price)
                    current_price = df['close'].iloc[i-1]
                    confidence = 1.0 - abs(denormalized_pred - current_price) / current_price
                    df.iloc[i, df.columns.get_loc('prediction_confidence')] = max(0, min(1, confidence))
                    
                except Exception as e:
                    print(f"Warning: Could not generate prediction for index {i}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error generating predictions: {e}")

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check if conditions are met for entering a position"""
        if index < 1 or index >= len(df):
            return False
        
        # Check if we have a valid prediction
        if pd.isna(df['ml_prediction'].iloc[index]):
            return False
        
        prediction = df['ml_prediction'].iloc[index]
        current_price = df['close'].iloc[index]
        confidence = df.get('prediction_confidence', pd.Series([0.5] * len(df))).iloc[index]
        
        # Entry condition: prediction is higher than current price with sufficient confidence
        price_increase_threshold = 0.005  # 0.5% minimum predicted increase
        confidence_threshold = 0.6  # 60% confidence threshold
        
        predicted_return = (prediction - current_price) / current_price
        
        return (predicted_return > price_increase_threshold and 
                confidence > confidence_threshold)

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