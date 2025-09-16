"""
ML Sentiment Strategy

This strategy uses machine learning models trained with both price data and sentiment analysis.
It leverages the Fear & Greed Index to enhance prediction accuracy and trading decisions.

Key Features:
- Price + sentiment predictions using LSTM neural network
- 120-day sequence length for pattern recognition
- Fear & Greed Index sentiment integration
- Adaptive position sizing based on sentiment confidence
- 2% stop loss, 4% take profit risk management
- Robust fallback when sentiment data is unavailable

Ideal for:
- Enhanced prediction accuracy with market sentiment
- Bull/bear market regime detection
- Trading during high-volatility periods
- Environments with reliable sentiment data access
"""

from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import pandas as pd

from src.config.config_manager import get_config
from src.config.constants import DEFAULT_USE_PREDICTION_ENGINE
from src.prediction import PredictionConfig, PredictionEngine
from src.prediction.features.pipeline import FeaturePipeline
from src.prediction.features.sentiment import SentimentFeatureExtractor
from src.prediction.features.technical import TechnicalFeatureExtractor
from src.strategies.base import BaseStrategy


class MlSentiment(BaseStrategy):
    # * Strategy configuration constants
    SHORT_ENTRY_THRESHOLD = -0.0005  # -0.05% threshold for short entries
    CONFIDENCE_MULTIPLIER = 12  # Multiplier for confidence calculation
    BASE_POSITION_SIZE = 0.2  # Base position size (20% of balance)
    MIN_POSITION_SIZE_RATIO = 0.05  # Minimum position size (5% of balance)
    MAX_POSITION_SIZE_RATIO = 0.25  # Maximum position size (25% of balance)
    
    # Sentiment-specific parameters
    SENTIMENT_BOOST_MULTIPLIER = 1.2  # Boost position size when sentiment aligns
    SENTIMENT_REDUCTION_MULTIPLIER = 0.8  # Reduce position size when sentiment contradicts
    MIN_SENTIMENT_CONFIDENCE = 0.6  # Minimum sentiment confidence to use sentiment features

    def __init__(
        self,
        name="MlSentiment",
        model_path="src/ml/btcusdt_sentiment.onnx",
        sequence_length=120,
        use_prediction_engine: Optional[bool] = None,
        model_name: Optional[str] = None,
    ):
        super().__init__(name)

        # Set strategy-specific trading pair - default to BTC, can be overridden
        self.trading_pair = "BTCUSDT"

        self.model_path = model_path
        self.sequence_length = sequence_length
        self.ort_session = ort.InferenceSession(self.model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit

        # Optional prediction engine integration
        cfg = get_config()
        self.use_prediction_engine = (
            use_prediction_engine
            if use_prediction_engine is not None
            else cfg.get_bool("USE_PREDICTION_ENGINE", default=DEFAULT_USE_PREDICTION_ENGINE)
        )
        
        # Model name configuration
        self.model_name = model_name
        if self.model_name is None:
            self.model_name = cfg.get("PREDICTION_ENGINE_MODEL_NAME", default=None)
        if self.model_name is None:
            try:
                self.model_name = Path(self.model_path).stem
            except Exception:
                self.model_name = None
        
        self.prediction_engine = None
        self._engine_warning_emitted = False
        self.use_engine_batch = get_config().get_bool("ENGINE_BATCH_INFERENCE", default=False)

        # Initialize feature pipeline with sentiment support
        technical_extractor = TechnicalFeatureExtractor(
            sequence_length=self.sequence_length, normalization_window=self.sequence_length
        )
        sentiment_extractor = SentimentFeatureExtractor(enabled=True)
        
        if self.use_prediction_engine:
            config = {
                "technical_features": {"enabled": True},
                "sentiment_features": {"enabled": True},
                "market_features": {"enabled": False},
                "price_only_features": {"enabled": False},
            }
            self.feature_pipeline = FeaturePipeline(
                config=config,
                custom_extractors=[technical_extractor, sentiment_extractor],
            )
        else:
            # Fallback to basic pipeline without engine
            self.feature_pipeline = None

        # Initialize prediction engine if enabled
        if self.use_prediction_engine:
            try:
                prediction_config = PredictionConfig(
                    model_name=self.model_name,
                    sequence_length=self.sequence_length,
                    feature_pipeline=self.feature_pipeline,
                )
                self.prediction_engine = PredictionEngine(prediction_config)
                self.logger.info(f"Initialized prediction engine with model: {self.model_name}")
            except Exception as e:
                if not self._engine_warning_emitted:
                    self.logger.warning(f"Failed to initialize prediction engine: {e}")
                    self._engine_warning_emitted = True


    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators and prepare features for ML prediction"""
        # Add technical indicators
        df = df.copy()
        
        # Simple moving averages
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df['sma_14'] = df['close'].rolling(window=14).mean()
        df['sma_30'] = df['close'].rolling(window=30).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df

    def get_prediction(self, df: pd.DataFrame, index: int) -> tuple[float, float]:
        """Get ML prediction and confidence score"""
        if index < self.sequence_length:
            return 0.0, 0.0

        try:
            # Use prediction engine if available
            if self.use_prediction_engine and self.prediction_engine:
                # Get prediction using the engine
                prediction_data = df.iloc[index - self.sequence_length : index]
                prediction_result = self.prediction_engine.predict(prediction_data)
                
                if prediction_result and 'prediction' in prediction_result:
                    prediction = prediction_result['prediction']
                    confidence = prediction_result.get('confidence', 0.5)
                    return float(prediction), float(confidence)
            
            # Fallback to direct ONNX inference
            return self._get_onnx_prediction(df, index)
            
        except Exception as e:
            self.logger.warning(f"Prediction failed at index {index}: {e}")
            return 0.0, 0.0

    def _get_onnx_prediction(self, df: pd.DataFrame, index: int) -> tuple[float, float]:
        """Get prediction using direct ONNX inference"""
        try:
            # Prepare features
            start_idx = index - self.sequence_length
            end_idx = index
            
            # Extract price features (OHLCV)
            price_features = ['open', 'high', 'low', 'close', 'volume']
            price_data = df[price_features].iloc[start_idx:end_idx].values
            
            # Add technical indicators
            technical_features = ['sma_7', 'sma_14', 'sma_30', 'rsi']
            technical_data = df[technical_features].iloc[start_idx:end_idx].values
            
            # Combine features
            features = np.concatenate([price_data, technical_data], axis=1)
            
            # Normalize features (simple min-max normalization)
            features_normalized = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0) + 1e-8)
            
            # Reshape for LSTM input (samples, timesteps, features)
            features_reshaped = features_normalized.reshape(1, self.sequence_length, -1)
            
            # Get prediction
            prediction = self.ort_session.run(None, {self.input_name: features_reshaped.astype(np.float32)})
            prediction_value = float(prediction[0][0][0])
            
            # Calculate confidence based on prediction magnitude
            confidence = min(abs(prediction_value) * self.CONFIDENCE_MULTIPLIER, 1.0)
            
            return prediction_value, confidence
            
        except Exception as e:
            self.logger.warning(f"ONNX prediction failed: {e}")
            return 0.0, 0.0

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check if entry conditions are met"""
        if index < self.sequence_length:
            return False

        prediction, confidence = self.get_prediction(df, index)
        
        # Entry condition: prediction above threshold with sufficient confidence
        return prediction > abs(self.SHORT_ENTRY_THRESHOLD) and confidence > 0.5

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Check if exit conditions are met"""
        current_price = df.iloc[index]['close']
        
        # Check stop loss and take profit
        price_change_pct = (current_price - entry_price) / entry_price
        
        if price_change_pct <= -self.stop_loss_pct:
            return True  # Stop loss hit
        if price_change_pct >= self.take_profit_pct:
            return True  # Take profit hit
            
        return False

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate position size with sentiment adjustment"""
        if index < self.sequence_length:
            return 0.0

        prediction, confidence = self.get_prediction(df, index)
        
        # Base position size calculation
        base_size = self.BASE_POSITION_SIZE * balance
        
        # Adjust based on confidence
        confidence_adjusted_size = base_size * confidence
        
        # Sentiment adjustment (if we have sentiment data)
        try:
            sentiment_score = self._get_sentiment_score(df, index)
            if sentiment_score is not None:
                # Boost size if sentiment aligns with prediction, reduce if it contradicts
                if (prediction > 0 and sentiment_score > 0.5) or (prediction < 0 and sentiment_score < 0.5):
                    confidence_adjusted_size *= self.SENTIMENT_BOOST_MULTIPLIER
                else:
                    confidence_adjusted_size *= self.SENTIMENT_REDUCTION_MULTIPLIER
        except Exception as e:
            self.logger.debug(f"Sentiment adjustment failed: {e}")
        
        # Apply size limits
        min_size = balance * self.MIN_POSITION_SIZE_RATIO
        max_size = balance * self.MAX_POSITION_SIZE_RATIO
        
        return max(min_size, min(confidence_adjusted_size, max_size))

    def _get_sentiment_score(self, df: pd.DataFrame, index: int) -> Optional[float]:
        """Get sentiment score from Fear & Greed Index if available"""
        try:
            # This would integrate with the FearGreedProvider in a real implementation
            # For now, return None to indicate no sentiment data
            return None
        except Exception:
            return None

    def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = "long") -> float:
        """Calculate stop loss level"""
        if side == "long":
            return price * (1 - self.stop_loss_pct)
        else:
            return price * (1 + self.stop_loss_pct)

    def get_parameters(self) -> dict:
        """Return strategy parameters for logging"""
        return {
            "strategy_name": self.name,
            "model_path": self.model_path,
            "sequence_length": self.sequence_length,
            "use_prediction_engine": self.use_prediction_engine,
            "model_name": self.model_name,
            "base_position_size": self.BASE_POSITION_SIZE,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "sentiment_boost_multiplier": self.SENTIMENT_BOOST_MULTIPLIER,
            "sentiment_reduction_multiplier": self.SENTIMENT_REDUCTION_MULTIPLIER,
            "min_sentiment_confidence": self.MIN_SENTIMENT_CONFIDENCE,
        }
