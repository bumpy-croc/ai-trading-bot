"""
Prediction Engine

Main prediction engine that strategies use for ML predictions.
Provides a unified interface for model predictions with caching and error handling.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
import time
import logging

from .config import PredictionConfig
from .models.onnx_runner import OnnxRunner, ModelPrediction
from .features.pipeline import FeaturePipeline


@dataclass
class PredictionResult:
    """Result of a prediction request"""
    price: Optional[float]
    confidence: float
    direction: int  # 1 for up, 0 for neutral, -1 for down
    model_name: Optional[str]
    timestamp: pd.Timestamp
    error: Optional[str] = None


class PredictionEngine:
    """
    Main prediction engine that provides ML predictions to trading strategies.
    
    This engine handles:
    - Model loading and inference
    - Feature extraction from market data
    - Prediction caching
    - Error handling and fallbacks
    """
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        """
        Initialize prediction engine.
        
        Args:
            config: Prediction configuration. If None, loads from config manager.
        """
        self.config = config or PredictionConfig.from_config_manager()
        self.config.validate()
        
        self.logger = logging.getLogger(__name__)
        
        # Model runners for different symbols/models
        self._model_runners: Dict[str, OnnxRunner] = {}
        
        # Feature pipeline
        self._feature_pipeline = FeaturePipeline()
        
        # Cache for predictions
        self._prediction_cache: Dict[str, PredictionResult] = {}
        
        # Initialize default model
        self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Initialize the default model for BTC predictions"""
        try:
            default_model_path = f"{self.config.model_registry_path}/btcusdt_price.onnx"
            self._model_runners["btcusdt_price"] = OnnxRunner(default_model_path, self.config)
            self.logger.info(f"Loaded default model: {default_model_path}")
        except Exception as e:
            self.logger.warning(f"Could not load default model: {e}")
    
    def predict(self, data: pd.DataFrame, symbol: str = "BTCUSDT", 
                model_name: Optional[str] = None) -> PredictionResult:
        """
        Generate prediction for given market data.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            symbol: Trading symbol (e.g., "BTCUSDT")
            model_name: Specific model to use. If None, uses default for symbol.
            
        Returns:
            PredictionResult with prediction details
        """
        try:
            # Determine model name
            if model_name is None:
                model_name = self._get_default_model_name(symbol)
            
            # Check cache
            cache_key = self._get_cache_key(data, symbol, model_name)
            cached_result = self._get_cached_prediction(cache_key)
            if cached_result:
                return cached_result
            
            # Get model runner
            model_runner = self._get_model_runner(model_name)
            if not model_runner:
                return self._create_error_result(f"Model {model_name} not available")
            
            # Extract features
            features = self._extract_features(data, model_name)
            if features is None:
                return self._create_error_result("Feature extraction failed")
            
            # Make prediction
            start_time = time.time()
            prediction = model_runner.predict(features)
            inference_time = time.time() - start_time
            
            # Check latency requirement
            if inference_time > self.config.max_prediction_latency:
                self.logger.warning(f"Prediction latency {inference_time:.3f}s exceeds limit {self.config.max_prediction_latency}s")
            
            # Convert to result
            result = self._convert_to_result(prediction, inference_time)
            
            # Cache result
            self._cache_prediction(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return self._create_error_result(str(e))
    
    def _get_default_model_name(self, symbol: str) -> str:
        """Get default model name for a symbol"""
        symbol_lower = symbol.lower()
        if self.config.enable_sentiment:
            return f"{symbol_lower}_sentiment"
        else:
            return f"{symbol_lower}_price"
    
    def _get_model_runner(self, model_name: str) -> Optional[OnnxRunner]:
        """Get or load model runner for given model name"""
        if model_name not in self._model_runners:
            try:
                model_path = f"{self.config.model_registry_path}/{model_name}.onnx"
                self._model_runners[model_name] = OnnxRunner(model_path, self.config)
                self.logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")
                return None
        
        return self._model_runners.get(model_name)
    
    def _extract_features(self, data: pd.DataFrame, model_name: str) -> Optional[np.ndarray]:
        """Extract features from market data for the specified model"""
        try:
            # For now, use simple price features
            # This should be expanded based on the feature pipeline
            if len(data) < 120:  # Minimum sequence length
                return None
            
            # Use last 120 rows for sequence
            sequence_data = data.tail(120)
            
            # Extract OHLCV features and normalize
            price_features = ['open', 'high', 'low', 'close', 'volume']
            feature_array = sequence_data[price_features].values
            
            # Simple min-max normalization
            for i in range(feature_array.shape[1]):
                col = feature_array[:, i]
                min_val, max_val = col.min(), col.max()
                if max_val != min_val:
                    feature_array[:, i] = (col - min_val) / (max_val - min_val)
                else:
                    feature_array[:, i] = 0.5
            
            # Reshape for model: (batch_size, sequence_length, features)
            return feature_array.astype(np.float32)[np.newaxis, :, :]
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            return None
    
    def _convert_to_result(self, prediction: ModelPrediction, inference_time: float) -> PredictionResult:
        """Convert ModelPrediction to PredictionResult"""
        # Calculate direction based on confidence and price
        direction = 1 if prediction.confidence > self.config.min_confidence_threshold else 0
        if prediction.price and len(self._prediction_cache) > 0:
            # Compare with recent predictions to determine direction
            recent_predictions = list(self._prediction_cache.values())[-5:]
            avg_recent_price = np.mean([p.price for p in recent_predictions if p.price])
            if prediction.price < avg_recent_price * 0.98:  # 2% drop threshold
                direction = -1
        
        return PredictionResult(
            price=prediction.price,
            confidence=prediction.confidence,
            direction=direction,
            model_name=prediction.model_name,
            timestamp=pd.Timestamp.now(),
            error=None
        )
    
    def _create_error_result(self, error_message: str) -> PredictionResult:
        """Create error result"""
        return PredictionResult(
            price=None,
            confidence=0.0,
            direction=0,
            model_name=None,
            timestamp=pd.Timestamp.now(),
            error=error_message
        )
    
    def _get_cache_key(self, data: pd.DataFrame, symbol: str, model_name: str) -> str:
        """Generate cache key for prediction"""
        # Use last row timestamp and symbol/model as key
        last_timestamp = data.index[-1] if len(data) > 0 else pd.Timestamp.now()
        return f"{symbol}_{model_name}_{last_timestamp}"
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[PredictionResult]:
        """Get cached prediction if still valid"""
        if cache_key in self._prediction_cache:
            cached = self._prediction_cache[cache_key]
            # Check if cache is still valid (within TTL)
            age = (pd.Timestamp.now() - cached.timestamp).total_seconds()
            if age < self.config.model_cache_ttl:
                return cached
            else:
                # Remove expired cache
                del self._prediction_cache[cache_key]
        return None
    
    def _cache_prediction(self, cache_key: str, result: PredictionResult):
        """Cache prediction result"""
        self._prediction_cache[cache_key] = result
        
        # Cleanup old cache entries to prevent memory growth
        if len(self._prediction_cache) > 1000:
            # Remove oldest 200 entries
            old_keys = list(self._prediction_cache.keys())[:200]
            for key in old_keys:
                del self._prediction_cache[key]
    
    def clear_cache(self):
        """Clear prediction cache"""
        self._prediction_cache.clear()
    
    def get_available_models(self) -> list:
        """Get list of available model names"""
        return list(self._model_runners.keys())