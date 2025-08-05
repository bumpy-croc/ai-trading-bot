"""
Core Prediction Engine

This module provides the main PredictionEngine class that serves as the unified
facade for all prediction operations, orchestrating configuration, feature
engineering, and model inference.
"""

import pandas as pd
import numpy as np
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import PredictionConfig
from .features.pipeline import FeaturePipeline
from .models.registry import PredictionModelRegistry
from .models.onnx_runner import ModelPrediction
from .exceptions import (
    PredictionEngineError,
    InvalidInputError,
    ModelNotFoundError,
    FeatureExtractionError,
    PredictionTimeoutError
)


@dataclass
class PredictionResult:
    """Result of a prediction engine operation"""
    price: float
    confidence: float
    direction: int  # 1, 0, -1
    model_name: str
    timestamp: datetime
    inference_time: float
    features_used: int
    cache_hit: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PredictionEngine:
    """Main prediction engine facade that orchestrates all components"""
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        """
        Initialize prediction engine with configuration
        
        Args:
            config: Optional prediction configuration. If None, loads from ConfigManager
        """
        self.config = config or PredictionConfig.from_config_manager()
        self.config.validate()
        
        # Initialize components
        self.feature_pipeline = FeaturePipeline(
            enable_sentiment=self.config.enable_sentiment,
            enable_market_microstructure=self.config.enable_market_microstructure,
            cache_ttl=self.config.feature_cache_ttl
        )
        
        self.model_registry = PredictionModelRegistry(self.config)
        
        # Performance tracking
        self._prediction_count = 0
        self._total_inference_time = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        self._feature_extraction_time = 0.0
    
    def predict(self, data: pd.DataFrame, model_name: Optional[str] = None) -> PredictionResult:
        """
        Main prediction method - unified interface for all predictions
        
        Args:
            data: Input market data (OHLCV format)
            model_name: Optional specific model to use. If None, uses default model
            
        Returns:
            PredictionResult: Unified prediction result with metadata
        """
        start_time = time.time()
        
        try:
            # Validate input data
            self._validate_input_data(data)
            
            # Extract features
            feature_start_time = time.time()
            features = self._extract_features(data)
            feature_time = time.time() - feature_start_time
            self._feature_extraction_time = feature_time
            
            # Get model for prediction
            model = self._get_model(model_name)
            
            # Make prediction
            prediction = model.predict(features)
            
            # Calculate total inference time
            inference_time = time.time() - start_time
            
            # Check if cache was hit (from feature pipeline)
            cache_hit = self._was_cache_hit()
            
            # Create unified result
            result = PredictionResult(
                price=prediction.price,
                confidence=prediction.confidence,
                direction=prediction.direction,
                model_name=prediction.model_name,
                timestamp=datetime.now(timezone.utc),
                inference_time=inference_time,
                features_used=features.shape[1] if hasattr(features, 'shape') else 0,
                cache_hit=cache_hit,
                metadata={
                    'data_length': len(data),
                    'feature_extraction_time': feature_time,
                    'model_inference_time': prediction.inference_time,
                    'config_version': self._get_config_version()
                }
            )
            
            # Update performance statistics
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            # Check for timeout
            total_time = time.time() - start_time
            if total_time > self.config.max_prediction_latency:
                error_msg = f"Prediction timeout after {total_time:.3f}s (max: {self.config.max_prediction_latency}s)"
                error = PredictionTimeoutError(error_msg)
            else:
                error = e
            
            # Return error result
            return PredictionResult(
                price=0.0,
                confidence=0.0,
                direction=0,
                model_name=model_name or "unknown",
                timestamp=datetime.now(timezone.utc),
                inference_time=time.time() - start_time,
                features_used=0,
                error=str(error),
                metadata={
                    'error_type': type(error).__name__,
                    'data_length': len(data) if isinstance(data, pd.DataFrame) else 0
                }
            )
    
    def predict_batch(self, data_batches: List[pd.DataFrame], 
                     model_name: Optional[str] = None) -> List[PredictionResult]:
        """
        Batch prediction for multiple data sets
        
        Args:
            data_batches: List of market data DataFrames
            model_name: Optional specific model to use
            
        Returns:
            List[PredictionResult]: List of prediction results
        """
        results = []
        
        # Get model once for efficiency
        try:
            model = self._get_model(model_name)
        except Exception as e:
            # If model loading fails, return error results for all batches
            error_result_template = PredictionResult(
                price=0.0,
                confidence=0.0,
                direction=0,
                model_name=model_name or "unknown",
                timestamp=datetime.now(timezone.utc),
                inference_time=0.0,
                features_used=0,
                error=str(e),
                metadata={'error_type': type(e).__name__}
            )
            return [error_result_template for _ in data_batches]
        
        for i, data in enumerate(data_batches):
            start_time = time.time()
            
            try:
                # Validate input data
                self._validate_input_data(data)
                
                # Extract features
                feature_start_time = time.time()
                features = self._extract_features(data)
                feature_time = time.time() - feature_start_time
                
                # Make prediction with pre-loaded model
                prediction = model.predict(features)
                
                # Calculate total inference time
                inference_time = time.time() - start_time
                
                # Create result
                result = PredictionResult(
                    price=prediction.price,
                    confidence=prediction.confidence,
                    direction=prediction.direction,
                    model_name=prediction.model_name,
                    timestamp=datetime.now(timezone.utc),
                    inference_time=inference_time,
                    features_used=features.shape[1] if hasattr(features, 'shape') else 0,
                    cache_hit=self._was_cache_hit(),
                    metadata={
                        'data_length': len(data),
                        'feature_extraction_time': feature_time,
                        'model_inference_time': prediction.inference_time,
                        'batch_index': i,
                        'batch_size': len(data_batches)
                    }
                )
                
                # Update performance statistics
                self._update_performance_stats(result)
                
                results.append(result)
                
            except Exception as e:
                # Create error result for this batch
                error_result = PredictionResult(
                    price=0.0,
                    confidence=0.0,
                    direction=0,
                    model_name=model_name or prediction.model_name if 'prediction' in locals() else "unknown",
                    timestamp=datetime.now(timezone.utc),
                    inference_time=time.time() - start_time,
                    features_used=0,
                    error=str(e),
                    metadata={
                        'error_type': type(e).__name__,
                        'batch_index': i,
                        'batch_size': len(data_batches),
                        'data_length': len(data) if isinstance(data, pd.DataFrame) else 0
                    }
                )
                results.append(error_result)
        
        return results
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self.model_registry.list_models()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict containing model information
        """
        model = self.model_registry.get_model(model_name)
        if not model:
            return {}
        
        return {
            'name': model_name,
            'path': model.model_path,
            'metadata': model.model_metadata,
            'loaded': True,
            'inference_time_avg': self._get_model_avg_inference_time(model_name)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics"""
        return {
            'total_predictions': self._prediction_count,
            'avg_inference_time': (
                self._total_inference_time / self._prediction_count 
                if self._prediction_count > 0 else 0.0
            ),
            'cache_hit_rate': (
                self._cache_hits / (self._cache_hits + self._cache_misses) 
                if (self._cache_hits + self._cache_misses) > 0 else 0.0
            ),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'available_models': len(self.get_available_models()),
            'avg_feature_extraction_time': self._feature_extraction_time
        }
    
    def clear_caches(self) -> None:
        """Clear all caches"""
        self.feature_pipeline.clear_cache()
        # Reset performance stats related to caching
        self._cache_hits = 0
        self._cache_misses = 0
    
    def reload_models(self) -> None:
        """Reload all models"""
        self.model_registry.reload_models()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of all components
        
        Returns:
            Dict containing health status of all components
        """
        health = {
            'status': 'healthy',
            'components': {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Check feature pipeline
        try:
            # Test with minimal valid data
            test_data = pd.DataFrame({
                'open': [100.0, 101.0, 102.0] * 40,  # 120 rows minimum
                'high': [102.0, 103.0, 104.0] * 40,
                'low': [99.0, 100.0, 101.0] * 40,
                'close': [101.0, 102.0, 103.0] * 40,
                'volume': [1000, 1100, 1200] * 40
            })
            features = self.feature_pipeline.transform(test_data, use_cache=False)
            health['components']['feature_pipeline'] = {
                'status': 'healthy',
                'test_features_count': features.shape[1] if hasattr(features, 'shape') else 0
            }
        except Exception as e:
            health['components']['feature_pipeline'] = {
                'status': 'error',
                'error': str(e)
            }
            health['status'] = 'degraded'
        
        # Check model registry
        try:
            models = self.model_registry.list_models()
            default_model = self.model_registry.get_default_model()
            health['components']['model_registry'] = {
                'status': 'healthy',
                'available_models': len(models),
                'model_names': models,
                'default_model': default_model.model_path if default_model else None
            }
        except Exception as e:
            health['components']['model_registry'] = {
                'status': 'error',
                'error': str(e)
            }
            health['status'] = 'degraded'
        
        # Check configuration
        try:
            self.config.validate()
            health['components']['configuration'] = {
                'status': 'healthy',
                'config_version': self._get_config_version()
            }
        except Exception as e:
            health['components']['configuration'] = {
                'status': 'error',
                'error': str(e)
            }
            health['status'] = 'degraded'
        
        return health
    
    # Private methods
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data has required columns and sufficient length"""
        if not isinstance(data, pd.DataFrame):
            raise InvalidInputError("Input data must be a pandas DataFrame")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise InvalidInputError(f"Missing required columns: {missing_columns}")
        
        if len(data) < 120:  # Minimum for LSTM sequence
            raise InvalidInputError(f"Insufficient data: {len(data)} rows, minimum 120 required")
        
        # Check for invalid values
        if data[required_columns].isnull().any().any():
            raise InvalidInputError("Input data contains null values")
        
        if (data[required_columns] <= 0).any().any():
            raise InvalidInputError("Input data contains non-positive values")
    
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features using feature pipeline"""
        try:
            return self.feature_pipeline.transform(data, use_cache=True)
        except Exception as e:
            raise FeatureExtractionError(f"Feature extraction failed: {str(e)}")
    
    def _get_model(self, model_name: Optional[str]):
        """Get model for prediction"""
        if model_name:
            model = self.model_registry.get_model(model_name)
            if not model:
                raise ModelNotFoundError(f"Model '{model_name}' not found")
            return model
        
        # Use default model
        default_model = self.model_registry.get_default_model()
        if not default_model:
            raise ModelNotFoundError("No models available for prediction")
        
        return default_model
    
    def _was_cache_hit(self) -> bool:
        """Check if last operation was a cache hit"""
        # * This would be implemented based on feature pipeline cache tracking
        # For now, return False as placeholder
        return False
    
    def _get_config_version(self) -> str:
        """Get configuration version/hash for tracking"""
        return f"v1.0-{hash(str(self.config))}"
    
    def _update_performance_stats(self, result: PredictionResult) -> None:
        """Update internal performance statistics"""
        self._prediction_count += 1
        self._total_inference_time += result.inference_time
        
        if result.cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
    
    def _get_model_avg_inference_time(self, model_name: str) -> float:
        """Get average inference time for specific model"""
        # * This would be tracked per model in a future enhancement  
        # For now, return global average
        return (
            self._total_inference_time / self._prediction_count 
            if self._prediction_count > 0 else 0.0
        )