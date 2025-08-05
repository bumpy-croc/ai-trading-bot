"""
Core Prediction Engine

This module provides the main PredictionEngine class that serves as the orchestrator
and facade, connecting configuration, feature engineering, model integration, and
providing a unified prediction interface for strategies.
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


class PredictionEngineError(Exception):
    """Base exception for prediction engine errors"""
    pass


class InvalidInputError(PredictionEngineError):
    """Raised when input data is invalid"""
    pass


class ModelNotFoundError(PredictionEngineError):
    """Raised when requested model is not available"""
    pass


class FeatureExtractionError(PredictionEngineError):
    """Raised when feature extraction fails"""
    pass


class PredictionTimeoutError(PredictionEngineError):
    """Raised when prediction takes too long"""
    pass


class PredictionEngine:
    """Main prediction engine facade that orchestrates all components"""
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        """Initialize prediction engine with configuration"""
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
        self._model_inference_times: Dict[str, List[float]] = {}
    
    def predict(self, data: pd.DataFrame, model_name: Optional[str] = None) -> PredictionResult:
        """Main prediction method - unified interface for all predictions"""
        start_time = time.time()
        
        try:
            # Validate input data
            self._validate_input_data(data)
            
            # Extract features
            feature_start = time.time()
            features = self._extract_features(data)
            feature_time = time.time() - feature_start
            self._feature_extraction_time += feature_time
            
            # Get model for prediction
            model = self._get_model(model_name)
            
            # Make prediction
            prediction = model.predict(features)
            
            # Calculate total inference time
            inference_time = time.time() - start_time
            
            # Track model inference time
            if prediction.model_name not in self._model_inference_times:
                self._model_inference_times[prediction.model_name] = []
            self._model_inference_times[prediction.model_name].append(prediction.inference_time)
            
            # Create unified result
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
                    'config_version': self._get_config_version(),
                    'feature_pipeline_stats': self.feature_pipeline.get_performance_stats()
                }
            )
            
            # Update performance statistics
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            # Return error result
            return PredictionResult(
                price=0.0,
                confidence=0.0,
                direction=0,
                model_name=model_name or "unknown",
                timestamp=datetime.now(timezone.utc),
                inference_time=time.time() - start_time,
                features_used=0,
                error=str(e),
                metadata={'error_type': type(e).__name__}
            )
    
    def predict_batch(self, data_batches: List[pd.DataFrame], 
                     model_name: Optional[str] = None) -> List[PredictionResult]:
        """Batch prediction for multiple data sets"""
        results = []
        batch_start_time = time.time()
        
        # Pre-load model to avoid reloading for each prediction
        try:
            model = self._get_model(model_name)
        except Exception as e:
            # Return error results for all batches
            error_result = PredictionResult(
                price=0.0,
                confidence=0.0,
                direction=0,
                model_name=model_name or "unknown",
                timestamp=datetime.now(timezone.utc),
                inference_time=0.0,
                features_used=0,
                error=str(e),
                metadata={'error_type': type(e).__name__, 'batch_error': True}
            )
            return [error_result] * len(data_batches)
        
        for i, data in enumerate(data_batches):
            result = self.predict(data, model_name)
            result.metadata['batch_index'] = i
            result.metadata['batch_size'] = len(data_batches)
            results.append(result)
        
        batch_time = time.time() - batch_start_time
        
        # Add batch metadata to all results
        for result in results:
            result.metadata['batch_total_time'] = batch_time
            result.metadata['batch_avg_time'] = batch_time / len(data_batches)
        
        return results
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self.model_registry.list_models()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        model = self.model_registry.get_model(model_name)
        if not model:
            return {}
        
        return {
            'name': model_name,
            'metadata': model.model_metadata,
            'loaded': True,
            'inference_time_avg': self._get_model_avg_inference_time(model_name),
            'prediction_count': len(self._model_inference_times.get(model_name, [])),
            'path': model.model_path
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics"""
        return {
            'total_predictions': self._prediction_count,
            'avg_inference_time': (
                self._total_inference_time / self._prediction_count 
                if self._prediction_count > 0 else 0.0
            ),
            'avg_feature_extraction_time': (
                self._feature_extraction_time / self._prediction_count
                if self._prediction_count > 0 else 0.0
            ),
            'cache_hit_rate': (
                self._cache_hits / (self._cache_hits + self._cache_misses) 
                if (self._cache_hits + self._cache_misses) > 0 else 0.0
            ),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'available_models': len(self.get_available_models()),
            'model_stats': {
                name: {
                    'avg_inference_time': self._get_model_avg_inference_time(name),
                    'prediction_count': len(times),
                    'min_inference_time': min(times) if times else 0.0,
                    'max_inference_time': max(times) if times else 0.0
                }
                for name, times in self._model_inference_times.items()
            },
            'feature_pipeline_stats': self.feature_pipeline.get_performance_stats()
        }
    
    def clear_caches(self) -> None:
        """Clear all caches"""
        self.feature_pipeline.clear_cache()
        # Reset performance stats
        self._cache_hits = 0
        self._cache_misses = 0
    
    def reload_models(self) -> None:
        """Reload all models"""
        self.model_registry = PredictionModelRegistry(self.config)
        # Clear model-specific performance stats
        self._model_inference_times = {}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of all components"""
        health = {
            'status': 'healthy',
            'components': {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Check feature pipeline
        try:
            # Test with minimal data
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
                'test_features_count': features.shape[1] if hasattr(features, 'shape') else 0,
                'test_data_length': len(test_data)
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
                'models': models,
                'default_model': default_model.model_path if default_model else None
            }
        except Exception as e:
            health['components']['model_registry'] = {
                'status': 'error',
                'error': str(e)
            }
            health['status'] = 'degraded'
        
        # Test end-to-end prediction
        try:
            if health['components']['feature_pipeline']['status'] == 'healthy':
                test_prediction = self.predict(test_data)
                if test_prediction.error:
                    raise Exception(test_prediction.error)
                
                health['components']['end_to_end'] = {
                    'status': 'healthy',
                    'test_prediction_time': test_prediction.inference_time,
                    'test_model_used': test_prediction.model_name
                }
        except Exception as e:
            health['components']['end_to_end'] = {
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
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features using feature pipeline"""
        try:
            return self.feature_pipeline.transform(data, use_cache=True)
        except Exception as e:
            raise FeatureExtractionError(f"Feature extraction failed: {e}")
    
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
        # This would be implemented based on feature pipeline cache tracking
        pipeline_stats = self.feature_pipeline.get_performance_stats()
        return pipeline_stats.get('cache_hits', 0) > pipeline_stats.get('cache_misses', 1)
    
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
        times = self._model_inference_times.get(model_name, [])
        return sum(times) / len(times) if times else 0.0