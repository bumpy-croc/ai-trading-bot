"""
Core prediction engine implementation.

This module provides the main PredictionEngine class that serves as the orchestrator
and facade, connecting configuration, feature engineering, model integration, and
providing a unified prediction interface for strategies.
"""
import time
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from .config import PredictionConfig
from .features.pipeline import FeaturePipeline
from .models.registry import PredictionModelRegistry
from .models.onnx_runner import ModelPrediction
from .exceptions import (
    PredictionEngineError, InvalidInputError, ModelNotFoundError,
    FeatureExtractionError, PredictionTimeoutError, InsufficientDataError,
    ModelInferenceError
)


@dataclass
class PredictionResult:
    """Result of a prediction engine operation."""
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
    """
    Main prediction engine facade that orchestrates all components.
    
    This class serves as the main entry point for making predictions,
    coordinating the feature pipeline, model registry, and providing
    a unified interface for strategies.
    """
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        """
        Initialize prediction engine with configuration.
        
        Args:
            config: PredictionConfig instance, or None to load from ConfigManager
        """
        # Load and validate configuration
        self.config = config or PredictionConfig.from_config_manager()
        try:
            self.config.validate()
        except Exception as e:
            raise PredictionEngineError(f"Invalid configuration: {e}")
        
        # Initialize components
        try:
            self.feature_pipeline = FeaturePipeline(
                enable_sentiment=self.config.enable_sentiment,
                enable_market_microstructure=self.config.enable_market_microstructure,
                cache_ttl=self.config.feature_cache_ttl
            )
            
            self.model_registry = PredictionModelRegistry(self.config)
        except Exception as e:
            raise PredictionEngineError(f"Failed to initialize prediction engine: {e}")
        
        # Performance tracking
        self._prediction_count = 0
        self._total_inference_time = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        self._last_prediction_time = 0.0
        self._feature_extraction_time = 0.0
        
        # Engine state
        self._initialized = True
    
    def predict(self, data: pd.DataFrame, model_name: Optional[str] = None) -> PredictionResult:
        """
        Main prediction method - unified interface for all predictions.
        
        Args:
            data: OHLCV DataFrame with market data
            model_name: Optional specific model to use, defaults to default model
            
        Returns:
            PredictionResult with prediction and metadata
            
        Raises:
            PredictionEngineError: If prediction fails
        """
        start_time = time.time()
        prediction_start = start_time
        
        try:
            # Validate input data
            self._validate_input_data(data)
            
            # Extract features
            feature_start = time.time()
            features = self._extract_features(data)
            self._feature_extraction_time = time.time() - feature_start
            
            # Check for cache hit in feature pipeline
            cache_hit = self._was_cache_hit()
            if cache_hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
            
            # Get model for prediction
            model = self._get_model(model_name)
            
            # Make prediction with timeout
            model_start = time.time()
            if self.config.max_prediction_latency > 0:
                # Implement basic timeout check
                prediction = model.predict(features)
                model_time = time.time() - model_start
                
                if model_time > self.config.max_prediction_latency:
                    raise PredictionTimeoutError(
                        timeout_seconds=self.config.max_prediction_latency,
                        actual_time=model_time
                    )
            else:
                prediction = model.predict(features)
                model_time = time.time() - model_start
            
            # Calculate total inference time
            total_inference_time = time.time() - start_time
            
            # Create unified result
            result = PredictionResult(
                price=prediction.price,
                confidence=prediction.confidence,
                direction=prediction.direction,
                model_name=prediction.model_name,
                timestamp=datetime.now(timezone.utc),
                inference_time=total_inference_time,
                features_used=features.shape[1] if hasattr(features, 'shape') else 0,
                cache_hit=cache_hit,
                metadata={
                    'data_length': len(data),
                    'feature_extraction_time': self._feature_extraction_time,
                    'model_inference_time': model_time,
                    'config_version': self._get_config_version(),
                    'model_metadata': prediction.model_metadata
                }
            )
            
            # Update performance statistics
            self._update_performance_stats(result)
            
            return result
            
        except (InvalidInputError, ModelNotFoundError, FeatureExtractionError, 
                PredictionTimeoutError, ModelInferenceError) as e:
            # Re-raise known prediction errors
            inference_time = time.time() - start_time
            self._last_prediction_time = inference_time
            raise e
            
        except Exception as e:
            # Handle unexpected errors
            inference_time = time.time() - start_time
            self._last_prediction_time = inference_time
            
            error_result = PredictionResult(
                price=0.0,
                confidence=0.0,
                direction=0,
                model_name=model_name or "unknown",
                timestamp=datetime.now(timezone.utc),
                inference_time=inference_time,
                features_used=0,
                error=str(e),
                metadata={
                    'error_type': type(e).__name__,
                    'prediction_failed': True
                }
            )
            
            # Still update stats for failed predictions
            self._prediction_count += 1
            self._total_inference_time += inference_time
            
            return error_result
    
    def predict_batch(self, data_batches: List[pd.DataFrame], 
                     model_name: Optional[str] = None) -> List[PredictionResult]:
        """
        Batch prediction for multiple data sets.
        
        Args:
            data_batches: List of OHLCV DataFrames
            model_name: Optional specific model to use
            
        Returns:
            List of PredictionResult objects
        """
        if not data_batches:
            return []
        
        results = []
        batch_start = time.time()
        
        # Process each batch
        for i, data in enumerate(data_batches):
            try:
                result = self.predict(data, model_name)
                result.metadata['batch_index'] = i
                result.metadata['batch_size'] = len(data_batches)
                results.append(result)
            except Exception as e:
                # Create error result for failed batch item
                error_result = PredictionResult(
                    price=0.0,
                    confidence=0.0,
                    direction=0,
                    model_name=model_name or "unknown",
                    timestamp=datetime.now(timezone.utc),
                    inference_time=0.0,
                    features_used=0,
                    error=str(e),
                    metadata={
                        'batch_index': i,
                        'batch_size': len(data_batches),
                        'batch_error': True,
                        'error_type': type(e).__name__
                    }
                )
                results.append(error_result)
        
        # Update batch metadata
        batch_time = time.time() - batch_start
        for result in results:
            result.metadata['total_batch_time'] = batch_time
            result.metadata['avg_batch_time'] = batch_time / len(data_batches)
        
        return results
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return self.model_registry.list_models()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        # Get metadata from registry
        metadata = self.model_registry.get_model_metadata(model_name)
        if not metadata:
            return {}
        
        # Try to get runtime info if model is loaded
        model = self.model_registry.get_model(model_name)
        runtime_info = {}
        if model:
            runtime_info = model.get_model_info()
        
        return {
            'name': model_name,
            'metadata': metadata,
            'loaded': model is not None,
            'runtime_info': runtime_info,
            'inference_time_avg': self._get_model_avg_inference_time(model_name)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        cache_requests = self._cache_hits + self._cache_misses
        
        return {
            'total_predictions': self._prediction_count,
            'avg_inference_time': (
                self._total_inference_time / self._prediction_count 
                if self._prediction_count > 0 else 0.0
            ),
            'last_prediction_time': self._last_prediction_time,
            'avg_feature_extraction_time': self._feature_extraction_time,
            'cache_hit_rate': (
                self._cache_hits / cache_requests 
                if cache_requests > 0 else 0.0
            ),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'available_models': len(self.get_available_models()),
            'feature_pipeline_stats': self.feature_pipeline.get_cache_stats(),
            'model_registry_stats': self.model_registry.get_registry_stats()
        }
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        self.feature_pipeline.clear_cache()
        self.model_registry.clear_cache()
        
        # Reset performance stats related to cache
        self._cache_hits = 0
        self._cache_misses = 0
    
    def reload_models(self) -> None:
        """Reload all models."""
        self.model_registry.reload_models()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of all components.
        
        Returns:
            Dictionary with health status of all components
        """
        health = {
            'status': 'healthy',
            'components': {},
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'engine_initialized': self._initialized
        }
        
        # Check feature pipeline
        try:
            # Test with minimal data
            test_data = pd.DataFrame({
                'open': [100.0, 101.0, 102.0] * 50,  # Ensure enough data
                'high': [102.0, 103.0, 104.0] * 50,
                'low': [99.0, 100.0, 101.0] * 50,
                'close': [101.0, 102.0, 103.0] * 50,
                'volume': [1000, 1100, 1200] * 50
            })
            
            features = self.feature_pipeline.transform(test_data, use_cache=False)
            health['components']['feature_pipeline'] = {
                'status': 'healthy',
                'test_features_count': features.shape[1] if hasattr(features, 'shape') else 0,
                'cache_stats': self.feature_pipeline.get_cache_stats()
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
            registry_health = self.model_registry.health_check()
            
            health['components']['model_registry'] = {
                'status': registry_health['status'],
                'available_models': len(models),
                'default_model': default_model.model_name if default_model else None,
                'registry_health': registry_health,
                'registry_stats': self.model_registry.get_registry_stats()
            }
            
            if registry_health['status'] != 'healthy':
                health['status'] = 'degraded'
        except Exception as e:
            health['components']['model_registry'] = {
                'status': 'error',
                'error': str(e)
            }
            health['status'] = 'degraded'
        
        # Test end-to-end prediction if components are healthy
        if health['status'] == 'healthy':
            try:
                test_result = self.predict(test_data)
                health['components']['end_to_end'] = {
                    'status': 'healthy' if not test_result.error else 'error',
                    'test_prediction': {
                        'price': test_result.price,
                        'confidence': test_result.confidence,
                        'direction': test_result.direction,
                        'inference_time': test_result.inference_time,
                        'error': test_result.error
                    }
                }
                
                if test_result.error:
                    health['status'] = 'degraded'
                    
            except Exception as e:
                health['components']['end_to_end'] = {
                    'status': 'error',
                    'error': str(e)
                }
                health['status'] = 'degraded'
        
        return health
    
    # Private methods
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data has required columns and sufficient length."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise InvalidInputError(
                f"Missing required columns: {missing_columns}",
                invalid_columns=missing_columns
            )
        
        if len(data) < 120:  # Minimum for LSTM sequence
            raise InsufficientDataError(
                required_rows=120,
                actual_rows=len(data),
                data_type="market data"
            )
        
        # Check for null values
        null_columns = [col for col in required_columns if data[col].isnull().any()]
        if null_columns:
            raise InvalidInputError(
                f"Columns contain null values: {null_columns}",
                invalid_columns=null_columns
            )
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features using feature pipeline."""
        try:
            return self.feature_pipeline.transform(data, use_cache=True)
        except Exception as e:
            raise FeatureExtractionError(f"Feature extraction failed: {e}")
    
    def _get_model(self, model_name: Optional[str]):
        """Get model for prediction."""
        if model_name:
            model = self.model_registry.get_model(model_name)
            if not model:
                available_models = self.model_registry.list_models()
                raise ModelNotFoundError(model_name, available_models)
            return model
        
        # Use default model
        default_model = self.model_registry.get_default_model()
        if not default_model:
            available_models = self.model_registry.list_models()
            if not available_models:
                raise ModelNotFoundError("No models available in registry")
            else:
                raise ModelNotFoundError(
                    "No default model set", 
                    available_models=available_models
                )
        
        return default_model
    
    def _was_cache_hit(self) -> bool:
        """Check if last feature extraction was a cache hit."""
        # This is a simplified implementation
        # In a real implementation, this would track cache hits more accurately
        stats = self.feature_pipeline.get_cache_stats()
        return stats.get('hit_rate', 0) > 0
    
    def _get_config_version(self) -> str:
        """Get configuration version/hash for tracking."""
        config_str = str(self.config)
        return f"v1.0-{hash(config_str) % 10000:04d}"
    
    def _update_performance_stats(self, result: PredictionResult) -> None:
        """Update internal performance statistics."""
        self._prediction_count += 1
        self._total_inference_time += result.inference_time
        self._last_prediction_time = result.inference_time
    
    def _get_model_avg_inference_time(self, model_name: str) -> float:
        """Get average inference time for specific model."""
        # This would need to be tracked per model in a real implementation
        # For now, return the overall average
        return (
            self._total_inference_time / self._prediction_count 
            if self._prediction_count > 0 else 0.0
        )