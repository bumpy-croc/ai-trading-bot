"""
Prediction Engine Package

This package contains the prediction engine components including:
- Configuration management
- Feature engineering pipeline  
- Model registry and inference
- Core prediction engine
- Exception handling

The main entry point is the PredictionEngine class which provides a unified
interface for making ML predictions in trading strategies.

Example usage:
    from prediction import PredictionEngine, PredictionConfig
    
    # Create engine with default configuration
    engine = PredictionEngine()
    
    # Make a prediction
    result = engine.predict(ohlcv_data)
    print(f"Predicted price: {result.price}, confidence: {result.confidence}")
    
    # Get engine health status
    health = engine.health_check()
    print(f"Engine status: {health['status']}")
"""

# Core engine components
from .engine import PredictionEngine, PredictionResult
from .config import PredictionConfig

# Feature pipeline
from .features.pipeline import FeaturePipeline

# Model components
from .models.registry import PredictionModelRegistry
from .models.onnx_runner import OnnxRunner, ModelPrediction

# Exception classes
from .exceptions import (
    PredictionEngineError,
    InvalidInputError,
    ModelNotFoundError,
    ModelLoadError,
    FeatureExtractionError,
    PredictionTimeoutError,
    InsufficientDataError,
    ConfigurationError,
    ModelInferenceError,
    CacheError,
    ValidationError
)

# Public API
__all__ = [
    # Main engine
    'PredictionEngine',
    'PredictionResult',
    'PredictionConfig',
    
    # Components
    'FeaturePipeline',
    'PredictionModelRegistry',
    'OnnxRunner',
    'ModelPrediction',
    
    # Exceptions
    'PredictionEngineError',
    'InvalidInputError',
    'ModelNotFoundError',
    'ModelLoadError',
    'FeatureExtractionError',
    'PredictionTimeoutError',
    'InsufficientDataError',
    'ConfigurationError',
    'ModelInferenceError',
    'CacheError',
    'ValidationError'
]


# Convenience factory functions
def create_engine(config=None, **kwargs):
    """
    Create a PredictionEngine with optional configuration overrides.
    
    Args:
        config: Optional PredictionConfig instance
        **kwargs: Configuration overrides (e.g., enable_sentiment=True)
        
    Returns:
        PredictionEngine instance
        
    Example:
        engine = create_engine(enable_sentiment=True, model_cache_ttl=1200)
    """
    if config is None:
        config = PredictionConfig.from_config_manager()
    
    # Apply any configuration overrides
    if kwargs:
        config_dict = {
            'prediction_horizons': config.prediction_horizons,
            'min_confidence_threshold': config.min_confidence_threshold,
            'max_prediction_latency': config.max_prediction_latency,
            'model_registry_path': config.model_registry_path,
            'enable_sentiment': config.enable_sentiment,
            'enable_market_microstructure': config.enable_market_microstructure,
            'feature_cache_ttl': config.feature_cache_ttl,
            'model_cache_ttl': config.model_cache_ttl
        }
        
        # Update with provided overrides
        config_dict.update(kwargs)
        
        # Create new config with overrides
        config = PredictionConfig(**config_dict)
    
    return PredictionEngine(config)


def quick_predict(data, model_name=None, **engine_kwargs):
    """
    Quick prediction function for one-off predictions.
    
    Args:
        data: OHLCV DataFrame
        model_name: Optional model name
        **engine_kwargs: Configuration overrides for engine
        
    Returns:
        PredictionResult
        
    Example:
        result = quick_predict(ohlcv_data, model_name="btc_price_v2")
    """
    engine = create_engine(**engine_kwargs)
    return engine.predict(data, model_name)


def get_available_models(registry_path=None):
    """
    Get list of available models without creating a full engine.
    
    Args:
        registry_path: Optional path to model registry
        
    Returns:
        List of model names
    """
    if registry_path:
        from .config import PredictionConfig
        config = PredictionConfig.from_config_manager()
        config.model_registry_path = registry_path
        registry = PredictionModelRegistry(config)
    else:
        registry = PredictionModelRegistry()
    
    return registry.list_models()


# Version information
__version__ = "1.0.0"
__author__ = "AI Trading Bot Team"
__description__ = "Unified prediction engine for ML-based trading strategies"