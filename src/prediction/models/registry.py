"""
Model registry for managing prediction models.

This module provides a registry for loading, managing, and accessing
multiple ML models in a centralized way.
"""
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .onnx_runner import OnnxRunner, ModelPrediction
from config.constants import DEFAULT_MODEL_REGISTRY_PATH


# Normalization functions for different model types
def _minmax_price_normalize(df: pd.DataFrame, seq_len: int) -> Tuple[pd.DataFrame, List[str]]:
    """Min-max normalization for price-based models."""
    df = df.copy()
    
    # Basic price features
    price_features = ['open', 'high', 'low', 'close']
    for feature in price_features:
        if feature in df.columns:
            feature_values = df[feature].dropna()
            if len(feature_values) > 0:
                min_val = feature_values.min()
                max_val = feature_values.max()
                if max_val != min_val:
                    df[f'{feature}_normalized'] = (df[feature] - min_val) / (max_val - min_val)
                else:
                    df[f'{feature}_normalized'] = 0.5
    
    # Volume normalization
    if 'volume' in df.columns:
        volume_values = df['volume'].dropna()
        if len(volume_values) > 0:
            min_vol = volume_values.min()
            max_vol = volume_values.max()
            if max_vol != min_vol:
                df['volume_normalized'] = (df['volume'] - min_vol) / (max_vol - min_vol)
            else:
                df['volume_normalized'] = 0.5
    
    # Fill NaN values
    df = df.ffill().fillna(0)
    
    feature_names = ['open_normalized', 'high_normalized', 'low_normalized', 
                    'close_normalized', 'volume_normalized']
    
    return df, feature_names


def _log_return_normalize(df: pd.DataFrame, seq_len: int) -> Tuple[pd.DataFrame, List[str]]:
    """Log return normalization for advanced models."""
    EPSILON = 1e-8
    df = df.copy()
    
    # Calculate log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    
    # High-low range as percentage of close
    df['hl_range'] = ((df['high'] - df['low']) / (df['close'] + EPSILON)).fillna(0)
    
    # Volume z-score over rolling window
    rolling_mean = df['volume'].rolling(window=20, min_periods=1).mean()
    rolling_std = df['volume'].rolling(window=20, min_periods=1).std()
    df['volume_z'] = ((df['volume'] - rolling_mean) / 
                     (rolling_std + EPSILON)).fillna(0.0)
    
    return df, ["log_return", "hl_range", "volume_z"]


class PredictionModelRegistry:
    """
    Registry for managing prediction models.
    
    This class provides a centralized way to load, manage, and access
    multiple ML models for prediction. It handles model discovery,
    loading, caching, and provides a unified interface.
    """
    
    def __init__(self, config=None):
        """
        Initialize the model registry.
        
        Args:
            config: PredictionConfig instance with registry settings
        """
        self.config = config
        self.registry_path = config.model_registry_path if config else DEFAULT_MODEL_REGISTRY_PATH
        self.cache_ttl = config.model_cache_ttl if config else 600
        
        # Model storage: {name: (runner, last_accessed_time)}
        self._models: Dict[str, tuple] = {}
        
        # Model metadata cache
        self._model_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Default model name
        self._default_model: Optional[str] = None
        
        # Initialize registry
        self._discover_models()
    
    def _discover_models(self) -> None:
        """Discover and register available models."""
        registry_path = Path(self.registry_path)
        
        if not registry_path.exists():
            # Create directory if it doesn't exist
            registry_path.mkdir(parents=True, exist_ok=True)
            return
        
        # Look for ONNX files in the registry path
        model_files = list(registry_path.glob("*.onnx"))
        
        # Register known models with their configurations
        self._register_known_models(registry_path)
        
        # Auto-discover other ONNX files
        for model_file in model_files:
            model_name = model_file.stem
            if model_name not in self._model_metadata:
                self._register_generic_model(model_file)
    
    def _register_known_models(self, registry_path: Path) -> None:
        """Register known models with their specific configurations."""
        # Register BTC price model (original)
        btc_price_path = registry_path / "btcusdt_price.onnx"
        if btc_price_path.exists():
            self._model_metadata["btc_price_minmax"] = {
                'path': str(btc_price_path),
                'normalization_fn': _minmax_price_normalize,
                'expected_features': 5,
                'sequence_length': 120,
                'model_type': 'price_prediction',
                'description': 'BTC price prediction model with min-max normalization',
                'created_date': datetime.fromtimestamp(btc_price_path.stat().st_mtime).isoformat()
            }
            if self._default_model is None:
                self._default_model = "btc_price_minmax"
        
        # Register BTC price model v2 (log returns)
        btc_price_v2_path = registry_path / "btcusdt_price_v2.onnx"
        if btc_price_v2_path.exists():
            self._model_metadata["btc_price_v2"] = {
                'path': str(btc_price_v2_path),
                'normalization_fn': _log_return_normalize,
                'expected_features': 3,
                'sequence_length': 120,
                'model_type': 'price_prediction',
                'description': 'BTC price prediction model v2 with log return normalization',
                'created_date': datetime.fromtimestamp(btc_price_v2_path.stat().st_mtime).isoformat()
            }
            # Prefer v2 if available
            self._default_model = "btc_price_v2"
        
        # Register sentiment model
        btc_sentiment_path = registry_path / "btcusdt_sentiment.onnx"
        if btc_sentiment_path.exists():
            self._model_metadata["btc_sentiment"] = {
                'path': str(btc_sentiment_path),
                'normalization_fn': _minmax_price_normalize,  # Default normalization
                'expected_features': 14,  # Includes sentiment features
                'sequence_length': 120,
                'model_type': 'sentiment_prediction',
                'description': 'BTC prediction model with sentiment analysis',
                'created_date': datetime.fromtimestamp(btc_sentiment_path.stat().st_mtime).isoformat()
            }
    
    def _register_generic_model(self, model_path: Path) -> None:
        """Register a generic ONNX model with default settings."""
        model_name = model_path.stem
        
        self._model_metadata[model_name] = {
            'path': str(model_path),
            'normalization_fn': _minmax_price_normalize,  # Default normalization
            'expected_features': None,  # Auto-detect
            'sequence_length': 120,
            'model_type': 'generic',
            'description': f'Auto-discovered model: {model_name}',
            'created_date': datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
        }
    
    def get_model(self, model_name: str) -> Optional[OnnxRunner]:
        """
        Get a model runner by name.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            OnnxRunner instance or None if not found
        """
        if model_name not in self._model_metadata:
            return None
        
        # Check if model is cached and not expired
        if model_name in self._models:
            runner, last_accessed = self._models[model_name]
            if time.time() - last_accessed < self.cache_ttl:
                # Update access time
                self._models[model_name] = (runner, time.time())
                return runner
            else:
                # Remove expired model
                del self._models[model_name]
        
        # Load the model
        try:
            model_config = self._model_metadata[model_name]
            runner = OnnxRunner(
                model_path=model_config['path'],
                model_name=model_name,
                normalization_fn=model_config['normalization_fn'],
                expected_features=model_config['expected_features'],
                sequence_length=model_config['sequence_length'],
                model_metadata=model_config
            )
            
            # Cache the model
            self._models[model_name] = (runner, time.time())
            
            return runner
            
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            return None
    
    def get_default_model(self) -> Optional[OnnxRunner]:
        """Get the default model runner."""
        if self._default_model:
            return self.get_model(self._default_model)
        
        # If no default set, use the first available model
        available_models = self.list_models()
        if available_models:
            return self.get_model(available_models[0])
        
        return None
    
    def list_models(self) -> List[str]:
        """Get a list of available model names."""
        return list(self._model_metadata.keys())
    
    def get_model_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific model."""
        return self._model_metadata.get(model_name)
    
    def get_all_model_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all registered models."""
        return self._model_metadata.copy()
    
    def set_default_model(self, model_name: str) -> bool:
        """
        Set the default model.
        
        Args:
            model_name: Name of the model to set as default
            
        Returns:
            True if successful, False if model not found
        """
        if model_name in self._model_metadata:
            self._default_model = model_name
            return True
        return False
    
    def reload_models(self) -> None:
        """Reload all models and clear cache."""
        self._models.clear()
        self._model_metadata.clear()
        self._default_model = None
        self._discover_models()
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._models.clear()
    
    def cleanup_expired_models(self) -> int:
        """Remove expired models from cache and return count removed."""
        current_time = time.time()
        expired_models = [
            name for name, (_, last_accessed) in self._models.items()
            if current_time - last_accessed > self.cache_ttl
        ]
        
        for model_name in expired_models:
            del self._models[model_name]
        
        return len(expired_models)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the model registry."""
        current_time = time.time()
        
        # Count loaded models
        loaded_models = len(self._models)
        
        # Count total available models
        total_models = len(self._model_metadata)
        
        # Calculate cache statistics
        recent_models = sum(
            1 for _, last_accessed in self._models.values()
            if current_time - last_accessed < 300  # Last 5 minutes
        )
        
        return {
            'total_models': total_models,
            'loaded_models': loaded_models,
            'recent_models': recent_models,
            'default_model': self._default_model,
            'registry_path': self.registry_path,
            'cache_ttl': self.cache_ttl,
            'available_models': self.list_models()
        }
    
    def warmup_models(self, sample_data: pd.DataFrame, 
                      model_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Warm up models with sample data.
        
        Args:
            sample_data: Sample DataFrame for warmup predictions
            model_names: List of models to warm up, or None for all
            
        Returns:
            Dict mapping model names to success status
        """
        if model_names is None:
            model_names = self.list_models()
        
        results = {}
        for model_name in model_names:
            try:
                model = self.get_model(model_name)
                if model:
                    model.warmup(sample_data)
                    results[model_name] = True
                else:
                    results[model_name] = False
            except Exception:
                results[model_name] = False
        
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the model registry."""
        health = {
            'status': 'healthy',
            'registry_accessible': False,
            'models': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Check if registry path is accessible
        try:
            registry_path = Path(self.registry_path)
            health['registry_accessible'] = registry_path.exists() and registry_path.is_dir()
        except Exception as e:
            health['registry_error'] = str(e)
        
        # Check each model
        for model_name in self.list_models():
            model_health = {'status': 'unknown', 'loadable': False}
            
            try:
                # Try to get model metadata
                metadata = self.get_model_metadata(model_name)
                if metadata:
                    model_health['has_metadata'] = True
                    model_health['model_path'] = metadata['path']
                    
                    # Check if model file exists
                    model_path = Path(metadata['path'])
                    model_health['file_exists'] = model_path.exists()
                    
                    if model_path.exists():
                        # Try to load the model
                        model = self.get_model(model_name)
                        if model:
                            model_health['loadable'] = True
                            model_health['status'] = 'healthy'
                        else:
                            model_health['status'] = 'load_failed'
                    else:
                        model_health['status'] = 'file_missing'
                else:
                    model_health['status'] = 'no_metadata'
            
            except Exception as e:
                model_health['status'] = 'error'
                model_health['error'] = str(e)
            
            health['models'][model_name] = model_health
            
            # Update overall health status
            if model_health['status'] != 'healthy':
                health['status'] = 'degraded'
        
        return health