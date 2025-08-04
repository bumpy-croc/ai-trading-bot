"""
Tests for the core PredictionEngine class.

This module contains comprehensive tests for the main prediction engine
functionality including prediction, batch processing, health monitoring,
and error handling.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from prediction import (
    PredictionEngine, PredictionResult, PredictionConfig,
    PredictionEngineError, InvalidInputError, ModelNotFoundError,
    FeatureExtractionError, PredictionTimeoutError, InsufficientDataError
)
from prediction.features.pipeline import FeaturePipeline
from prediction.models.registry import PredictionModelRegistry
from prediction.models.onnx_runner import ModelPrediction


class TestPredictionEngine:
    """Test suite for PredictionEngine class."""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        n_rows = 150  # Ensure enough data for sequence length
        np.random.seed(42)  # For reproducible tests
        
        dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='1H')
        
        # Generate realistic OHLCV data
        close_prices = 50000 + np.cumsum(np.random.randn(n_rows) * 100)
        high_prices = close_prices + np.random.uniform(0, 500, n_rows)
        low_prices = close_prices - np.random.uniform(0, 500, n_rows)
        open_prices = close_prices + np.random.uniform(-200, 200, n_rows)
        volumes = np.random.uniform(100, 10000, n_rows)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = Mock(spec=PredictionConfig)
        config.prediction_horizons = [1]
        config.min_confidence_threshold = 0.6
        config.max_prediction_latency = 0.1
        config.model_registry_path = "src/ml"
        config.enable_sentiment = False
        config.enable_market_microstructure = False
        config.feature_cache_ttl = 300
        config.model_cache_ttl = 600
        config.validate = Mock()
        return config
    
    @pytest.fixture
    def mock_model_prediction(self):
        """Create a mock model prediction for testing."""
        return ModelPrediction(
            price=51000.0,
            confidence=0.8,
            direction=1,
            model_name="test_model",
            inference_time=0.05,
            model_metadata={'test': 'metadata'}
        )
    
    def test_engine_initialization_success(self, mock_config):
        """Test successful engine initialization."""
        with patch('prediction.engine.PredictionConfig.from_config_manager', return_value=mock_config), \
             patch('prediction.engine.FeaturePipeline') as mock_pipeline, \
             patch('prediction.engine.PredictionModelRegistry') as mock_registry:
            
            engine = PredictionEngine(mock_config)
            
            assert engine.config == mock_config
            assert engine._initialized is True
            mock_config.validate.assert_called_once()
            mock_pipeline.assert_called_once()
            mock_registry.assert_called_once()
    
    def test_engine_initialization_with_invalid_config(self):
        """Test engine initialization with invalid configuration."""
        invalid_config = Mock(spec=PredictionConfig)
        invalid_config.validate.side_effect = ValueError("Invalid config")
        
        with pytest.raises(PredictionEngineError, match="Invalid configuration"):
            PredictionEngine(invalid_config)
    
    def test_predict_success(self, sample_ohlcv_data, mock_config, mock_model_prediction):
        """Test successful prediction."""
        with patch('prediction.engine.PredictionConfig.from_config_manager', return_value=mock_config), \
             patch('prediction.engine.FeaturePipeline') as mock_pipeline_class, \
             patch('prediction.engine.PredictionModelRegistry') as mock_registry_class:
            
            # Setup mocks
            mock_pipeline = Mock()
            mock_pipeline.transform.return_value = sample_ohlcv_data  # Features
            mock_pipeline.get_cache_stats.return_value = {'hit_rate': 0.5}
            mock_pipeline_class.return_value = mock_pipeline
            
            mock_model = Mock()
            mock_model.predict.return_value = mock_model_prediction
            
            mock_registry = Mock()
            mock_registry.get_default_model.return_value = mock_model
            mock_registry_class.return_value = mock_registry
            
            # Create engine and make prediction
            engine = PredictionEngine(mock_config)
            result = engine.predict(sample_ohlcv_data)
            
            # Assertions
            assert isinstance(result, PredictionResult)
            assert result.price == 51000.0
            assert result.confidence == 0.8
            assert result.direction == 1
            assert result.model_name == "test_model"
            assert result.error is None
            assert result.features_used == len(sample_ohlcv_data.columns)
            assert 'data_length' in result.metadata
            assert 'feature_extraction_time' in result.metadata
            
            # Verify method calls
            mock_pipeline.transform.assert_called_once_with(sample_ohlcv_data, use_cache=True)
            mock_model.predict.assert_called_once()
    
    def test_predict_with_invalid_input(self, mock_config):
        """Test prediction with invalid input data."""
        with patch('prediction.engine.PredictionConfig.from_config_manager', return_value=mock_config), \
             patch('prediction.engine.FeaturePipeline'), \
             patch('prediction.engine.PredictionModelRegistry'):
            
            engine = PredictionEngine(mock_config)
            
            # Test with missing columns
            invalid_data = pd.DataFrame({'price': [100, 101, 102]})
            
            with pytest.raises(InvalidInputError, match="Missing required columns"):
                engine.predict(invalid_data)
    
    def test_predict_with_insufficient_data(self, mock_config):
        """Test prediction with insufficient data."""
        with patch('prediction.engine.PredictionConfig.from_config_manager', return_value=mock_config), \
             patch('prediction.engine.FeaturePipeline'), \
             patch('prediction.engine.PredictionModelRegistry'):
            
            engine = PredictionEngine(mock_config)
            
            # Create data with too few rows
            insufficient_data = pd.DataFrame({
                'open': [100, 101],
                'high': [102, 103],
                'low': [99, 100],
                'close': [101, 102],
                'volume': [1000, 1100]
            })
            
            with pytest.raises(InsufficientDataError, match="Insufficient market data"):
                engine.predict(insufficient_data)
    
    def test_predict_with_model_not_found(self, sample_ohlcv_data, mock_config):
        """Test prediction when requested model is not found."""
        with patch('prediction.engine.PredictionConfig.from_config_manager', return_value=mock_config), \
             patch('prediction.engine.FeaturePipeline') as mock_pipeline_class, \
             patch('prediction.engine.PredictionModelRegistry') as mock_registry_class:
            
            # Setup mocks
            mock_pipeline = Mock()
            mock_pipeline.transform.return_value = sample_ohlcv_data
            mock_pipeline_class.return_value = mock_pipeline
            
            mock_registry = Mock()
            mock_registry.get_model.return_value = None
            mock_registry.list_models.return_value = ['model1', 'model2']
            mock_registry_class.return_value = mock_registry
            
            engine = PredictionEngine(mock_config)
            
            with pytest.raises(ModelNotFoundError, match="Model 'nonexistent' not found"):
                engine.predict(sample_ohlcv_data, model_name="nonexistent")
    
    def test_predict_with_timeout(self, sample_ohlcv_data, mock_config, mock_model_prediction):
        """Test prediction timeout handling."""
        mock_config.max_prediction_latency = 0.001  # Very short timeout
        
        with patch('prediction.engine.PredictionConfig.from_config_manager', return_value=mock_config), \
             patch('prediction.engine.FeaturePipeline') as mock_pipeline_class, \
             patch('prediction.engine.PredictionModelRegistry') as mock_registry_class:
            
            # Setup mocks
            mock_pipeline = Mock()
            mock_pipeline.transform.return_value = sample_ohlcv_data
            mock_pipeline.get_cache_stats.return_value = {'hit_rate': 0}
            mock_pipeline_class.return_value = mock_pipeline
            
            # Mock slow model
            mock_model = Mock()
            def slow_predict(features):
                import time
                time.sleep(0.01)  # Longer than timeout
                return mock_model_prediction
            mock_model.predict.side_effect = slow_predict
            
            mock_registry = Mock()
            mock_registry.get_default_model.return_value = mock_model
            mock_registry_class.return_value = mock_registry
            
            engine = PredictionEngine(mock_config)
            
            with pytest.raises(PredictionTimeoutError):
                engine.predict(sample_ohlcv_data)
    
    def test_predict_batch_success(self, sample_ohlcv_data, mock_config, mock_model_prediction):
        """Test successful batch prediction."""
        with patch('prediction.engine.PredictionConfig.from_config_manager', return_value=mock_config), \
             patch('prediction.engine.FeaturePipeline') as mock_pipeline_class, \
             patch('prediction.engine.PredictionModelRegistry') as mock_registry_class:
            
            # Setup mocks
            mock_pipeline = Mock()
            mock_pipeline.transform.return_value = sample_ohlcv_data
            mock_pipeline.get_cache_stats.return_value = {'hit_rate': 0.5}
            mock_pipeline_class.return_value = mock_pipeline
            
            mock_model = Mock()
            mock_model.predict.return_value = mock_model_prediction
            
            mock_registry = Mock()
            mock_registry.get_default_model.return_value = mock_model
            mock_registry_class.return_value = mock_registry
            
            engine = PredictionEngine(mock_config)
            
            # Create batch data
            batch_data = [sample_ohlcv_data, sample_ohlcv_data.copy()]
            results = engine.predict_batch(batch_data)
            
            assert len(results) == 2
            for i, result in enumerate(results):
                assert isinstance(result, PredictionResult)
                assert result.metadata['batch_index'] == i
                assert result.metadata['batch_size'] == 2
                assert 'total_batch_time' in result.metadata
    
    def test_predict_batch_empty(self, mock_config):
        """Test batch prediction with empty input."""
        with patch('prediction.engine.PredictionConfig.from_config_manager', return_value=mock_config), \
             patch('prediction.engine.FeaturePipeline'), \
             patch('prediction.engine.PredictionModelRegistry'):
            
            engine = PredictionEngine(mock_config)
            results = engine.predict_batch([])
            
            assert results == []
    
    def test_get_available_models(self, mock_config):
        """Test getting available models."""
        with patch('prediction.engine.PredictionConfig.from_config_manager', return_value=mock_config), \
             patch('prediction.engine.FeaturePipeline'), \
             patch('prediction.engine.PredictionModelRegistry') as mock_registry_class:
            
            mock_registry = Mock()
            mock_registry.list_models.return_value = ['model1', 'model2', 'model3']
            mock_registry_class.return_value = mock_registry
            
            engine = PredictionEngine(mock_config)
            models = engine.get_available_models()
            
            assert models == ['model1', 'model2', 'model3']
            mock_registry.list_models.assert_called_once()
    
    def test_get_model_info(self, mock_config):
        """Test getting model information."""
        with patch('prediction.engine.PredictionConfig.from_config_manager', return_value=mock_config), \
             patch('prediction.engine.FeaturePipeline'), \
             patch('prediction.engine.PredictionModelRegistry') as mock_registry_class:
            
            mock_model = Mock()
            mock_model.get_model_info.return_value = {'runtime': 'info'}
            
            mock_registry = Mock()
            mock_registry.get_model_metadata.return_value = {'metadata': 'info'}
            mock_registry.get_model.return_value = mock_model
            mock_registry_class.return_value = mock_registry
            
            engine = PredictionEngine(mock_config)
            info = engine.get_model_info('test_model')
            
            assert info['name'] == 'test_model'
            assert info['metadata'] == {'metadata': 'info'}
            assert info['loaded'] is True
            assert info['runtime_info'] == {'runtime': 'info'}
    
    def test_get_performance_stats(self, sample_ohlcv_data, mock_config, mock_model_prediction):
        """Test getting performance statistics."""
        with patch('prediction.engine.PredictionConfig.from_config_manager', return_value=mock_config), \
             patch('prediction.engine.FeaturePipeline') as mock_pipeline_class, \
             patch('prediction.engine.PredictionModelRegistry') as mock_registry_class:
            
            # Setup mocks
            mock_pipeline = Mock()
            mock_pipeline.transform.return_value = sample_ohlcv_data
            mock_pipeline.get_cache_stats.return_value = {'cache': 'stats'}
            mock_pipeline_class.return_value = mock_pipeline
            
            mock_model = Mock()
            mock_model.predict.return_value = mock_model_prediction
            
            mock_registry = Mock()
            mock_registry.get_default_model.return_value = mock_model
            mock_registry.get_registry_stats.return_value = {'registry': 'stats'}
            mock_registry_class.return_value = mock_registry
            
            engine = PredictionEngine(mock_config)
            
            # Make a prediction to generate stats
            engine.predict(sample_ohlcv_data)
            
            stats = engine.get_performance_stats()
            
            assert stats['total_predictions'] == 1
            assert 'avg_inference_time' in stats
            assert 'cache_hit_rate' in stats
            assert 'feature_pipeline_stats' in stats
            assert 'model_registry_stats' in stats
    
    def test_clear_caches(self, mock_config):
        """Test clearing all caches."""
        with patch('prediction.engine.PredictionConfig.from_config_manager', return_value=mock_config), \
             patch('prediction.engine.FeaturePipeline') as mock_pipeline_class, \
             patch('prediction.engine.PredictionModelRegistry') as mock_registry_class:
            
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry
            
            engine = PredictionEngine(mock_config)
            engine.clear_caches()
            
            mock_pipeline.clear_cache.assert_called_once()
            mock_registry.clear_cache.assert_called_once()
    
    def test_reload_models(self, mock_config):
        """Test reloading models."""
        with patch('prediction.engine.PredictionConfig.from_config_manager', return_value=mock_config), \
             patch('prediction.engine.FeaturePipeline'), \
             patch('prediction.engine.PredictionModelRegistry') as mock_registry_class:
            
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry
            
            engine = PredictionEngine(mock_config)
            engine.reload_models()
            
            mock_registry.reload_models.assert_called_once()
    
    def test_health_check_healthy(self, mock_config):
        """Test health check when all components are healthy."""
        with patch('prediction.engine.PredictionConfig.from_config_manager', return_value=mock_config), \
             patch('prediction.engine.FeaturePipeline') as mock_pipeline_class, \
             patch('prediction.engine.PredictionModelRegistry') as mock_registry_class:
            
            # Setup healthy mocks
            mock_pipeline = Mock()
            mock_pipeline.transform.return_value = pd.DataFrame({'test': [1, 2, 3]})
            mock_pipeline.get_cache_stats.return_value = {'cache': 'stats'}
            mock_pipeline_class.return_value = mock_pipeline
            
            mock_model = Mock()
            mock_model.model_name = 'test_model'
            mock_model.predict.return_value = ModelPrediction(
                price=100.0, confidence=0.8, direction=1, 
                model_name='test', inference_time=0.01, model_metadata={}
            )
            
            mock_registry = Mock()
            mock_registry.list_models.return_value = ['model1']
            mock_registry.get_default_model.return_value = mock_model
            mock_registry.health_check.return_value = {'status': 'healthy'}
            mock_registry.get_registry_stats.return_value = {'registry': 'stats'}
            mock_registry_class.return_value = mock_registry
            
            engine = PredictionEngine(mock_config)
            health = engine.health_check()
            
            assert health['status'] == 'healthy'
            assert health['engine_initialized'] is True
            assert 'feature_pipeline' in health['components']
            assert 'model_registry' in health['components']
            assert 'end_to_end' in health['components']
    
    def test_health_check_degraded(self, mock_config):
        """Test health check when components are degraded."""
        with patch('prediction.engine.PredictionConfig.from_config_manager', return_value=mock_config), \
             patch('prediction.engine.FeaturePipeline') as mock_pipeline_class, \
             patch('prediction.engine.PredictionModelRegistry') as mock_registry_class:
            
            # Setup degraded mocks
            mock_pipeline = Mock()
            mock_pipeline.transform.side_effect = Exception("Feature extraction failed")
            mock_pipeline_class.return_value = mock_pipeline
            
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry
            
            engine = PredictionEngine(mock_config)
            health = engine.health_check()
            
            assert health['status'] == 'degraded'
            assert health['components']['feature_pipeline']['status'] == 'error'
    
    def test_error_handling_in_predict(self, sample_ohlcv_data, mock_config):
        """Test error handling in predict method."""
        with patch('prediction.engine.PredictionConfig.from_config_manager', return_value=mock_config), \
             patch('prediction.engine.FeaturePipeline') as mock_pipeline_class, \
             patch('prediction.engine.PredictionModelRegistry') as mock_registry_class:
            
            # Setup mocks to raise exception
            mock_pipeline = Mock()
            mock_pipeline.transform.side_effect = Exception("Unexpected error")
            mock_pipeline_class.return_value = mock_pipeline
            
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry
            
            engine = PredictionEngine(mock_config)
            result = engine.predict(sample_ohlcv_data)
            
            # Should return error result instead of raising
            assert isinstance(result, PredictionResult)
            assert result.error == "Unexpected error"
            assert result.price == 0.0
            assert result.confidence == 0.0
            assert result.direction == 0
            assert result.metadata['prediction_failed'] is True