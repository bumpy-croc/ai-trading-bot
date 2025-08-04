"""
Integration tests for the prediction engine.

These tests verify that the prediction engine components work together
correctly in realistic scenarios.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock

from prediction import PredictionEngine, PredictionConfig, create_engine, quick_predict
from prediction.exceptions import PredictionEngineError


class TestPredictionEngineIntegration:
    """Integration test suite for prediction engine."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for integration testing."""
        n_rows = 200
        np.random.seed(123)
        
        # Generate realistic BTC-like price data
        close_prices = 45000 + np.cumsum(np.random.randn(n_rows) * 50)
        high_prices = close_prices + np.random.uniform(0, 200, n_rows)
        low_prices = close_prices - np.random.uniform(0, 200, n_rows)
        open_prices = close_prices + np.random.uniform(-100, 100, n_rows)
        volumes = np.random.uniform(1000, 50000, n_rows)
        
        return pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
    
    def test_engine_creation_and_basic_functionality(self, sample_data):
        """Test creating engine and basic functionality without real models."""
        # Mock the components to avoid requiring real ONNX models
        with patch('prediction.features.pipeline.TechnicalIndicators') as mock_indicators, \
             patch('prediction.models.registry.Path.exists', return_value=False), \
             patch('prediction.models.registry.Path.mkdir'):
            
            # Mock technical indicators
            mock_indicators.calculate_all.return_value = sample_data.copy()
            
            # Create engine
            engine = create_engine(
                enable_sentiment=False,
                enable_market_microstructure=False,
                model_cache_ttl=600
            )
            
            assert engine is not None
            assert isinstance(engine, PredictionEngine)
            assert engine._initialized is True
    
    def test_config_creation_and_validation(self):
        """Test configuration creation and validation."""
        with patch('prediction.config.get_config') as mock_get_config:
            # Mock config manager
            mock_config_manager = Mock()
            mock_config_manager.get_list.return_value = ['1']
            mock_config_manager.get_float.side_effect = lambda key, default: default
            mock_config_manager.get.side_effect = lambda key, default: default
            mock_config_manager.get_bool.side_effect = lambda key, default: default
            mock_config_manager.get_int.side_effect = lambda key, default: default
            mock_get_config.return_value = mock_config_manager
            
            # Create config
            config = PredictionConfig.from_config_manager()
            
            # Validate
            config.validate()  # Should not raise
            
            assert config.prediction_horizons == [1]
            assert config.min_confidence_threshold == 0.6
            assert config.enable_sentiment is False
    
    def test_feature_pipeline_integration(self, sample_data):
        """Test feature pipeline integration with technical indicators."""
        with patch('prediction.features.pipeline.TechnicalIndicators') as mock_indicators:
            # Mock technical indicators to return data with indicators
            enriched_data = sample_data.copy()
            enriched_data['sma_20'] = enriched_data['close'].rolling(20).mean()
            enriched_data['rsi'] = 50.0  # Simplified RSI
            mock_indicators.calculate_all.return_value = enriched_data
            
            from prediction.features.pipeline import FeaturePipeline
            
            pipeline = FeaturePipeline(
                enable_sentiment=False,
                enable_market_microstructure=False,
                cache_ttl=300
            )
            
            # Transform data
            features = pipeline.transform(sample_data, use_cache=False)
            
            # Verify features were extracted
            assert len(features) == len(sample_data)
            assert 'open_normalized' in features.columns
            assert 'close_normalized' in features.columns
            assert 'volume_normalized' in features.columns
    
    def test_quick_predict_function(self, sample_data):
        """Test the quick_predict convenience function."""
        with patch('prediction.features.pipeline.TechnicalIndicators') as mock_indicators, \
             patch('prediction.models.registry.Path.exists', return_value=False), \
             patch('prediction.models.registry.Path.mkdir'):
            
            # Mock technical indicators
            mock_indicators.calculate_all.return_value = sample_data.copy()
            
            # Test that quick_predict can be called (though it will fail without models)
            try:
                result = quick_predict(sample_data)
                # If we get here, the function structure is working
                # The result may have an error due to no models, which is expected
                assert hasattr(result, 'error') or hasattr(result, 'price')
            except Exception as e:
                # Expected to fail without real models, but shouldn't be an import error
                assert "Model" in str(e) or "not found" in str(e)
    
    def test_error_propagation(self, sample_data):
        """Test that errors are properly propagated through the system."""
        with patch('prediction.features.pipeline.TechnicalIndicators') as mock_indicators, \
             patch('prediction.models.registry.Path.exists', return_value=False):
            
            # Mock indicators to raise an error
            mock_indicators.calculate_all.side_effect = RuntimeError("Indicator calculation failed")
            
            engine = create_engine()
            
            # This should handle the error gracefully
            result = engine.predict(sample_data)
            
            # Should return error result instead of crashing
            assert result.error is not None
            assert "failed" in result.error.lower()
    
    def test_health_check_integration(self):
        """Test health check functionality."""
        with patch('prediction.features.pipeline.TechnicalIndicators') as mock_indicators, \
             patch('prediction.models.registry.Path.exists', return_value=False), \
             patch('prediction.models.registry.Path.mkdir'):
            
            # Mock technical indicators
            mock_indicators.calculate_all.return_value = pd.DataFrame({
                'open': [100] * 150,
                'high': [102] * 150,
                'low': [98] * 150,
                'close': [101] * 150,
                'volume': [1000] * 150
            })
            
            engine = create_engine()
            health = engine.health_check()
            
            # Should have basic health check structure
            assert 'status' in health
            assert 'components' in health
            assert 'timestamp' in health
            assert health['engine_initialized'] is True
    
    def test_performance_stats_tracking(self, sample_data):
        """Test that performance statistics are tracked."""
        with patch('prediction.features.pipeline.TechnicalIndicators') as mock_indicators, \
             patch('prediction.models.registry.Path.exists', return_value=False), \
             patch('prediction.models.registry.Path.mkdir'):
            
            mock_indicators.calculate_all.return_value = sample_data.copy()
            
            engine = create_engine()
            
            # Get initial stats
            stats_before = engine.get_performance_stats()
            assert stats_before['total_predictions'] == 0
            
            # Make a prediction (will likely fail, but should still track stats)
            engine.predict(sample_data)
            
            # Get stats after
            stats_after = engine.get_performance_stats()
            assert stats_after['total_predictions'] == 1
            assert 'avg_inference_time' in stats_after
    
    def test_cache_functionality(self, sample_data):
        """Test that caching functionality works."""
        with patch('prediction.features.pipeline.TechnicalIndicators') as mock_indicators, \
             patch('prediction.models.registry.Path.exists', return_value=False):
            
            mock_indicators.calculate_all.return_value = sample_data.copy()
            
            engine = create_engine()
            
            # Test cache clearing
            engine.clear_caches()  # Should not raise
            
            # Test that cache stats are available
            stats = engine.get_performance_stats()
            assert 'cache_hit_rate' in stats
            assert 'feature_pipeline_stats' in stats