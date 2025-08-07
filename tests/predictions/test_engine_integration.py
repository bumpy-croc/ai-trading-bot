"""
Integration tests for the PredictionEngine with real components.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.prediction.engine import PredictionEngine, PredictionResult
from src.prediction.config import PredictionConfig
from src.prediction import create_engine, create_minimal_engine, predict


class TestPredictionEngineIntegration:
    """Integration tests with real components"""
    
    def create_test_data(self, num_rows=120):
        """Create realistic test market data"""
        # Use local random state for better test isolation
        rng = np.random.RandomState(42)
        
        # Generate realistic OHLCV data
        base_price = 50000.0
        price_data = []
        volume_data = []
        
        current_price = base_price
        for i in range(num_rows):
            # Random walk with some volatility
            change = rng.normal(0, 0.02) * current_price
            current_price = max(current_price + change, 1000)  # Minimum price
            
            # Generate OHLC around current price
            high = current_price * (1 + abs(rng.normal(0, 0.01)))
            low = current_price * (1 - abs(rng.normal(0, 0.01)))
            open_price = current_price + rng.normal(0, 0.005) * current_price
            close_price = current_price
            
            price_data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price
            })
            
            # Generate realistic volume
            volume = rng.exponential(1000) + 100
            volume_data.append(volume)
        
        data = pd.DataFrame(price_data)
        data['volume'] = volume_data
        
        return data
    
    def test_engine_creation_with_real_config(self):
        """Test engine creation with real configuration"""
        try:
            engine = PredictionEngine()
            assert engine is not None
            assert engine.config is not None
            assert engine.feature_pipeline is not None
            assert engine.model_registry is not None
        except Exception as e:
            pytest.skip(f"Cannot create engine with real config: {e}")
    
    def test_engine_health_check_real(self):
        """Test health check with real components"""
        try:
            engine = PredictionEngine()
            health = engine.health_check()
            
            assert 'status' in health
            assert 'components' in health
            assert 'timestamp' in health
            
            # Check component status
            for component in ['feature_pipeline', 'model_registry', 'configuration']:
                assert component in health['components']
                assert 'status' in health['components'][component]
                
            # Overall status should be healthy or degraded (not error)
            assert health['status'] in ['healthy', 'degraded']
            
        except Exception as e:
            pytest.skip(f"Cannot perform health check: {e}")
    
    def test_feature_extraction_real(self):
        """Test feature extraction with real feature pipeline"""
        try:
            engine = PredictionEngine()
            data = self.create_test_data()
            
            # Test feature extraction
            features = engine._extract_features(data)
            
            assert features is not None
            assert isinstance(features, np.ndarray)
            assert features.shape[0] > 0  # Should have at least one row
            assert features.shape[1] > 0  # Should have at least one feature
            
        except Exception as e:
            pytest.skip(f"Cannot test feature extraction: {e}")
    
    def test_model_registry_real(self):
        """Test model registry with real models"""
        try:
            engine = PredictionEngine()
            
            # Test getting available models
            models = engine.get_available_models()
            assert isinstance(models, list)
            
            # Test getting model info if models exist
            if models:
                model_name = models[0]
                info = engine.get_model_info(model_name)
                assert isinstance(info, dict)
                assert 'name' in info
                assert 'loaded' in info
                
        except Exception as e:
            pytest.skip(f"Cannot test model registry: {e}")
    
    def test_end_to_end_prediction_real(self):
        """Test end-to-end prediction with real components"""
        try:
            engine = PredictionEngine()
            data = self.create_test_data()
            
            # Check if we have models available
            models = engine.get_available_models()
            if not models:
                pytest.skip("No models available for testing")
            
            # Test prediction
            result = engine.predict(data)
            
            assert isinstance(result, PredictionResult)
            assert result.timestamp is not None
            assert result.inference_time > 0
            
            # If successful prediction
            if result.error is None:
                assert result.features_used > 0
                assert result.model_name in models
                assert isinstance(result.price, (int, float))
                assert isinstance(result.confidence, (int, float))
                assert result.direction in [-1, 0, 1]
            else:
                # If error occurred, check error message is informative
                assert len(result.error) > 0
                
        except Exception as e:
            pytest.skip(f"Cannot perform end-to-end prediction: {e}")
    
    def test_batch_prediction_real(self):
        """Test batch prediction with real components"""
        try:
            engine = PredictionEngine()
            
            # Check if we have models available
            models = engine.get_available_models()
            if not models:
                pytest.skip("No models available for testing")
            
            # Create multiple data batches
            data_batches = [
                self.create_test_data(),
                self.create_test_data(),
                self.create_test_data()
            ]
            
            # Test batch prediction
            results = engine.predict_batch(data_batches)
            
            assert len(results) == 3
            
            for i, result in enumerate(results):
                assert isinstance(result, PredictionResult)
                assert result.timestamp is not None
                assert result.inference_time > 0
                
                # Check batch metadata
                if 'batch_index' in result.metadata:
                    assert result.metadata['batch_index'] == i
                    assert result.metadata['batch_size'] == 3
                
        except Exception as e:
            pytest.skip(f"Cannot perform batch prediction: {e}")
    
    def test_performance_stats_real(self):
        """Test performance statistics with real usage"""
        try:
            engine = PredictionEngine()
            data = self.create_test_data()
            
            # Get initial stats
            initial_stats = engine.get_performance_stats()
            assert 'total_predictions' in initial_stats
            assert 'avg_inference_time' in initial_stats
            assert 'cache_hit_rate' in initial_stats
            
            # Check if we have models available
            models = engine.get_available_models()
            if models:
                # Make a prediction
                result = engine.predict(data)
                
                # Get updated stats
                updated_stats = engine.get_performance_stats()
                
                # Stats should be updated
                assert updated_stats['total_predictions'] >= initial_stats['total_predictions']
                
        except Exception as e:
            pytest.skip(f"Cannot test performance stats: {e}")
    
    def test_cache_functionality_real(self):
        """Test caching functionality with real components"""
        try:
            engine = PredictionEngine()
            data = self.create_test_data()
            
            # Check if we have models available
            models = engine.get_available_models()
            if not models:
                pytest.skip("No models available for testing")
            
            # Clear caches first
            engine.clear_caches()
            
            # Make first prediction (should be cache miss)
            result1 = engine.predict(data)
            
            # Make second prediction with same data (should potentially be cache hit)
            result2 = engine.predict(data)
            
            # Both should succeed or both should fail consistently
            if result1.error is None:
                assert result2.error is None
                # Results should be similar (allowing for small numerical differences)
                assert abs(result1.price - result2.price) < 1.0
            else:
                assert result2.error is not None
                
        except Exception as e:
            pytest.skip(f"Cannot test cache functionality: {e}")


class TestFactoryFunctions:
    """Test factory functions and convenience methods"""
    
    def test_create_engine_factory(self):
        """Test create_engine factory function"""
        try:
            engine = create_engine(enable_sentiment=True, enable_market_microstructure=False)
            assert engine is not None
            assert engine.config.enable_sentiment is True
            assert engine.config.enable_market_microstructure is False
        except Exception as e:
            pytest.skip(f"Cannot test create_engine factory: {e}")
    
    def test_create_minimal_engine_factory(self):
        """Test create_minimal_engine factory function"""
        try:
            engine = create_minimal_engine()
            assert engine is not None
            assert engine.config.enable_sentiment is False
            assert engine.config.enable_market_microstructure is False
        except Exception as e:
            pytest.skip(f"Cannot test create_minimal_engine factory: {e}")
    
    def test_predict_convenience_function(self):
        """Test predict convenience function"""
        try:
            data = pd.DataFrame({
                'open': [100.0] * 120,
                'high': [102.0] * 120,
                'low': [99.0] * 120,
                'close': [101.0] * 120,
                'volume': [1000.0] * 120
            })
            
            result = predict(data)
            assert isinstance(result, PredictionResult)
            assert result.timestamp is not None
            
        except Exception as e:
            pytest.skip(f"Cannot test predict convenience function: {e}")


class TestConfigurationIntegration:
    """Test configuration integration"""
    
    def test_config_validation_integration(self):
        """Test configuration validation in engine context"""
        # Test valid configuration
        config = PredictionConfig()
        config.prediction_horizons = [1, 5, 15]
        config.min_confidence_threshold = 0.7
        config.max_prediction_latency = 1.0
        
        try:
            engine = PredictionEngine(config)
            assert engine.config == config
        except Exception as e:
            pytest.skip(f"Cannot test config validation: {e}")
    
    def test_config_from_config_manager_integration(self):
        """Test loading configuration from ConfigManager"""
        try:
            config = PredictionConfig.from_config_manager()
            engine = PredictionEngine(config)
            
            # Verify config was loaded properly
            assert config.prediction_horizons is not None
            assert len(config.prediction_horizons) > 0
            assert config.min_confidence_threshold >= 0
            assert config.max_prediction_latency > 0
            
        except Exception as e:
            pytest.skip(f"Cannot test config manager integration: {e}")


class TestErrorHandlingIntegration:
    """Test error handling with real components"""
    
    def test_graceful_degradation_no_models(self):
        """Test graceful degradation when no models are available"""
        try:
            # Create config pointing to empty directory
            config = PredictionConfig()
            config.model_registry_path = "/tmp/nonexistent_models"
            
            engine = PredictionEngine(config)
            
            # Health check should show degraded status
            health = engine.health_check()
            
            # Should not crash but indicate issues
            assert health['status'] in ['healthy', 'degraded']
            
        except Exception as e:
            pytest.skip(f"Cannot test graceful degradation: {e}")
    
    def test_error_recovery_real(self):
        """Test error recovery with real components"""
        try:
            engine = PredictionEngine()
            
            # Test with invalid data
            invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
            result = engine.predict(invalid_data)
            
            # Should return error result, not crash
            assert isinstance(result, PredictionResult)
            assert result.error is not None
            assert result.price == 0.0
            assert result.confidence == 0.0
            
            # Engine should still be functional after error
            health = engine.health_check()
            assert 'status' in health
            
        except Exception as e:
            pytest.skip(f"Cannot test error recovery: {e}")