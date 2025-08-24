"""
Tests for prediction engine configuration.
"""

import os
from unittest.mock import patch

import pytest

from src.prediction.config import PredictionConfig


class TestPredictionConfig:
    """Test the prediction engine configuration."""

    def test_prediction_config_loading_with_environment_variables(self):
        """Test loading prediction configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "PREDICTION_HORIZONS": "1,5,20",
                "MIN_CONFIDENCE_THRESHOLD": "0.7",
                "MAX_PREDICTION_LATENCY": "0.05",
                "MODEL_REGISTRY_PATH": "custom/ml/path",
                "ENABLE_SENTIMENT": "true",
                "ENABLE_MARKET_MICROSTRUCTURE": "false",
                "FEATURE_CACHE_TTL": "600",
                "MODEL_CACHE_TTL": "1200",
            },
        ):
            config = PredictionConfig.from_config_manager()

            # Test list parsing for prediction horizons
            assert config.prediction_horizons == [1, 5, 20]

            # Test float parsing
            assert config.min_confidence_threshold == 0.7
            assert config.max_prediction_latency == 0.05

            # Test string values
            assert config.model_registry_path == "custom/ml/path"

            # Test boolean parsing
            assert config.enable_sentiment is True
            assert config.enable_market_microstructure is False

            # Test integer parsing
            assert config.feature_cache_ttl == 600
            assert config.model_cache_ttl == 1200

    def test_prediction_config_defaults(self):
        """Test prediction configuration default values."""
        # Clear environment to test defaults
        _prediction_keys = [
            "PREDICTION_HORIZONS",
            "MIN_CONFIDENCE_THRESHOLD",
            "MAX_PREDICTION_LATENCY",
            "MODEL_REGISTRY_PATH",
            "ENABLE_SENTIMENT",
            "ENABLE_MARKET_MICROSTRUCTURE",
            "FEATURE_CACHE_TTL",
            "MODEL_CACHE_TTL",
        ]

        with patch.dict(os.environ, {}, clear=True):
            config = PredictionConfig.from_config_manager()

            # Test default values
            assert config.prediction_horizons == [1]
            assert config.min_confidence_threshold == 0.6
            assert config.max_prediction_latency == 0.1
            assert config.model_registry_path == "src/ml"
            assert config.enable_sentiment is False
            assert config.enable_market_microstructure is False
            assert config.feature_cache_ttl == 3600
            assert config.model_cache_ttl == 600

    def test_prediction_config_validation_success(self):
        """Test prediction configuration validation with valid values."""
        valid_config = PredictionConfig(
            prediction_horizons=[1, 5, 20],
            min_confidence_threshold=0.7,
            max_prediction_latency=0.1,
            model_registry_path="src/ml",
            enable_sentiment=False,
            enable_market_microstructure=False,
            feature_cache_ttl=3600,
            model_cache_ttl=600,
        )

        # Should not raise any exception
        valid_config.validate()

    def test_prediction_config_validation_empty_horizons(self):
        """Test prediction configuration validation with empty horizons."""
        invalid_config = PredictionConfig(
            prediction_horizons=[],  # Empty horizons
            min_confidence_threshold=0.7,
            max_prediction_latency=0.1,
            model_registry_path="src/ml",
            enable_sentiment=False,
            enable_market_microstructure=False,
            feature_cache_ttl=3600,
            model_cache_ttl=600,
        )

        with pytest.raises(ValueError, match="At least one prediction horizon"):
            invalid_config.validate()

    def test_prediction_config_validation_negative_horizons(self):
        """Test prediction configuration validation with negative horizons."""
        invalid_config = PredictionConfig(
            prediction_horizons=[1, -5, 20],  # Negative horizon
            min_confidence_threshold=0.7,
            max_prediction_latency=0.1,
            model_registry_path="src/ml",
            enable_sentiment=False,
            enable_market_microstructure=False,
            feature_cache_ttl=3600,
            model_cache_ttl=600,
        )

        with pytest.raises(ValueError, match="Prediction horizons must be positive"):
            invalid_config.validate()

    def test_prediction_config_validation_invalid_confidence(self):
        """Test prediction configuration validation with invalid confidence threshold."""
        # Test confidence > 1
        invalid_config_high = PredictionConfig(
            prediction_horizons=[1],
            min_confidence_threshold=1.5,  # > 1
            max_prediction_latency=0.1,
            model_registry_path="src/ml",
            enable_sentiment=False,
            enable_market_microstructure=False,
            feature_cache_ttl=3600,
            model_cache_ttl=600,
        )

        with pytest.raises(ValueError, match="Confidence threshold must be between 0 and 1"):
            invalid_config_high.validate()

        # Test confidence < 0
        invalid_config_low = PredictionConfig(
            prediction_horizons=[1],
            min_confidence_threshold=-0.1,  # < 0
            max_prediction_latency=0.1,
            model_registry_path="src/ml",
            enable_sentiment=False,
            enable_market_microstructure=False,
            feature_cache_ttl=3600,
            model_cache_ttl=600,
        )

        with pytest.raises(ValueError, match="Confidence threshold must be between 0 and 1"):
            invalid_config_low.validate()

    def test_prediction_config_validation_negative_latency(self):
        """Test prediction configuration validation with negative latency."""
        invalid_config = PredictionConfig(
            prediction_horizons=[1],
            min_confidence_threshold=0.7,
            max_prediction_latency=-0.1,  # Negative latency
            model_registry_path="src/ml",
            enable_sentiment=False,
            enable_market_microstructure=False,
            feature_cache_ttl=3600,
            model_cache_ttl=600,
        )

        with pytest.raises(ValueError, match="Prediction latency must be positive"):
            invalid_config.validate()

    def test_prediction_config_validation_invalid_cache_ttl(self):
        """Test prediction configuration validation with invalid cache TTL values."""
        # Test negative feature cache TTL
        invalid_config_feature = PredictionConfig(
            prediction_horizons=[1],
            min_confidence_threshold=0.7,
            max_prediction_latency=0.1,
            model_registry_path="src/ml",
            enable_sentiment=False,
            enable_market_microstructure=False,
            feature_cache_ttl=-300,  # Negative TTL
            model_cache_ttl=600,
        )

        with pytest.raises(ValueError, match="Feature cache TTL must be positive"):
            invalid_config_feature.validate()

        # Test negative model cache TTL
        invalid_config_model = PredictionConfig(
            prediction_horizons=[1],
            min_confidence_threshold=0.7,
            max_prediction_latency=0.1,
            model_registry_path="src/ml",
            enable_sentiment=False,
            enable_market_microstructure=False,
            feature_cache_ttl=3600,
            model_cache_ttl=-600,  # Negative TTL
        )

        with pytest.raises(ValueError, match="Model cache TTL must be positive"):
            invalid_config_model.validate()

    def test_prediction_config_str_representation(self):
        """Test prediction configuration string representation."""
        config = PredictionConfig(
            prediction_horizons=[1, 5],
            min_confidence_threshold=0.8,
            max_prediction_latency=0.05,
            model_registry_path="src/ml",
            enable_sentiment=True,
            enable_market_microstructure=False,
            feature_cache_ttl=3600,
            model_cache_ttl=600,
        )

        config_str = str(config)

        # Check that important values are in the string representation
        assert "horizons=[1, 5]" in config_str
        assert "confidence_threshold=0.8" in config_str
        assert "max_latency=0.05s" in config_str
        assert "sentiment=True" in config_str
        assert "microstructure=False" in config_str

    def test_prediction_config_integration_with_config_manager(self):
        """Test that PredictionConfig integrates properly with ConfigManager."""
        from config.config_manager import get_config

        with patch.dict(
            os.environ,
            {
                "PREDICTION_HORIZONS": "1,10",
                "MIN_CONFIDENCE_THRESHOLD": "0.75",
                "ENABLE_SENTIMENT": "true",
            },
        ):
            # Verify ConfigManager can retrieve the values
            config_manager = get_config()

            horizons_list = config_manager.get_list("PREDICTION_HORIZONS")
            assert horizons_list == ["1", "10"]

            confidence = config_manager.get_float("MIN_CONFIDENCE_THRESHOLD")
            assert confidence == 0.75

            sentiment = config_manager.get_bool("ENABLE_SENTIMENT")
            assert sentiment is True

            # Verify PredictionConfig can load from ConfigManager
            prediction_config = PredictionConfig.from_config_manager()
            assert prediction_config.prediction_horizons == [1, 10]
            assert prediction_config.min_confidence_threshold == 0.75
            assert prediction_config.enable_sentiment is True
