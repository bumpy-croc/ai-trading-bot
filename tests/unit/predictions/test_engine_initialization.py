"""Tests covering PredictionEngine initialization."""

from unittest.mock import patch

from src.prediction.config import PredictionConfig
from src.prediction.engine import PredictionEngine


class TestPredictionEngineInit:
    """Test PredictionEngine initialization"""

    def test_init_with_config(self):
        """Test initialization with custom config"""
        config = PredictionConfig(
            prediction_horizons=[1],
            enable_sentiment=True,
            enable_market_microstructure=True,
            feature_cache_ttl=600,
        )

        with patch("src.prediction.engine.PredictionModelRegistry") as mock_registry:
            with patch("src.prediction.engine.FeaturePipeline") as mock_pipeline:
                PredictionEngine(config)

                mock_registry.assert_called_once_with(config, None)
                mock_pipeline.assert_called_once()

    def test_init_without_config(self):
        """Test initialization without config (uses default)"""
        with patch("src.prediction.engine.PredictionModelRegistry") as mock_registry:
            with patch("src.prediction.engine.FeaturePipeline") as mock_pipeline:
                PredictionEngine()

                default_config = PredictionConfig.from_config_manager()
                mock_registry.assert_called_once_with(default_config, None)
                mock_pipeline.assert_called_once()
