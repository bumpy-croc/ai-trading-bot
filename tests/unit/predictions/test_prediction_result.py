"""Tests for the PredictionResult dataclass."""

from datetime import datetime, timezone

from src.prediction.engine import PredictionResult


class TestPredictionResult:
    """Test PredictionResult dataclass"""

    def test_prediction_result_creation(self):
        """Test creating a PredictionResult"""
        timestamp = datetime.now(timezone.utc)
        result = PredictionResult(
            price=100.5,
            confidence=0.85,
            direction=1,
            model_name="test_model",
            timestamp=timestamp,
            inference_time=0.05,
            features_used=10,
        )

        assert result.price == 100.5
        assert result.confidence == 0.85
        assert result.direction == 1
        assert result.model_name == "test_model"
        assert result.timestamp == timestamp
        assert result.inference_time == 0.05
        assert result.features_used == 10
        assert result.cache_hit is False
        assert result.error is None
        assert result.metadata == {}

    def test_prediction_result_with_error(self):
        """Test creating a PredictionResult with error"""
        result = PredictionResult(
            price=0.0,
            confidence=0.0,
            direction=0,
            model_name="test_model",
            timestamp=datetime.now(timezone.utc),
            inference_time=0.1,
            features_used=0,
            error="Test error message",
        )

        assert result.error == "Test error message"
        assert result.price == 0.0
        assert result.confidence == 0.0
