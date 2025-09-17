"""
Tests for prediction engine model components.
"""

import json
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest

from src.prediction.config import PredictionConfig
from src.prediction.models.onnx_runner import ModelPrediction, OnnxRunner
from src.prediction.models.registry import PredictionModelRegistry
from src.prediction.utils.caching import ModelCache, cache_prediction


class TestPredictionConfig:
    """Test PredictionConfig class"""

    def test_default_initialization(self):
        """Test default configuration initialization"""
        config = PredictionConfig()

        assert config.prediction_horizons == [1]
        assert config.min_confidence_threshold == 0.6
        assert config.max_prediction_latency == 0.1
        assert config.model_registry_path == "src/ml"
        assert config.enable_sentiment is False
        assert config.enable_market_microstructure is False
        assert config.feature_cache_ttl == 3600
        assert config.model_cache_ttl == 600

    def test_custom_initialization(self):
        """Test custom configuration initialization"""
        config = PredictionConfig(
            prediction_horizons=[1, 5, 10],
            min_confidence_threshold=0.8,
            model_registry_path="custom/path",
        )

        assert config.prediction_horizons == [1, 5, 10]
        assert config.min_confidence_threshold == 0.8
        assert config.model_registry_path == "custom/path"


class TestModelCache:
    """Test ModelCache class"""

    def test_cache_operations(self):
        """Test basic cache operations"""
        cache = ModelCache(ttl=1)

        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Test missing key
        assert cache.get("nonexistent") is None

        # Test cache size
        assert cache.size() == 1

        # Test clear
        cache.clear()
        assert cache.size() == 0
        assert cache.get("key1") is None

    def test_cache_expiration(self):
        """Test cache TTL expiration"""
        import time

        cache = ModelCache(ttl=0.1)  # 100ms TTL
        cache.set("key1", "value1")

        # Should be available immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(0.2)

        # Should be expired
        assert cache.get("key1") is None

    def test_cache_decorator(self):
        """Test cache decorator functionality"""
        call_count = 0

        @cache_prediction(ttl=1)
        def test_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = test_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call should use cache
        result2 = test_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment

        # Different argument should call function
        result3 = test_function(3)
        assert result3 == 6
        assert call_count == 2


class TestOnnxRunner:
    """Test OnnxRunner class"""

    def setup_method(self):
        """Set up test configuration"""
        self.config = PredictionConfig(
            prediction_horizons=[1],
            min_confidence_threshold=0.6,
            max_prediction_latency=0.1,
            model_registry_path="src/ml",
        )

    @patch("onnxruntime.InferenceSession")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"sequence_length": 120, "feature_count": 5}',
    )
    def test_model_loading(self, mock_file, mock_session):
        """Test ONNX model loading"""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance

        # Use proper path format instead of filename string
        model_path = "/tmp/test_model.onnx"
        runner = OnnxRunner(model_path, self.config)

        assert runner.session is not None
        assert runner.model_metadata is not None
        assert runner.model_metadata["sequence_length"] == 120
        mock_session.assert_called_once_with(model_path, providers=["CPUExecutionProvider"])

    @patch("onnxruntime.InferenceSession")
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_model_loading_no_metadata(self, mock_file, mock_session):
        """Test model loading with missing metadata file"""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance

        # Use proper path format instead of filename string
        model_path = "/tmp/test_model.onnx"
        runner = OnnxRunner(model_path, self.config)

        # Should use default metadata
        assert runner.model_metadata["sequence_length"] == 120
        assert runner.model_metadata["feature_count"] == 5

    @patch("onnxruntime.InferenceSession")
    def test_input_preparation(self, mock_session):
        """Test feature input preparation"""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance

        with patch("builtins.open", mock_open(read_data='{"sequence_length": 120}')):
            # Use proper path format instead of filename string
            model_path = "/tmp/test_model.onnx"
            runner = OnnxRunner(model_path, self.config)

            # Test with 2D input
            features = np.random.rand(120, 5).astype(np.float32)
            prepared = runner._prepare_input(features)

            assert prepared.shape == (1, 120, 5)
            assert prepared.dtype == np.float32

    @patch("onnxruntime.InferenceSession")
    def test_prediction_processing(self, mock_session):
        """Test prediction output processing"""
        mock_session_instance = Mock()
        mock_session_instance.run.return_value = [np.array([[[0.05]]])]  # Positive prediction
        mock_session.return_value = mock_session_instance

        with patch("builtins.open", mock_open(read_data='{"sequence_length": 120}')):
            # Use proper path format instead of filename string
            model_path = "/tmp/test_model.onnx"
            runner = OnnxRunner(model_path, self.config)

            # Mock model output
            output = np.array([[[0.05]]])
            result = runner._process_output(output)

            assert result["price"] > 0
            assert result["direction"] == 1
            assert 0 <= result["confidence"] <= 1

    @patch("onnxruntime.InferenceSession")
    def test_full_prediction_flow(self, mock_session):
        """Test complete prediction flow"""
        mock_session_instance = Mock()
        mock_session_instance.run.return_value = [np.array([[[0.02]]])]

        # Mock the get_inputs method to return a list with a mock input
        mock_input = Mock()
        mock_input.name = "input"
        mock_session_instance.get_inputs.return_value = [mock_input]

        mock_session.return_value = mock_session_instance

        with patch("builtins.open", mock_open(read_data='{"sequence_length": 120}')):
            # Use proper path format instead of filename string
            model_path = "/tmp/test_model.onnx"
            runner = OnnxRunner(model_path, self.config)

            # Create test features
            features = np.random.rand(120, 5).astype(np.float32)

            # Run prediction
            prediction = runner.predict(features)

            assert isinstance(prediction, ModelPrediction)
            # model_name should be basename of the path
            assert prediction.model_name == "test_model.onnx"
            assert prediction.inference_time >= 0
            assert isinstance(prediction.confidence, float)
            assert prediction.direction in [-1, 0, 1]

    @patch("onnxruntime.InferenceSession")
    def test_input_preparation_different_shapes(self, mock_session):
        """Test input preparation with different input shapes"""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance

        with patch("builtins.open", mock_open(read_data='{"sequence_length": 120}')):
            # Use proper path format instead of filename string
            model_path = "/tmp/test_model.onnx"
            runner = OnnxRunner(model_path, self.config)

            # Test 1D input
            features_1d = np.random.rand(5).astype(np.float32)
            result = runner._prepare_input(features_1d)
            assert result.shape == (1, 1, 5)
            assert result.dtype == np.float32

            # Test 2D input
            features_2d = np.random.rand(10, 5).astype(np.float32)
            result = runner._prepare_input(features_2d)
            assert result.shape == (1, 10, 5)
            assert result.dtype == np.float32

            # Test 3D input
            features_3d = np.random.rand(2, 10, 5).astype(np.float32)
            result = runner._prepare_input(features_3d)
            assert result.shape == (2, 10, 5)
            assert result.dtype == np.float32

            # Test invalid input shape
            features_4d = np.random.rand(2, 3, 4, 5).astype(np.float32)
            with pytest.raises(ValueError, match="Features must be 1D, 2D, or 3D array"):
                runner._prepare_input(features_4d)

    @patch("onnxruntime.InferenceSession")
    def test_normalize_features_zero_std_fix(self, mock_session):
        """Test that ZeroDivisionError is prevented when std is 0.0"""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance

        # Mock metadata with zero std values
        metadata = {
            "normalization_params": {
                "feature1": {"mean": 0.0, "std": 0.0},
                "feature2": {"mean": 1.0, "std": 0.0},
                "feature3": {"mean": 0.5, "std": 1.0},
            }
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(metadata))):
            # Use proper path format instead of filename string
            model_path = "/tmp/test_model.onnx"
            runner = OnnxRunner(model_path, self.config)

            # Create 3D features
            features = np.random.rand(1, 10, 3).astype(np.float32)

            # This should not raise ZeroDivisionError
            normalized = runner._normalize_features(features)

            assert normalized.shape == features.shape
            assert not np.isnan(normalized).any()
            assert not np.isinf(normalized).any()

    @patch("onnxruntime.InferenceSession")
    def test_normalize_features_invalid_shape_fix(self, mock_session):
        """Test that IndexError is prevented with invalid input shapes"""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance

        metadata = {"normalization_params": {"feature1": {"mean": 0.0, "std": 1.0}}}

        with patch("builtins.open", mock_open(read_data=json.dumps(metadata))):
            # Use proper path format instead of filename string
            model_path = "/tmp/test_model.onnx"
            runner = OnnxRunner(model_path, self.config)

            # Test with 2D features (should raise ValueError)
            features_2d = np.random.rand(10, 1).astype(np.float32)
            with pytest.raises(ValueError, match="Features must be 3D for normalization"):
                runner._normalize_features(features_2d)

            # Test with 1D features (should raise ValueError)
            features_1d = np.random.rand(5).astype(np.float32)
            with pytest.raises(ValueError, match="Features must be 3D for normalization"):
                runner._normalize_features(features_1d)


class TestPredictionModelRegistry:
    """Structured-only PredictionModelRegistry tests"""

    def _write_bundle(self, root, symbol: str, model_type: str, version: str, timeframe: str = "1h"):
        import json
        base = root / symbol / model_type / version
        base.mkdir(parents=True, exist_ok=True)
        # empty onnx file (stub runner will handle failures)
        (base / "model.onnx").write_bytes(b"")
        (base / "metadata.json").write_text(json.dumps({"symbol": symbol, "model_type": model_type, "timeframe": timeframe, "version_id": version}))
        return base

    def test_loading_structured_bundles(self, tmp_path):
        cfg = PredictionConfig(model_registry_path=str(tmp_path))
        self._write_bundle(tmp_path, "BTCUSDT", "basic", "2025-09-17_1h_v1")
        reg = PredictionModelRegistry(cfg)
        bundles = reg.list_bundles()
        assert len(bundles) == 1
        b = bundles[0]
        assert b.symbol == "BTCUSDT"
        assert b.model_type == "basic"

    def test_select_and_default_runner(self, tmp_path):
        cfg = PredictionConfig(model_registry_path=str(tmp_path))
        self._write_bundle(tmp_path, "BTCUSDT", "basic", "2025-09-17_1h_v1")
        reg = PredictionModelRegistry(cfg)
        bundle = reg.select_bundle(symbol="BTCUSDT", model_type="basic", timeframe="1h")
        assert bundle is not None
        runner = reg.get_default_runner()
        assert runner is not None

    def test_reload_models(self, tmp_path):
        cfg = PredictionConfig(model_registry_path=str(tmp_path))
        self._write_bundle(tmp_path, "BTCUSDT", "basic", "2025-09-17_1h_v1")
        reg = PredictionModelRegistry(cfg)
        assert len(reg.list_bundles()) == 1
        reg.reload_models()
        assert len(reg.list_bundles()) == 1
