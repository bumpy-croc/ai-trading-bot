"""Tests for ml.training_pipeline.gpu_config module."""

import platform
from unittest.mock import MagicMock, patch

import pytest


class TestConfigureGpu:
    """Tests for configure_gpu function."""

    def test_returns_none_when_tensorflow_unavailable(self):
        """Test that None is returned when TensorFlow is not available."""
        with patch.dict("sys.modules", {"tensorflow": None}):
            # Need to reload module to pick up the change
            import importlib

            import src.ml.training_pipeline.gpu_config as gpu_config

            # Simulate TensorFlow not available
            original_available = gpu_config._TENSORFLOW_AVAILABLE
            gpu_config._TENSORFLOW_AVAILABLE = False

            try:
                result = gpu_config.configure_gpu()
                assert result is None
            finally:
                gpu_config._TENSORFLOW_AVAILABLE = original_available

    def test_detects_apple_silicon(self):
        """Test detection of Apple Silicon Mac."""
        from src.ml.training_pipeline.gpu_config import configure_gpu

        with patch("platform.system", return_value="Darwin"):
            with patch("platform.machine", return_value="arm64"):
                with patch(
                    "src.ml.training_pipeline.gpu_config._TENSORFLOW_AVAILABLE", True
                ):
                    with patch(
                        "src.ml.training_pipeline.gpu_config._TENSORFLOW_METAL_AVAILABLE",
                        False,
                    ):
                        result = configure_gpu()
                        # Without tensorflow-metal, should return None
                        assert result is None

    def test_returns_gpu_name_on_apple_silicon_with_metal(self):
        """Test GPU name returned on Apple Silicon with Metal."""
        from src.ml.training_pipeline.gpu_config import configure_gpu

        mock_tf = MagicMock()
        mock_device = MagicMock()
        mock_device.name = "GPU:0"
        mock_tf.config.list_physical_devices.return_value = [mock_device]

        with patch("platform.system", return_value="Darwin"):
            with patch("platform.machine", return_value="arm64"):
                with patch(
                    "src.ml.training_pipeline.gpu_config._TENSORFLOW_AVAILABLE", True
                ):
                    with patch(
                        "src.ml.training_pipeline.gpu_config._TENSORFLOW_METAL_AVAILABLE",
                        True,
                    ):
                        with patch(
                            "src.ml.training_pipeline.gpu_config.tf", mock_tf
                        ):
                            result = configure_gpu()
                            assert result == "GPU:0"

    def test_returns_none_when_no_apple_gpu(self):
        """Test None returned when no Apple GPU detected."""
        from src.ml.training_pipeline.gpu_config import configure_gpu

        mock_tf = MagicMock()
        mock_tf.config.list_physical_devices.return_value = []

        with patch("platform.system", return_value="Darwin"):
            with patch("platform.machine", return_value="arm64"):
                with patch(
                    "src.ml.training_pipeline.gpu_config._TENSORFLOW_AVAILABLE", True
                ):
                    with patch(
                        "src.ml.training_pipeline.gpu_config._TENSORFLOW_METAL_AVAILABLE",
                        True,
                    ):
                        with patch(
                            "src.ml.training_pipeline.gpu_config.tf", mock_tf
                        ):
                            result = configure_gpu()
                            assert result is None

    def test_detects_nvidia_gpu(self):
        """Test detection of NVIDIA GPU."""
        from src.ml.training_pipeline.gpu_config import configure_gpu

        mock_tf = MagicMock()
        mock_device = MagicMock()
        mock_device.name = "/physical_device:GPU:0"
        mock_tf.config.list_physical_devices.return_value = [mock_device]
        mock_tf.config.experimental.set_memory_growth = MagicMock()

        with patch("platform.system", return_value="Linux"):
            with patch("platform.machine", return_value="x86_64"):
                with patch(
                    "src.ml.training_pipeline.gpu_config._TENSORFLOW_AVAILABLE", True
                ):
                    with patch(
                        "src.ml.training_pipeline.gpu_config.tf", mock_tf
                    ):
                        result = configure_gpu()
                        assert result == "/physical_device:GPU:0"
                        mock_tf.config.experimental.set_memory_growth.assert_called()

    def test_handles_memory_growth_exception(self):
        """Test handling of memory growth configuration failure."""
        from src.ml.training_pipeline.gpu_config import configure_gpu

        mock_tf = MagicMock()
        mock_device = MagicMock()
        mock_device.name = "/physical_device:GPU:0"
        mock_tf.config.list_physical_devices.return_value = [mock_device]
        mock_tf.config.experimental.set_memory_growth.side_effect = RuntimeError(
            "Cannot set memory growth"
        )

        with patch("platform.system", return_value="Linux"):
            with patch("platform.machine", return_value="x86_64"):
                with patch(
                    "src.ml.training_pipeline.gpu_config._TENSORFLOW_AVAILABLE", True
                ):
                    with patch(
                        "src.ml.training_pipeline.gpu_config.tf", mock_tf
                    ):
                        # Should not raise, just log warning
                        result = configure_gpu()
                        assert result == "/physical_device:GPU:0"

    def test_returns_none_when_no_gpu(self):
        """Test None returned when no GPU detected."""
        from src.ml.training_pipeline.gpu_config import configure_gpu

        mock_tf = MagicMock()
        mock_tf.config.list_physical_devices.return_value = []

        with patch("platform.system", return_value="Linux"):
            with patch("platform.machine", return_value="x86_64"):
                with patch(
                    "src.ml.training_pipeline.gpu_config._TENSORFLOW_AVAILABLE", True
                ):
                    with patch(
                        "src.ml.training_pipeline.gpu_config.tf", mock_tf
                    ):
                        result = configure_gpu()
                        assert result is None

    def test_handles_gpu_detection_exception(self):
        """Test handling of GPU detection failure."""
        from src.ml.training_pipeline.gpu_config import configure_gpu

        mock_tf = MagicMock()
        mock_tf.config.list_physical_devices.side_effect = Exception("GPU detection failed")

        with patch("platform.system", return_value="Linux"):
            with patch("platform.machine", return_value="x86_64"):
                with patch(
                    "src.ml.training_pipeline.gpu_config._TENSORFLOW_AVAILABLE", True
                ):
                    with patch(
                        "src.ml.training_pipeline.gpu_config.tf", mock_tf
                    ):
                        result = configure_gpu()
                        assert result is None


class TestGetComputeDevice:
    """Tests for get_compute_device function."""

    def test_returns_cpu_when_tensorflow_unavailable(self):
        """Test that CPU is returned when TensorFlow not available."""
        from src.ml.training_pipeline.gpu_config import get_compute_device

        import src.ml.training_pipeline.gpu_config as gpu_config

        original_available = gpu_config._TENSORFLOW_AVAILABLE
        gpu_config._TENSORFLOW_AVAILABLE = False

        try:
            result = get_compute_device()
            assert result == "CPU"
        finally:
            gpu_config._TENSORFLOW_AVAILABLE = original_available

    def test_returns_gpu_name_when_available(self):
        """Test GPU name returned when GPU available."""
        from src.ml.training_pipeline.gpu_config import get_compute_device

        mock_tf = MagicMock()
        mock_device = MagicMock()
        mock_device.name = "GPU:0"
        mock_tf.config.list_physical_devices.return_value = [mock_device]

        with patch(
            "src.ml.training_pipeline.gpu_config._TENSORFLOW_AVAILABLE", True
        ):
            with patch("src.ml.training_pipeline.gpu_config.tf", mock_tf):
                result = get_compute_device()
                assert result == "GPU:0"

    def test_returns_cpu_when_no_gpu(self):
        """Test CPU returned when no GPU available."""
        from src.ml.training_pipeline.gpu_config import get_compute_device

        mock_tf = MagicMock()
        mock_tf.config.list_physical_devices.return_value = []

        with patch(
            "src.ml.training_pipeline.gpu_config._TENSORFLOW_AVAILABLE", True
        ):
            with patch("src.ml.training_pipeline.gpu_config.tf", mock_tf):
                result = get_compute_device()
                assert result == "CPU"

    def test_returns_cpu_on_exception(self):
        """Test CPU returned on exception."""
        from src.ml.training_pipeline.gpu_config import get_compute_device

        mock_tf = MagicMock()
        mock_tf.config.list_physical_devices.side_effect = Exception("Error")

        with patch(
            "src.ml.training_pipeline.gpu_config._TENSORFLOW_AVAILABLE", True
        ):
            with patch("src.ml.training_pipeline.gpu_config.tf", mock_tf):
                result = get_compute_device()
                assert result == "CPU"


@pytest.mark.fast
class TestGpuConfigIntegration:
    """Integration tests for GPU configuration."""

    def test_module_constants_defined(self):
        """Test that module constants are properly defined."""
        from src.ml.training_pipeline.gpu_config import (
            _TENSORFLOW_AVAILABLE,
            _TENSORFLOW_METAL_AVAILABLE,
        )

        # These should be booleans
        assert isinstance(_TENSORFLOW_AVAILABLE, bool)
        assert isinstance(_TENSORFLOW_METAL_AVAILABLE, bool)

    def test_consistent_device_reporting(self):
        """Test that get_compute_device is consistent with configure_gpu."""
        from src.ml.training_pipeline.gpu_config import (
            configure_gpu,
            get_compute_device,
        )

        mock_tf = MagicMock()
        mock_device = MagicMock()
        mock_device.name = "GPU:0"
        mock_tf.config.list_physical_devices.return_value = [mock_device]
        mock_tf.config.experimental.set_memory_growth = MagicMock()

        with patch("platform.system", return_value="Linux"):
            with patch("platform.machine", return_value="x86_64"):
                with patch(
                    "src.ml.training_pipeline.gpu_config._TENSORFLOW_AVAILABLE", True
                ):
                    with patch(
                        "src.ml.training_pipeline.gpu_config.tf", mock_tf
                    ):
                        configured = configure_gpu()
                        current = get_compute_device()
                        # Both should report the same GPU
                        assert configured == current == "GPU:0"
