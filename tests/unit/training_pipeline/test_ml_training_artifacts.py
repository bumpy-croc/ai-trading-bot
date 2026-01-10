"""Unit tests for ML training pipeline artifacts module."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pytest

# Use non-interactive backend for tests
matplotlib.use("Agg")

from src.ml.training_pipeline.artifacts import (
    ArtifactPaths,
    convert_to_onnx,
    create_training_plots,
    evaluate_model_performance,
    save_artifacts,
    validate_model_robustness,
)


@pytest.mark.fast
class TestArtifactPaths:
    """Test ArtifactPaths dataclass."""

    def test_initialization(self, tmp_path):
        # Arrange
        directory = tmp_path / "model_dir"
        keras_path = directory / "model.keras"
        onnx_path = directory / "model.onnx"
        metadata_path = directory / "metadata.json"
        plot_path = directory / "plot.png"

        # Act
        paths = ArtifactPaths(
            directory=directory,
            keras_path=keras_path,
            onnx_path=onnx_path,
            metadata_path=metadata_path,
            plot_path=plot_path,
        )

        # Assert
        assert paths.directory == directory
        assert paths.keras_path == keras_path
        assert paths.onnx_path == onnx_path
        assert paths.metadata_path == metadata_path
        assert paths.plot_path == plot_path

    def test_optional_paths_can_be_none(self, tmp_path):
        # Arrange
        directory = tmp_path / "model_dir"
        keras_path = directory / "model.keras"
        metadata_path = directory / "metadata.json"

        # Act
        paths = ArtifactPaths(
            directory=directory,
            keras_path=keras_path,
            onnx_path=None,
            metadata_path=metadata_path,
            plot_path=None,
        )

        # Assert
        assert paths.onnx_path is None
        assert paths.plot_path is None


@pytest.mark.fast
class TestCreateTrainingPlots:
    """Test create_training_plots function."""

    def test_creates_plot_when_enabled(self, tmp_path):
        # Arrange
        history = MagicMock()
        history.history = {
            "loss": [0.5, 0.4, 0.3],
            "val_loss": [0.6, 0.5, 0.4],
            "rmse": [0.7, 0.6, 0.5],
            "val_rmse": [0.8, 0.7, 0.6],
        }

        model = MagicMock()
        model.predict.return_value = np.array([[0.5], [0.6], [0.7]])

        X_test = np.random.rand(10, 120, 15).astype(np.float32)
        y_test = np.random.rand(10).astype(np.float32)
        feature_names = ["feature1", "feature2"]

        # Act
        result = create_training_plots(
            history,
            model,
            X_test,
            y_test,
            feature_names,
            symbol="BTCUSDT",
            model_type="basic",
            output_dir=tmp_path,
            enable_plots=True,
        )

        # Assert
        assert result is not None
        assert result.exists()
        assert result.name == "BTCUSDT_basic_training.png"

    def test_returns_none_when_disabled(self, tmp_path):
        # Arrange
        history = MagicMock()
        model = MagicMock()
        X_test = np.random.rand(10, 120, 15).astype(np.float32)
        y_test = np.random.rand(10).astype(np.float32)
        feature_names = ["feature1"]

        # Act
        result = create_training_plots(
            history,
            model,
            X_test,
            y_test,
            feature_names,
            symbol="BTCUSDT",
            model_type="basic",
            output_dir=tmp_path,
            enable_plots=False,
        )

        # Assert
        assert result is None

    def test_handles_plotting_exceptions_gracefully(self, tmp_path):
        # Arrange
        history = MagicMock()
        history.history = {"loss": [0.5], "val_loss": [0.6]}

        model = MagicMock()
        model.predict.side_effect = RuntimeError("Prediction error")

        X_test = np.random.rand(10, 120, 15).astype(np.float32)
        y_test = np.random.rand(10).astype(np.float32)
        feature_names = ["feature1"]

        # Act
        result = create_training_plots(
            history,
            model,
            X_test,
            y_test,
            feature_names,
            symbol="BTCUSDT",
            model_type="basic",
            output_dir=tmp_path,
            enable_plots=True,
        )

        # Assert - should return None on error, not raise
        assert result is None


@pytest.mark.fast
class TestValidateModelRobustness:
    """Test validate_model_robustness function."""

    def test_base_performance_calculation(self):
        # Arrange
        model = MagicMock()
        model.predict.return_value = np.array([[0.5], [0.6], [0.7]])

        X_test = np.random.rand(3, 120, 10).astype(np.float32)
        y_test = np.array([0.5, 0.6, 0.7])
        feature_names = ["feature1", "feature2"]

        # Act
        result = validate_model_robustness(
            model, X_test, y_test, feature_names, has_sentiment=False
        )

        # Assert
        assert "base_performance" in result
        assert "mse" in result["base_performance"]
        assert "rmse" in result["base_performance"]
        assert result["base_performance"]["mse"] >= 0
        assert result["base_performance"]["rmse"] >= 0

    def test_sentiment_ablation_when_has_sentiment(self):
        # Arrange
        model = MagicMock()
        model.predict.side_effect = [
            np.array([[0.5], [0.6], [0.7]]),  # Base prediction
            np.array([[0.4], [0.5], [0.6]]),  # No sentiment prediction
        ]

        X_test = np.random.rand(3, 120, 10).astype(np.float32)
        y_test = np.array([0.5, 0.6, 0.7])
        feature_names = ["feature1", "sentiment_score", "feature3"]

        # Act
        result = validate_model_robustness(model, X_test, y_test, feature_names, has_sentiment=True)

        # Assert
        assert "no_sentiment_performance" in result
        assert "mse" in result["no_sentiment_performance"]
        assert "rmse" in result["no_sentiment_performance"]
        assert "degradation_pct" in result["no_sentiment_performance"]

    def test_no_sentiment_ablation_when_no_sentiment(self):
        # Arrange
        model = MagicMock()
        model.predict.return_value = np.array([[0.5], [0.6], [0.7]])

        X_test = np.random.rand(3, 120, 10).astype(np.float32)
        y_test = np.array([0.5, 0.6, 0.7])
        feature_names = ["feature1", "feature2"]

        # Act
        result = validate_model_robustness(
            model, X_test, y_test, feature_names, has_sentiment=False
        )

        # Assert
        assert "no_sentiment_performance" not in result

    def test_validates_input_tensor_shape(self):
        # Arrange
        model = MagicMock()
        X_test = np.random.rand(10, 120).astype(np.float32)  # Missing feature dimension
        y_test = np.array([0.5] * 10)
        feature_names = ["feature1"]

        # Act & Assert
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            validate_model_robustness(model, X_test, y_test, feature_names, has_sentiment=False)

    def test_validates_sentiment_feature_indices(self):
        # Arrange
        model = MagicMock()
        model.predict.side_effect = [
            np.array([[0.5], [0.6]]),
            np.array([[0.4], [0.5]]),
        ]

        X_test = np.random.rand(2, 120, 3).astype(np.float32)
        y_test = np.array([0.5, 0.6])
        # Invalid: sentiment index exceeds feature dimension
        feature_names = ["f1", "f2", "f3", "sentiment_invalid"]

        # Act & Assert
        with pytest.raises(ValueError, match="exceed feature dimension"):
            validate_model_robustness(model, X_test, y_test, feature_names, has_sentiment=True)


@pytest.mark.fast
class TestEvaluateModelPerformance:
    """Test evaluate_model_performance function."""

    def test_basic_performance_evaluation(self):
        # Arrange
        model = MagicMock()
        model.evaluate.side_effect = [
            (0.1, 0.3),  # train_loss, train_rmse
            (0.2, 0.4),  # test_loss, test_rmse
        ]
        model.predict.return_value = np.array([[0.5], [0.6], [0.7]])

        X_train = np.random.rand(10, 120, 5).astype(np.float32)
        y_train = np.random.rand(10).astype(np.float32)
        X_test = np.random.rand(3, 120, 5).astype(np.float32)
        y_test = np.array([0.5, 0.6, 0.7])

        # Act
        result = evaluate_model_performance(model, X_train, y_train, X_test, y_test)

        # Assert
        assert "train_loss" in result
        assert "test_loss" in result
        assert "train_rmse" in result
        assert "test_rmse" in result
        assert "mape" in result
        assert result["train_loss"] == 0.1
        assert result["test_loss"] == 0.2
        assert result["train_rmse"] == 0.3
        assert result["test_rmse"] == 0.4

    def test_mape_calculation_with_scaler(self):
        # Arrange
        from sklearn.preprocessing import MinMaxScaler

        model = MagicMock()
        model.evaluate.side_effect = [(0.1, 0.3), (0.2, 0.4)]
        model.predict.return_value = np.array([[0.5], [0.6], [0.7]])

        X_train = np.random.rand(10, 120, 5).astype(np.float32)
        y_train = np.random.rand(10).astype(np.float32)
        X_test = np.random.rand(3, 120, 5).astype(np.float32)
        y_test = np.array([0.5, 0.6, 0.7])

        scaler = MinMaxScaler()
        scaler.fit([[0.0], [1.0]])

        # Act
        result = evaluate_model_performance(
            model, X_train, y_train, X_test, y_test, close_scaler=scaler
        )

        # Assert
        assert "mape" in result
        assert result["mape"] >= 0

    def test_mape_handles_near_zero_values(self):
        # Arrange
        model = MagicMock()
        model.evaluate.side_effect = [(0.1, 0.3), (0.2, 0.4)]
        model.predict.return_value = np.array([[0.5], [0.6], [0.7]])

        X_train = np.random.rand(10, 120, 5).astype(np.float32)
        y_train = np.random.rand(10).astype(np.float32)
        X_test = np.random.rand(3, 120, 5).astype(np.float32)
        y_test = np.array([1e-10, 1e-9, 1e-8])  # Very small values

        # Act
        result = evaluate_model_performance(model, X_train, y_train, X_test, y_test)

        # Assert - should not raise division by zero
        assert "mape" in result
        assert result["mape"] <= 1000  # Capped at 1000% (inclusive)

    def test_mape_caps_extreme_outliers(self):
        # Arrange
        model = MagicMock()
        model.evaluate.side_effect = [(0.1, 0.3), (0.2, 0.4)]
        model.predict.return_value = np.array([[100.0], [100.0], [100.0]])

        X_train = np.random.rand(10, 120, 5).astype(np.float32)
        y_train = np.random.rand(10).astype(np.float32)
        X_test = np.random.rand(3, 120, 5).astype(np.float32)
        y_test = np.array([1e-10, 1e-10, 1e-10])  # Predictions vastly different from actuals

        # Act
        result = evaluate_model_performance(model, X_train, y_train, X_test, y_test)

        # Assert - individual errors capped at 1000%, mean should be at most 1000%
        assert result["mape"] <= 1000.0


@pytest.mark.fast
class TestConvertToOnnx:
    """Test convert_to_onnx function."""

    @patch("subprocess.run")
    @patch("tempfile.mkdtemp")
    def test_successful_conversion(self, mock_mkdtemp, mock_run, tmp_path):
        # Arrange
        mock_mkdtemp.return_value = str(tmp_path / "temp")
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        model = MagicMock()
        output_path = tmp_path / "model.onnx"
        output_path.touch()

        # Act
        result = convert_to_onnx(model, output_path)

        # Assert
        assert result == output_path
        mock_run.assert_called_once()

    @patch("subprocess.run")
    @patch("tempfile.mkdtemp")
    def test_conversion_failure(self, mock_mkdtemp, mock_run, tmp_path):
        # Arrange
        mock_mkdtemp.return_value = str(tmp_path / "temp")
        mock_run.return_value = MagicMock(returncode=1, stderr="Conversion error")

        model = MagicMock()
        output_path = tmp_path / "model.onnx"

        # Act
        result = convert_to_onnx(model, output_path)

        # Assert
        assert result is None

    @patch("subprocess.run")
    @patch("tempfile.mkdtemp")
    def test_conversion_timeout(self, mock_mkdtemp, mock_run, tmp_path):
        # Arrange
        mock_mkdtemp.return_value = str(tmp_path / "temp")
        mock_run.side_effect = subprocess.TimeoutExpired("tf2onnx", 300)

        model = MagicMock()
        output_path = tmp_path / "model.onnx"

        # Act
        result = convert_to_onnx(model, output_path)

        # Assert
        assert result is None

    @patch("subprocess.run")
    @patch("tempfile.mkdtemp")
    def test_conversion_exception(self, mock_mkdtemp, mock_run, tmp_path):
        # Arrange
        mock_mkdtemp.return_value = str(tmp_path / "temp")
        mock_run.side_effect = Exception("Unexpected error")

        model = MagicMock()
        output_path = tmp_path / "model.onnx"

        # Act
        result = convert_to_onnx(model, output_path)

        # Assert - should return None on exception, not raise
        assert result is None


@pytest.mark.fast
class TestSaveArtifacts:
    """Test save_artifacts function."""

    def test_creates_directory_structure(self, tmp_path):
        # Arrange
        model = MagicMock()
        model.save = MagicMock()

        metadata = {"test": "data"}
        version_id = "2024-01-01_10h_v1"

        # Act
        result = save_artifacts(
            model,
            symbol="BTCUSDT",
            model_type="basic",
            registry_root=tmp_path,
            metadata=metadata,
            version_id=version_id,
            enable_onnx=False,
        )

        # Assert
        expected_dir = tmp_path / "BTCUSDT" / "basic" / version_id
        assert expected_dir.exists()
        assert result.directory == expected_dir

    def test_saves_keras_model(self, tmp_path):
        # Arrange
        model = MagicMock()
        model.save = MagicMock()

        metadata = {"test": "data"}
        version_id = "2024-01-01_10h_v1"

        # Act
        result = save_artifacts(
            model,
            symbol="BTCUSDT",
            model_type="basic",
            registry_root=tmp_path,
            metadata=metadata,
            version_id=version_id,
            enable_onnx=False,
        )

        # Assert
        expected_keras_path = tmp_path / "BTCUSDT" / "basic" / version_id / "model.keras"
        assert result.keras_path == expected_keras_path
        model.save.assert_called_once_with(expected_keras_path)

    def test_saves_metadata_json(self, tmp_path):
        # Arrange
        model = MagicMock()
        model.save = MagicMock()

        metadata = {"test": "data", "number": 123}
        version_id = "2024-01-01_10h_v1"

        # Act
        result = save_artifacts(
            model,
            symbol="BTCUSDT",
            model_type="basic",
            registry_root=tmp_path,
            metadata=metadata,
            version_id=version_id,
            enable_onnx=False,
        )

        # Assert
        assert result.metadata_path.exists()
        with open(result.metadata_path, encoding="utf-8") as f:
            saved_metadata = json.load(f)
        assert saved_metadata == metadata

    @patch("src.ml.training_pipeline.artifacts.convert_to_onnx")
    def test_converts_to_onnx_when_enabled(self, mock_convert, tmp_path):
        # Arrange
        model = MagicMock()
        model.save = MagicMock()

        metadata = {"test": "data"}
        version_id = "2024-01-01_10h_v1"
        onnx_path = tmp_path / "BTCUSDT" / "basic" / version_id / "model.onnx"
        mock_convert.return_value = onnx_path

        # Act
        result = save_artifacts(
            model,
            symbol="BTCUSDT",
            model_type="basic",
            registry_root=tmp_path,
            metadata=metadata,
            version_id=version_id,
            enable_onnx=True,
        )

        # Assert
        assert result.onnx_path == onnx_path
        mock_convert.assert_called_once()

    @patch("src.ml.training_pipeline.artifacts.convert_to_onnx")
    def test_skips_onnx_when_disabled(self, mock_convert, tmp_path):
        # Arrange
        model = MagicMock()
        model.save = MagicMock()

        metadata = {"test": "data"}
        version_id = "2024-01-01_10h_v1"

        # Act
        result = save_artifacts(
            model,
            symbol="BTCUSDT",
            model_type="basic",
            registry_root=tmp_path,
            metadata=metadata,
            version_id=version_id,
            enable_onnx=False,
        )

        # Assert
        assert result.onnx_path is None
        mock_convert.assert_not_called()

    def test_creates_latest_symlink(self, tmp_path):
        # Arrange
        model = MagicMock()
        model.save = MagicMock()

        metadata = {"test": "data"}
        version_id = "2024-01-01_10h_v1"

        # Act
        save_artifacts(
            model,
            symbol="BTCUSDT",
            model_type="basic",
            registry_root=tmp_path,
            metadata=metadata,
            version_id=version_id,
            enable_onnx=False,
        )

        # Assert
        latest_link = tmp_path / "BTCUSDT" / "basic" / "latest"
        assert latest_link.is_symlink()
        assert latest_link.resolve().name == version_id

    def test_updates_existing_symlink(self, tmp_path):
        # Arrange
        model = MagicMock()
        model.save = MagicMock()

        # First version
        version_id_1 = "2024-01-01_10h_v1"
        save_artifacts(
            model,
            symbol="BTCUSDT",
            model_type="basic",
            registry_root=tmp_path,
            metadata={"version": 1},
            version_id=version_id_1,
            enable_onnx=False,
        )

        # Second version
        version_id_2 = "2024-01-01_11h_v1"

        # Act
        save_artifacts(
            model,
            symbol="BTCUSDT",
            model_type="basic",
            registry_root=tmp_path,
            metadata={"version": 2},
            version_id=version_id_2,
            enable_onnx=False,
        )

        # Assert
        latest_link = tmp_path / "BTCUSDT" / "basic" / "latest"
        assert latest_link.resolve().name == version_id_2
