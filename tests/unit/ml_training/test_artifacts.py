"""Unit tests for ML training pipeline artifacts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ml.training_pipeline.artifacts import (
    ArtifactPaths,
    create_training_plots,
    evaluate_model_performance,
    validate_model_robustness,
)


@pytest.mark.fast
class TestArtifactPaths:
    """Test ArtifactPaths dataclass."""

    def test_creates_artifact_paths(self, tmp_path: Path):
        # Arrange
        directory = tmp_path / "models" / "BTCUSDT" / "basic" / "2025-01-15_14h_v1"
        directory.mkdir(parents=True)
        keras_path = directory / "model.keras"
        onnx_path = directory / "model.onnx"
        metadata_path = directory / "metadata.json"
        plot_path = directory / "training_plot.png"

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


@pytest.mark.fast
class TestCreateTrainingPlots:
    """Test training plot generation."""

    def test_returns_none_when_plots_disabled(self, tmp_path: Path):
        # Arrange
        mock_history = MagicMock()
        mock_model = MagicMock()
        X_test = np.random.rand(10, 60, 5)
        y_test = np.random.rand(10)

        # Act
        result = create_training_plots(
            mock_history,
            mock_model,
            X_test,
            y_test,
            ["close_scaled"],
            "BTCUSDT",
            "basic",
            tmp_path,
            enable_plots=False,
        )

        # Assert
        assert result is None

    @patch("src.ml.training_pipeline.artifacts.plt")
    def test_creates_plot_file_when_enabled(self, mock_plt, tmp_path: Path):
        # Arrange
        mock_history = MagicMock()
        mock_history.history = {
            "loss": [0.1, 0.05, 0.02],
            "val_loss": [0.15, 0.08, 0.03],
        }
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[100], [101], [102]])
        X_test = np.random.rand(3, 60, 5)
        y_test = np.array([99, 100, 101])

        # Act
        result = create_training_plots(
            mock_history,
            mock_model,
            X_test,
            y_test,
            ["close_scaled", "volume_scaled"],
            "ETHUSDT",
            "sentiment",
            tmp_path,
            enable_plots=True,
        )

        # Assert
        assert result is not None
        assert "ETHUSDT_sentiment_training.png" in str(result)
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()

    @patch("src.ml.training_pipeline.artifacts.plt")
    def test_handles_plot_errors_gracefully(self, mock_plt, tmp_path: Path):
        # Arrange
        mock_plt.savefig.side_effect = OSError("Failed to write file")
        mock_history = MagicMock()
        mock_history.history = {"loss": [0.1], "val_loss": [0.15]}
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[100]])
        X_test = np.random.rand(1, 60, 5)
        y_test = np.array([99])

        # Act - should not raise exception
        result = create_training_plots(
            mock_history,
            mock_model,
            X_test,
            y_test,
            ["close_scaled"],
            "BTCUSDT",
            "basic",
            tmp_path,
            enable_plots=True,
        )

        # Assert - returns None on error
        assert result is None


@pytest.mark.fast
class TestValidateModelRobustness:
    """Test model robustness validation."""

    def test_calculates_base_performance(self):
        # Arrange
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[100.5], [101.0], [102.2]])
        X_test = np.random.rand(3, 60, 5)
        y_test = np.array([100, 101, 102])
        feature_names = ["close_scaled", "volume_scaled", "rsi_scaled"]

        # Act
        result = validate_model_robustness(mock_model, X_test, y_test, feature_names, has_sentiment=False)

        # Assert
        assert "base_performance" in result
        assert "mse" in result["base_performance"]
        assert "rmse" in result["base_performance"]
        assert result["base_performance"]["mse"] >= 0
        assert result["base_performance"]["rmse"] >= 0

    def test_evaluates_sentiment_impact_when_present(self):
        # Arrange
        mock_model = MagicMock()
        # First call for base, second for no sentiment
        mock_model.predict.side_effect = [
            np.array([[100.5], [101.0], [102.2]]),  # base prediction
            np.array([[100.3], [100.9], [102.5]]),  # no sentiment prediction
        ]
        X_test = np.random.rand(3, 60, 5)
        y_test = np.array([100, 101, 102])
        feature_names = ["close_scaled", "volume_scaled", "sentiment_score", "sentiment_volume"]

        # Act
        result = validate_model_robustness(mock_model, X_test, y_test, feature_names, has_sentiment=True)

        # Assert
        assert "no_sentiment_performance" in result
        assert "mse" in result["no_sentiment_performance"]
        assert "rmse" in result["no_sentiment_performance"]
        assert "degradation_pct" in result["no_sentiment_performance"]

    def test_skips_sentiment_evaluation_when_absent(self):
        # Arrange
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[100.5], [101.0]])
        X_test = np.random.rand(2, 60, 5)
        y_test = np.array([100, 101])
        feature_names = ["close_scaled", "volume_scaled"]

        # Act
        result = validate_model_robustness(mock_model, X_test, y_test, feature_names, has_sentiment=False)

        # Assert
        assert "no_sentiment_performance" not in result
        assert mock_model.predict.call_count == 1  # Only base prediction


@pytest.mark.fast
class TestEvaluateModelPerformance:
    """Test model performance evaluation."""

    def test_evaluates_train_and_test_metrics(self):
        # Arrange
        mock_model = MagicMock()
        mock_model.evaluate.side_effect = [
            (0.05, 0.22),  # train: loss, rmse
            (0.08, 0.28),  # test: loss, rmse
        ]
        mock_model.predict.return_value = np.array([[100.5], [101.2], [102.1]])

        X_train = np.random.rand(10, 60, 5)
        y_train = np.random.rand(10)
        X_test = np.random.rand(3, 60, 5)
        y_test = np.array([100, 101, 102])

        # Act
        result = evaluate_model_performance(mock_model, X_train, y_train, X_test, y_test, close_scaler=None)

        # Assert - check for actual returned metrics
        assert "train_loss" in result
        assert "train_rmse" in result
        assert "test_loss" in result
        assert "test_rmse" in result
        assert "mape" in result  # Mean Absolute Percentage Error is returned, not MAE
        assert result["train_loss"] == 0.05
        assert result["test_rmse"] == 0.28

    def test_handles_scaler_when_provided(self):
        # Arrange
        mock_model = MagicMock()
        mock_model.evaluate.side_effect = [(0.05, 0.22), (0.08, 0.28)]
        mock_model.predict.return_value = np.array([[0.5], [0.6], [0.7]])

        mock_scaler = MagicMock()
        mock_scaler.inverse_transform.return_value = np.array([[100], [101], [102]])

        X_train = np.random.rand(10, 60, 5)
        y_train = np.random.rand(10)
        X_test = np.random.rand(3, 60, 5)
        y_test = np.array([0.5, 0.55, 0.6])

        # Act
        result = evaluate_model_performance(mock_model, X_train, y_train, X_test, y_test, close_scaler=mock_scaler)

        # Assert
        assert mock_scaler.inverse_transform.called
        assert "test_rmse" in result

    def test_calculates_mape(self):
        # Arrange
        mock_model = MagicMock()
        mock_model.evaluate.side_effect = [(0.05, 0.22), (0.08, 0.28)]
        # Predictions close to actual values
        mock_model.predict.return_value = np.array([[100.5], [101.2], [102.1]])

        X_train = np.random.rand(5, 60, 5)
        y_train = np.random.rand(5)
        X_test = np.random.rand(3, 60, 5)
        y_test = np.array([100, 101, 102])

        # Act
        result = evaluate_model_performance(mock_model, X_train, y_train, X_test, y_test, close_scaler=None)

        # Assert - function calculates MAPE (Mean Absolute Percentage Error)
        assert "mape" in result
        assert result["mape"] >= 0  # MAPE should be non-negative
        assert isinstance(result["mape"], float)
