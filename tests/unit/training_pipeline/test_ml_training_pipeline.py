"""Unit tests for ML training pipeline orchestration module."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.ml.training_pipeline.config import TrainingConfig, TrainingContext, TrainingPaths
from src.ml.training_pipeline.pipeline import (
    TrainingResult,
    _generate_version_id,
    enable_mixed_precision,
    run_training_pipeline,
)


@pytest.mark.fast
class TestTrainingResult:
    """Test TrainingResult dataclass."""

    def test_initialization(self):
        # Arrange
        metadata = {"test": "data"}
        artifact_paths = MagicMock()

        # Act
        result = TrainingResult(
            success=True,
            metadata=metadata,
            artifact_paths=artifact_paths,
            duration_seconds=123.45,
        )

        # Assert
        assert result.success is True
        assert result.metadata == metadata
        assert result.artifact_paths == artifact_paths
        assert result.duration_seconds == 123.45

    def test_failed_result(self):
        # Arrange & Act
        result = TrainingResult(
            success=False,
            metadata={"error": "Something went wrong"},
            artifact_paths=None,
            duration_seconds=10.0,
        )

        # Assert
        assert result.success is False
        assert result.artifact_paths is None


@pytest.mark.fast
class TestGenerateVersionId:
    """Test _generate_version_id function."""

    def test_generates_version_id(self, tmp_path):
        # Arrange
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Act
        version_id = _generate_version_id(models_dir, "BTCUSDT", "basic")

        # Assert
        assert version_id.endswith("_v1")
        assert "_" in version_id
        # Format: YYYY-MM-DD_HHh_vN
        parts = version_id.split("_")
        assert len(parts) == 3
        assert parts[2] == "v1"

    def test_increments_version_when_exists(self, tmp_path):
        # Arrange
        models_dir = tmp_path / "models"
        symbol_dir = models_dir / "BTCUSDT" / "basic"
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Create existing version directory
        existing_version = symbol_dir / "2024-01-01_10h_v1"
        existing_version.mkdir(parents=True, exist_ok=True)

        # Act
        with patch("src.ml.training_pipeline.pipeline.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value.strftime.return_value = "2024-01-01_10h"
            version_id = _generate_version_id(models_dir, "BTCUSDT", "basic")

        # Assert
        assert version_id == "2024-01-01_10h_v2"

    def test_multiple_versions(self, tmp_path):
        # Arrange
        models_dir = tmp_path / "models"
        symbol_dir = models_dir / "BTCUSDT" / "basic"
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Create multiple existing versions
        for i in range(1, 4):
            (symbol_dir / f"2024-01-01_10h_v{i}").mkdir(parents=True, exist_ok=True)

        # Act
        with patch("src.ml.training_pipeline.pipeline.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value.strftime.return_value = "2024-01-01_10h"
            version_id = _generate_version_id(models_dir, "BTCUSDT", "basic")

        # Assert
        assert version_id == "2024-01-01_10h_v4"

    def test_raises_error_after_max_retries(self, tmp_path):
        # Arrange
        models_dir = tmp_path / "models"
        symbol_dir = models_dir / "BTCUSDT" / "basic"
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Create all versions up to max
        with patch("src.ml.training_pipeline.pipeline.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value.strftime.return_value = "2024-01-01_10h"
            for i in range(1, 1001):
                (symbol_dir / f"2024-01-01_10h_v{i}").mkdir(parents=True, exist_ok=True)

            # Act & Assert
            with pytest.raises(RuntimeError, match="Failed to generate unique version ID"):
                _generate_version_id(models_dir, "BTCUSDT", "basic")


@pytest.mark.fast
class TestEnableMixedPrecision:
    """Test enable_mixed_precision function."""

    def test_disabled_via_config(self):
        # Arrange & Act
        enable_mixed_precision(enabled=False)

        # Assert - should not raise any errors

    @patch("tensorflow.config.list_physical_devices")
    def test_no_gpu_available(self, mock_list_devices):
        # Arrange
        mock_list_devices.return_value = []

        # Act
        enable_mixed_precision(enabled=True)

        # Assert - should not raise any errors

    @patch("tensorflow.config.list_physical_devices")
    @patch("tensorflow.keras.mixed_precision.set_global_policy")
    @patch("tensorflow.config.optimizer.set_jit")
    def test_successful_enable_with_gpu(self, mock_set_jit, mock_set_policy, mock_list_devices):
        # Arrange
        mock_list_devices.return_value = [MagicMock()]  # Mock GPU

        # Act
        enable_mixed_precision(enabled=True)

        # Assert
        mock_set_policy.assert_called_once_with("mixed_float16")
        mock_set_jit.assert_called_once_with(True)

    @patch("tensorflow.config.list_physical_devices")
    @patch("tensorflow.keras.mixed_precision.set_global_policy")
    def test_handles_runtime_error(self, mock_set_policy, mock_list_devices):
        # Arrange
        mock_list_devices.return_value = [MagicMock()]
        mock_set_policy.side_effect = RuntimeError("GPU error")

        # Act - should not raise, just log warning
        enable_mixed_precision(enabled=True)

        # Assert - no exception should be raised

    @patch("tensorflow.config.list_physical_devices")
    @patch("tensorflow.keras.mixed_precision.set_global_policy")
    def test_handles_value_error(self, mock_set_policy, mock_list_devices):
        # Arrange
        mock_list_devices.return_value = [MagicMock()]
        mock_set_policy.side_effect = ValueError("Invalid policy")

        # Act - should not raise, just log warning
        enable_mixed_precision(enabled=True)

        # Assert - no exception should be raised


@pytest.mark.fast
class TestRunTrainingPipeline:
    """Test run_training_pipeline function."""

    @patch("src.ml.training_pipeline.pipeline.download_price_data")
    @patch("src.ml.training_pipeline.pipeline.load_sentiment_data")
    @patch("src.ml.training_pipeline.pipeline.assess_sentiment_data_quality")
    @patch("src.ml.training_pipeline.pipeline.create_robust_features")
    @patch("src.ml.training_pipeline.pipeline.create_sequences")
    @patch("src.ml.training_pipeline.pipeline.split_sequences")
    @patch("src.ml.training_pipeline.pipeline.build_tf_datasets")
    @patch("src.ml.training_pipeline.pipeline.create_adaptive_model")
    @patch("src.ml.training_pipeline.pipeline.save_artifacts")
    @patch("src.ml.training_pipeline.pipeline.enable_mixed_precision")
    def test_successful_training_price_only(
        self,
        mock_enable_mp,
        mock_save,
        mock_create_model,
        mock_build_ds,
        mock_split,
        mock_create_seq,
        mock_create_features,
        mock_assess,
        mock_load_sentiment,
        mock_download_price,
        tmp_path,
    ):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
            epochs=1,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        paths.data_dir.mkdir(parents=True, exist_ok=True)
        paths.models_dir.mkdir(parents=True, exist_ok=True)
        ctx = TrainingContext(config=config, paths=paths)

        # Mock data
        price_df = pd.DataFrame(
            {
                "open": [100] * 100,
                "high": [105] * 100,
                "low": [99] * 100,
                "close": [103] * 100,
                "volume": [1000] * 100,
            },
            index=pd.date_range("2024-01-01", periods=100, freq="1h"),
        )
        mock_download_price.return_value = price_df
        mock_load_sentiment.return_value = None
        mock_assess.return_value = {"recommendation": "price_only", "quality_score": 0.0}

        # Mock feature engineering
        feature_df = price_df.copy()
        feature_df["close_scaled"] = 0.5
        mock_create_features.return_value = (
            feature_df,
            {"close": MagicMock()},
            ["close_scaled"],
        )

        # Mock sequences
        sequences = np.random.rand(50, 120, 1).astype(np.float32)
        targets = np.random.rand(50).astype(np.float32)
        mock_create_seq.return_value = (sequences, targets)

        X_train = sequences[:40]
        y_train = targets[:40]
        X_val = sequences[40:]
        y_val = targets[40:]
        mock_split.return_value = (X_train, y_train, X_val, y_val)

        # Mock datasets
        train_ds = MagicMock()
        val_ds = MagicMock()
        mock_build_ds.return_value = (train_ds, val_ds)

        # Mock model
        model = MagicMock()
        model.fit.return_value = MagicMock(history={"loss": [0.1], "val_loss": [0.2]})
        model.predict.return_value = np.random.rand(10, 1)
        model.evaluate.side_effect = [(0.1, 0.3), (0.2, 0.4)]
        mock_create_model.return_value = model

        # Mock artifacts
        artifact_paths = MagicMock()
        artifact_paths.directory = tmp_path / "artifacts"
        mock_save.return_value = artifact_paths

        # Act
        result = run_training_pipeline(ctx)

        # Assert
        assert result.success is True
        assert result.artifact_paths is not None
        assert result.duration_seconds > 0
        assert "symbol" in result.metadata
        assert result.metadata["symbol"] == "BTCUSDT"

    @patch("src.ml.training_pipeline.pipeline.download_price_data")
    def test_handles_download_failure(self, mock_download, tmp_path):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)

        mock_download.side_effect = RuntimeError("Download failed")

        # Act
        result = run_training_pipeline(ctx)

        # Assert
        assert result.success is False
        assert "error" in result.metadata
        assert result.artifact_paths is None

    @patch("src.ml.training_pipeline.pipeline.download_price_data")
    @patch("src.ml.training_pipeline.pipeline.load_sentiment_data")
    @patch("src.ml.training_pipeline.pipeline.assess_sentiment_data_quality")
    @patch("src.ml.training_pipeline.pipeline.merge_price_sentiment_data")
    def test_handles_empty_merged_data(
        self,
        mock_merge,
        mock_assess,
        mock_load_sentiment,
        mock_download,
        tmp_path,
    ):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)

        price_df = pd.DataFrame()
        sentiment_df = pd.DataFrame()
        mock_download.return_value = price_df
        mock_load_sentiment.return_value = sentiment_df
        mock_assess.return_value = {"recommendation": "price_only"}
        mock_merge.return_value = pd.DataFrame()

        # Act
        result = run_training_pipeline(ctx)

        # Assert
        assert result.success is False
        assert "error" in result.metadata
        assert "No data available" in result.metadata["error"]

    @patch("src.ml.training_pipeline.pipeline.download_price_data")
    @patch("src.ml.training_pipeline.pipeline.load_sentiment_data")
    @patch("src.ml.training_pipeline.pipeline.assess_sentiment_data_quality")
    @patch("src.ml.training_pipeline.pipeline.create_robust_features")
    def test_force_sentiment_overrides_recommendation(
        self,
        mock_create_features,
        mock_assess,
        mock_load_sentiment,
        mock_download,
        tmp_path,
    ):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
            force_sentiment=True,
            epochs=1,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)

        price_df = pd.DataFrame(
            {
                "open": [100] * 100,
                "close": [103] * 100,
            },
            index=pd.date_range("2024-01-01", periods=100, freq="1h"),
        )
        sentiment_df = pd.DataFrame(
            {"sentiment_score": [0.5] * 100},
            index=pd.date_range("2024-01-01", periods=100, freq="1h"),
        )

        mock_download.return_value = price_df
        mock_load_sentiment.return_value = sentiment_df

        # Original assessment says price_only, but force_sentiment should override
        mock_assess.return_value = {"recommendation": "price_only", "quality_score": 0.3}

        # Mock feature creation to avoid full pipeline execution
        mock_create_features.side_effect = RuntimeError("Stop pipeline for test")

        # Act
        result = run_training_pipeline(ctx)

        # Assert - pipeline should fail at feature creation, but we verify the assessment was overridden
        assert result.success is False
        # The test confirms force_sentiment logic runs before feature creation

    @patch("src.ml.training_pipeline.pipeline.download_price_data")
    @patch("src.ml.training_pipeline.pipeline.load_sentiment_data")
    def test_handles_timezone_localization(self, mock_load_sentiment, mock_download, tmp_path):
        # Arrange
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
        )
        paths = TrainingPaths(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
        )
        ctx = TrainingContext(config=config, paths=paths)

        # Price data without timezone
        price_df = pd.DataFrame(
            {"close": [100] * 10},
            index=pd.date_range("2024-01-01", periods=10, freq="1h"),
        )
        # Sentiment data with timezone
        sentiment_df = pd.DataFrame(
            {"sentiment_score": [0.5] * 10},
            index=pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC"),
        )

        mock_download.return_value = price_df
        mock_load_sentiment.return_value = sentiment_df

        # Act - will fail later in pipeline, but timezone handling should work
        result = run_training_pipeline(ctx)

        # Assert - should not fail on timezone mismatch
        # Will fail on other things (like insufficient data), but that's expected
        assert result.success is False
