"""Unit tests for ML training pipeline orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ml.training_pipeline.pipeline import (
    TrainingResult,
    _generate_version_id,
    enable_mixed_precision,
)


@pytest.mark.fast
class TestGenerateVersionId:
    """Test auto-incrementing version ID generation."""

    def test_generates_v1_for_new_timestamp(self, tmp_path: Path):
        # Arrange
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        symbol = "BTCUSDT"
        model_type = "basic"

        # Act
        version_id = _generate_version_id(models_dir, symbol, model_type)

        # Assert
        assert version_id.endswith("_v1")
        assert len(version_id.split("_")) == 3  # YYYY-MM-DD_HHh_v1

    def test_increments_version_when_collision(self, tmp_path: Path):
        # Arrange
        models_dir = tmp_path / "models"
        symbol_dir = models_dir / "BTCUSDT" / "basic"
        symbol_dir.mkdir(parents=True)

        # Create existing version directories with same timestamp
        with patch("src.ml.training_pipeline.pipeline.datetime") as mock_dt:
            mock_dt.utcnow.return_value.strftime.return_value = "2025-01-15_14h"

            # Create v1 and v2 to force v3
            (symbol_dir / "2025-01-15_14h_v1").mkdir()
            (symbol_dir / "2025-01-15_14h_v2").mkdir()

            # Act
            version_id = _generate_version_id(models_dir, "BTCUSDT", "basic")

        # Assert
        assert version_id == "2025-01-15_14h_v3"

    def test_handles_multiple_increments(self, tmp_path: Path):
        # Arrange
        models_dir = tmp_path / "models"
        symbol_dir = models_dir / "ETHUSDT" / "sentiment"
        symbol_dir.mkdir(parents=True)

        with patch("src.ml.training_pipeline.pipeline.datetime") as mock_dt:
            mock_dt.utcnow.return_value.strftime.return_value = "2025-01-15_09h"

            # Create many versions
            for i in range(1, 6):
                (symbol_dir / f"2025-01-15_09h_v{i}").mkdir()

            # Act
            version_id = _generate_version_id(models_dir, "ETHUSDT", "sentiment")

        # Assert
        assert version_id == "2025-01-15_09h_v6"

    def test_case_insensitive_symbol(self, tmp_path: Path):
        # Arrange
        models_dir = tmp_path / "models"

        # Act
        version_id = _generate_version_id(models_dir, "btcusdt", "basic")

        # Assert - should uppercase symbol internally
        assert version_id.endswith("_v1")


@pytest.mark.fast
class TestEnableMixedPrecision:
    """Test mixed precision configuration."""

    def test_logs_when_explicitly_disabled(self, caplog):
        # Arrange
        caplog.set_level(logging.INFO)

        # Act
        enable_mixed_precision(enabled=False)

        # Assert
        assert any(
            "explicitly disabled via configuration" in record.message for record in caplog.records
        )

    @patch("src.ml.training_pipeline.pipeline.tf")
    def test_logs_when_no_gpu_detected(self, mock_tf, caplog):
        # Arrange
        mock_tf.config.list_physical_devices.return_value = []
        caplog.set_level(logging.INFO)

        # Act
        enable_mixed_precision(enabled=True)

        # Assert
        assert any("no GPU detected" in record.message for record in caplog.records)

    @patch("src.ml.training_pipeline.pipeline.tf")
    def test_enables_mixed_precision_with_gpu(self, mock_tf, caplog):
        # Arrange
        mock_tf.config.list_physical_devices.return_value = [MagicMock()]  # Mock GPU
        caplog.set_level(logging.INFO)

        # Act
        enable_mixed_precision(enabled=True)

        # Assert
        mock_tf.keras.mixed_precision.set_global_policy.assert_called_once_with("mixed_float16")
        mock_tf.config.optimizer.set_jit.assert_called_once_with(True)
        assert any("Enabled mixed precision and XLA" in record.message for record in caplog.records)

    @patch("src.ml.training_pipeline.pipeline.tf")
    def test_handles_mixed_precision_failure_gracefully(self, mock_tf, caplog):
        # Arrange
        mock_tf.config.list_physical_devices.return_value = [MagicMock()]
        mock_tf.keras.mixed_precision.set_global_policy.side_effect = RuntimeError("GPU error")
        caplog.set_level(logging.WARNING)

        # Act - should not raise exception
        enable_mixed_precision(enabled=True)

        # Assert
        assert any(
            "Failed to enable mixed precision" in record.message for record in caplog.records
        )


@pytest.mark.fast
class TestTrainingResult:
    """Test TrainingResult dataclass."""

    def test_creates_successful_result(self):
        # Arrange
        metadata = {"symbol": "BTCUSDT", "model_type": "basic"}
        artifact_paths = MagicMock()

        # Act
        result = TrainingResult(
            success=True, metadata=metadata, artifact_paths=artifact_paths, duration_seconds=120.5
        )

        # Assert
        assert result.success is True
        assert result.metadata == metadata
        assert result.artifact_paths == artifact_paths
        assert result.duration_seconds == 120.5

    def test_creates_failed_result(self):
        # Arrange
        error_metadata = {"error": "Training failed due to insufficient data"}

        # Act
        result = TrainingResult(
            success=False, metadata=error_metadata, artifact_paths=None, duration_seconds=5.2
        )

        # Assert
        assert result.success is False
        assert result.metadata["error"] == "Training failed due to insufficient data"
        assert result.artifact_paths is None
        assert result.duration_seconds == 5.2
