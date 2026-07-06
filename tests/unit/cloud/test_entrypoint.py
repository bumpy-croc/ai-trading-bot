"""Unit tests for the SageMaker training container entrypoint."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.ml.cloud.entrypoint import parse_hyperparameters, run_training


@pytest.mark.fast
class TestParseHyperparameters:
    """Tests for hyperparameter parsing (SageMaker passes all values as strings)."""

    def test_defaults_preserve_current_behavior(self) -> None:
        # Act
        parsed = parse_hyperparameters({})

        # Assert
        assert parsed["symbol"] == "BTCUSDT"
        assert parsed["timeframe"] == "1h"
        assert parsed["epochs"] == 300
        assert parsed["batch_size"] == 32
        assert parsed["sequence_length"] == 120
        assert parsed["force_sentiment"] is False
        assert parsed["force_price_only"] is False
        assert parsed["mixed_precision"] is True
        assert parsed["model_type"] == "cnn_lstm"
        assert parsed["model_variant"] == "default"

    def test_model_type_and_variant_are_parsed(self) -> None:
        # Act
        parsed = parse_hyperparameters({"model_type": "tcn", "model_variant": "lightweight"})

        # Assert
        assert parsed["model_type"] == "tcn"
        assert parsed["model_variant"] == "lightweight"

    def test_numeric_strings_converted(self) -> None:
        # Act
        parsed = parse_hyperparameters({"epochs": "50", "batch_size": "64"})

        # Assert
        assert parsed["epochs"] == 50
        assert parsed["batch_size"] == 64


@pytest.mark.fast
class TestRunTrainingThreading:
    """Tests that parsed hyperparameters reach TrainingConfig."""

    def _base_params(self, **overrides) -> dict:
        params = parse_hyperparameters(
            {"start_date": "2026-05-01T00:00:00", "end_date": "2026-06-01T00:00:00"}
        )
        params.update(overrides)
        return params

    def _run_with_stub_pipeline(self, params: dict) -> tuple[int, MagicMock]:
        """Run entrypoint.run_training against a stubbed pipeline module.

        The pipeline module imports TensorFlow at import time; stubbing it in
        sys.modules keeps these tests fast and isolated from TF.
        """
        fake_pipeline = MagicMock()
        fake_pipeline.run_training_pipeline.return_value = MagicMock(
            success=True, duration_seconds=1.0, artifact_paths=None
        )
        with patch.dict(sys.modules, {"src.ml.training_pipeline.pipeline": fake_pipeline}):
            rc = run_training(params)
        return rc, fake_pipeline.run_training_pipeline

    def test_model_type_and_variant_reach_training_config(self) -> None:
        # Arrange
        params = self._base_params(model_type="tcn_attention", model_variant="deep")

        # Act
        rc, mock_run = self._run_with_stub_pipeline(params)

        # Assert
        assert rc == 0
        ctx = mock_run.call_args.args[0]
        assert ctx.config.model_type == "tcn_attention"
        assert ctx.config.model_variant == "deep"

    def test_defaults_reach_training_config(self) -> None:
        # Arrange
        params = self._base_params()

        # Act
        rc, mock_run = self._run_with_stub_pipeline(params)

        # Assert
        assert rc == 0
        ctx = mock_run.call_args.args[0]
        assert ctx.config.model_type == "cnn_lstm"
        assert ctx.config.model_variant == "default"

    def test_invalid_model_type_fails_before_training(self) -> None:
        # Arrange - TrainingConfig validation must reject unknown architectures
        params = self._base_params(model_type="transformer")
        fake_pipeline = MagicMock()

        # Act
        with (
            patch.dict(sys.modules, {"src.ml.training_pipeline.pipeline": fake_pipeline}),
            patch("src.ml.cloud.entrypoint.write_failure_file") as mock_failure,
        ):
            rc = run_training(params)

        # Assert
        assert rc == 1
        fake_pipeline.run_training_pipeline.assert_not_called()
        mock_failure.assert_called_once()
