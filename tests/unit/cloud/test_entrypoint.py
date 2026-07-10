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

    def test_target_type_defaults_preserve_current_behavior(self) -> None:
        parsed = parse_hyperparameters({})
        assert parsed["target_type"] == "regression"
        assert parsed["target_horizon"] == 1

    def test_target_type_and_horizon_are_parsed(self) -> None:
        parsed = parse_hyperparameters({"target_type": "binary_direction", "target_horizon": "6"})
        assert parsed["target_type"] == "binary_direction"
        assert parsed["target_horizon"] == 6

    def test_primary_model_type_defaults_to_none(self) -> None:
        parsed = parse_hyperparameters({})
        assert parsed["primary_model_type"] is None

    def test_primary_model_type_empty_string_parses_to_none(self) -> None:
        """orchestrator.py's _build_job_spec sends "" for an unset
        primary_model_type (SageMaker hyperparameters must be strings) --
        must round-trip back to None, not the literal empty string."""
        parsed = parse_hyperparameters({"primary_model_type": ""})
        assert parsed["primary_model_type"] is None

    def test_primary_model_type_is_parsed(self) -> None:
        parsed = parse_hyperparameters({"primary_model_type": "basic"})
        assert parsed["primary_model_type"] == "basic"

    def test_use_mock_data_defaults_to_false(self) -> None:
        parsed = parse_hyperparameters({})
        assert parsed["use_mock_data"] is False

    def test_use_mock_data_is_parsed(self) -> None:
        parsed = parse_hyperparameters({"use_mock_data": "true"})
        assert parsed["use_mock_data"] is True


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

    def test_target_type_and_horizon_reach_training_config(self) -> None:
        # Arrange
        params = self._base_params(
            model_type="tft", target_type="binary_direction", target_horizon=6
        )

        # Act
        rc, mock_run = self._run_with_stub_pipeline(params)

        # Assert
        assert rc == 0
        ctx = mock_run.call_args.args[0]
        assert ctx.config.target_type == "binary_direction"
        assert ctx.config.target_horizon == 6

    def test_target_type_defaults_reach_training_config(self) -> None:
        params = self._base_params()

        rc, mock_run = self._run_with_stub_pipeline(params)

        assert rc == 0
        ctx = mock_run.call_args.args[0]
        assert ctx.config.target_type == "regression"
        assert ctx.config.target_horizon == 1

    def test_primary_model_type_reaches_training_config(self) -> None:
        params = self._base_params(target_type="meta_label", primary_model_type="basic")

        rc, mock_run = self._run_with_stub_pipeline(params)

        assert rc == 0
        ctx = mock_run.call_args.args[0]
        assert ctx.config.target_type == "meta_label"
        assert ctx.config.primary_model_type == "basic"

    def test_primary_model_type_default_reaches_training_config_as_none(self) -> None:
        params = self._base_params()

        rc, mock_run = self._run_with_stub_pipeline(params)

        assert rc == 0
        ctx = mock_run.call_args.args[0]
        assert ctx.config.primary_model_type is None

    def test_use_mock_data_reaches_training_config(self) -> None:
        params = self._base_params(use_mock_data=True)

        rc, mock_run = self._run_with_stub_pipeline(params)

        assert rc == 0
        ctx = mock_run.call_args.args[0]
        assert ctx.config.use_mock_data is True

    def test_use_mock_data_default_reaches_training_config_as_false(self) -> None:
        params = self._base_params()

        rc, mock_run = self._run_with_stub_pipeline(params)

        assert rc == 0
        ctx = mock_run.call_args.args[0]
        assert ctx.config.use_mock_data is False

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
