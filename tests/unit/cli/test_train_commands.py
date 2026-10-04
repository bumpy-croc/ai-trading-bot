"""Tests for cli/commands/train_commands.py::train_model_main.

Focused on Phase 2b item 1 (CLI/config threading): --target-type/
--target-horizon must actually reach the TrainingConfig passed into
run_training_pipeline, not just be accepted by argparse.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _base_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        symbol="BTCUSDT",
        start_date="2024-01-01",
        end_date="2024-01-31",
        timeframe="1h",
        epochs=1,
        batch_size=32,
        sequence_length=10,
        force_sentiment=False,
        force_price_only=False,
        skip_plots=True,
        skip_robustness=True,
        skip_onnx=True,
        disable_mixed_precision=True,
        model_type="cnn_lstm",
        model_variant="default",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.mark.fast
class TestTrainModelMainTargetType:
    @patch("cli.commands.train_commands.run_training_pipeline")
    @patch("cli.commands.train_commands._TENSORFLOW_AVAILABLE", True)
    def test_target_type_and_horizon_reach_training_config(self, mock_run):
        from cli.commands.train_commands import train_model_main

        mock_run.return_value = MagicMock(
            success=True, duration_seconds=1.0, metadata={"evaluation_results": {}}
        )
        args = _base_args(target_type="binary_direction", target_horizon=6)

        train_model_main(args)

        ctx = mock_run.call_args.args[0]
        assert ctx.config.target_type == "binary_direction"
        assert ctx.config.target_horizon == 6

    @patch("cli.commands.train_commands.run_training_pipeline")
    @patch("cli.commands.train_commands._TENSORFLOW_AVAILABLE", True)
    def test_missing_target_type_defaults_to_regression(self, mock_run):
        """args without target_type/target_horizon at all (e.g. an older
        caller not yet updated) must still work -- getattr fallback."""
        from cli.commands.train_commands import train_model_main

        mock_run.return_value = MagicMock(
            success=True, duration_seconds=1.0, metadata={"evaluation_results": {}}
        )
        args = argparse.Namespace(
            symbol="BTCUSDT",
            start_date="2024-01-01",
            end_date="2024-01-31",
            timeframe="1h",
            epochs=1,
            batch_size=32,
            sequence_length=10,
            force_sentiment=False,
            force_price_only=False,
        )

        train_model_main(args)

        ctx = mock_run.call_args.args[0]
        assert ctx.config.target_type == "regression"
        assert ctx.config.target_horizon == 1

    @patch("cli.commands.train_commands.run_training_pipeline")
    @patch("cli.commands.train_commands._TENSORFLOW_AVAILABLE", True)
    def test_primary_model_type_reaches_training_config(self, mock_run):
        """Phase 2b item 3: --primary-model-type must reach TrainingConfig
        for --target-type meta_label to be trainable at all."""
        from cli.commands.train_commands import train_model_main

        mock_run.return_value = MagicMock(
            success=True, duration_seconds=1.0, metadata={"evaluation_results": {}}
        )
        args = _base_args(target_type="meta_label", primary_model_type="basic")

        train_model_main(args)

        ctx = mock_run.call_args.args[0]
        assert ctx.config.target_type == "meta_label"
        assert ctx.config.primary_model_type == "basic"

    @patch("cli.commands.train_commands.run_training_pipeline")
    @patch("cli.commands.train_commands._TENSORFLOW_AVAILABLE", True)
    def test_missing_primary_model_type_defaults_to_none(self, mock_run):
        from cli.commands.train_commands import train_model_main

        mock_run.return_value = MagicMock(
            success=True, duration_seconds=1.0, metadata={"evaluation_results": {}}
        )
        args = _base_args()

        train_model_main(args)

        ctx = mock_run.call_args.args[0]
        assert ctx.config.primary_model_type is None


@pytest.mark.fast
class TestTrainModelMainSetLatest:
    """GH #1242: training promotes only on an explicit set_latest opt-in."""

    @staticmethod
    def _config_for(args):
        from cli.commands.train_commands import train_model_main

        with (
            patch("cli.commands.train_commands.run_training_pipeline") as mock_run,
            patch("cli.commands.train_commands._TENSORFLOW_AVAILABLE", True),
        ):
            mock_run.return_value = MagicMock(
                success=True, duration_seconds=1.0, metadata={"evaluation_results": {}}
            )
            train_model_main(args)
        return mock_run.call_args.args[0].config

    def test_default_does_not_set_latest(self):
        assert self._config_for(_base_args()).set_latest is False

    def test_missing_attribute_does_not_set_latest(self):
        args = _base_args()
        del args.model_type
        assert self._config_for(args).set_latest is False

    def test_flag_reaches_training_config(self):
        assert self._config_for(_base_args(set_latest=True)).set_latest is True

    @pytest.mark.parametrize("set_latest", [True, False])
    def test_promotion_status_is_reported(self, set_latest, capsys):
        from cli.commands.train_commands import train_model_main

        version_dir = Path("/registry/BTCUSDT/basic/2024-01-01_00h_v1")
        with (
            patch("cli.commands.train_commands.run_training_pipeline") as mock_run,
            patch("cli.commands.train_commands._TENSORFLOW_AVAILABLE", True),
        ):
            mock_run.return_value = MagicMock(
                success=True,
                duration_seconds=1.0,
                metadata={"evaluation_results": {}},
                artifact_paths=MagicMock(directory=version_dir),
            )
            train_model_main(_base_args(set_latest=set_latest))

        out = capsys.readouterr().out
        assert ("'latest' now points to" in out) is set_latest
        assert ("'latest' was NOT changed" in out) is not set_latest


@pytest.mark.fast
class TestTrainPriceModelMainSetLatest:
    """The legacy price trainer wrote ``latest`` unconditionally (GH #1242)."""

    @staticmethod
    def _run(tmp_path, **overrides):
        import numpy as np
        import pandas as pd

        from cli.commands import train_commands

        n = 40
        frame = pd.DataFrame({"close": np.linspace(1.0, 2.0, n)})
        model = MagicMock()
        model.fit.return_value = MagicMock(history={"loss": [0.1]})
        model.evaluate.return_value = (0.1, 0.1)
        model.predict.return_value = np.ones((8, 1), dtype=np.float32)
        args = argparse.Namespace(
            symbol="BTCUSDT",
            start_date="2024-01-01",
            end_date="2024-01-31",
            timeframe="1h",
            epochs=1,
            batch_size=8,
            sequence_length=4,
            **overrides,
        )
        with (
            patch.object(train_commands, "_TENSORFLOW_AVAILABLE", True),
            patch.object(train_commands, "configure_gpu", return_value=None),
            patch.object(train_commands, "_download_price_frame", return_value=frame),
            patch.object(
                train_commands,
                "_prepare_price_only_sequences",
                return_value=(
                    np.zeros((n, 4, 5), dtype=np.float32),
                    np.zeros(n, dtype=np.float32),
                    ["f"] * 5,
                ),
            ),
            patch.object(train_commands, "_build_price_only_model", return_value=model),
            patch.object(train_commands, "callbacks", MagicMock()),
            patch.object(train_commands, "MODEL_REGISTRY", tmp_path),
            patch("src.ml.training_pipeline.artifacts.convert_to_onnx"),
        ):
            return train_commands.train_price_model_main(args)

    def test_default_leaves_latest_alone(self, tmp_path):
        assert self._run(tmp_path) == 0

        type_dir = tmp_path / "BTCUSDT" / "basic"
        assert len(list(type_dir.glob("*_v1"))) == 1
        assert not (type_dir / "latest").exists()
        assert not (type_dir / "latest").is_symlink()

    def test_set_latest_points_latest_at_new_bundle(self, tmp_path):
        assert self._run(tmp_path, set_latest=True) == 0

        type_dir = tmp_path / "BTCUSDT" / "basic"
        (bundle,) = type_dir.glob("*_v1")
        assert (type_dir / "latest").resolve() == bundle.resolve()
