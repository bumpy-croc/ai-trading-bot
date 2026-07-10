"""Tests for cli/commands/train_commands.py::train_model_main.

Focused on Phase 2b item 1 (CLI/config threading): --target-type/
--target-horizon must actually reach the TrainingConfig passed into
run_training_pipeline, not just be accepted by argparse.
"""

from __future__ import annotations

import argparse
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
