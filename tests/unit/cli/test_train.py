"""Tests for atb train commands."""

import argparse
from unittest.mock import patch

import pytest

from cli.commands.train import _handle_model


class TestTrainModelCommand:
    """Tests for the train model command."""

    def test_trains_model_successfully(self):
        """Test that model training succeeds."""
        # Arrange
        args = argparse.Namespace(
            args=[
                "BTCUSDT",
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-12-01",
                "--timeframe",
                "1d",
                "--epochs",
                "10",
                "--batch-size",
                "32",
                "--sequence-length",
                "120",
                "--skip-plots",
                "--skip-robustness",
            ]
        )

        # Act
        with (
            patch("cli.commands.train._TRAINING_AVAILABLE", True),
            patch("cli.commands.train.train_model_main") as mock_train,
        ):

            mock_train.return_value = 0

            result = _handle_model(args)

            # Assert
            assert result == 0

    def test_returns_error_when_training_unavailable(self):
        """Test that error is returned when training is unavailable."""
        # Arrange
        args = argparse.Namespace(
            symbol="BTCUSDT",
        )

        # Act
        with patch("cli.commands.train._TRAINING_AVAILABLE", False):
            result = _handle_model(args)

            # Assert
            # Should return error when training not available
            assert result is not None

    def test_target_type_and_horizon_reach_train_model_main(self):
        """Phase 2b item 1: --target-type/--target-horizon must reach
        train_model_main's args namespace (TrainingConfig construction is
        train_model_main's own responsibility, tested separately)."""
        args = argparse.Namespace(
            args=[
                "BTCUSDT",
                "--target-type",
                "binary_direction",
                "--target-horizon",
                "6",
            ]
        )

        with (
            patch("cli.commands.train._TRAINING_AVAILABLE", True),
            patch("cli.commands.train.train_model_main") as mock_train,
        ):
            mock_train.return_value = 0
            _handle_model(args)

            parsed_args = mock_train.call_args.args[0]
            assert parsed_args.target_type == "binary_direction"
            assert parsed_args.target_horizon == 6

    def test_target_type_defaults_preserve_current_behavior(self):
        args = argparse.Namespace(args=["BTCUSDT"])

        with (
            patch("cli.commands.train._TRAINING_AVAILABLE", True),
            patch("cli.commands.train.train_model_main") as mock_train,
        ):
            mock_train.return_value = 0
            _handle_model(args)

            parsed_args = mock_train.call_args.args[0]
            assert parsed_args.target_type == "regression"
            assert parsed_args.target_horizon == 1

    def test_unknown_target_type_rejected_by_argparse(self):
        args = argparse.Namespace(args=["BTCUSDT", "--target-type", "not_a_real_target"])

        with patch("cli.commands.train._TRAINING_AVAILABLE", True):
            with pytest.raises(SystemExit):
                _handle_model(args)
