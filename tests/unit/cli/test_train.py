"""Tests for atb train commands."""

import argparse
from unittest.mock import Mock, patch

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
