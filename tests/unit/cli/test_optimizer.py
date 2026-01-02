"""Tests for atb optimizer command."""

import argparse
import json
import uuid
from unittest.mock import Mock, patch

import pytest

from cli.commands.optimizer import _handle
from src.infrastructure.runtime.paths import get_project_root


class TestOptimizerHandle:
    """Tests for the _handle function."""

    @pytest.fixture
    def default_args(self):
        """Provides default optimizer arguments."""
        return argparse.Namespace(
            strategy="ml_basic",
            symbol="BTCUSDT",
            timeframe="1h",
            days=30,
            initial_balance=10000,
            provider="mock",
            no_cache=False,
            output="artifacts/optimizer_report.json",
            seed=42,
            no_validate=False,
            persist=False,
        )

    @pytest.fixture
    def temp_output_path(self):
        """Provides a temporary output path within the project directory."""
        project_root = get_project_root()
        unique_name = f"test_optimizer_{uuid.uuid4().hex[:8]}.json"
        output_path = project_root / "artifacts" / unique_name
        yield output_path
        # Cleanup after test
        if output_path.exists():
            output_path.unlink()

    @pytest.fixture
    def mock_experiment_result(self):
        """Provides a mock experiment result."""
        mock_result = Mock()
        mock_result.total_trades = 10
        mock_result.win_rate = 60.0
        mock_result.total_return = 15.5
        mock_result.annualized_return = 25.0
        mock_result.max_drawdown = 8.5
        mock_result.sharpe_ratio = 1.5
        mock_result.final_balance = 11550.0
        return mock_result

    def test_runs_baseline_experiment_successfully(
        self, default_args, mock_experiment_result, temp_output_path
    ):
        """Test that baseline experiment runs successfully."""
        # Arrange
        default_args.output = str(temp_output_path)

        with (
            patch("src.optimizer.runner.ExperimentRunner") as mock_runner_class,
            patch("src.optimizer.analyzer.PerformanceAnalyzer") as mock_analyzer_class,
        ):

            mock_runner = Mock()
            mock_runner.run.return_value = mock_experiment_result
            mock_runner_class.return_value = mock_runner

            mock_analyzer = Mock()
            mock_analyzer.analyze.return_value = []
            mock_analyzer_class.return_value = mock_analyzer

            # Act
            result = _handle(default_args)

            # Assert
            assert result == 0
            mock_runner.run.assert_called_once()
            assert temp_output_path.exists()

    def test_generates_suggestions_from_analyzer(
        self, default_args, mock_experiment_result, temp_output_path
    ):
        """Test that suggestions are generated from performance analyzer."""
        # Arrange
        default_args.output = str(temp_output_path)

        mock_suggestion = Mock()
        mock_suggestion.target = "risk_per_trade"
        mock_suggestion.change = {"MlBasic.risk_per_trade": 0.02}
        mock_suggestion.rationale = "Increase risk for better returns"
        mock_suggestion.expected_delta = 5.0
        mock_suggestion.confidence = 0.8

        with (
            patch("src.optimizer.runner.ExperimentRunner") as mock_runner_class,
            patch("src.optimizer.analyzer.PerformanceAnalyzer") as mock_analyzer_class,
        ):

            mock_runner = Mock()
            mock_runner.run.return_value = mock_experiment_result
            mock_runner_class.return_value = mock_runner

            mock_analyzer = Mock()
            mock_analyzer.analyze.return_value = [mock_suggestion]
            mock_analyzer_class.return_value = mock_analyzer

            # Act
            result = _handle(default_args)

            # Assert
            assert result == 0
            assert temp_output_path.exists()

            with open(temp_output_path) as f:
                report = json.load(f)
                assert len(report["suggestions"]) == 1
                assert report["suggestions"][0]["target"] == "risk_per_trade"

    def test_skips_validation_when_no_validate_flag_set(
        self, default_args, mock_experiment_result, temp_output_path
    ):
        """Test that validation is skipped when no_validate flag is set."""
        # Arrange
        default_args.output = str(temp_output_path)
        default_args.no_validate = True

        with (
            patch("src.optimizer.runner.ExperimentRunner") as mock_runner_class,
            patch("src.optimizer.analyzer.PerformanceAnalyzer") as mock_analyzer_class,
        ):

            mock_runner = Mock()
            mock_runner.run.return_value = mock_experiment_result
            mock_runner_class.return_value = mock_runner

            mock_analyzer = Mock()
            mock_analyzer.analyze.return_value = []
            mock_analyzer_class.return_value = mock_analyzer

            # Act
            result = _handle(default_args)

            # Assert
            assert result == 0
            # Validation should not be called when no_validate is True
            assert mock_runner.run.call_count == 1  # Only baseline

    def test_returns_error_on_exception(self, default_args):
        """Test that error is returned when exception occurs."""
        # Arrange
        with patch("src.optimizer.runner.ExperimentRunner") as mock_runner_class:
            mock_runner_class.side_effect = Exception("Experiment failed")

            # Act
            result = _handle(default_args)

            # Assert
            assert result == 1

    def test_rejects_invalid_output_path(self, default_args):
        """Test that invalid output path is rejected."""
        # Arrange
        default_args.output = "../../../etc/passwd"

        # Act
        result = _handle(default_args)

        # Assert
        assert result == 1
