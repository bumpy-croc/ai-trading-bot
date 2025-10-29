"""Tests for the atb test command."""

import argparse
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cli.commands.test import _handle_test, register


class TestHandleTest:
    """Tests for the _handle_test function."""

    def test_runs_test_runner_with_category(self):
        """Test that the command runs test runner with the specified category."""
        ns = argparse.Namespace(
            category="unit",
            file=None,
            markers=None,
            coverage=False,
            verbose=False,
            quiet=False,
            pytest_args=None,
        )

        with patch("cli.commands.test.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = _handle_test(ns)

            assert result == 0
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "run_tests.py" in str(args[1])
            assert "unit" in args

    def test_passes_through_pytest_args(self):
        """Test that pytest args are passed through to the test runner."""
        ns = argparse.Namespace(
            category="unit",
            file=None,
            markers=None,
            coverage=False,
            verbose=False,
            quiet=False,
            pytest_args=["-k", "test_example"],
        )

        with patch("cli.commands.test.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = _handle_test(ns)

            assert result == 0
            args = mock_run.call_args[0][0]
            assert "--pytest-args" in args
            assert "-k" in args
            assert "test_example" in args

    def test_handles_file_argument(self):
        """Test that file argument is passed to test runner."""
        ns = argparse.Namespace(
            category=None,
            file="test_example.py",
            markers=None,
            coverage=False,
            verbose=False,
            quiet=False,
            pytest_args=None,
        )

        with patch("cli.commands.test.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = _handle_test(ns)

            assert result == 0
            args = mock_run.call_args[0][0]
            assert "--file" in args
            assert "test_example.py" in args

    def test_handles_markers_argument(self):
        """Test that markers argument is passed to test runner."""
        ns = argparse.Namespace(
            category=None,
            file=None,
            markers="not integration",
            coverage=False,
            verbose=False,
            quiet=False,
            pytest_args=None,
        )

        with patch("cli.commands.test.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = _handle_test(ns)

            assert result == 0
            args = mock_run.call_args[0][0]
            assert "--markers" in args
            assert "not integration" in args

    def test_handles_coverage_flag(self):
        """Test that coverage flag is passed to test runner."""
        ns = argparse.Namespace(
            category="unit",
            file=None,
            markers=None,
            coverage=True,
            verbose=False,
            quiet=False,
            pytest_args=None,
        )

        with patch("cli.commands.test.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = _handle_test(ns)

            assert result == 0
            args = mock_run.call_args[0][0]
            assert "--coverage" in args

    def test_handles_verbose_flag(self):
        """Test that verbose flag is passed to test runner."""
        ns = argparse.Namespace(
            category="unit",
            file=None,
            markers=None,
            coverage=False,
            verbose=True,
            quiet=False,
            pytest_args=None,
        )

        with patch("cli.commands.test.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = _handle_test(ns)

            assert result == 0
            args = mock_run.call_args[0][0]
            assert "--verbose" in args

    def test_returns_nonzero_on_failure(self):
        """Test that command returns non-zero exit code when tests fail."""
        ns = argparse.Namespace(
            category="unit",
            file=None,
            markers=None,
            coverage=False,
            verbose=False,
            quiet=False,
            pytest_args=None,
        )

        with patch("cli.commands.test.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=1)

            result = _handle_test(ns)

            assert result == 1

    def test_handles_keyboard_interrupt(self):
        """Test that KeyboardInterrupt is handled gracefully."""
        ns = argparse.Namespace(
            category="unit",
            file=None,
            markers=None,
            coverage=False,
            verbose=False,
            quiet=False,
            pytest_args=None,
        )

        with patch("cli.commands.test.subprocess.run") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()

            result = _handle_test(ns)

            assert result == 1

    def test_handles_file_not_found(self):
        """Test that FileNotFoundError is handled gracefully."""
        ns = argparse.Namespace(
            category="unit",
            file=None,
            markers=None,
            coverage=False,
            verbose=False,
            quiet=False,
            pytest_args=None,
        )

        with patch("cli.commands.test.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("run_tests.py not found")

            result = _handle_test(ns)

            assert result == 1

    def test_handles_permission_error(self):
        """Test that PermissionError is handled gracefully."""
        ns = argparse.Namespace(
            category="unit",
            file=None,
            markers=None,
            coverage=False,
            verbose=False,
            quiet=False,
            pytest_args=None,
        )

        with patch("cli.commands.test.subprocess.run") as mock_run:
            mock_run.side_effect = PermissionError("Permission denied")

            result = _handle_test(ns)

            assert result == 1


class TestRegister:
    """Tests for the register function."""

    def test_registers_test_command(self):
        """Test that the test command is registered with the parser."""
        mock_subparsers = Mock()
        mock_parser = Mock()
        mock_subparsers.add_parser.return_value = mock_parser
        mock_parser.add_argument = Mock()
        mock_parser.set_defaults = Mock()

        register(mock_subparsers)

        mock_subparsers.add_parser.assert_called_once_with(
            "test",
            help="Run test suite",
            description="Run the test suite using the project's test runner",
        )
