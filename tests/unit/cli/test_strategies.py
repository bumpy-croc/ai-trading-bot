"""Tests for atb strategies commands."""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from cli.commands.strategies import _get_modified_strategy_files, _handle_version


class TestGetModifiedStrategyFiles:
    """Tests for the _get_modified_strategy_files function."""

    def test_returns_specified_files(self):
        """Test that specified files are returned."""
        # Arrange
        specified = ["src/strategies/ml_basic.py"]

        # Act
        result = _get_modified_strategy_files(specified)

        # Assert
        assert len(result) == 1
        assert result[0].name == "ml_basic.py"

    def test_gets_files_from_git_when_not_specified(self):
        """Test that git staged files are retrieved when not specified."""
        # Arrange & Act
        with patch("cli.commands.strategies.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="src/strategies/ml_basic.py\nsrc/strategies/ml_adaptive.py\n",
            )

            result = _get_modified_strategy_files(None)

            # Assert
            assert len(result) == 2
            mock_run.assert_called_once()

    def test_filters_out_non_strategy_files(self):
        """Test that non-strategy files are filtered out."""
        # Arrange & Act
        with patch("cli.commands.strategies.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="src/strategies/__init__.py\nsrc/strategies/components/base.py\n",
            )

            result = _get_modified_strategy_files(None)

            # Assert
            assert len(result) == 0

    def test_returns_empty_list_when_git_fails(self):
        """Test that empty list is returned when git command fails."""
        # Arrange & Act
        with patch("cli.commands.strategies.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")

            result = _get_modified_strategy_files(None)

            # Assert
            assert len(result) == 0


class TestHandleVersion:
    """Tests for the _handle_version function."""

    def test_processes_strategy_files_successfully(self):
        """Test that strategy files are processed successfully."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_file = Path(tmpdir) / "ml_basic.py"
            strategy_file.touch()

            args = argparse.Namespace(
                strategies_cmd="version",
                strategy=[str(strategy_file)],
                yes=True,
            )

            # Act
            with (
                patch("cli.commands.strategies._get_modified_strategy_files") as mock_get_files,
                patch("cli.commands.strategies._process_strategy_file") as mock_process,
            ):

                mock_get_files.return_value = [strategy_file]
                mock_process.return_value = True

                result = _handle_version(args)

                # Assert
                assert result == 0
                mock_process.assert_called_once()

    def test_handles_no_strategy_files(self):
        """Test that no strategy files is handled gracefully."""
        # Arrange
        args = argparse.Namespace(
            strategies_cmd="version",
            strategy=None,
            yes=False,
        )

        # Act
        with patch("cli.commands.strategies._get_modified_strategy_files") as mock_get_files:
            mock_get_files.return_value = []

            result = _handle_version(args)

            # Assert
            assert result == 0

    def test_auto_confirms_with_yes_flag(self):
        """Test that auto-confirm works with yes flag."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_file = Path(tmpdir) / "ml_basic.py"
            strategy_file.touch()

            args = argparse.Namespace(
                strategies_cmd="version",
                strategy=[str(strategy_file)],
                yes=True,
            )

            # Act
            with (
                patch("cli.commands.strategies._get_modified_strategy_files") as mock_get_files,
                patch("cli.commands.strategies._process_strategy_file") as mock_process,
            ):

                mock_get_files.return_value = [strategy_file]
                mock_process.return_value = True

                result = _handle_version(args)

                # Assert
                assert result == 0
                # Verify auto_confirm was passed as True
                call_args = mock_process.call_args
                assert call_args[1]["auto_confirm"] is True

    def test_processes_multiple_strategy_files(self):
        """Test that multiple strategy files are processed."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "ml_basic.py"
            file2 = Path(tmpdir) / "ml_adaptive.py"
            file1.touch()
            file2.touch()

            args = argparse.Namespace(
                strategies_cmd="version",
                strategy=[str(file1), str(file2)],
                yes=True,
            )

            # Act
            with (
                patch("cli.commands.strategies._get_modified_strategy_files") as mock_get_files,
                patch("cli.commands.strategies._process_strategy_file") as mock_process,
            ):

                mock_get_files.return_value = [file1, file2]
                mock_process.return_value = True

                result = _handle_version(args)

                # Assert
                assert result == 0
                assert mock_process.call_count == 2
