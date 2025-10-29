"""Tests for atb dev commands (quality and clean)."""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cli.commands.dev import _clean, _quality


class TestQualityCommand:
    """Tests for the _quality function."""

    def test_runs_all_quality_tools(self):
        """Test that all quality tools are executed."""
        ns = argparse.Namespace()

        with patch("cli.commands.dev.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

            result = _quality(ns)

            assert result == 0
            assert mock_run.call_count == 4  # black, ruff, mypy, bandit

    def test_returns_nonzero_when_tool_fails(self):
        """Test that command returns 1 when a tool fails."""
        ns = argparse.Namespace()

        with patch("cli.commands.dev.subprocess.run") as mock_run:
            # First tool passes, second fails, rest pass
            mock_run.side_effect = [
                Mock(returncode=0, stdout="", stderr=""),
                Mock(returncode=1, stdout="", stderr="error output"),
                Mock(returncode=0, stdout="", stderr=""),
                Mock(returncode=0, stdout="", stderr=""),
            ]

            result = _quality(ns)

            assert result == 1

    def test_returns_nonzero_when_tool_missing(self):
        """Test that command returns 1 when a required tool is missing."""
        ns = argparse.Namespace()

        with patch("cli.commands.dev.subprocess.run") as mock_run:
            # First tool missing, rest pass
            mock_run.side_effect = [
                FileNotFoundError("black not found"),
                Mock(returncode=0, stdout="", stderr=""),
                Mock(returncode=0, stdout="", stderr=""),
                Mock(returncode=0, stdout="", stderr=""),
            ]

            result = _quality(ns)

            assert result == 1

    def test_prints_stderr_on_failure(self):
        """Test that stderr is printed when a tool fails."""
        ns = argparse.Namespace()

        with patch("cli.commands.dev.subprocess.run") as mock_run:
            mock_run.side_effect = [
                Mock(returncode=1, stdout="", stderr="formatting errors found"),
                Mock(returncode=0, stdout="", stderr=""),
                Mock(returncode=0, stdout="", stderr=""),
                Mock(returncode=0, stdout="", stderr=""),
            ]

            with patch("builtins.print") as mock_print:
                result = _quality(ns)

                assert result == 1
                # Check that stderr was printed
                printed_output = " ".join(str(call[0][0]) for call in mock_print.call_args_list)
                assert "formatting errors found" in printed_output

    def test_handles_exception_in_tool_execution(self):
        """Test that exceptions during tool execution are handled."""
        ns = argparse.Namespace()

        with patch("cli.commands.dev.subprocess.run") as mock_run:
            mock_run.side_effect = [
                Mock(returncode=0, stdout="", stderr=""),
                Exception("Unexpected error"),
                Mock(returncode=0, stdout="", stderr=""),
                Mock(returncode=0, stdout="", stderr=""),
            ]

            result = _quality(ns)

            assert result == 1

    def test_all_tools_executed_even_if_some_fail(self):
        """Test that all tools are executed even if earlier ones fail."""
        ns = argparse.Namespace()

        with patch("cli.commands.dev.subprocess.run") as mock_run:
            mock_run.side_effect = [
                Mock(returncode=1, stdout="", stderr="error 1"),
                Mock(returncode=1, stdout="", stderr="error 2"),
                Mock(returncode=0, stdout="", stderr=""),
                Mock(returncode=0, stdout="", stderr=""),
            ]

            result = _quality(ns)

            assert result == 1
            assert mock_run.call_count == 4


class TestCleanCommand:
    """Tests for the _clean function."""

    def test_removes_cache_directories(self):
        """Test that cache directories are removed."""
        ns = argparse.Namespace()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test cache directories
            pytest_cache = tmppath / ".pytest_cache"
            ruff_cache = tmppath / ".ruff_cache"
            mypy_cache = tmppath / ".mypy_cache"

            pytest_cache.mkdir()
            ruff_cache.mkdir()
            mypy_cache.mkdir()

            # Verify they exist
            assert pytest_cache.exists()
            assert ruff_cache.exists()
            assert mypy_cache.exists()

            # Mock PROJECT_ROOT to point to temp directory
            with patch("cli.commands.dev.PROJECT_ROOT", tmppath):
                result = _clean(ns)

                assert result == 0
                # Verify directories were removed
                assert not pytest_cache.exists()
                assert not ruff_cache.exists()
                assert not mypy_cache.exists()

    def test_removes_build_directories(self):
        """Test that build directories are removed."""
        ns = argparse.Namespace()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test build directories
            build_dir = tmppath / "build"
            dist_dir = tmppath / "dist"

            build_dir.mkdir()
            dist_dir.mkdir()

            assert build_dir.exists()
            assert dist_dir.exists()

            with patch("cli.commands.dev.PROJECT_ROOT", tmppath):
                result = _clean(ns)

                assert result == 0
                assert not build_dir.exists()
                assert not dist_dir.exists()

    def test_removes_egg_info_directories(self):
        """Test that egg-info directories are removed."""
        ns = argparse.Namespace()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test egg-info directory
            egg_info = tmppath / "test_package.egg-info"
            egg_info.mkdir()

            assert egg_info.exists()

            with patch("cli.commands.dev.PROJECT_ROOT", tmppath):
                result = _clean(ns)

                assert result == 0
                assert not egg_info.exists()

    def test_removes_pycache_directories(self):
        """Test that __pycache__ directories are removed."""
        ns = argparse.Namespace()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create nested __pycache__ directories
            src_dir = tmppath / "src"
            src_dir.mkdir()
            pycache1 = src_dir / "__pycache__"
            pycache1.mkdir()

            tests_dir = tmppath / "tests"
            tests_dir.mkdir()
            pycache2 = tests_dir / "__pycache__"
            pycache2.mkdir()

            assert pycache1.exists()
            assert pycache2.exists()

            with patch("cli.commands.dev.PROJECT_ROOT", tmppath):
                result = _clean(ns)

                assert result == 0
                assert not pycache1.exists()
                assert not pycache2.exists()

    def test_handles_missing_directories_gracefully(self):
        """Test that clean handles missing directories without error."""
        ns = argparse.Namespace()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Don't create any directories - should handle gracefully
            with patch("cli.commands.dev.PROJECT_ROOT", tmppath):
                result = _clean(ns)

                assert result == 0

    def test_does_not_remove_directories_outside_project_root(self):
        """Test that clean only removes directories inside project root."""
        ns = argparse.Namespace()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            project_root = tmppath / "project"
            project_root.mkdir()

            # Create cache directory outside project root (should not be removed)
            outside_cache = tmppath / ".pytest_cache"
            outside_cache.mkdir()

            with patch("cli.commands.dev.PROJECT_ROOT", project_root):
                result = _clean(ns)

                assert result == 0
                # Directory outside project root should still exist
                assert outside_cache.exists()
