"""Tests for atb dev commands (quality and clean)."""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from cli.commands.dev import _clean, _dashboard, _quality, _setup, _venv


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
                printed_output = " ".join(
                    str(call[0][0]) for call in mock_print.call_args_list if call[0]
                )
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


class TestSetupCommand:
    """Tests for the _setup function."""

    def test_runs_setup_successfully(self):
        """Test that setup runs successfully."""
        # Arrange
        ns = argparse.Namespace()

        # Act
        with (
            patch("cli.commands.dev._check_requirements") as mock_check,
            patch("cli.commands.dev._setup_environment_file") as mock_setup_env,
            patch("cli.commands.dev._install_python_dependencies") as mock_install,
            patch("cli.commands.dev._setup_postgresql") as mock_setup_pg,
            patch("cli.commands.dev._run_migrations") as mock_migrate,
            patch("cli.commands.dev._test_setup") as mock_test,
            patch("cli.commands.dev._print_next_steps") as mock_print_steps,
        ):

            mock_check.return_value = True
            mock_install.return_value = True
            mock_setup_pg.return_value = True
            mock_migrate.return_value = True
            mock_test.return_value = True

            result = _setup(ns)

            # Assert
            assert result == 0
            mock_check.assert_called_once()
            mock_setup_env.assert_called_once()
            mock_install.assert_called_once()
            mock_setup_pg.assert_called_once()
            mock_migrate.assert_called_once()
            mock_test.assert_called_once()
            mock_print_steps.assert_called_once()

    def test_returns_error_when_requirements_missing(self):
        """Test that error is returned when requirements are missing."""
        # Arrange
        ns = argparse.Namespace()

        # Act
        with patch("cli.commands.dev._check_requirements") as mock_check:
            mock_check.return_value = False

            result = _setup(ns)

            # Assert
            assert result == 1


class TestVenvCommand:
    """Tests for the _venv function."""

    def test_creates_venv_successfully(self):
        """Test that venv is created successfully."""
        # Arrange
        ns = argparse.Namespace()

        # Act
        with (
            patch("cli.commands.dev.subprocess.run") as mock_run,
            patch("cli.commands.dev.PROJECT_ROOT") as mock_root,
        ):

            mock_root.__truediv__ = lambda self, other: Path(f"/tmp/{other}")
            mock_run.return_value = Mock(returncode=0)

            result = _venv(ns)

            # Assert
            assert result == 0
            # Should be called multiple times (venv creation, pip upgrade, install -e ., install requirements)
            assert mock_run.call_count >= 4

    def test_returns_error_when_venv_creation_fails(self):
        """Test that error is returned when venv creation fails."""
        # Arrange
        ns = argparse.Namespace()

        # Act
        with (
            patch("cli.commands.dev.subprocess.run") as mock_run,
            patch("cli.commands.dev.PROJECT_ROOT") as mock_root,
            patch("cli.commands.dev.Path") as mock_path_class,
        ):

            mock_venv_path = Mock()
            mock_venv_path.exists.return_value = False
            mock_root.__truediv__ = lambda self, other: mock_venv_path

            mock_run.return_value = Mock(returncode=1)

            result = _venv(ns)

            # Assert
            assert result == 1


class TestDashboardCommand:
    """Tests for the _dashboard function."""

    def test_starts_dashboard_successfully(self):
        """Test that dashboard starts successfully."""
        # Arrange
        ns = argparse.Namespace()

        # Act
        with (
            patch(
                "src.dashboards.monitoring.dashboard.MonitoringDashboard"
            ) as mock_dashboard_class,
            patch("src.infrastructure.logging.config.configure_logging"),
            patch.dict("os.environ", {"PORT": "8090", "HOST": "0.0.0.0"}),
        ):

            mock_dashboard = Mock()
            mock_dashboard.run.return_value = None
            mock_dashboard_class.return_value = mock_dashboard

            result = _dashboard(ns)

            # Assert
            assert result == 0
            mock_dashboard.run.assert_called_once()

    def test_returns_error_when_dashboard_fails(self):
        """Test that error is returned when dashboard fails to start."""
        # Arrange
        ns = argparse.Namespace()

        # Act
        with (
            patch(
                "src.dashboards.monitoring.dashboard.MonitoringDashboard"
            ) as mock_dashboard_class,
            patch("src.infrastructure.logging.config.configure_logging"),
            patch.dict("os.environ", {"PORT": "8090", "HOST": "0.0.0.0"}),
        ):

            mock_dashboard_class.side_effect = Exception("Failed to start")

            result = _dashboard(ns)

            # Assert
            assert result == 1
