"""Tests for atb dashboards commands."""

import argparse
from unittest.mock import Mock, patch

import pytest

from cli.commands.dashboards import _handle_list, _handle_run


class TestDashboardsListCommand:
    """Tests for the dashboards list command."""

    def test_lists_available_dashboards(self):
        """Test that available dashboards are listed."""
        # Arrange
        args = argparse.Namespace(dash_cmd="list")

        mock_dashboard = Mock()
        mock_dashboard.summary = "Monitoring dashboard for live trading"

        # Act
        with patch("cli.commands.dashboards.discover_dashboards") as mock_discover:
            mock_discover.return_value = {"monitoring": mock_dashboard}

            result = _handle_list(args)

            # Assert
            assert result == 0
            mock_discover.assert_called_once()

    def test_handles_no_dashboards_found(self):
        """Test that no dashboards found is handled gracefully."""
        # Arrange
        args = argparse.Namespace(dash_cmd="list")

        # Act
        with patch("cli.commands.dashboards.discover_dashboards") as mock_discover:
            mock_discover.return_value = {}

            result = _handle_list(args)

            # Assert
            assert result == 0


class TestDashboardsRunCommand:
    """Tests for the dashboards run command."""

    def test_runs_dashboard_successfully(self):
        """Test that dashboard runs successfully."""
        # Arrange
        args = argparse.Namespace(
            dash_cmd="run",
            name="monitoring",
            host="127.0.0.1",
            port=8000,
            debug=False,
            db_url=None,
            update_interval=None,
            logs_dir=None,
        )

        mock_dashboard = Mock()
        mock_dashboard.module_name = "src.dashboards.monitoring"
        mock_dashboard.object_name = "MonitoringDashboard"

        mock_module = Mock()
        mock_dashboard_class = Mock()
        mock_dashboard_instance = Mock()
        mock_dashboard_class.return_value = mock_dashboard_instance
        setattr(mock_module, "MonitoringDashboard", mock_dashboard_class)

        # Act
        with (
            patch("cli.commands.dashboards.discover_dashboards") as mock_discover,
            patch("cli.commands.dashboards._import_module") as mock_import,
            patch("cli.commands.dashboards.call_with_supported_params") as mock_call,
        ):

            mock_discover.return_value = {"monitoring": mock_dashboard}
            mock_import.return_value = mock_module
            mock_call.side_effect = [mock_dashboard_instance, None]

            result = _handle_run(args)

            # Assert
            assert result == 0

    def test_returns_error_when_dashboard_not_found(self):
        """Test that error is returned when dashboard is not found."""
        # Arrange
        args = argparse.Namespace(
            dash_cmd="run",
            name="nonexistent",
            host="127.0.0.1",
            port=8000,
            debug=False,
            db_url=None,
            update_interval=None,
            logs_dir=None,
        )

        # Act
        with patch("cli.commands.dashboards.discover_dashboards") as mock_discover:
            mock_discover.return_value = {}

            result = _handle_run(args)

            # Assert
            assert result == 1
