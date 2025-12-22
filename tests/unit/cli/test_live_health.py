"""Tests for atb live-health command."""

import argparse
import json
import os
from unittest.mock import Mock, patch, MagicMock
from http.server import HTTPServer

import pytest

from cli.commands.live_health import _handle, _run_health_server, _HealthCheckHandler


class TestHealthCheckHandler:
    """Tests for the _HealthCheckHandler class."""

    def test_health_endpoint_returns_200(self):
        """Test that /health endpoint returns 200 status."""
        # Arrange
        with patch.object(_HealthCheckHandler, "handle") as mock_handle:
            mock_request = Mock()
            handler = _HealthCheckHandler(mock_request, ("127.0.0.1", 8000), Mock())
            handler.path = "/health"

            with (
                patch.object(handler, "send_response") as mock_send_response,
                patch.object(handler, "send_header"),
                patch.object(handler, "end_headers"),
                patch.object(handler, "wfile") as mock_wfile,
            ):

                mock_wfile.write = Mock()

                # Act
                handler.do_GET()

                # Assert
                mock_send_response.assert_called_once_with(200)

    def test_health_endpoint_returns_json(self):
        """Test that /health endpoint returns JSON response."""
        # Arrange
        with patch.object(_HealthCheckHandler, "handle") as mock_handle:
            mock_request = Mock()
            handler = _HealthCheckHandler(mock_request, ("127.0.0.1", 8000), Mock())
            handler.path = "/health"

            with (
                patch.object(handler, "send_response"),
                patch.object(handler, "send_header") as mock_send_header,
                patch.object(handler, "end_headers"),
                patch.object(handler, "wfile") as mock_wfile,
            ):

                mock_wfile.write = Mock()

                # Act
                handler.do_GET()

                # Assert
                mock_send_header.assert_any_call("Content-Type", "application/json")

    def test_health_endpoint_returns_expected_fields(self):
        """Test that /health endpoint returns expected fields."""
        # Arrange
        with patch.object(_HealthCheckHandler, "handle") as mock_handle:
            mock_request = Mock()
            handler = _HealthCheckHandler(mock_request, ("127.0.0.1", 8000), Mock())
            handler.path = "/health"

            written_data = None

            def capture_write(data):
                nonlocal written_data
                written_data = data

            with (
                patch.object(handler, "send_response"),
                patch.object(handler, "send_header"),
                patch.object(handler, "end_headers"),
                patch.object(handler, "wfile") as mock_wfile,
            ):

                mock_wfile.write = capture_write

                # Act
                handler.do_GET()

                # Assert
                assert written_data is not None
                response = json.loads(written_data.decode())
                assert response["status"] == "healthy"
                assert "timestamp" in response
                assert response["service"] == "ai-trading-bot"

    def test_status_endpoint_returns_200(self):
        """Test that /status endpoint returns 200 status."""
        # Arrange
        with patch.object(_HealthCheckHandler, "handle") as mock_handle:
            mock_request = Mock()
            handler = _HealthCheckHandler(mock_request, ("127.0.0.1", 8000), Mock())
            handler.path = "/status"

            # Patch the imports at their actual locations
            with (
                patch.object(handler, "send_response") as mock_send_response,
                patch.object(handler, "send_header"),
                patch.object(handler, "end_headers"),
                patch.object(handler, "wfile") as mock_wfile,
                patch("config.config_manager.get_config") as mock_get_config,
                patch("database.manager.DatabaseManager") as mock_db_manager,
                patch("data_providers.binance_provider.BinanceProvider") as mock_binance,
            ):

                # Set up mock config
                mock_cfg = Mock()
                mock_provider = Mock()
                mock_provider.is_available.return_value = True
                mock_provider.provider_name = "TestProvider"
                mock_cfg.providers = [mock_provider]
                mock_get_config.return_value = mock_cfg

                # Set up mock database
                mock_db = Mock()
                mock_session = Mock()
                mock_session.__enter__ = Mock(return_value=mock_session)
                mock_session.__exit__ = Mock(return_value=False)
                mock_session.execute = Mock()
                mock_db.get_session.return_value = mock_session
                mock_db_manager.return_value = mock_db

                # Set up mock Binance provider
                import pandas as pd

                mock_binance_instance = Mock()
                mock_binance_instance.get_live_data.return_value = pd.DataFrame({"close": [50000]})
                mock_binance.return_value = mock_binance_instance

                mock_wfile.write = Mock()

                # Act
                handler.do_GET()

                # Assert
                mock_send_response.assert_called_once_with(200)

    def test_unknown_endpoint_returns_404(self):
        """Test that unknown endpoint returns 404."""
        # Arrange
        with patch.object(_HealthCheckHandler, "handle") as mock_handle:
            mock_request = Mock()
            handler = _HealthCheckHandler(mock_request, ("127.0.0.1", 8000), Mock())
            handler.path = "/unknown"

            with patch.object(handler, "send_error") as mock_send_error:
                # Act
                handler.do_GET()

                # Assert
                mock_send_error.assert_called_once_with(404, "Not Found")


class TestRunHealthServer:
    """Tests for the _run_health_server function."""

    def test_starts_http_server_on_specified_port(self):
        """Test that HTTP server is started on specified port."""
        # Arrange & Act
        with patch("cli.commands.live_health.HTTPServer") as mock_http_server:
            mock_server_instance = Mock()
            mock_http_server.return_value = mock_server_instance

            # Start server in a way that doesn't block
            mock_server_instance.serve_forever.side_effect = KeyboardInterrupt()

            try:
                _run_health_server(8000)
            except KeyboardInterrupt:
                pass

            # Assert
            mock_http_server.assert_called_once()
            args, kwargs = mock_http_server.call_args
            assert args[0] == ("", 8000)
            assert args[1] == _HealthCheckHandler

    def test_handles_port_already_in_use(self):
        """Test that error is handled when port is already in use."""
        # Arrange & Act
        with patch("cli.commands.live_health.HTTPServer") as mock_http_server:
            # Simulate "Address already in use" error (errno 48)
            error = OSError("Address already in use")
            error.errno = 48
            mock_http_server.side_effect = error

            # Should not raise, just print warning
            _run_health_server(8000)

            # Assert - function completes without raising


class TestHandleLiveHealth:
    """Tests for the _handle function."""

    def test_starts_health_server_in_background(self):
        """Test that health server is started in background thread."""
        # Arrange
        args = argparse.Namespace(args=["ml_basic", "--paper-trading"])

        # Act
        with (
            patch("cli.commands.live_health.threading.Thread") as mock_thread,
            patch("cli.commands.live_health.forward_to_module_main") as mock_forward,
            patch.dict(os.environ, {}, clear=True),
        ):

            mock_forward.return_value = 0

            result = _handle(args)

            # Assert
            assert result == 0
            mock_thread.assert_called_once()
            args, kwargs = mock_thread.call_args
            assert kwargs["target"] == _run_health_server
            assert kwargs["daemon"] is True

    def test_uses_default_port_8000(self):
        """Test that default port 8000 is used when no PORT env var."""
        # Arrange
        args = argparse.Namespace(args=[])

        # Act
        with (
            patch("cli.commands.live_health.threading.Thread") as mock_thread,
            patch("cli.commands.live_health.forward_to_module_main") as mock_forward,
            patch.dict(os.environ, {}, clear=True),
        ):

            mock_forward.return_value = 0

            result = _handle(args)

            # Assert
            assert result == 0
            args, kwargs = mock_thread.call_args
            assert kwargs["args"] == (8000,)

    def test_uses_custom_port_from_env(self):
        """Test that custom port from PORT env var is used."""
        # Arrange
        args = argparse.Namespace(args=[])

        # Act
        with (
            patch("cli.commands.live_health.threading.Thread") as mock_thread,
            patch("cli.commands.live_health.forward_to_module_main") as mock_forward,
            patch.dict(os.environ, {"PORT": "9000"}),
        ):

            mock_forward.return_value = 0

            result = _handle(args)

            # Assert
            assert result == 0
            args, kwargs = mock_thread.call_args
            assert kwargs["args"] == (9000,)

    def test_forwards_args_to_live_runner(self):
        """Test that arguments are forwarded to live runner."""
        # Arrange
        args = argparse.Namespace(args=["ml_basic", "--paper-trading", "--symbol", "ETHUSDT"])

        # Act
        with (
            patch("cli.commands.live_health.threading.Thread"),
            patch("cli.commands.live_health.forward_to_module_main") as mock_forward,
            patch.dict(os.environ, {}, clear=True),
        ):

            mock_forward.return_value = 0

            result = _handle(args)

            # Assert
            assert result == 0
            mock_forward.assert_called_once_with(
                "src.live.runner", ["ml_basic", "--paper-trading", "--symbol", "ETHUSDT"]
            )

    def test_filters_out_port_argument(self):
        """Test that --port argument is filtered out before forwarding."""
        # Arrange
        args = argparse.Namespace(args=["ml_basic", "--port", "9000", "--paper-trading"])

        # Act
        with (
            patch("cli.commands.live_health.threading.Thread"),
            patch("cli.commands.live_health.forward_to_module_main") as mock_forward,
            patch.dict(os.environ, {}, clear=True),
        ):

            mock_forward.return_value = 0

            result = _handle(args)

            # Assert
            assert result == 0
            # --port and its value should be filtered out
            mock_forward.assert_called_once_with("src.live.runner", ["ml_basic", "--paper-trading"])

    def test_filters_out_help_argument(self):
        """Test that --help argument is filtered out before forwarding."""
        # Arrange
        args = argparse.Namespace(args=["ml_basic", "--help", "--paper-trading"])

        # Act
        with (
            patch("cli.commands.live_health.threading.Thread"),
            patch("cli.commands.live_health.forward_to_module_main") as mock_forward,
            patch.dict(os.environ, {}, clear=True),
        ):

            mock_forward.return_value = 0

            result = _handle(args)

            # Assert
            assert result == 0
            # --help should be filtered out
            mock_forward.assert_called_once_with("src.live.runner", ["ml_basic", "--paper-trading"])

    def test_handles_empty_args(self):
        """Test that empty args are handled correctly."""
        # Arrange
        args = argparse.Namespace(args=None)

        # Act
        with (
            patch("cli.commands.live_health.threading.Thread"),
            patch("cli.commands.live_health.forward_to_module_main") as mock_forward,
            patch.dict(os.environ, {}, clear=True),
        ):

            mock_forward.return_value = 0

            result = _handle(args)

            # Assert
            assert result == 0
            mock_forward.assert_called_once_with("src.live.runner", [])

    def test_returns_forward_result_code(self):
        """Test that return code from forward is passed through."""
        # Arrange
        args = argparse.Namespace(args=[])

        # Act
        with (
            patch("cli.commands.live_health.threading.Thread"),
            patch("cli.commands.live_health.forward_to_module_main") as mock_forward,
            patch.dict(os.environ, {}, clear=True),
        ):

            mock_forward.return_value = 1

            result = _handle(args)

            # Assert
            assert result == 1
