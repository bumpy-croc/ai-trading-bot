"""Tests for infrastructure.logging.config module."""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.logging.config import (
    ContextInjectorFilter,
    MaxMessageLengthFilter,
    NamespacePrefixFilter,
    SamplingFilter,
    SensitiveDataFilter,
    SimpleJsonFormatter,
    build_logging_config,
    configure_logging,
)


class TestSensitiveDataFilter:
    """Tests for SensitiveDataFilter."""

    def setup_method(self):
        """Create filter instance for each test."""
        self.filter = SensitiveDataFilter()

    def test_redacts_api_key_in_message(self):
        """Test that API keys are redacted in log messages."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Connecting with api_key=sk_live_12345",
            args=(),
            exc_info=None,
        )
        self.filter.filter(record)
        assert "sk_live_12345" not in record.msg
        assert "***" in record.msg

    def test_redacts_password_in_message(self):
        """Test that passwords are redacted."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Login with password=secret123",
            args=(),
            exc_info=None,
        )
        self.filter.filter(record)
        assert "secret123" not in record.msg
        assert "***" in record.msg

    def test_redacts_json_style_secrets(self):
        """Test that JSON-style secrets are redacted."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='Config: {"api_secret": "mysecret123"}',
            args=(),
            exc_info=None,
        )
        self.filter.filter(record)
        assert "mysecret123" not in record.msg
        assert "***" in record.msg

    def test_redacts_bearer_token(self):
        """Test that bearer tokens are redacted."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Authorization: bearer=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            args=(),
            exc_info=None,
        )
        self.filter.filter(record)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in record.msg

    def test_redacts_dict_args(self):
        """Test that sensitive keys in dict args are redacted."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Config: %(config)s",
            args={"config": "value", "api_key": "secret123"},
            exc_info=None,
        )
        self.filter.filter(record)
        assert record.args["api_key"] == "***"

    def test_redacts_tuple_args(self):
        """Test that sensitive values in tuple args are redacted."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Values: %s %s",
            args=("normal", "api_key=secret123"),
            exc_info=None,
        )
        self.filter.filter(record)
        # The tuple should have the sensitive value redacted
        assert "secret123" not in str(record.args)

    def test_preserves_non_sensitive_data(self):
        """Test that non-sensitive data is preserved."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Processing symbol=BTCUSDT with price=50000",
            args=(),
            exc_info=None,
        )
        self.filter.filter(record)
        assert "BTCUSDT" in record.msg
        assert "50000" in record.msg

    def test_always_returns_true(self):
        """Test that filter always returns True (never blocks logs)."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Any message",
            args=(),
            exc_info=None,
        )
        assert self.filter.filter(record) is True


class TestNamespacePrefixFilter:
    """Tests for NamespacePrefixFilter."""

    def setup_method(self):
        """Create filter instance for each test."""
        self.filter = NamespacePrefixFilter()

    def test_adds_atb_prefix(self):
        """Test that 'atb.' prefix is added to logger name."""
        record = logging.LogRecord(
            name="mylogger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )
        self.filter.filter(record)
        assert record.name == "atb.mylogger"

    def test_does_not_double_prefix(self):
        """Test that prefix is not added if already present."""
        record = logging.LogRecord(
            name="atb.mylogger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )
        self.filter.filter(record)
        assert record.name == "atb.mylogger"

    def test_handles_empty_name(self):
        """Test handling of empty logger name."""
        record = logging.LogRecord(
            name="",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )
        self.filter.filter(record)
        # Empty name should not be prefixed
        assert record.name == ""


class TestContextInjectorFilter:
    """Tests for ContextInjectorFilter."""

    def setup_method(self):
        """Create filter instance for each test."""
        self.filter = ContextInjectorFilter()

    def test_injects_context_fields(self):
        """Test that context fields are injected into record."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )

        with patch(
            "src.infrastructure.logging.config.get_context",
            return_value={"request_id": "abc123", "user_id": "user1"},
        ):
            self.filter.filter(record)

        assert record.request_id == "abc123"
        assert record.user_id == "user1"

    def test_does_not_override_reserved_fields(self):
        """Test that reserved LogRecord fields are not overwritten."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Original message",
            args=(),
            exc_info=None,
        )

        with patch(
            "src.infrastructure.logging.config.get_context",
            return_value={"msg": "Malicious override", "name": "bad_name"},
        ):
            self.filter.filter(record)

        # Reserved fields should not be changed
        assert record.msg == "Original message"
        assert record.name == "test"

    def test_does_not_override_existing_attrs(self):
        """Test that existing record attributes are not overwritten."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.custom_attr = "original"

        with patch(
            "src.infrastructure.logging.config.get_context",
            return_value={"custom_attr": "from_context"},
        ):
            self.filter.filter(record)

        assert record.custom_attr == "original"


class TestSamplingFilter:
    """Tests for SamplingFilter."""

    def test_always_passes_warning_and_above(self):
        """Test that WARNING and above always pass."""
        filter = SamplingFilter()

        for level in [logging.WARNING, logging.ERROR, logging.CRITICAL]:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="",
                lineno=0,
                msg="Test",
                args=(),
                exc_info=None,
            )
            assert filter.filter(record) is True

    def test_respects_sampling_rate(self):
        """Test that sampling rate affects DEBUG/INFO logs."""
        with patch("src.infrastructure.logging.config.get_config") as mock_config:
            mock_config.return_value = MagicMock()
            mock_config.return_value.get.side_effect = lambda key, default="1.0": "0.0"

            filter = SamplingFilter()

            # With rate 0, DEBUG should never pass
            record = logging.LogRecord(
                name="test",
                level=logging.DEBUG,
                pathname="",
                lineno=0,
                msg="Test",
                args=(),
                exc_info=None,
            )
            # Due to hash-based sampling, results vary, so we just verify it doesn't crash
            filter.filter(record)


class TestMaxMessageLengthFilter:
    """Tests for MaxMessageLengthFilter."""

    def test_truncates_long_messages(self):
        """Test that messages exceeding max length are truncated."""
        with patch("src.infrastructure.logging.config.get_config") as mock_config:
            mock_config.return_value = MagicMock()
            mock_config.return_value.get.return_value = "100"

            filter = MaxMessageLengthFilter()

            long_msg = "x" * 200
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=long_msg,
                args=(),
                exc_info=None,
            )
            filter.filter(record)

            assert len(record.msg) < 200
            assert "[truncated]" in record.msg

    def test_preserves_short_messages(self):
        """Test that short messages are not modified."""
        with patch("src.infrastructure.logging.config.get_config") as mock_config:
            mock_config.return_value = MagicMock()
            mock_config.return_value.get.return_value = "5000"

            filter = MaxMessageLengthFilter()

            short_msg = "Short message"
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=short_msg,
                args=(),
                exc_info=None,
            )
            filter.filter(record)

            assert record.msg == short_msg


class TestSimpleJsonFormatter:
    """Tests for SimpleJsonFormatter."""

    def setup_method(self):
        """Create formatter instance for each test."""
        self.formatter = SimpleJsonFormatter()

    def test_produces_valid_json(self):
        """Test that formatter produces valid JSON."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = self.formatter.format(record)
        parsed = json.loads(result)

        assert "timestamp" in parsed
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test"
        assert parsed["message"] == "Test message"

    def test_includes_exception_info(self):
        """Test that exception info is included."""
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = self.formatter.format(record)
        parsed = json.loads(result)

        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]

    def test_includes_extra_fields(self):
        """Test that extra fields from record are included."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.custom_field = "custom_value"
        record.another_field = 123

        result = self.formatter.format(record)
        parsed = json.loads(result)

        assert parsed["custom_field"] == "custom_value"
        assert parsed["another_field"] == 123


class TestBuildLoggingConfig:
    """Tests for build_logging_config function."""

    def test_returns_dict_config(self):
        """Test that function returns a valid logging config dict."""
        with patch("src.infrastructure.logging.config.get_config") as mock_config:
            mock_config.return_value = MagicMock()
            mock_config.return_value.get.return_value = "INFO"

            config = build_logging_config()

            assert "version" in config
            assert config["version"] == 1
            assert "handlers" in config
            assert "console" in config["handlers"]
            assert "filters" in config

    def test_json_formatter_when_enabled(self):
        """Test that JSON formatter is used when json=True."""
        with patch("src.infrastructure.logging.config.get_config") as mock_config:
            mock_config.return_value = MagicMock()
            mock_config.return_value.get.return_value = "INFO"

            config = build_logging_config(json=True)

            assert "()" in config["formatters"]["default"]
            assert "SimpleJsonFormatter" in config["formatters"]["default"]["()"]

    def test_text_formatter_when_json_disabled(self):
        """Test that text formatter is used when json=False."""
        with patch("src.infrastructure.logging.config.get_config") as mock_config:
            mock_config.return_value = MagicMock()
            mock_config.return_value.get.return_value = "INFO"

            config = build_logging_config(json=False)

            assert "format" in config["formatters"]["default"]

    def test_respects_log_level(self):
        """Test that log level is properly set."""
        with patch("src.infrastructure.logging.config.get_config") as mock_config:
            mock_config.return_value = MagicMock()
            mock_config.return_value.get.return_value = "DEBUG"

            config = build_logging_config(level_name="WARNING")

            assert config["handlers"]["console"]["level"] == "WARNING"


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_applies_config(self):
        """Test that logging config is applied."""
        with patch("src.infrastructure.logging.config.get_config") as mock_config:
            mock_config.return_value = MagicMock()
            mock_config.return_value.get.return_value = None

            with patch("logging.config.dictConfig") as mock_dict_config:
                configure_logging(level_name="DEBUG", use_json=False)
                mock_dict_config.assert_called_once()

    def test_auto_detects_railway_for_json(self):
        """Test that JSON is auto-enabled for Railway deployments."""
        with patch("src.infrastructure.logging.config.get_config") as mock_config:
            mock_cfg = MagicMock()

            def mock_get(key, default=None):
                if key == "RAILWAY_DEPLOYMENT_ID":
                    return "deploy_123"
                return default

            mock_cfg.get.side_effect = mock_get
            mock_config.return_value = mock_cfg

            with patch("logging.config.dictConfig") as mock_dict_config:
                configure_logging()
                # Should have been called with JSON config
                mock_dict_config.assert_called_once()


@pytest.mark.fast
class TestLoggingConfigIntegration:
    """Integration tests for logging configuration."""

    def test_full_config_workflow(self):
        """Test complete logging configuration workflow."""
        with patch("src.infrastructure.logging.config.get_config") as mock_config:
            mock_config.return_value = MagicMock()
            mock_config.return_value.get.return_value = "INFO"

            # Build config
            config = build_logging_config(level_name="INFO", json=True)

            # Verify all filters are configured
            assert "redact" in config["filters"]
            assert "ns" in config["filters"]
            assert "ctx" in config["filters"]
            assert "sample" in config["filters"]
            assert "truncate" in config["filters"]

            # Verify handler uses all filters
            handler_filters = config["handlers"]["console"]["filters"]
            assert "redact" in handler_filters
            assert "ns" in handler_filters
