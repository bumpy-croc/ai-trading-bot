"""Unit tests for network retry infrastructure.

Tests cover:
- Successful calls (no retry)
- Retryable exceptions (network errors)
- Retryable HTTP status codes
- Non-retryable errors
- Exponential backoff calculation
- Jitter behavior
- Max retries exhaustion
"""

import time
from unittest.mock import Mock, patch

import pytest
import requests

from src.infrastructure.network_retry import (
    RETRYABLE_EXCEPTIONS,
    RETRYABLE_STATUS_CODES,
    _calculate_delay,
    with_network_retry,
)


class TestCalculateDelay:
    """Tests for _calculate_delay function."""

    def test_first_attempt_uses_base_delay(self):
        """Test first attempt (attempt=0) uses base delay."""
        delay = _calculate_delay(
            attempt=0,
            base_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=False,
        )
        assert delay == 1.0

    def test_exponential_backoff(self):
        """Test delay increases exponentially."""
        delays = []
        for attempt in range(5):
            delay = _calculate_delay(
                attempt=attempt,
                base_delay=1.0,
                max_delay=1000.0,  # High max to not cap
                exponential_base=2.0,
                jitter=False,
            )
            delays.append(delay)

        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]

    def test_delay_capped_at_max(self):
        """Test delay is capped at max_delay."""
        delay = _calculate_delay(
            attempt=10,  # Would be 1024 without cap
            base_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=False,
        )
        assert delay == 60.0

    def test_jitter_adds_variance(self):
        """Test jitter adds randomness within expected range."""
        delays = set()
        for _ in range(100):
            delay = _calculate_delay(
                attempt=0,
                base_delay=10.0,
                max_delay=60.0,
                exponential_base=2.0,
                jitter=True,
            )
            delays.add(round(delay, 2))

        # With jitter, we should see varied delays
        assert len(delays) > 1

        # All delays should be within ±25% + minimum of 0.1
        for delay in delays:
            assert 0.1 <= delay <= 12.5  # 10 ± 25%

    def test_jitter_ensures_minimum_delay(self):
        """Test jitter never produces delay below 0.1."""
        for _ in range(100):
            delay = _calculate_delay(
                attempt=0,
                base_delay=0.1,  # Very small base
                max_delay=60.0,
                exponential_base=2.0,
                jitter=True,
            )
            assert delay >= 0.1

    def test_custom_exponential_base(self):
        """Test custom exponential base works."""
        delays = []
        for attempt in range(4):
            delay = _calculate_delay(
                attempt=attempt,
                base_delay=1.0,
                max_delay=1000.0,
                exponential_base=3.0,
                jitter=False,
            )
            delays.append(delay)

        assert delays == [1.0, 3.0, 9.0, 27.0]


class TestWithNetworkRetrySuccess:
    """Tests for successful calls (no retry needed)."""

    def test_successful_call_returns_result(self):
        """Test successful call returns result without retry."""
        call_count = {"count": 0}

        @with_network_retry(max_retries=3)
        def successful_function():
            call_count["count"] += 1
            return "success"

        result = successful_function()

        assert result == "success"
        assert call_count["count"] == 1  # Only called once

    def test_preserves_function_metadata(self):
        """Test decorator preserves function metadata."""

        @with_network_retry()
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == """My docstring."""


class TestWithNetworkRetryNetworkErrors:
    """Tests for retryable network exceptions."""

    def test_retries_on_timeout(self):
        """Test retries on requests.exceptions.Timeout."""
        call_count = {"count": 0}

        @with_network_retry(max_retries=3, base_delay=0.01)
        def timeout_then_succeed():
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise requests.exceptions.Timeout("timeout")
            return "success"

        result = timeout_then_succeed()

        assert result == "success"
        assert call_count["count"] == 3

    def test_retries_on_connection_error(self):
        """Test retries on requests.exceptions.ConnectionError."""
        call_count = {"count": 0}

        @with_network_retry(max_retries=3, base_delay=0.01)
        def connection_error_then_succeed():
            call_count["count"] += 1
            if call_count["count"] < 2:
                raise requests.exceptions.ConnectionError("connection failed")
            return "success"

        result = connection_error_then_succeed()

        assert result == "success"
        assert call_count["count"] == 2

    def test_raises_after_max_retries_exhausted(self):
        """Test raises exception after max retries exhausted."""
        call_count = {"count": 0}

        @with_network_retry(max_retries=3, base_delay=0.01)
        def always_fails():
            call_count["count"] += 1
            raise requests.exceptions.Timeout("timeout")

        with pytest.raises(requests.exceptions.Timeout):
            always_fails()

        assert call_count["count"] == 4  # Initial + 3 retries


class TestWithNetworkRetryHTTPErrors:
    """Tests for HTTP status code based retries."""

    def test_retries_on_429(self):
        """Test retries on 429 Too Many Requests."""
        call_count = {"count": 0}

        @with_network_retry(max_retries=3, base_delay=0.01)
        def rate_limited_then_succeed():
            call_count["count"] += 1
            if call_count["count"] < 2:
                response = Mock()
                response.status_code = 429
                error = requests.exceptions.HTTPError(response=response)
                raise error
            return "success"

        result = rate_limited_then_succeed()

        assert result == "success"
        assert call_count["count"] == 2

    def test_retries_on_503(self):
        """Test retries on 503 Service Unavailable."""
        call_count = {"count": 0}

        @with_network_retry(max_retries=3, base_delay=0.01)
        def service_unavailable_then_succeed():
            call_count["count"] += 1
            if call_count["count"] < 2:
                response = Mock()
                response.status_code = 503
                error = requests.exceptions.HTTPError(response=response)
                raise error
            return "success"

        result = service_unavailable_then_succeed()

        assert result == "success"

    def test_no_retry_on_400(self):
        """Test no retry on 400 Bad Request."""
        call_count = {"count": 0}

        @with_network_retry(max_retries=3, base_delay=0.01)
        def bad_request():
            call_count["count"] += 1
            response = Mock()
            response.status_code = 400
            error = requests.exceptions.HTTPError(response=response)
            raise error

        with pytest.raises(requests.exceptions.HTTPError):
            bad_request()

        assert call_count["count"] == 1  # No retries

    def test_no_retry_on_401(self):
        """Test no retry on 401 Unauthorized."""
        call_count = {"count": 0}

        @with_network_retry(max_retries=3, base_delay=0.01)
        def unauthorized():
            call_count["count"] += 1
            response = Mock()
            response.status_code = 401
            error = requests.exceptions.HTTPError(response=response)
            raise error

        with pytest.raises(requests.exceptions.HTTPError):
            unauthorized()

        assert call_count["count"] == 1  # No retries

    def test_no_retry_on_404(self):
        """Test no retry on 404 Not Found."""
        call_count = {"count": 0}

        @with_network_retry(max_retries=3, base_delay=0.01)
        def not_found():
            call_count["count"] += 1
            response = Mock()
            response.status_code = 404
            error = requests.exceptions.HTTPError(response=response)
            raise error

        with pytest.raises(requests.exceptions.HTTPError):
            not_found()

        assert call_count["count"] == 1


class TestWithNetworkRetryNonNetworkErrors:
    """Tests for non-network exceptions (should not retry)."""

    def test_no_retry_on_value_error(self):
        """Test no retry on ValueError."""
        call_count = {"count": 0}

        @with_network_retry(max_retries=3, base_delay=0.01)
        def value_error():
            call_count["count"] += 1
            raise ValueError("invalid value")

        with pytest.raises(ValueError):
            value_error()

        assert call_count["count"] == 1  # No retries

    def test_no_retry_on_key_error(self):
        """Test no retry on KeyError."""
        call_count = {"count": 0}

        @with_network_retry(max_retries=3, base_delay=0.01)
        def key_error():
            call_count["count"] += 1
            raise KeyError("missing key")

        with pytest.raises(KeyError):
            key_error()

        assert call_count["count"] == 1


class TestWithNetworkRetryCustomConfig:
    """Tests for custom retry configuration."""

    def test_custom_retryable_status_codes(self):
        """Test custom retryable status codes work."""
        call_count = {"count": 0}

        @with_network_retry(
            max_retries=3,
            base_delay=0.01,
            retryable_status_codes={418},  # I'm a teapot - custom
        )
        def teapot_then_succeed():
            call_count["count"] += 1
            if call_count["count"] < 2:
                response = Mock()
                response.status_code = 418
                error = requests.exceptions.HTTPError(response=response)
                raise error
            return "success"

        result = teapot_then_succeed()

        assert result == "success"
        assert call_count["count"] == 2

    def test_custom_retryable_exceptions(self):
        """Test custom retryable exception types work."""

        class CustomNetworkError(Exception):
            pass

        call_count = {"count": 0}

        @with_network_retry(
            max_retries=3,
            base_delay=0.01,
            retryable_exceptions=(CustomNetworkError,),
        )
        def custom_error_then_succeed():
            call_count["count"] += 1
            if call_count["count"] < 2:
                raise CustomNetworkError("custom network issue")
            return "success"

        result = custom_error_then_succeed()

        assert result == "success"
        assert call_count["count"] == 2

    def test_custom_max_delay(self):
        """Test custom max_delay caps delays."""
        with patch("src.infrastructure.network_retry.time.sleep") as mock_sleep:
            call_count = {"count": 0}

            @with_network_retry(
                max_retries=5,
                base_delay=1.0,
                max_delay=2.0,  # Cap at 2 seconds
                jitter=False,
            )
            def always_timeout():
                call_count["count"] += 1
                raise requests.exceptions.Timeout("timeout")

            with pytest.raises(requests.exceptions.Timeout):
                always_timeout()

            # Check sleep calls were capped
            sleep_args = [call[0][0] for call in mock_sleep.call_args_list]
            for delay in sleep_args:
                assert delay <= 2.0


class TestWithNetworkRetryTiming:
    """Tests for retry timing behavior."""

    def test_exponential_backoff_timing(self):
        """Test delays follow exponential backoff pattern."""
        with patch("src.infrastructure.network_retry.time.sleep") as mock_sleep:
            call_count = {"count": 0}

            @with_network_retry(
                max_retries=3,
                base_delay=1.0,
                exponential_base=2.0,
                jitter=False,
            )
            def always_timeout():
                call_count["count"] += 1
                raise requests.exceptions.Timeout("timeout")

            with pytest.raises(requests.exceptions.Timeout):
                always_timeout()

            # Check sleep was called with exponential delays
            sleep_args = [call[0][0] for call in mock_sleep.call_args_list]
            assert sleep_args == [1.0, 2.0, 4.0]


class TestRetryableConstants:
    """Tests for module-level constants."""

    def test_retryable_status_codes_defined(self):
        """Test RETRYABLE_STATUS_CODES contains expected codes."""
        assert 429 in RETRYABLE_STATUS_CODES  # Too Many Requests
        assert 500 in RETRYABLE_STATUS_CODES  # Internal Server Error
        assert 502 in RETRYABLE_STATUS_CODES  # Bad Gateway
        assert 503 in RETRYABLE_STATUS_CODES  # Service Unavailable
        assert 504 in RETRYABLE_STATUS_CODES  # Gateway Timeout

        # Should not include client errors
        assert 400 not in RETRYABLE_STATUS_CODES
        assert 401 not in RETRYABLE_STATUS_CODES
        assert 404 not in RETRYABLE_STATUS_CODES

    def test_retryable_exceptions_defined(self):
        """Test RETRYABLE_EXCEPTIONS contains expected types."""
        assert requests.exceptions.Timeout in RETRYABLE_EXCEPTIONS
        assert requests.exceptions.ConnectionError in RETRYABLE_EXCEPTIONS
        assert ConnectionResetError in RETRYABLE_EXCEPTIONS
        assert ConnectionRefusedError in RETRYABLE_EXCEPTIONS


class TestWithNetworkRetryHTTPErrorEdgeCases:
    """Edge case tests for HTTP error handling."""

    def test_http_error_with_no_response(self):
        """Test HTTP error with no response object."""
        call_count = {"count": 0}

        @with_network_retry(max_retries=3, base_delay=0.01)
        def http_error_no_response():
            call_count["count"] += 1
            error = requests.exceptions.HTTPError()
            error.response = None
            raise error

        with pytest.raises(requests.exceptions.HTTPError):
            http_error_no_response()

        # Should not retry since we can't determine status code
        assert call_count["count"] == 1

    def test_http_error_with_no_status_code(self):
        """Test HTTP error with response but no status_code."""
        call_count = {"count": 0}

        @with_network_retry(max_retries=3, base_delay=0.01)
        def http_error_no_status():
            call_count["count"] += 1
            response = Mock(spec=[])  # No status_code attribute
            error = requests.exceptions.HTTPError(response=response)
            raise error

        with pytest.raises(requests.exceptions.HTTPError):
            http_error_no_status()

        # Should not retry
        assert call_count["count"] == 1
