"""Tests for Binance ban expiry timestamp parsing."""

import time

import pytest

from src.data_providers.binance_provider import _parse_ban_expiry


@pytest.mark.fast
class TestParseBanExpiry:
    """Tests for _parse_ban_expiry function."""

    def test_valid_ban_message_returns_positive_seconds(self):
        # Ban 60 seconds in the future
        future_ms = int(time.time() * 1000) + 60_000
        msg = f"Way too much request weight used; IP banned until {future_ms}."
        result = _parse_ban_expiry(msg)
        assert result is not None
        assert 55 < result < 65  # Allow for test execution time

    def test_past_timestamp_returns_zero(self):
        past_ms = int(time.time() * 1000) - 10_000
        msg = f"IP banned until {past_ms}. Please use WebSocket."
        result = _parse_ban_expiry(msg)
        assert result == 0

    def test_no_timestamp_returns_none(self):
        msg = "APIError(code=-1003): Too many requests"
        result = _parse_ban_expiry(msg)
        assert result is None

    def test_malformed_timestamp_returns_none(self):
        msg = "banned until abc123"
        result = _parse_ban_expiry(msg)
        assert result is None

    def test_short_number_not_matched(self):
        # 10-digit number (seconds, not milliseconds) should not match
        msg = "banned until 1774720000"
        result = _parse_ban_expiry(msg)
        assert result is None

    def test_real_binance_error_format(self):
        future_ms = int(time.time() * 1000) + 300_000
        msg = (
            f"APIError(code=-1003): Way too much request weight used; "
            f"IP banned until {future_ms}. Please use WebSocket Streams "
            f"for live updates to avoid bans."
        )
        result = _parse_ban_expiry(msg)
        assert result is not None
        assert 295 < result < 305
