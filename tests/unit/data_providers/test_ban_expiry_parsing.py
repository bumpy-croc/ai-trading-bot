"""Tests for Binance ban expiry timestamp parsing."""

import pytest

from src.data_providers.binance_provider import _parse_ban_expiry

# Fixed reference time for deterministic tests
FIXED_NOW_MS = 1_700_000_000_000


@pytest.mark.fast
class TestParseBanExpiry:
    """Tests for _parse_ban_expiry function."""

    def test_valid_ban_message_returns_positive_seconds(self):
        future_ms = FIXED_NOW_MS + 60_000
        msg = f"Way too much request weight used; IP banned until {future_ms}."
        result = _parse_ban_expiry(msg, now_ms=FIXED_NOW_MS)
        assert result == 60.0

    def test_past_timestamp_returns_zero(self):
        past_ms = FIXED_NOW_MS - 10_000
        msg = f"IP banned until {past_ms}. Please use WebSocket."
        result = _parse_ban_expiry(msg, now_ms=FIXED_NOW_MS)
        assert result == 0

    def test_no_timestamp_returns_none(self):
        msg = "APIError(code=-1003): Too many requests"
        result = _parse_ban_expiry(msg, now_ms=FIXED_NOW_MS)
        assert result is None

    def test_malformed_timestamp_returns_none(self):
        msg = "banned until abc123"
        result = _parse_ban_expiry(msg, now_ms=FIXED_NOW_MS)
        assert result is None

    def test_short_number_not_matched(self):
        # 10-digit number (seconds, not milliseconds) should not match
        msg = "banned until 1774720000"
        result = _parse_ban_expiry(msg, now_ms=FIXED_NOW_MS)
        assert result is None

    def test_real_binance_error_format(self):
        future_ms = FIXED_NOW_MS + 300_000
        msg = (
            f"APIError(code=-1003): Way too much request weight used; "
            f"IP banned until {future_ms}. Please use WebSocket Streams "
            f"for live updates to avoid bans."
        )
        result = _parse_ban_expiry(msg, now_ms=FIXED_NOW_MS)
        assert result == 300.0

    def test_defaults_to_current_time_when_now_ms_not_provided(self):
        # Use a timestamp far in the future so it's always positive
        far_future_ms = 9_999_999_999_999
        msg = f"banned until {far_future_ms}"
        result = _parse_ban_expiry(msg)
        assert result is not None
        assert result > 0
