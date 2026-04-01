"""Tests for KlineBuffer — thread-safe rolling kline history for WebSocket events."""

import threading
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.engines.live.kline_buffer import KlineBuffer

# --- Fixtures ---


def _make_seed_df(n: int = 5, start_ms: int = 1_700_000_000_000, interval_ms: int = 3_600_000):
    """Build a DataFrame matching BinanceProvider.get_live_data() output format."""
    timestamps = [
        pd.Timestamp(start_ms + i * interval_ms, unit="ms") for i in range(n)
    ]
    data = {
        "open": [100.0 + i for i in range(n)],
        "high": [110.0 + i for i in range(n)],
        "low": [90.0 + i for i in range(n)],
        "close": [105.0 + i for i in range(n)],
        "volume": [1000.0 + i for i in range(n)],
    }
    df = pd.DataFrame(data, index=timestamps)
    df.index.name = "timestamp"
    return df


def _make_provider(seed_df: pd.DataFrame) -> MagicMock:
    """Create a mock provider that returns seed_df from get_live_data."""
    provider = MagicMock()
    provider.get_live_data = MagicMock(return_value=seed_df)
    return provider


def _make_kline_event(ts_ms: int, o: float, h: float, l: float, c: float, v: float, closed: bool) -> dict:
    """Build a kline WebSocket event dict."""
    return {
        "k": {
            "t": ts_ms,
            "o": str(o),
            "h": str(h),
            "l": str(l),
            "c": str(c),
            "v": str(v),
            "x": closed,
        }
    }


# --- Tests ---


@pytest.mark.fast
class TestKlineBufferSeeding:
    """Verify initial REST seeding from provider."""

    def test_seeds_from_provider(self):
        """Buffer seeds its DataFrame via provider.get_live_data on init."""
        seed_df = _make_seed_df(5)
        provider = _make_provider(seed_df)

        buf = KlineBuffer("BTCUSDT", "1h", provider)

        provider.get_live_data.assert_called_once_with("BTCUSDT", "1h", limit=500)
        result = buf.get_dataframe()
        assert len(result) == 5
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_seed_preserves_values(self):
        """Seeded DataFrame values match provider output exactly."""
        seed_df = _make_seed_df(3)
        provider = _make_provider(seed_df)

        buf = KlineBuffer("BTCUSDT", "1h", provider)
        result = buf.get_dataframe()

        pd.testing.assert_frame_equal(result, seed_df)


@pytest.mark.fast
class TestOpenCandleUpdate:
    """Open candle updates modify the tail row in-place when timestamps match."""

    def test_updates_tail_on_matching_timestamp(self):
        """An open-candle event with the same timestamp as tail updates OHLCV in-place."""
        seed_df = _make_seed_df(5)
        provider = _make_provider(seed_df)
        buf = KlineBuffer("BTCUSDT", "1h", provider)

        tail_ts_ms = int(seed_df.index[-1].timestamp() * 1000)
        event = _make_kline_event(tail_ts_ms, 200.0, 220.0, 190.0, 210.0, 5000.0, closed=False)

        buf.on_kline(event)

        result = buf.get_dataframe()
        assert len(result) == 5  # no row added
        assert result.iloc[-1]["close"] == pytest.approx(210.0)
        assert result.iloc[-1]["high"] == pytest.approx(220.0)
        assert result.iloc[-1]["volume"] == pytest.approx(5000.0)


@pytest.mark.fast
class TestCandleClose:
    """Candle close events replace the current tail when timestamps match."""

    def test_close_replaces_tail(self):
        """A closed-candle event with matching timestamp replaces the tail row."""
        seed_df = _make_seed_df(5)
        provider = _make_provider(seed_df)
        buf = KlineBuffer("BTCUSDT", "1h", provider)

        tail_ts_ms = int(seed_df.index[-1].timestamp() * 1000)
        event = _make_kline_event(tail_ts_ms, 300.0, 330.0, 290.0, 310.0, 9000.0, closed=True)

        buf.on_kline(event)

        result = buf.get_dataframe()
        assert len(result) == 5
        assert result.iloc[-1]["close"] == pytest.approx(310.0)


@pytest.mark.fast
class TestNewCandleRollWindow:
    """Events with timestamp newer than tail roll the window (drop first, append)."""

    def test_new_candle_rolls_window(self):
        """A closed event with event_ts > tail_ts drops the oldest row and appends."""
        seed_df = _make_seed_df(5)
        provider = _make_provider(seed_df)
        buf = KlineBuffer("BTCUSDT", "1h", provider)

        # Timestamp one interval after the last candle
        new_ts_ms = int(seed_df.index[-1].timestamp() * 1000) + 3_600_000
        event = _make_kline_event(new_ts_ms, 400.0, 440.0, 390.0, 410.0, 7000.0, closed=True)

        buf.on_kline(event)

        result = buf.get_dataframe()
        assert len(result) == 5  # window size preserved
        assert result.iloc[-1]["close"] == pytest.approx(410.0)
        # First row should now be what was previously the second row
        assert result.iloc[0]["open"] == pytest.approx(seed_df.iloc[1]["open"])

    def test_open_candle_new_timestamp_rolls_window(self):
        """An open (not closed) event with event_ts > tail_ts also rolls the window."""
        seed_df = _make_seed_df(5)
        provider = _make_provider(seed_df)
        buf = KlineBuffer("BTCUSDT", "1h", provider)

        new_ts_ms = int(seed_df.index[-1].timestamp() * 1000) + 3_600_000
        event = _make_kline_event(new_ts_ms, 500.0, 550.0, 490.0, 510.0, 8000.0, closed=False)

        buf.on_kline(event)

        result = buf.get_dataframe()
        assert len(result) == 5
        assert result.iloc[-1]["open"] == pytest.approx(500.0)


@pytest.mark.fast
class TestStaleEvent:
    """Stale events (older than tail) are silently ignored."""

    def test_stale_event_ignored(self):
        """An event with timestamp older than the tail is dropped."""
        seed_df = _make_seed_df(5)
        provider = _make_provider(seed_df)
        buf = KlineBuffer("BTCUSDT", "1h", provider)

        original = buf.get_dataframe().copy()

        # Timestamp older than the tail
        stale_ts_ms = int(seed_df.index[-2].timestamp() * 1000)
        event = _make_kline_event(stale_ts_ms, 999.0, 999.0, 999.0, 999.0, 999.0, closed=True)

        buf.on_kline(event)

        result = buf.get_dataframe()
        pd.testing.assert_frame_equal(result, original)


@pytest.mark.fast
class TestThreadSafety:
    """Concurrent access does not corrupt the buffer."""

    def test_concurrent_updates(self):
        """Multiple threads writing kline events do not crash or corrupt length."""
        seed_df = _make_seed_df(500)
        provider = _make_provider(seed_df)
        buf = KlineBuffer("BTCUSDT", "1h", provider)

        base_ts_ms = int(seed_df.index[-1].timestamp() * 1000)
        errors = []

        def writer(offset: int):
            """Write a batch of kline events."""
            try:
                for i in range(50):
                    ts = base_ts_ms + (offset * 50 + i) * 3_600_000
                    event = _make_kline_event(ts, 100.0, 110.0, 90.0, 105.0, 1000.0, closed=True)
                    buf.on_kline(event)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            assert not t.is_alive(), "Thread did not finish in time"

        assert not errors, f"Errors during concurrent access: {errors}"
        result = buf.get_dataframe()
        assert len(result) == 500  # window size preserved


@pytest.mark.fast
class TestIsFresh:
    """The is_fresh property checks recency of updates."""

    def test_fresh_after_init(self):
        """Buffer is fresh immediately after construction."""
        seed_df = _make_seed_df(3)
        provider = _make_provider(seed_df)
        buf = KlineBuffer("BTCUSDT", "1h", provider)

        assert buf.is_fresh is True

    def test_stale_after_timeout(self):
        """Buffer reports not fresh when _last_update is older than 120 seconds."""
        seed_df = _make_seed_df(3)
        provider = _make_provider(seed_df)
        buf = KlineBuffer("BTCUSDT", "1h", provider)

        # Manually set _last_update to the past
        buf._last_update = datetime.now(UTC) - timedelta(seconds=130)

        assert buf.is_fresh is False

    def test_fresh_after_kline_event(self):
        """Receiving a kline event resets the freshness timer."""
        seed_df = _make_seed_df(3)
        provider = _make_provider(seed_df)
        buf = KlineBuffer("BTCUSDT", "1h", provider)

        buf._last_update = datetime.now(UTC) - timedelta(seconds=130)
        assert buf.is_fresh is False

        tail_ts_ms = int(seed_df.index[-1].timestamp() * 1000)
        event = _make_kline_event(tail_ts_ms, 100.0, 110.0, 90.0, 105.0, 1000.0, closed=False)
        buf.on_kline(event)

        assert buf.is_fresh is True


@pytest.mark.fast
class TestResyncFromRest:
    """resync_from_rest replaces the entire DataFrame."""

    def test_resync_replaces_dataframe(self):
        """After resync, the buffer contains the provider's fresh data."""
        seed_df = _make_seed_df(5)
        provider = _make_provider(seed_df)
        buf = KlineBuffer("BTCUSDT", "1h", provider)

        new_df = _make_seed_df(3, start_ms=1_800_000_000_000)
        new_provider = _make_provider(new_df)

        buf.resync_from_rest(new_provider, "BTCUSDT", "1h")

        result = buf.get_dataframe()
        assert len(result) == 3
        pd.testing.assert_frame_equal(result, new_df)


@pytest.mark.fast
class TestGetDataframeReturnsCopy:
    """get_dataframe returns a defensive copy."""

    def test_modifying_copy_does_not_affect_buffer(self):
        """Mutating the returned DataFrame leaves the buffer unchanged."""
        seed_df = _make_seed_df(5)
        provider = _make_provider(seed_df)
        buf = KlineBuffer("BTCUSDT", "1h", provider)

        copy = buf.get_dataframe()
        copy.iloc[-1, copy.columns.get_loc("close")] = 999999.0

        original = buf.get_dataframe()
        assert original.iloc[-1]["close"] != pytest.approx(999999.0)


@pytest.mark.fast
class TestEdgeCases:
    """Edge cases and malformed events."""

    def test_event_without_k_key_ignored(self):
        """An event dict missing the 'k' key is silently ignored."""
        seed_df = _make_seed_df(5)
        provider = _make_provider(seed_df)
        buf = KlineBuffer("BTCUSDT", "1h", provider)

        original = buf.get_dataframe().copy()
        buf.on_kline({})  # no "k" key
        buf.on_kline({"something": "else"})

        result = buf.get_dataframe()
        pd.testing.assert_frame_equal(result, original)

    def test_empty_kline_dict_ignored(self):
        """An event with an empty kline dict is ignored."""
        seed_df = _make_seed_df(5)
        provider = _make_provider(seed_df)
        buf = KlineBuffer("BTCUSDT", "1h", provider)

        original = buf.get_dataframe().copy()
        buf.on_kline({"k": {}})

        result = buf.get_dataframe()
        pd.testing.assert_frame_equal(result, original)
