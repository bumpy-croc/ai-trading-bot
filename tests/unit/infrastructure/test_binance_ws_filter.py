"""Unit tests for ``BinanceWSKeepaliveFilter``.

Covers GH #608: the upstream python-binance ws_api connection drops every
~2 minutes with WebSocket close code 1011, and the unretrieved-task
exception clutters logs. The filter rate-limits these to one record per
window and emits a periodic summary.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close, CloseCode

from src.infrastructure.logging.binance_ws_filter import BinanceWSKeepaliveFilter

pytestmark = pytest.mark.unit


def _make_record(
    msg: str,
    *,
    exc: BaseException | None = None,
    name: str = "atb.asyncio",
    args: tuple[Any, ...] = (),
) -> logging.LogRecord:
    """Build a LogRecord with optional exception info attached."""
    exc_info = None
    if exc is not None:
        try:
            raise exc
        except BaseException:
            import sys

            exc_info = sys.exc_info()
    record = logging.LogRecord(
        name=name,
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=exc_info,
    )
    return record


def _keepalive_exc() -> ConnectionClosedError:
    """Return a ConnectionClosedError matching the python-binance fingerprint."""
    return ConnectionClosedError(
        rcvd=None,
        sent=Close(code=CloseCode.INTERNAL_ERROR, reason="keepalive ping timeout"),
    )


# Fingerprint requires both markers in the formatted text. We synthesize a
# realistic record by mentioning ``keepalive_websocket`` in the message
# (mirroring the path in the upstream traceback) and attaching an exception
# whose text contains ``keepalive ping timeout``.
def _keepalive_record() -> logging.LogRecord:
    return _make_record(
        "Task exception was never retrieved (keepalive_websocket.py)",
        exc=_keepalive_exc(),
    )


class TestKeepaliveFingerprint:
    """The filter must only act on records that match the keepalive fingerprint."""

    def test_passes_through_unrelated_record(self):
        """Generic error records are never suppressed."""
        f = BinanceWSKeepaliveFilter(window_seconds=60)
        record = _make_record("some unrelated error", exc=ValueError("nope"))

        assert f.filter(record) is True
        assert f.suppressed_count == 0

    def test_passes_through_partial_match(self):
        """Only one of the two markers — must not match."""
        f = BinanceWSKeepaliveFilter(window_seconds=60)
        # Has 'keepalive_websocket' but no 'keepalive ping timeout' text
        record = _make_record("Task exception (keepalive_websocket.py)", exc=ValueError("nope"))

        assert f.filter(record) is True
        assert f.suppressed_count == 0

    def test_matches_on_exception_text(self):
        """Markers can come from the exception traceback text."""
        f = BinanceWSKeepaliveFilter(window_seconds=60)
        record = _keepalive_record()

        # First match passes; would suppress on second
        assert f.filter(record) is True


class TestRateLimiting:
    """First match per window passes; rest are dropped and counted."""

    def test_first_match_passes(self):
        f = BinanceWSKeepaliveFilter(window_seconds=60)

        assert f.filter(_keepalive_record()) is True
        assert f.suppressed_count == 0

    def test_subsequent_matches_in_window_dropped(self):
        f = BinanceWSKeepaliveFilter(window_seconds=60)
        f.filter(_keepalive_record())  # establishes window

        for _ in range(10):
            assert f.filter(_keepalive_record()) is False

        assert f.suppressed_count == 10

    def test_window_rollover_releases_next_record(self):
        """After the window expires, the next match passes through again."""
        f = BinanceWSKeepaliveFilter(window_seconds=0.05)

        assert f.filter(_keepalive_record()) is True
        assert f.filter(_keepalive_record()) is False
        assert f.filter(_keepalive_record()) is False
        assert f.suppressed_count == 2

        # Wait for the window to roll over
        time.sleep(0.07)

        assert f.filter(_keepalive_record()) is True
        # Counter resets at rollover
        assert f.suppressed_count == 0


class TestSummaryEmission:
    """Window rollover with suppressed records emits a summary log."""

    def test_emits_summary_on_rollover_with_suppressions(self, caplog):
        f = BinanceWSKeepaliveFilter(window_seconds=0.05)

        f.filter(_keepalive_record())  # passes
        f.filter(_keepalive_record())  # suppressed
        f.filter(_keepalive_record())  # suppressed
        time.sleep(0.07)

        with caplog.at_level(logging.WARNING, logger=BinanceWSKeepaliveFilter.SUMMARY_LOGGER_NAME):
            f.filter(_keepalive_record())  # rollover triggers summary, then passes

        summary_records = [
            r for r in caplog.records if r.name == BinanceWSKeepaliveFilter.SUMMARY_LOGGER_NAME
        ]
        assert len(summary_records) == 1
        assert summary_records[0].levelno == logging.WARNING
        assert "Suppressed 2" in summary_records[0].getMessage()

    def test_no_summary_when_window_rolls_with_zero_suppressions(self, caplog):
        f = BinanceWSKeepaliveFilter(window_seconds=0.05)

        f.filter(_keepalive_record())  # passes
        time.sleep(0.07)

        with caplog.at_level(logging.WARNING, logger=BinanceWSKeepaliveFilter.SUMMARY_LOGGER_NAME):
            f.filter(_keepalive_record())  # rollover, but nothing was suppressed

        summary_records = [
            r for r in caplog.records if r.name == BinanceWSKeepaliveFilter.SUMMARY_LOGGER_NAME
        ]
        assert summary_records == []


class TestThreadSafety:
    """The filter is invoked from multiple threads in production."""

    def test_concurrent_filter_calls_consistent(self):
        """Total dropped + total passed equals total submitted."""
        f = BinanceWSKeepaliveFilter(window_seconds=60)
        passed: list[bool] = []
        passed_lock = threading.Lock()
        n_threads = 8
        per_thread = 50

        def worker():
            for _ in range(per_thread):
                result = f.filter(_keepalive_record())
                with passed_lock:
                    passed.append(result)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = n_threads * per_thread
        passed_count = sum(1 for p in passed if p)
        suppressed_count_obs = sum(1 for p in passed if not p)
        # Exactly one record is allowed through per window; under heavy
        # contention with a 60s window the first-arriver wins.
        assert passed_count == 1
        assert suppressed_count_obs == total - 1
        assert f.suppressed_count == total - 1


class TestSafety:
    """The filter must never raise — failure to filter is preferable to crashing."""

    def test_record_with_unformattable_args_does_not_raise(self):
        """Record with %-args that don't match should still classify safely."""

        class Boom:
            def __str__(self) -> str:
                raise RuntimeError("boom")

        f = BinanceWSKeepaliveFilter(window_seconds=60)
        # %s format will call str() on Boom() and raise inside getMessage().
        record = logging.LogRecord(
            name="atb.asyncio",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="oops %s",
            args=(Boom(),),
            exc_info=None,
        )

        # Must not raise; non-keepalive record passes through.
        assert f.filter(record) is True
