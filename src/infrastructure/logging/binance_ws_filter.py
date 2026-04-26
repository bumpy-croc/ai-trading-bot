"""Rate-limit Binance WebSocket keepalive-timeout exception noise.

`python-binance==1.0.36` multiplexes margin user-data subscriptions over a
shared `ws_api` connection. That socket is closed periodically by Binance with
WebSocket close code ``1011 keepalive ping timeout``. The library's
ThreadedApiManager surfaces the close as an unretrieved asyncio task
exception (``Task exception was never retrieved``), which the asyncio default
exception handler logs at WARNING+ level on the ``atb.asyncio`` logger.

The library's reconnect machinery recovers automatically — a fresh task is
spawned on the next subscribe attempt — so live trading is unaffected. But
on prod the message fires every ~2 minutes (~720/day), drowning real signals.

This filter keeps the *first* occurrence per ``WINDOW_SECONDS`` so an
operator can still see the full traceback once, suppresses the rest, and
emits a single summary at the end of each window if any were suppressed.

Tracking issue: GH #608.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback


class BinanceWSKeepaliveFilter(logging.Filter):
    """Rate-limit ``Task exception was never retrieved`` keepalive-timeout noise.

    Attached to the console handler in ``build_logging_config``. Inspects the
    record's message text and exception traceback for the
    python-binance keepalive-timeout fingerprint; passes the first match per
    window through, drops subsequent matches, and emits a synthetic summary
    log when the window rolls over so operators retain visibility.

    Thread-safe: a logging filter can be invoked from multiple threads
    (asyncio default handler runs on the loop thread; other handlers run on
    callers). State is guarded by a lock.

    Non-matching records pass through unchanged.
    """

    # Markers that must all appear (in message text or exception traceback)
    # for a record to be classified as keepalive-timeout noise.
    KEEPALIVE_MARKERS: tuple[str, ...] = (
        "keepalive_websocket",
        "keepalive ping timeout",
    )

    # Window length in seconds. One traceback per window is preserved; the
    # rest are counted and summarized.
    DEFAULT_WINDOW_SECONDS: float = 60.0

    SUMMARY_LOGGER_NAME: str = "atb.binance.ws_filter"

    def __init__(
        self,
        window_seconds: float | None = None,
        summary_logger_name: str | None = None,
    ) -> None:
        """Initialize the filter.

        Args:
            window_seconds: Length of each rate-limit window. Defaults to
                ``DEFAULT_WINDOW_SECONDS``.
            summary_logger_name: Logger to emit suppression summaries on.
                Defaults to ``SUMMARY_LOGGER_NAME``.
        """
        super().__init__()
        self._window_seconds: float = (
            float(window_seconds) if window_seconds is not None else self.DEFAULT_WINDOW_SECONDS
        )
        self._summary_logger = logging.getLogger(summary_logger_name or self.SUMMARY_LOGGER_NAME)
        self._lock = threading.Lock()
        self._suppressed_count: int = 0
        self._window_start: float | None = None

    @property
    def suppressed_count(self) -> int:
        """Number of records suppressed in the current window. Diagnostic."""
        with self._lock:
            return self._suppressed_count

    def filter(self, record: logging.LogRecord) -> bool:
        """Return True to keep the record, False to drop it.

        Non-matching records always pass through. For matching records:
        - The first record in a window passes through (full traceback).
        - Subsequent records in the same window are dropped and counted.
        - Window rollover emits a summary on a separate logger and starts
          a new window with the current record passed through.
        """
        if not self._is_keepalive_noise(record):
            return True

        now = time.monotonic()
        with self._lock:
            window_open = (
                self._window_start is not None and (now - self._window_start) < self._window_seconds
            )
            if not window_open:
                # Window rolled over (or first ever match) — emit summary
                # for the previous window if anything was suppressed, then
                # reset and let this record through.
                suppressed_to_report = self._suppressed_count
                window_len = self._window_seconds if self._window_start is not None else 0.0
                self._window_start = now
                self._suppressed_count = 0
                emit_summary = suppressed_to_report > 0
            else:
                self._suppressed_count += 1
                emit_summary = False
                suppressed_to_report = 0
                window_len = 0.0

        if emit_summary:
            # Done outside the lock to avoid recursion if the summary logger
            # itself routes through this filter (it should not, but be safe).
            self._summary_logger.warning(
                "Suppressed %d Binance WS keepalive-timeout exception(s) "
                "in last %.0fs (reconnect handled by python-binance).",
                suppressed_to_report,
                window_len,
            )

        # First match in a fresh window passes; subsequent matches are dropped.
        return not window_open

    def _is_keepalive_noise(self, record: logging.LogRecord) -> bool:
        """Inspect record message and exception text for the keepalive fingerprint."""
        try:
            text = record.getMessage()
        except Exception:
            text = str(record.msg) if record.msg is not None else ""
        if record.exc_info:
            try:
                text = text + "\n" + "".join(traceback.format_exception(*record.exc_info))
            except Exception:
                # If formatting fails for any reason, fall back to the
                # message text alone — never raise from a logging filter.
                pass
        elif record.exc_text:
            text = text + "\n" + record.exc_text
        return all(marker in text for marker in self.KEEPALIVE_MARKERS)
