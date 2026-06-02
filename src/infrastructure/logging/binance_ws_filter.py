"""Rate-limit Binance WebSocket keepalive-timeout exception noise.

`python-binance==1.0.36` multiplexes margin user-data subscriptions over a
shared `ws_api` connection, and that connection fails recoverably in two
ways the library surfaces as an unretrieved asyncio task exception
(``Task exception was never retrieved``) on the ``atb.asyncio`` logger:

  * the socket is closed by Binance with WebSocket close code
    ``1011 keepalive ping timeout`` (the original #609 target); or
  * the ws_api ``userDataStream.subscribe`` request times out after 10s and
    raises ``BinanceWebsocketUnableToConnect('Request timed out')``. This is
    what actually fires on prod every ~2 minutes (~720/day) — there is NO
    "keepalive ping timeout" text, so #609's single fingerprint missed it
    entirely (GH #608 follow-up).

The library's reconnect machinery recovers automatically — a fresh task is
spawned on the next subscribe attempt — so live trading is unaffected, but
the noise drowns real signals.

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
    record's message text and exception traceback for any of the
    python-binance recoverable-ws-churn fingerprints
    (see ``KEEPALIVE_MARKER_GROUPS``); passes the first match per window
    through, drops subsequent matches, and emits a synthetic summary log
    when the window rolls over so operators retain visibility.

    Thread-safe: a logging filter can be invoked from multiple threads
    (asyncio default handler runs on the loop thread; other handlers run on
    callers). State is guarded by a lock.

    Non-matching records pass through unchanged.
    """

    # A record is classified as recoverable python-binance ws churn noise
    # when *every* marker in *any one* group appears in its message text or
    # exception traceback. Two distinct signatures exist (GH #608):
    #
    #   * 1011 keepalive-ping-timeout close — the original #609 target. Rare
    #     in practice; kept for completeness.
    #   * ws_api subscribe timeout — what actually fires on prod every ~2min:
    #     the margin user-data subscription is multiplexed over the shared
    #     ws_api socket, whose request times out after 10s and surfaces as an
    #     unretrieved task exception. There is NO "keepalive ping timeout"
    #     text in this case, so it must be matched on its own fingerprint.
    #     The "binance/ws/" anchor keeps this from swallowing a
    #     BinanceWebsocketUnableToConnect raised by our own code.
    KEEPALIVE_MARKER_GROUPS: tuple[tuple[str, ...], ...] = (
        ("keepalive_websocket", "keepalive ping timeout"),
        (
            "Task exception was never retrieved",
            "BinanceWebsocketUnableToConnect",
            "binance/ws/",
        ),
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
        return any(
            all(marker in text for marker in group)
            for group in self.KEEPALIVE_MARKER_GROUPS
        )
