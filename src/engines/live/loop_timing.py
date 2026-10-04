"""Live trading-loop timing helpers (#486).

Small, self-contained helpers the trading loop uses for cadence and data
freshness: ``sleep_with_interrupt`` (interruptible sleep), ``calculate_adaptive_interval``
(activity/time-of-day-aware poll interval), and ``is_data_fresh`` (candle-age /
WS-buffer freshness gate). Moved verbatim out of ``LiveTradingEngine`` (a
mechanical ``self.`` -> ``state.`` rewrite against an engine backref); the
engine keeps thin delegating wrappers, so the trading loop and the
``LiveMarketDataCoordinator`` (which calls ``state._is_data_fresh``) are
unchanged.

Leaf helpers — no order placement, balance mutation, or engine-method calls.
The only state the coordinator keeps is the stale-data streak the latched-condition
monitor reads (``stale_data_observation``). Runs on the trading-loop thread.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

import pandas as pd

from src.config.constants import DEFAULT_SLEEP_POLL_INTERVAL
from src.engines.live.kline_buffer import timeframe_to_ms

logger = logging.getLogger(__name__)

# A freshness gate that has been failing this long is no longer a blip: the loop
# is skipping entries AND in-engine exits for the whole streak, so it is
# announced as a latched condition instead of a debug line. Long enough to ride
# out a WebSocket reconnect, longer than the largest poll interval so one slow sample cannot latch.
STALE_DATA_LATCH_SECONDS = 600.0

if TYPE_CHECKING:
    from src.engines.live.execution.position_tracker import LivePositionTracker
    from src.engines.live.kline_buffer import KlineBuffer


class LiveLoopTimingEngineState(Protocol):
    """Engine state the loop-timing helpers read at call time."""

    stop_event: threading.Event
    base_check_interval: int
    min_check_interval: int
    max_check_interval: int
    data_freshness_threshold: int
    timeframe: str | None
    live_position_tracker: LivePositionTracker
    _ws_kline_active: bool
    _kline_buffer: KlineBuffer | None
    _ws_kline_provider: Any


@dataclass(frozen=True)
class StaleDataObservation:
    """Whether the freshness gate has been failing long enough to announce.

    Attributes:
        stale: True once the gate has failed continuously for longer than
            ``STALE_DATA_LATCH_SECONDS``.
        elapsed_seconds: Length of the current failing streak (0 when fresh).
        reason: Operator-facing description, or None when not stale.
    """

    stale: bool
    elapsed_seconds: float
    reason: str | None = None


class LiveLoopTimingCoordinator:
    """Owns the trading loop's cadence + data-freshness helpers."""

    def __init__(
        self,
        engine_state: LiveLoopTimingEngineState,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind to the engine's live state (see protocol for the surface)."""
        self._state = engine_state
        self._clock = clock
        self._stale_since: float | None = None
        self._stale_reason: str | None = None

    def sleep_with_interrupt(self, seconds: float) -> None:
        """Sleep in small increments to allow for interrupt and float seconds"""
        state = self._state
        end_time = time.time() + seconds
        poll_interval = DEFAULT_SLEEP_POLL_INTERVAL  # Use configurable interval instead of 0.1
        while time.time() < end_time:
            if state.stop_event.is_set():
                break
            time.sleep(min(poll_interval, end_time - time.time()))

    def calculate_adaptive_interval(self, current_price: float | None = None) -> int:
        """Calculate adaptive check interval based on recent trading activity and market conditions"""
        state = self._state
        # Base interval from configuration (float: off-hours scaling is fractional)
        interval: float = state.base_check_interval

        # Factor in recent trading activity — use naive UTC for comparison
        # since positions from the database store naive UTC timestamps
        now_naive = datetime.now(UTC).replace(tzinfo=None)
        one_hour_ago = now_naive - timedelta(hours=1)
        recent_trades = len(
            [
                p
                for p in state.live_position_tracker.positions.values()
                if p.entry_time is not None and p.entry_time.replace(tzinfo=None) > one_hour_ago
            ]
        )
        if recent_trades > 0:
            # More frequent checks if we have recent activity
            interval = max(state.min_check_interval, interval // 2)
        elif state.live_position_tracker.position_count == 0:
            # Less frequent checks if no active positions
            interval = min(state.max_check_interval, interval * 2)

        # Consider time of day (basic market hours awareness)
        current_hour = datetime.now(UTC).hour
        if current_hour < 6 or current_hour > 22:  # Off-hours (UTC)
            interval = min(state.max_check_interval, interval * 1.5)

        return int(interval)

    def is_data_fresh(self, df: pd.DataFrame) -> bool:
        """Check if the data is fresh enough to warrant processing.

        When the WS kline cache is active and healthy, uses buffer freshness
        (candle timestamps stay static for higher timeframes like 1h). On the
        REST fallback the tail bar is the *forming* candle, whose open time is
        up to one interval old by construction, so its age is judged against
        ``interval + data_freshness_threshold``: a tail bar older than that is
        genuinely missing the current candle. Every verdict feeds the stale-data
        streak behind ``stale_data_observation``.
        """
        fresh, reason = self._evaluate_freshness(df)
        self._record_freshness(fresh, reason)
        return fresh

    def stale_data_observation(self) -> StaleDataObservation:
        """Has the freshness gate been blocking the loop long enough to announce?"""
        if self._stale_since is None:
            return StaleDataObservation(stale=False, elapsed_seconds=0.0)
        elapsed = max(0.0, self._clock() - self._stale_since)
        if elapsed <= STALE_DATA_LATCH_SECONDS:
            return StaleDataObservation(stale=False, elapsed_seconds=elapsed)
        return StaleDataObservation(
            stale=True,
            elapsed_seconds=elapsed,
            reason=(
                f"market data has failed the freshness gate for {int(elapsed)}s "
                f"({self._stale_reason}); the trading loop is skipping entries and "
                "in-engine exits"
            ),
        )

    def _record_freshness(self, fresh: bool, reason: str | None) -> None:
        """Track the failing streak; log its edges once instead of every tick."""
        if fresh:
            if self._stale_since is not None:
                logger.info(
                    "Market data fresh again after %ds", int(self._clock() - self._stale_since)
                )
                self._stale_since = None
                self._stale_reason = None
            return
        if self._stale_since is None:
            self._stale_since = self._clock()
            logger.warning("Market data failed the freshness gate: %s", reason)
        self._stale_reason = reason

    def _evaluate_freshness(self, df: pd.DataFrame) -> tuple[bool, str | None]:
        """Return ``(fresh, reason_if_stale)`` for the frame the loop just fetched."""
        state = self._state
        if df is None or df.empty:
            return False, "empty market-data frame"

        # Bypass candle-timestamp check only when WS stream is confirmed healthy
        if (
            state._ws_kline_active
            and state._kline_buffer
            and state._ws_kline_provider
            and getattr(state._ws_kline_provider, "ws_healthy", False)
        ):
            if state._kline_buffer.is_fresh:
                return True, None
            return False, "WebSocket kline buffer has stopped updating"

        latest_timestamp = df.index[-1] if hasattr(df.index[-1], "timestamp") else datetime.now(UTC)
        if isinstance(latest_timestamp, str):
            try:
                latest_timestamp = pd.to_datetime(latest_timestamp)
            except (ValueError, TypeError):
                return True, None  # Assume fresh if we can't parse timestamp

        # Normalizes to UTC to avoid naive/aware datetime comparisons.
        if isinstance(latest_timestamp, pd.Timestamp):
            if latest_timestamp.tz is None:
                latest_timestamp = latest_timestamp.tz_localize(UTC)
            else:
                latest_timestamp = latest_timestamp.tz_convert(UTC)
        elif isinstance(latest_timestamp, datetime):
            if latest_timestamp.tzinfo is None:
                latest_timestamp = latest_timestamp.replace(tzinfo=UTC)
            else:
                latest_timestamp = latest_timestamp.astimezone(UTC)

        age_seconds = (datetime.now(UTC) - latest_timestamp).total_seconds()
        max_age_seconds = state.data_freshness_threshold + self._candle_interval_seconds()
        if age_seconds <= max_age_seconds:
            return True, None
        return False, (
            f"latest candle opened {int(age_seconds)}s ago, limit {int(max_age_seconds)}s "
            f"(timeframe {getattr(state, 'timeframe', None) or 'unknown'})"
        )

    def _candle_interval_seconds(self) -> float:
        """Seconds per candle for the engine's timeframe, or 0 when unknown.

        Unknown means the strict threshold-only rule: never relax the gate on a
        timeframe this code cannot size.
        """
        timeframe = getattr(self._state, "timeframe", None)
        interval_ms = timeframe_to_ms(timeframe) if isinstance(timeframe, str) else None
        return interval_ms / 1000.0 if interval_ms else 0.0
