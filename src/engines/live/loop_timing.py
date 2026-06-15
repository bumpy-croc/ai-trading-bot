"""Live trading-loop timing helpers (#486).

Small, self-contained helpers the trading loop uses for cadence and data
freshness: ``sleep_with_interrupt`` (interruptible sleep), ``calculate_adaptive_interval``
(activity/time-of-day-aware poll interval), and ``is_data_fresh`` (candle-age /
WS-buffer freshness gate). Moved verbatim out of ``LiveTradingEngine`` (a
mechanical ``self.`` -> ``state.`` rewrite against an engine backref); the
engine keeps thin delegating wrappers, so the trading loop and the
``LiveMarketDataCoordinator`` (which calls ``state._is_data_fresh``) are
unchanged.

Leaf helpers — no order placement, balance mutation, or engine-method calls;
the coordinator holds no state of its own. Runs on the trading-loop thread.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

import pandas as pd

from src.config.constants import DEFAULT_SLEEP_POLL_INTERVAL

if TYPE_CHECKING:
    from src.engines.live.execution.position_tracker import LivePositionTracker
    from src.engines.live.kline_buffer import KlineBuffer


class LiveLoopTimingEngineState(Protocol):
    """Engine state the loop-timing helpers read at call time."""

    stop_event: threading.Event
    base_check_interval: int
    min_check_interval: int
    max_check_interval: int
    data_freshness_threshold: float
    live_position_tracker: LivePositionTracker
    _ws_kline_active: bool
    _kline_buffer: KlineBuffer | None
    _ws_kline_provider: Any


class LiveLoopTimingCoordinator:
    """Owns the trading loop's cadence + data-freshness helpers."""

    def __init__(self, engine_state: LiveLoopTimingEngineState) -> None:
        """Bind to the engine's live state (see protocol for the surface)."""
        self._state = engine_state

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

        When WS kline cache is active, uses buffer freshness instead of
        candle timestamps (which stay static for higher timeframes like 1h).
        """
        state = self._state
        if df is None or df.empty:
            return False

        # Bypass candle-timestamp check only when WS stream is confirmed healthy
        if (
            state._ws_kline_active
            and state._kline_buffer
            and state._ws_kline_provider
            and getattr(state._ws_kline_provider, "ws_healthy", False)
        ):
            return state._kline_buffer.is_fresh

        latest_timestamp = df.index[-1] if hasattr(df.index[-1], "timestamp") else datetime.now(UTC)
        if isinstance(latest_timestamp, str):
            try:
                latest_timestamp = pd.to_datetime(latest_timestamp)
            except (ValueError, TypeError):
                return True  # Assume fresh if we can't parse timestamp

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
        return age_seconds <= state.data_freshness_threshold
