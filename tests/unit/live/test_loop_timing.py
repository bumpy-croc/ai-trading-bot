"""Unit tests for LiveLoopTimingCoordinator (#486).

The trading-loop cadence + data-freshness helpers were extracted from
LiveTradingEngine into ``LiveLoopTimingCoordinator`` (the engine keeps thin
delegating wrappers). These tests drive the coordinator directly via an
autospec'd engine-state backref, covering the main branch of each helper:
the stop-event break in ``sleep_with_interrupt``, the activity/no-position
scaling in ``calculate_adaptive_interval``, and the WS-buffer vs candle-age
paths in ``is_data_fresh``.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from unittest.mock import MagicMock, create_autospec

import pandas as pd
import pytest

from src.engines.live.loop_timing import (
    LiveLoopTimingCoordinator,
    LiveLoopTimingEngineState,
)

pytestmark = pytest.mark.fast


def _make_state(**overrides) -> MagicMock:
    """Autospec'd engine-state backref with sane loop-timing defaults."""
    state = create_autospec(LiveLoopTimingEngineState, instance=True)
    state.stop_event = threading.Event()
    state.base_check_interval = 60
    state.min_check_interval = 10
    state.max_check_interval = 300
    state.data_freshness_threshold = 120.0
    state.live_position_tracker = MagicMock()
    state.live_position_tracker.positions = {}
    state.live_position_tracker.position_count = 0
    state._ws_kline_active = False
    state._kline_buffer = None
    state._ws_kline_provider = None
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


def test_sleep_with_interrupt_breaks_when_stop_event_set():
    state = _make_state()
    state.stop_event.set()  # already stopping -> must return promptly
    coordinator = LiveLoopTimingCoordinator(state)

    start = time.time()
    coordinator.sleep_with_interrupt(5.0)
    elapsed = time.time() - start

    assert elapsed < 1.0  # did not sleep the full 5s


def test_calculate_adaptive_interval_no_positions_backs_off():
    # No open positions -> interval doubles (capped at max).
    state = _make_state()
    interval = LiveLoopTimingCoordinator(state).calculate_adaptive_interval()
    assert interval == 120  # 60 * 2


def test_calculate_adaptive_interval_recent_activity_speeds_up():
    pos = MagicMock()
    pos.entry_time = datetime.now(UTC)  # entered within the last hour
    state = _make_state()
    state.live_position_tracker.positions = {"p1": pos}
    state.live_position_tracker.position_count = 1

    interval = LiveLoopTimingCoordinator(state).calculate_adaptive_interval()
    assert interval == 30  # max(min_check_interval=10, 60 // 2)


def test_is_data_fresh_false_on_empty_frame():
    state = _make_state()
    assert LiveLoopTimingCoordinator(state).is_data_fresh(pd.DataFrame()) is False


def test_is_data_fresh_uses_ws_buffer_when_stream_healthy():
    buffer = MagicMock()
    buffer.is_fresh = True
    provider = MagicMock()
    provider.ws_healthy = True
    state = _make_state(_ws_kline_active=True, _kline_buffer=buffer, _ws_kline_provider=provider)

    df = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex([pd.Timestamp.now(tz=UTC)]))
    assert LiveLoopTimingCoordinator(state).is_data_fresh(df) is True
    buffer.is_fresh = False
    assert LiveLoopTimingCoordinator(state).is_data_fresh(df) is False


def test_is_data_fresh_recent_candle_is_fresh_stale_candle_is_not():
    state = _make_state()  # WS inactive -> candle-age path
    coordinator = LiveLoopTimingCoordinator(state)

    fresh_df = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex([datetime.now(UTC)]))
    assert coordinator.is_data_fresh(fresh_df) is True

    stale_df = pd.DataFrame(
        {"close": [1.0]},
        index=pd.DatetimeIndex([datetime.now(UTC) - pd.Timedelta(seconds=600)]),
    )
    assert coordinator.is_data_fresh(stale_df) is False
