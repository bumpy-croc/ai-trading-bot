"""Unit tests for DB-durable short-guard rejection events (#990).

The SHORT-side margin inventory guard in ``_execute_live_order`` fail-closed
rejects short entries whenever free base-asset value exceeds the dust
threshold. These tests verify the rejections become observable as
``system_events`` rows without changing the guard's accept/reject behavior:

- A rejection writes a SHORT_ENTRY_BLOCKED event carrying symbol, side, free
  balance, threshold, signal metadata, and an open-position snapshot.
- A rejection episode (many consecutive cycles) writes a bounded number of
  rows: the first rejection, every Nth after it, and an episode-end summary.
- Event-write failures never propagate into the trading decision.
- The guard's decision itself is unchanged in both accept and reject paths.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from src.database.models import EventType
from src.engines.live.execution.execution_engine import (
    SHORT_GUARD_DUST_THRESHOLD_USD,
    SHORT_GUARD_EMIT_EVERY_N,
    SHORT_GUARD_EPISODE_GAP_SECONDS,
    LiveExecutionEngine,
)
from src.engines.shared.models import PositionSide

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakePosition:
    """Minimal open-position stand-in with real (JSON-safe) field values."""

    symbol: str
    side: PositionSide
    current_size: float
    quantity: float


def _make_margin_engine(*, free_base: float | None = 0.05):
    """Build a live LiveExecutionEngine in margin mode.

    ``free_base`` sets the mocked free base-asset balance (None -> get_balance
    returns None, the fail-closed branch).
    """
    mock_exchange = MagicMock()
    mock_exchange.place_order.return_value = MagicMock(
        order_id="test_order_123",
        average_price=2000.0,
        filled_quantity=0.025,
        commission=0.0,
        status="FILLED",
    )
    mock_exchange.is_margin_mode = True
    if free_base is None:
        mock_exchange.get_balance.return_value = None
    else:
        balance = MagicMock()
        balance.free = free_base
        mock_exchange.get_balance.return_value = balance

    engine = LiveExecutionEngine(
        enable_live_trading=True,
        exchange_interface=mock_exchange,
    )
    engine._normalize_quantity = MagicMock(return_value=0.025)
    engine.db_manager = MagicMock()
    engine.session_id = 7
    engine.strategy_name = "test_strategy"
    return engine


def _reject_short(engine, *, signal_context=None):
    """Run one short-entry attempt through the guard; return its result."""
    return engine._execute_live_order(
        symbol="ETHUSDT",
        side=PositionSide.SHORT,
        value=50.0,
        price=2000.0,
        signal_context=signal_context,
    )


def _logged_events(engine):
    """Return the list of log_event call kwargs recorded on the mock db."""
    return [call.kwargs for call in engine.db_manager.log_event.call_args_list]


# ---------------------------------------------------------------------------
# Rejection event payload
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_rejection_emits_event_with_payload():
    """A guard rejection writes one SHORT_ENTRY_BLOCKED row with full context."""
    engine = _make_margin_engine(free_base=0.05)  # ~$100 free at $2000

    result = _reject_short(engine, signal_context={"strength": 0.8, "confidence": 0.6})

    assert result == (None, None)
    events = _logged_events(engine)
    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == EventType.SHORT_ENTRY_BLOCKED
    assert event["severity"] == "warning"
    assert event["component"] == "execution"
    assert event["error_code"] == "SHORT_ENTRY_BLOCKED"
    assert event["session_id"] == 7

    details = event["details"]
    assert details["symbol"] == "ETHUSDT"
    assert details["side"] == "short"
    assert details["base_asset"] == "ETH"
    assert details["reason"] == "free_inventory_above_threshold"
    assert details["free_base_balance"] == pytest.approx(0.05)
    assert details["free_value_usd"] == pytest.approx(100.0)
    assert details["threshold_usd"] == pytest.approx(SHORT_GUARD_DUST_THRESHOLD_USD)
    assert details["price"] == pytest.approx(2000.0)
    assert details["signal"] == {"strength": 0.8, "confidence": 0.6}
    assert details["episode"]["reject_count"] == 1


@pytest.mark.fast
def test_rejection_includes_open_position_snapshot():
    """The event carries a compact snapshot of open positions when wired."""
    engine = _make_margin_engine(free_base=0.05)
    engine.position_snapshot_provider = lambda: [
        _FakePosition(symbol="ETHUSDT", side=PositionSide.LONG, current_size=0.02, quantity=0.01)
    ]

    _reject_short(engine)

    details = _logged_events(engine)[0]["details"]
    assert details["open_positions"] == [
        {"symbol": "ETHUSDT", "side": "long", "current_size": 0.02, "quantity": 0.01}
    ]


@pytest.mark.fast
def test_rejection_without_provider_has_null_snapshot():
    """No snapshot provider wired -> open_positions is None, event still writes."""
    engine = _make_margin_engine(free_base=0.05)

    _reject_short(engine)

    details = _logged_events(engine)[0]["details"]
    assert details["open_positions"] is None


@pytest.mark.fast
def test_snapshot_provider_failure_does_not_block_event():
    """A raising snapshot provider degrades to None instead of losing the row."""
    engine = _make_margin_engine(free_base=0.05)

    def _boom():
        raise RuntimeError("tracker unavailable")

    engine.position_snapshot_provider = _boom

    result = _reject_short(engine)

    assert result == (None, None)
    details = _logged_events(engine)[0]["details"]
    assert details["open_positions"] is None


# ---------------------------------------------------------------------------
# Fail-closed branches also emit
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_balance_lookup_error_emits_event():
    """The fail-closed lookup-error rejection is observable too."""
    engine = _make_margin_engine(free_base=0.05)
    engine.exchange_interface.get_balance.side_effect = ConnectionError("api down")

    result = _reject_short(engine)

    assert result == (None, None)
    details = _logged_events(engine)[0]["details"]
    assert details["reason"] == "balance_lookup_error"
    assert details["free_base_balance"] is None
    assert details["free_value_usd"] is None


@pytest.mark.fast
def test_balance_unavailable_emits_event():
    """get_balance returning None (fail-closed) is observable too."""
    engine = _make_margin_engine(free_base=None)

    result = _reject_short(engine)

    assert result == (None, None)
    details = _logged_events(engine)[0]["details"]
    assert details["reason"] == "balance_unavailable"
    assert details["free_base_balance"] is None


# ---------------------------------------------------------------------------
# Episode dedup
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_episode_dedup_bounds_rows():
    """30 consecutive rejections write 4 rows (1st + every Nth), not 30."""
    engine = _make_margin_engine(free_base=0.05)

    for _ in range(30):
        assert _reject_short(engine) == (None, None)

    events = _logged_events(engine)
    assert len(events) == 4  # rejections 1, 10, 20, 30 with N=10
    counts = [e["details"]["episode"]["reject_count"] for e in events]
    assert counts == [1, SHORT_GUARD_EMIT_EVERY_N, 2 * SHORT_GUARD_EMIT_EVERY_N, 30]
    assert all(e["error_code"] == "SHORT_ENTRY_BLOCKED" for e in events)


@pytest.mark.fast
def test_guard_pass_emits_episode_end_summary():
    """When the guard accepts again, the episode closes with a summary row."""
    engine = _make_margin_engine(free_base=0.05)
    for _ in range(3):
        _reject_short(engine)

    # Inventory cleared: guard passes and the order goes through.
    engine.exchange_interface.get_balance.return_value = MagicMock(free=0.0)
    result = _reject_short(engine)

    assert result[0] is not None  # order placed — accept path unchanged
    events = _logged_events(engine)
    summary = events[-1]
    assert summary["error_code"] == "SHORT_GUARD_EPISODE_END"
    assert summary["event_type"] == EventType.SHORT_ENTRY_BLOCKED
    details = summary["details"]
    assert details["end_reason"] == "guard_pass"
    assert details["rejections_total"] == 3
    assert details["base_asset"] == "ETH"


@pytest.mark.fast
def test_inactivity_gap_closes_episode_and_starts_fresh():
    """A rejection after a long quiet gap summarises the old episode and
    starts a new one (first-of-episode row emits again)."""
    engine = _make_margin_engine(free_base=0.05)
    now = 1000.0
    engine._monotonic = lambda: now

    for _ in range(2):
        _reject_short(engine)
    assert len(_logged_events(engine)) == 1  # first rejection only

    now += SHORT_GUARD_EPISODE_GAP_SECONDS + 1.0
    _reject_short(engine)

    events = _logged_events(engine)
    assert len(events) == 3
    gap_summary = events[1]
    assert gap_summary["error_code"] == "SHORT_GUARD_EPISODE_END"
    assert gap_summary["details"]["end_reason"] == "inactivity_gap"
    assert gap_summary["details"]["rejections_total"] == 2
    fresh = events[2]
    assert fresh["error_code"] == "SHORT_ENTRY_BLOCKED"
    assert fresh["details"]["episode"]["reject_count"] == 1


@pytest.mark.fast
def test_new_episode_after_guard_pass_emits_first_rejection():
    """After an episode closes via guard pass, the next rejection is a fresh
    first-of-episode row (the latch resets)."""
    engine = _make_margin_engine(free_base=0.05)
    _reject_short(engine)

    engine.exchange_interface.get_balance.return_value = MagicMock(free=0.0)
    _reject_short(engine)  # guard pass -> episode end summary

    blocked = MagicMock()
    blocked.free = 0.05
    engine.exchange_interface.get_balance.return_value = blocked
    _reject_short(engine)

    events = _logged_events(engine)
    assert [e["error_code"] for e in events] == [
        "SHORT_ENTRY_BLOCKED",
        "SHORT_GUARD_EPISODE_END",
        "SHORT_ENTRY_BLOCKED",
    ]
    assert events[-1]["details"]["episode"]["reject_count"] == 1


# ---------------------------------------------------------------------------
# Fault isolation and unchanged guard behavior
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_event_write_failure_does_not_propagate():
    """A DB failure in the event write never breaks the rejection decision."""
    engine = _make_margin_engine(free_base=0.05)
    engine.db_manager.log_event.side_effect = RuntimeError("db down")

    result = _reject_short(engine)

    assert result == (None, None)
    engine.exchange_interface.place_order.assert_not_called()


@pytest.mark.fast
def test_rejection_without_session_still_rejects():
    """No db_manager/session wired -> guard decision identical, no crash."""
    engine = _make_margin_engine(free_base=0.05)
    engine.db_manager = None
    engine.session_id = None

    result = _reject_short(engine)

    assert result == (None, None)
    engine.exchange_interface.place_order.assert_not_called()


@pytest.mark.fast
def test_guard_accept_is_silent_and_unchanged():
    """Sub-threshold free balance: order placed, no guard events written."""
    engine = _make_margin_engine(free_base=0.0001)  # $0.20 at $2000 — dust

    result = _reject_short(engine)

    assert result[0] is not None
    engine.exchange_interface.place_order.assert_called_once()
    guard_events = [
        e for e in _logged_events(engine) if e.get("event_type") == EventType.SHORT_ENTRY_BLOCKED
    ]
    assert guard_events == []


@pytest.mark.fast
def test_long_entries_never_touch_guard_events():
    """Long entries bypass the guard entirely — no events, order placed."""
    engine = _make_margin_engine(free_base=0.05)

    result = engine._execute_live_order(
        symbol="ETHUSDT",
        side=PositionSide.LONG,
        value=50.0,
        price=2000.0,
    )

    assert result[0] is not None
    guard_events = [
        e for e in _logged_events(engine) if e.get("event_type") == EventType.SHORT_ENTRY_BLOCKED
    ]
    assert guard_events == []


# ---------------------------------------------------------------------------
# Signal context threading
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_execute_entry_threads_signal_context_to_event():
    """signal_context passed to execute_entry reaches the rejection event."""
    engine = _make_margin_engine(free_base=0.05)

    result = engine.execute_entry(
        symbol="ETHUSDT",
        side=PositionSide.SHORT,
        size_fraction=0.1,
        base_price=2000.0,
        balance=1000.0,
        signal_context={"strength": 0.9, "confidence": 0.7},
    )

    assert result.success is False  # guard rejection unchanged
    details = _logged_events(engine)[0]["details"]
    assert details["signal"] == {"strength": 0.9, "confidence": 0.7}


@pytest.mark.fast
def test_entry_handler_passes_signal_metadata():
    """LiveEntryHandler forwards signal strength/confidence to the engine."""
    from src.engines.live.execution.entry_handler import LiveEntryHandler, LiveEntrySignal
    from src.engines.live.execution.execution_engine import EntryExecutionResult

    execution_engine = MagicMock()
    execution_engine.execute_entry.return_value = EntryExecutionResult(
        success=False, error="Failed to execute live order"
    )
    execution_model = MagicMock()
    execution_model.decide_fill.return_value = MagicMock(
        should_fill=True, fill_price=2000.0, liquidity="taker"
    )

    handler = LiveEntryHandler(
        execution_engine=execution_engine,
        execution_model=execution_model,
    )
    signal = LiveEntrySignal(
        should_enter=True,
        side=PositionSide.SHORT,
        size_fraction=0.1,
        signal_strength=0.42,
        signal_confidence=0.84,
    )

    handler.execute_entry(signal=signal, symbol="ETHUSDT", current_price=2000.0, balance=1000.0)

    call_kwargs = execution_engine.execute_entry.call_args.kwargs
    assert call_kwargs["signal_context"] == {"strength": 0.42, "confidence": 0.84}
