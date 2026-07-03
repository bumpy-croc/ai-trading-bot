"""Max-position cap enforcement across all live position-increasing paths.

Regression guards for the live cap bypass: entries were clamped at the
coordinator (long, component, legacy short) but the scale-in path could grow
a position past ``--max-position``:

- ``LivePositionTracker.apply_scale_in`` capped ``current_size`` at 1.0, not
  at the configured cap (only ``size`` was clamped), and
- ``LiveTradingEngine`` never wired its ``max_position_size`` into
  ``LiveExitHandler``, so the scale-in cap silently stayed at the constant
  default regardless of the runner's ``--max-position``.

Observed live: a ~10% entry plus a scale-in clamped only by the (daily
resetting) daily-risk budget reached ~16% exposure against a 10% cap.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from src.engines.live.execution.entry_handler import LiveEntryHandler
from src.engines.live.execution.exit_handler import LiveExitHandler
from src.engines.live.execution.position_tracker import LivePosition, LivePositionTracker
from src.engines.shared.models import PositionSide
from src.strategies.components import SignalDirection

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Runtime entry path: LiveEntryHandler.process_runtime_decision clamps
# ---------------------------------------------------------------------------


def _runtime_decision(notional: float) -> SimpleNamespace:
    """Plain runtime-decision stub (no MagicMock so float math stays real)."""
    return SimpleNamespace(
        signal=SimpleNamespace(direction=SignalDirection.BUY, strength=0.9, confidence=0.8),
        position_size=notional,
        metadata={},
    )


def test_runtime_path_size_clamped_to_max_position():
    handler = LiveEntryHandler(
        execution_engine=MagicMock(),
        execution_model=MagicMock(),
        max_position_size=0.2,
    )

    # 500 notional on a 1000 balance = 0.5 fraction, above the 0.2 cap.
    signal = handler.process_runtime_decision(
        runtime_decision=_runtime_decision(500.0),
        balance=1000.0,
        current_price=50000.0,
        current_time=datetime(2024, 1, 1, tzinfo=UTC),
    )

    assert signal.should_enter is True
    assert signal.size_fraction == pytest.approx(0.2)


def test_runtime_path_size_below_cap_unclamped():
    handler = LiveEntryHandler(
        execution_engine=MagicMock(),
        execution_model=MagicMock(),
        max_position_size=0.2,
    )

    signal = handler.process_runtime_decision(
        runtime_decision=_runtime_decision(150.0),
        balance=1000.0,
        current_price=50000.0,
        current_time=datetime(2024, 1, 1, tzinfo=UTC),
    )

    assert signal.should_enter is True
    assert signal.size_fraction == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Scale-in path: tracker caps growth at max_position_size
# ---------------------------------------------------------------------------


def _tracked_position(tracker: LivePositionTracker, *, size: float) -> LivePosition:
    position = LivePosition(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=size,
        entry_price=50000.0,
        entry_time=datetime(2024, 1, 1, tzinfo=UTC),
        entry_balance=1000.0,
        quantity=0.01,
        order_id="order-1",
    )
    tracker._positions["order-1"] = position
    return position


def test_apply_scale_in_caps_current_size_at_max_position():
    tracker = LivePositionTracker(db_manager=None)
    position = _tracked_position(tracker, size=0.10)

    result = tracker.apply_scale_in(
        order_id="order-1",
        delta_fraction=0.06,
        price=51000.0,
        threshold_level=0,
        fraction_of_original=0.06,
        max_position_size=0.10,
    )

    assert result is not None
    assert position.current_size == pytest.approx(0.10)
    assert position.size == pytest.approx(0.10)


def test_apply_scale_in_allows_growth_within_cap():
    tracker = LivePositionTracker(db_manager=None)
    position = _tracked_position(tracker, size=0.10)

    result = tracker.apply_scale_in(
        order_id="order-1",
        delta_fraction=0.05,
        price=51000.0,
        threshold_level=0,
        fraction_of_original=0.05,
        max_position_size=0.20,
    )

    assert result is not None
    assert position.current_size == pytest.approx(0.15)
    assert position.size == pytest.approx(0.15)


def test_apply_scale_in_never_shrinks_over_cap_position():
    """An adopted position already above the cap must keep its tracked size.

    Shrinking it here would diverge tracked state from real exchange holdings
    (CODE.md Position Fields: current_size drives SL re-placement and holdings
    checks). It just must not grow any further.
    """
    tracker = LivePositionTracker(db_manager=None)
    position = _tracked_position(tracker, size=0.16)

    result = tracker.apply_scale_in(
        order_id="order-1",
        delta_fraction=0.05,
        price=51000.0,
        threshold_level=0,
        fraction_of_original=0.05,
        max_position_size=0.10,
    )

    assert result is not None
    assert position.current_size == pytest.approx(0.16)
    assert position.size == pytest.approx(0.16)


# ---------------------------------------------------------------------------
# Scale-in path: exit handler clamps the delta by max-position headroom
# ---------------------------------------------------------------------------


def _make_exit_handler(*, max_position_size: float, scale_fraction: float, position) -> tuple:
    execution_engine = MagicMock()
    execution_engine.fee_rate = 0.0
    execution_engine.slippage_rate = 0.0
    position_tracker = MagicMock()
    position_tracker.positions = {"order-1": position}
    partial_manager = MagicMock()
    partial_manager.check_partial_exit.return_value = SimpleNamespace(
        should_exit=False, exit_fraction=None, target_index=None
    )
    partial_manager.check_scale_in.return_value = SimpleNamespace(
        should_scale=True, scale_fraction=scale_fraction, target_index=0
    )
    handler = LiveExitHandler(
        execution_engine=execution_engine,
        position_tracker=position_tracker,
        execution_model=MagicMock(),
        risk_manager=None,  # no daily-risk clamp: isolates the cap clamp
        partial_manager=partial_manager,
        max_position_size=max_position_size,
    )
    return handler, position_tracker


def _scale_in_position(*, current_size: float) -> Mock:
    position = Mock()
    position.symbol = "BTCUSDT"
    position.order_id = "order-1"
    position.entry_time = datetime(2024, 1, 1, tzinfo=UTC)
    position.current_size = current_size
    position.original_size = current_size
    position.size = current_size
    return position


def test_exit_handler_scale_in_clamped_to_max_position_headroom():
    position = _scale_in_position(current_size=0.08)
    handler, tracker = _make_exit_handler(
        max_position_size=0.10, scale_fraction=0.06, position=position
    )

    handler.check_partial_operations(
        df=MagicMock(),
        current_index=0,
        current_price=51000.0,
        current_balance=1000.0,
    )

    tracker.apply_scale_in.assert_called_once()
    kwargs = tracker.apply_scale_in.call_args.kwargs
    # Requested +0.06 but only 0.02 headroom below the 0.10 cap.
    assert kwargs["delta_fraction"] == pytest.approx(0.02)
    assert kwargs["max_position_size"] == pytest.approx(0.10)


def test_exit_handler_scale_in_skipped_when_at_cap():
    position = _scale_in_position(current_size=0.10)
    handler, tracker = _make_exit_handler(
        max_position_size=0.10, scale_fraction=0.06, position=position
    )

    handler.check_partial_operations(
        df=MagicMock(),
        current_index=0,
        current_price=51000.0,
        current_balance=1000.0,
    )

    tracker.apply_scale_in.assert_not_called()


def test_exit_handler_scale_in_within_headroom_unclamped():
    position = _scale_in_position(current_size=0.08)
    handler, tracker = _make_exit_handler(
        max_position_size=0.20, scale_fraction=0.06, position=position
    )

    handler.check_partial_operations(
        df=MagicMock(),
        current_index=0,
        current_price=51000.0,
        current_balance=1000.0,
    )

    tracker.apply_scale_in.assert_called_once()
    assert tracker.apply_scale_in.call_args.kwargs["delta_fraction"] == pytest.approx(0.06)


# ---------------------------------------------------------------------------
# Engine wiring: --max-position reaches the exit handler's scale-in cap
# ---------------------------------------------------------------------------


def test_engine_wires_max_position_size_into_exit_handler(monkeypatch):
    from tests.mocks import MockDatabaseManager

    monkeypatch.setattr("src.engines.live.trading_engine.DatabaseManager", MockDatabaseManager)
    from src.engines.live.trading_engine import LiveTradingEngine

    strategy = Mock()
    strategy.get_risk_overrides.return_value = None
    strategy.name = "test"
    data_provider = Mock()
    data_provider.get_current_price.return_value = 100.0

    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=data_provider,
        initial_balance=10_000.0,
        enable_live_trading=False,
        log_trades=False,
        max_position_size=0.2,
    )

    assert engine.live_exit_handler.max_position_size == pytest.approx(0.2)
    assert engine.live_entry_handler.max_position_size == pytest.approx(0.2)
