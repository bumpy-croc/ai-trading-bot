"""Unit tests for LiveEntryCoordinator.execute_entry_locked branches (#486).

The entry pipeline was extracted from LiveTradingEngine into
``LiveEntryCoordinator`` (the engine keeps thin delegating wrappers). These
tests target the real-money execution branches directly on the coordinator,
driving it with a mocked engine-state backref. They also pin the CODE.md
hardening applied after the extraction: the stop-loss gate now keys on
``stop_loss is not None`` (so a misconfigured ``0.0`` stop is no longer
silently skipped), and the SL-calc / risk-override failure paths log.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, create_autospec

import pytest

from src.config.constants import DEFAULT_STOP_LOSS_PCT
from src.engines.live.execution.entry_coordinator import (
    LiveEntryCoordinator,
    LiveEntryEngineState,
)
from src.engines.shared.models import PositionSide
from src.strategies.components import SignalDirection

pytestmark = pytest.mark.fast


def _make_position() -> MagicMock:
    """Return a minimal position mock for use in coordinator tests."""
    position = MagicMock()
    position.order_id = "order-1"
    position.entry_price = 50000.0
    position.quantity = 0.01
    position.entry_balance = 1000.0
    position.metadata = {}
    return position


def _make_result(position: MagicMock, **overrides) -> MagicMock:
    """Return a successful entry-result mock wrapping the given position."""
    result = MagicMock()
    result.executed = True
    result.position = position
    result.entry_fee = 1.0
    result.slippage_cost = 0.5
    result.ambiguous = False
    result.error = None
    for k, v in overrides.items():
        setattr(result, k, v)
    return result


def _make_state(position: MagicMock, result: MagicMock, **overrides) -> MagicMock:
    """Build a mocked engine-state backref with sane happy-path defaults.

    Spec'd against the coordinator's protocol so method-name drift is caught;
    every attribute the execution path reads is set explicitly (protocol
    attributes are annotation-only, so they must be assigned to be readable).
    """
    state = create_autospec(LiveEntryEngineState, instance=True)
    state.enable_live_trading = False
    state.current_balance = 1000.0
    state.max_position_size = 1.0
    state.trading_session_id = None
    state._close_only_mode = False
    # Protocol attributes are annotation-only (not in dir()), so a spec'd mock
    # won't auto-create them — assign each object attribute the path reads.
    state.exchange_interface = MagicMock()
    state.order_tracker = MagicMock()
    state.live_position_tracker = MagicMock()
    state.risk_manager = MagicMock()
    state.live_entry_handler = MagicMock()
    state.stop_loss_manager = MagicMock()
    state.db_manager = MagicMock()

    state.live_position_tracker.has_position_for_symbol.return_value = False
    state.live_position_tracker.position_count = 0
    state.risk_manager.get_max_concurrent_positions.return_value = 1
    state.live_entry_handler.execute_entry.return_value = result
    state.stop_loss_manager.place_protection.return_value = "sl-order-1"
    state._strategy_name.return_value = "test_strategy"

    for k, v in overrides.items():
        setattr(state, k, v)
    return state


def _call(state: MagicMock, *, side=PositionSide.LONG, stop_loss=49000.0, take_profit=51000.0):
    """Drive execute_entry_locked on a fresh LiveEntryCoordinator with BTCUSDT defaults."""
    coordinator = LiveEntryCoordinator(engine_state=state)
    coordinator.execute_entry_locked(
        symbol="BTCUSDT",
        side=side,
        size=0.1,
        price=50000.0,
        stop_loss=stop_loss,
        take_profit=take_profit,
        signal_strength=0.8,
        signal_confidence=0.7,
    )
    return coordinator


# ---------------------------------------------------------------------------
# Pre-execution guards
# ---------------------------------------------------------------------------


def test_duplicate_position_guard_skips_execution():
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.live_position_tracker.has_position_for_symbol.return_value = True

    _call(state)

    state.live_entry_handler.execute_entry.assert_not_called()


def test_max_concurrent_positions_guard_skips_execution():
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.live_position_tracker.position_count = 1
    state.risk_manager.get_max_concurrent_positions.return_value = 1

    _call(state)

    state.live_entry_handler.execute_entry.assert_not_called()


def test_size_capped_at_max_position_size():
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.max_position_size = 0.05  # below the requested 0.1

    coordinator = LiveEntryCoordinator(engine_state=state)
    coordinator.execute_entry_locked(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=0.1,
        price=50000.0,
        stop_loss=49000.0,
        take_profit=51000.0,
        signal_strength=0.8,
        signal_confidence=0.7,
    )

    # The handler is called with the capped size, not the requested 0.1.
    _, kwargs = state.live_entry_handler.execute_entry.call_args
    assert kwargs["signal"].size_fraction == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Happy path + tracking
# ---------------------------------------------------------------------------


def test_happy_path_tracks_position_and_registers_risk():
    position = _make_position()
    state = _make_state(position, _make_result(position))

    _call(state)

    state.live_position_tracker.open_position.assert_called_once()
    state.risk_manager.update_position.assert_called_once()
    # No-session path deducts the fee directly.
    assert state.current_balance == pytest.approx(1000.0 - 1.0)


def test_failed_execution_returns_without_tracking():
    position = _make_position()
    result = _make_result(position, executed=False, error="boom")
    result.position = None
    state = _make_state(position, result)

    _call(state)

    state.live_position_tracker.open_position.assert_not_called()


# ---------------------------------------------------------------------------
# Failure / emergency-close paths
# ---------------------------------------------------------------------------


def test_position_tracking_failure_triggers_emergency_close_and_refund():
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.enable_live_trading = True
    state.live_position_tracker.open_position.side_effect = RuntimeError("tracker down")

    _call(state)

    # Emergency close placed with AUTO_REPAY; fee refunded directly (no session).
    state.exchange_interface.place_order.assert_called_once()
    assert state.current_balance == pytest.approx(1000.0)  # -1.0 fee then +1.0 refund


def test_risk_manager_failure_closes_position():
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.risk_manager.update_position.side_effect = ValueError("risk sync failed")

    _call(state)

    state._execute_exit.assert_called_once()


def test_balance_update_failure_emergency_closes_with_session():
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.enable_live_trading = True
    state.trading_session_id = 42
    state.db_manager.atomic_balance_update.side_effect = RuntimeError("db down")

    _call(state)

    # Balance update failed before tracking → emergency close, never tracks.
    state.exchange_interface.place_order.assert_called_once()
    state.live_position_tracker.open_position.assert_not_called()


def test_ambiguous_submission_enters_close_only_mode_without_stop_loss():
    position = _make_position()
    state = _make_state(position, _make_result(position, ambiguous=True))
    state.enable_live_trading = True

    _call(state)

    state._enter_close_only_mode.assert_called_once()
    state.stop_loss_manager.place_protection.assert_not_called()


def test_stop_loss_placement_failure_triggers_emergency_exit():
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.enable_live_trading = True
    state.stop_loss_manager.place_protection.return_value = None  # placement failed

    _call(state)

    state._record_event.assert_called_once()
    state._execute_exit.assert_called_once()


def test_balance_update_failure_unconfirmed_emergency_close_enters_close_only():
    """Site 1: post-entry balance update fails, then the emergency close placement
    returns None (ambiguous — NOT a confirmed close). The position may remain open
    and unprotected on the exchange, so the engine must escalate to close-only
    rather than log a false success."""
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.enable_live_trading = True
    state.trading_session_id = 42
    state.db_manager.atomic_balance_update.side_effect = RuntimeError("db down")
    state.exchange_interface.place_order.return_value = None  # ambiguous placement

    _call(state)

    state.exchange_interface.place_order.assert_called_once()
    state._enter_close_only_mode.assert_called_once()


def test_tracking_failure_unconfirmed_emergency_close_enters_close_only_no_refund():
    """Site 2: position tracking fails and the emergency close returns None
    (ambiguous). The orphaned position may still be open on the exchange, so the
    engine must escalate to close-only and must NOT optimistically refund the entry
    fee (which would overstate the balance). No-session path: balance is the fee-
    deducted 999.0 (not the 1000.0 it would be if refunded)."""
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.enable_live_trading = True  # no trading_session_id → direct balance math
    state.live_position_tracker.open_position.side_effect = RuntimeError("tracker down")
    state.exchange_interface.place_order.return_value = None  # ambiguous placement

    _call(state)

    state.exchange_interface.place_order.assert_called_once()
    state._enter_close_only_mode.assert_called_once()
    # Unconfirmed close → entry fee stays charged (deducted, not refunded).
    assert state.current_balance == pytest.approx(999.0)


def test_tracking_failure_confirmed_emergency_close_refunds_fee():
    """Site 2 contrast: a CONFIRMED emergency close (place_order returns an order)
    refunds the entry fee as before — balance returns to 1000.0 — and does not
    enter close-only."""
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.enable_live_trading = True
    state.live_position_tracker.open_position.side_effect = RuntimeError("tracker down")
    state.exchange_interface.place_order.return_value = MagicMock()  # confirmed

    _call(state)

    state.exchange_interface.place_order.assert_called_once()
    state._enter_close_only_mode.assert_not_called()
    assert state.current_balance == pytest.approx(1000.0)  # -1.0 fee then +1.0 refund


# ---------------------------------------------------------------------------
# CODE.md hardening: stop-loss gate keys on `is not None` (#813 follow-up)
# ---------------------------------------------------------------------------


def test_zero_stop_loss_is_not_silently_skipped():
    """A 0.0 stop must enter the placement path (and fail there → emergency
    close) rather than being silently treated as "no stop-loss" and leaving the
    position unprotected. Regression guard for the truthy-check fix."""
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.enable_live_trading = True

    _call(state, stop_loss=0.0)

    state.stop_loss_manager.place_protection.assert_called_once()


def test_none_stop_loss_skips_placement():
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.enable_live_trading = True

    _call(state, stop_loss=None)

    state.stop_loss_manager.place_protection.assert_not_called()


# ---------------------------------------------------------------------------
# process_legacy_short_entry (legacy duck-typed short path, moved from engine)
# ---------------------------------------------------------------------------


def _make_short_state(*, is_runtime: bool, short_signal, overrides):
    """Minimal backref for the legacy short-entry path."""
    state = create_autospec(LiveEntryEngineState, instance=True)
    state._close_only_mode = False
    state._is_runtime_strategy.return_value = is_runtime
    state.current_balance = 1000.0
    state.max_position_size = 1.0
    strategy = MagicMock()
    strategy.check_short_entry_conditions.return_value = short_signal
    strategy.get_risk_overrides.return_value = overrides
    strategy.name = "legacy_strat"
    state.strategy = strategy
    state._extract_indicators.return_value = {}
    state._get_correlation_context.return_value = None
    # Protocol attributes are annotation-only, so a spec'd/autospec'd mock won't
    # auto-create them — assign the ones the short path reads.
    state.risk_manager = MagicMock()
    state.risk_manager.calculate_position_fraction.return_value = 0.5
    state.risk_manager.compute_sl_tp.return_value = (52000.0, 48000.0)
    # Identity: dynamic-risk adjustment leaves the size unchanged here.
    state._apply_dynamic_risk_adjustment.side_effect = lambda original_size, current_time: (
        original_size
    )
    # #802 exposure gate is routed through the live entry handler; with the
    # governor inert (default) it returns the size unchanged.
    state.live_entry_handler = MagicMock()
    state.live_entry_handler.apply_pre_order_gates.side_effect = lambda size, **kwargs: (size, None)
    return state


def _run_short_entry(state):
    coordinator = LiveEntryCoordinator(engine_state=state)
    # Isolate the decision logic from the real (locked, exchange-touching) entry.
    coordinator.execute_entry = MagicMock()  # type: ignore[method-assign]
    coordinator.process_legacy_short_entry(
        df=MagicMock(),
        current_index=0,
        symbol="BTCUSDT",
        current_price=50000.0,
        current_time=datetime(2024, 1, 1, tzinfo=UTC),
    )
    return coordinator


def test_legacy_short_entry_skipped_for_runtime_strategy():
    """Runtime/component strategies never take the legacy duck-typed short path."""
    state = _make_short_state(is_runtime=True, short_signal=True, overrides={"position_sizer": "x"})
    coordinator = _run_short_entry(state)
    coordinator.execute_entry.assert_not_called()


def test_legacy_short_entry_no_signal_is_noop():
    """No short signal → no entry executed."""
    state = _make_short_state(
        is_runtime=False, short_signal=None, overrides={"position_sizer": "x"}
    )
    coordinator = _run_short_entry(state)
    coordinator.execute_entry.assert_not_called()


def test_legacy_short_entry_executes_short_with_sizer_overrides():
    """Non-runtime strategy + signal + sizer/SL overrides → a SHORT entry is executed."""
    overrides = {"position_sizer": "x", "stop_loss_pct": 0.05, "take_profit_pct": 0.04}
    state = _make_short_state(is_runtime=False, short_signal=True, overrides=overrides)
    coordinator = _run_short_entry(state)

    coordinator.execute_entry.assert_called_once()
    kwargs = coordinator.execute_entry.call_args.kwargs
    assert kwargs["side"] == PositionSide.SHORT
    assert kwargs["size"] == pytest.approx(0.5)
    assert kwargs["price"] == pytest.approx(50000.0)
    assert kwargs["stop_loss"] == pytest.approx(52000.0)


def test_legacy_short_entry_zero_size_skips_execution():
    """A dynamic-risk reduction to zero size suppresses the entry."""
    state = _make_short_state(
        is_runtime=False, short_signal=True, overrides={"position_sizer": "x"}
    )
    state._apply_dynamic_risk_adjustment.side_effect = None
    state._apply_dynamic_risk_adjustment.return_value = 0.0
    coordinator = _run_short_entry(state)
    coordinator.execute_entry.assert_not_called()


def test_legacy_short_entry_without_position_sizer_suppresses_entry():
    """No `position_sizer` override → size forced to 0.0 (logged), no entry."""
    state = _make_short_state(is_runtime=False, short_signal=True, overrides={})
    coordinator = _run_short_entry(state)
    coordinator.execute_entry.assert_not_called()


def test_legacy_short_entry_logs_when_get_risk_overrides_raises(caplog):
    """A raising `get_risk_overrides()` is logged at WARNING and falls through to None."""
    import logging

    state = _make_short_state(is_runtime=False, short_signal=True, overrides={})
    state.strategy.get_risk_overrides.side_effect = ValueError("boom")
    with caplog.at_level(logging.WARNING):
        coordinator = _run_short_entry(state)
    # overrides=None → no position_sizer → size 0.0 → no entry executed
    coordinator.execute_entry.assert_not_called()
    assert any("get_risk_overrides() raised" in r.message for r in caplog.records)


def test_legacy_short_entry_size_clamped_to_max_position():
    """A sizer fraction above the cap is clamped on the legacy short path."""
    state = _make_short_state(
        is_runtime=False,
        short_signal=True,
        overrides={"position_sizer": "x", "stop_loss_pct": 0.05, "take_profit_pct": 0.04},
    )
    state.max_position_size = 0.2  # below the sizer's 0.5
    coordinator = _run_short_entry(state)

    coordinator.execute_entry.assert_called_once()
    assert coordinator.execute_entry.call_args.kwargs["size"] == pytest.approx(0.2)


def test_legacy_short_entry_sl_fallback_when_overrides_lack_sl_tp():
    """Sizer override present but no SL/TP keys → default-stop fallback branch."""
    state = _make_short_state(
        is_runtime=False, short_signal=True, overrides={"position_sizer": "x"}
    )
    # Real float so the `1 - take_profit_pct` short-TP math resolves.
    state.strategy.take_profit_pct = 0.04
    coordinator = _run_short_entry(state)

    coordinator.execute_entry.assert_called_once()
    kwargs = coordinator.execute_entry.call_args.kwargs
    assert kwargs["side"] == PositionSide.SHORT
    # Fallback stop = price * (1 + DEFAULT_STOP_LOSS_PCT); compute_sl_tp NOT used.
    assert kwargs["stop_loss"] == pytest.approx(50000.0 * (1 + DEFAULT_STOP_LOSS_PCT))
    assert kwargs["take_profit"] == pytest.approx(50000.0 * (1 - 0.04))
    state.risk_manager.compute_sl_tp.assert_not_called()


# ---------------------------------------------------------------------------
# Max-position cap regression: every entry path clamps above-cap requests
# ---------------------------------------------------------------------------


def _make_component_entry_state(*, max_position_size: float, notional: float) -> MagicMock:
    """Backref for the direct ComponentStrategy path of check_entry_conditions."""
    from src.strategies.components import Strategy as ComponentStrategy

    state = create_autospec(LiveEntryEngineState, instance=True)
    state._close_only_mode = False
    state._is_runtime_strategy.return_value = False
    state.current_balance = 1000.0
    state.max_position_size = max_position_size
    state.trading_session_id = None
    state.db_manager = None  # skip strategy-execution DB logging

    strategy = MagicMock(spec=ComponentStrategy)
    decision = MagicMock()
    decision.position_size = notional
    decision.signal.direction = SignalDirection.BUY
    decision.signal.strength = 0.9
    decision.signal.confidence = 0.8
    strategy.process_candle.return_value = decision
    strategy.get_stop_loss_price.return_value = 45000.0
    state.strategy = strategy

    state._extract_indicators.return_value = {}
    state._extract_sentiment_data.return_value = {}
    state._extract_ml_predictions.return_value = {}
    state._build_component_positions.return_value = []
    state._get_correlation_context.return_value = None
    state._resolve_take_profit_pct.return_value = 0.3
    # Identity: dynamic-risk adjustment leaves the size unchanged here.
    state._apply_dynamic_risk_adjustment.side_effect = lambda original_size, current_time: (
        original_size
    )
    # #802 exposure gate is routed through the live entry handler; with the
    # governor inert (default) it returns the size unchanged.
    state.live_entry_handler = MagicMock()
    state.live_entry_handler.apply_pre_order_gates.side_effect = lambda size, **kwargs: (size, None)
    return state


def _run_component_entry(state: MagicMock) -> MagicMock:
    """Drive check_entry_conditions on the component path with a 1-candle df."""
    import pandas as pd

    coordinator = LiveEntryCoordinator(engine_state=state)
    coordinator.execute_entry = MagicMock()  # type: ignore[method-assign]
    df = pd.DataFrame({"close": [50000.0]})
    coordinator.check_entry_conditions(
        df=df,
        current_index=0,
        symbol="BTCUSDT",
        current_price=50000.0,
        current_time=datetime(2024, 1, 1, tzinfo=UTC),
    )
    return coordinator


def test_component_path_entry_size_clamped_to_max_position():
    """A component-strategy notional above the cap is clamped before execution."""
    # 500 notional on a 1000 balance = 0.5 fraction, above the 0.2 cap.
    state = _make_component_entry_state(max_position_size=0.2, notional=500.0)

    coordinator = _run_component_entry(state)

    coordinator.execute_entry.assert_called_once()
    assert coordinator.execute_entry.call_args.kwargs["size"] == pytest.approx(0.2)


def test_component_path_entry_size_below_cap_unclamped():
    """A below-cap component-strategy notional passes through unchanged."""
    state = _make_component_entry_state(max_position_size=0.2, notional=150.0)

    coordinator = _run_component_entry(state)

    coordinator.execute_entry.assert_called_once()
    assert coordinator.execute_entry.call_args.kwargs["size"] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# ml_predictions logging (#914): signal-metadata prediction context must reach
# the strategy_executions row instead of the historical always-null value.
# ---------------------------------------------------------------------------


def _ml_signal(metadata: dict, *, confidence: float = 0.42):
    from src.strategies.components import Signal

    return Signal(
        direction=SignalDirection.HOLD, strength=0.0, confidence=confidence, metadata=metadata
    )


def _make_runtime_entry_state() -> MagicMock:
    """Backref for the runtime-strategy path of check_entry_conditions."""
    state = create_autospec(LiveEntryEngineState, instance=True)
    state._close_only_mode = False
    state._is_runtime_strategy.return_value = True
    state.current_balance = 1000.0
    state.max_position_size = 0.2
    state.timeframe = "1h"
    state.trading_session_id = 7
    state.db_manager = MagicMock()
    state.live_position_tracker = MagicMock()
    state.live_position_tracker.position_count = 0
    state.risk_manager = MagicMock()
    state.risk_manager.get_max_concurrent_positions.return_value = 1
    state._durable_peak_balance.return_value = 1000.0
    state.live_entry_handler = MagicMock()
    no_entry = MagicMock()
    no_entry.should_enter = False
    no_entry.side = None
    state.live_entry_handler.process_runtime_decision.return_value = no_entry
    state._extract_indicators.return_value = {}
    state._extract_sentiment_data.return_value = {}
    state._extract_ml_predictions.return_value = {}
    state._strategy_name.return_value = "ml_basic"
    return state


def _run_runtime_entry(state: MagicMock, runtime_decision: MagicMock) -> None:
    import pandas as pd

    coordinator = LiveEntryCoordinator(engine_state=state)
    coordinator.execute_entry = MagicMock()  # type: ignore[method-assign]
    df = pd.DataFrame({"close": [50000.0]})
    coordinator.check_entry_conditions(
        df=df,
        current_index=0,
        symbol="BTCUSDT",
        current_price=50000.0,
        current_time=datetime(2024, 1, 1, tzinfo=UTC),
        runtime_decision=runtime_decision,
    )


def _runtime_decision_with(signal) -> MagicMock:
    decision = MagicMock()
    decision.signal = signal
    decision.regime = None
    decision.risk_metrics = None
    decision.metadata = {}
    return decision


def test_runtime_entry_passes_the_durable_peak_to_the_handler():
    """The drawdown throttle inside process_runtime_decision must measure
    drawdown from the engine's durable peak, not the restart-resettable
    PerformanceTracker peak (#845 peak-reset class)."""
    state = _make_runtime_entry_state()
    state._durable_peak_balance.return_value = 1234.0
    decision = _runtime_decision_with(_ml_signal({}))

    _run_runtime_entry(state, decision)

    kwargs = state.live_entry_handler.process_runtime_decision.call_args.kwargs
    assert kwargs["peak_balance"] == 1234.0


def test_runtime_entry_logs_signal_ml_predictions():
    state = _make_runtime_entry_state()
    decision = _runtime_decision_with(
        _ml_signal(
            {
                "generator": "ml_basic_signal_generator",
                "prediction": 50750.0,
                "current_price": 50000.0,
                "predicted_return": 0.015,
                "engine_model_name": "BTCUSDT:1h:basic:v1",
            }
        )
    )

    _run_runtime_entry(state, decision)

    kwargs = state.db_manager.log_strategy_execution.call_args.kwargs
    assert kwargs["ml_predictions"]["prediction"] == 50750.0
    assert kwargs["ml_predictions"]["predicted_return"] == 0.015
    assert kwargs["ml_predictions"]["engine_model_name"] == "BTCUSDT:1h:basic:v1"


def test_runtime_entry_logs_prediction_failure():
    state = _make_runtime_entry_state()
    decision = _runtime_decision_with(
        _ml_signal(
            {"generator": "ml_basic_signal_generator", "reason": "prediction_failed", "index": 0},
            confidence=0.0,
        )
    )

    _run_runtime_entry(state, decision)

    kwargs = state.db_manager.log_strategy_execution.call_args.kwargs
    assert kwargs["ml_predictions"]["prediction_failed"] is True
    assert kwargs["ml_predictions"]["reason"] == "prediction_failed"


def test_runtime_entry_merges_signal_over_dataframe_columns():
    state = _make_runtime_entry_state()
    state._extract_ml_predictions.return_value = {"onnx_pred": 50700.0}
    decision = _runtime_decision_with(
        _ml_signal(
            {
                "generator": "ml_basic_signal_generator",
                "prediction": 50750.0,
                "predicted_return": 0.015,
            }
        )
    )

    _run_runtime_entry(state, decision)

    kwargs = state.db_manager.log_strategy_execution.call_args.kwargs
    assert kwargs["ml_predictions"]["onnx_pred"] == 50700.0
    assert kwargs["ml_predictions"]["prediction"] == 50750.0


def test_runtime_entry_non_ml_signal_logs_null_ml_predictions():
    state = _make_runtime_entry_state()
    decision = _runtime_decision_with(
        _ml_signal({"generator": "rsi_signal_generator", "reason": "insufficient_history"})
    )

    _run_runtime_entry(state, decision)

    kwargs = state.db_manager.log_strategy_execution.call_args.kwargs
    assert kwargs["ml_predictions"] is None


def test_component_entry_logs_signal_ml_predictions():
    state = _make_component_entry_state(max_position_size=0.2, notional=150.0)
    # _make_component_entry_state skips DB logging; enable it and provide the
    # state the logging block reads.
    state.db_manager = MagicMock()
    state.live_position_tracker = MagicMock()
    state.live_position_tracker.position_count = 0
    state.risk_manager = MagicMock()
    state.risk_manager.get_max_concurrent_positions.return_value = 1
    state._strategy_name.return_value = "ml_basic"
    decision = state.strategy.process_candle.return_value
    decision.signal = _ml_signal(
        {
            "generator": "ml_basic_signal_generator",
            "prediction": 50750.0,
            "predicted_return": 0.015,
            "engine_model_name": "BTCUSDT:1h:basic:v1",
        }
    )

    _run_component_entry(state)

    kwargs = state.db_manager.log_strategy_execution.call_args.kwargs
    assert kwargs["ml_predictions"]["prediction"] == 50750.0
    assert kwargs["ml_predictions"]["engine_model_name"] == "BTCUSDT:1h:basic:v1"
