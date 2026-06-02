"""Parity wiring tests between live and backtest engines.

These tests lock in three structural parity invariants that previously drifted
silently between the two engines:

1. **Crash recovery → risk_manager registration.** When a position is added to
   the live tracker via the reconciler path (no DB row, just a fill detected
   on the exchange), the engine must still register it with the risk manager
   — same as backtest does on every entry. Without this, per-symbol caps and
   correlation gates are bypassed for crash-recovered fills.

2. **CorrelationHandler wired into LiveEntryHandler.** Backtest's EntryHandler
   reduces position size for correlated exposure; live must too. Previously
   live built a CorrelationEngine but never wrapped it in a CorrelationHandler
   nor passed it down — so live entries silently over-concentrated in
   correlated pairs that backtest would have de-risked.

3. **process_candle current_positions parity.** When live invokes a direct
   ComponentStrategy via process_candle (no StrategyRuntime wrapper), it now
   passes the live positions list rather than ``None``. Backtest already
   passes positions via the runtime context. Strategies that consult
   current_positions (anti-pyramiding, correlation-aware sizing, exposure
   capping) get the same view in both engines.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, Mock

import pytest

from src.engines.live.trading_engine import LiveTradingEngine, Position, PositionSide
from tests.mocks import MockDatabaseManager


@pytest.fixture(autouse=True)
def mock_database_manager(monkeypatch):
    """Mock the DatabaseManager so engine init succeeds without a live DB."""
    original_init = MockDatabaseManager.__init__

    def patched_init(self, database_url=None):
        original_init(self, database_url)
        self._fallback_balance = 10_000.0

    monkeypatch.setattr(MockDatabaseManager, "__init__", patched_init)
    monkeypatch.setattr("src.engines.live.trading_engine.DatabaseManager", MockDatabaseManager)


def _make_engine() -> LiveTradingEngine:
    """Build a paper-mode engine with a mock strategy / data provider."""
    strategy = Mock()
    strategy.get_risk_overrides.return_value = None
    strategy.name = "test"
    data_provider = Mock()
    data_provider.get_current_price.return_value = 100.0
    return LiveTradingEngine(
        strategy=strategy,
        data_provider=data_provider,
        initial_balance=10_000.0,
        enable_live_trading=False,
        log_trades=False,
        fee_rate=0.0,
        slippage_rate=0.0,
    )


@pytest.mark.fast
class TestCorrelationHandlerWired:
    """The live engine must wire CorrelationHandler into LiveEntryHandler.

    Backtest does this at src/engines/backtest/engine.py:343-385. Live must
    match so live entries reduce size for correlated exposure the same way
    backtest entries do.
    """

    def test_correlation_handler_present_when_engine_available(self) -> None:
        engine = _make_engine()
        # Engine should have built a CorrelationHandler iff the underlying
        # CorrelationEngine initialized successfully — same condition as backtest.
        if engine.correlation_engine is not None:
            assert engine.correlation_handler is not None
            assert engine.live_entry_handler.correlation_handler is engine.correlation_handler

    def test_no_correlation_handler_when_engine_missing(self, monkeypatch) -> None:
        """If correlation engine init fails, handler must be None — never crash."""

        def boom(*args, **kwargs):
            raise RuntimeError("simulated correlation engine failure")

        monkeypatch.setattr("src.engines.live.trading_engine.CorrelationEngine", boom)
        engine = _make_engine()
        assert engine.correlation_engine is None
        assert engine.correlation_handler is None
        assert engine.live_entry_handler.correlation_handler is None

    def test_check_entry_passes_correlation_context(self, monkeypatch) -> None:
        """``_check_entry_conditions`` must pass symbol/timeframe/df/index to the
        entry handler so the correlation guard can fire.

        Locks in the fix for the silent no-op: previously
        ``process_runtime_decision`` was called without these args, so the
        guard at LiveEntryHandler.process_runtime_decision:208-222
        (``and df is not None and index is not None``) always evaluated False
        and ``CorrelationHandler.apply_correlation_control`` never ran in live —
        making the wired CorrelationHandler a no-op. Backtest passes the full
        context unconditionally (src/engines/backtest/execution/entry_handler.py:209-216).
        """
        import pandas as pd

        engine = _make_engine()
        engine.timeframe = "1h"
        engine._active_symbol = "TEST"
        # Force the runtime path so we hit the call site we patched.
        monkeypatch.setattr(engine, "_is_runtime_strategy", lambda: True)

        # Use a sentinel exception so we observe the call args without
        # needing to drive the rest of _check_entry_conditions (which logs
        # regime info that doesn't tolerate a Mock).
        class _Captured(Exception):
            def __init__(self, kwargs):
                self.kwargs = kwargs

        def fake_process(**kwargs):
            raise _Captured(kwargs)

        monkeypatch.setattr(engine.live_entry_handler, "process_runtime_decision", fake_process)

        df = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1.0]}
        )
        with pytest.raises(_Captured) as exc_info:
            engine._check_entry_conditions(
                df=df,
                current_index=0,
                symbol="TEST",
                current_price=100.0,
                current_time=datetime.now(UTC),
                runtime_decision=Mock(),
            )

        captured = exc_info.value.kwargs
        # All four context fields must reach the handler — these are exactly
        # the fields the correlation guard requires to be non-None.
        assert captured.get("symbol") == "TEST"
        assert captured.get("timeframe") == "1h"
        assert captured.get("df") is df
        assert captured.get("index") == 0


@pytest.mark.fast
class TestRiskManagerSweepRegistersRecoveredPositions:
    """``_ensure_positions_registered_with_risk_manager`` must register every
    tracked position with the risk manager.

    This closes the gap where ``PositionReconciler._reconcile_filled_entry``
    creates a position via ``track_recovered_position`` without registering
    it (reconciliation.py:454-487 has no risk_manager handle), so per-symbol
    caps and correlation gates would otherwise miss crash-recovered fills.
    """

    def test_sweep_registers_tracked_position(self) -> None:
        engine = _make_engine()
        engine.risk_manager = MagicMock()

        # Simulate the reconciler path: position appears in tracker but
        # was never passed through a risk_manager.update_position call.
        position = Position(
            symbol="TEST",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=100.0,
            entry_time=datetime.now(UTC),
            order_id="recovered-123",
            original_size=0.1,
            current_size=0.1,
        )
        engine.live_position_tracker.track_recovered_position(position, db_id=None)

        engine._ensure_positions_registered_with_risk_manager()

        engine.risk_manager.update_position.assert_called_once()
        kwargs = engine.risk_manager.update_position.call_args.kwargs
        assert kwargs["symbol"] == "TEST"
        assert kwargs["side"] == "long"
        assert kwargs["size"] == pytest.approx(0.1)
        assert kwargs["entry_price"] == pytest.approx(100.0)

    def test_sweep_drains_risk_manager_when_current_size_zero(self) -> None:
        """A position whose ``current_size`` was drained to 0.0 by partial
        exits but not yet popped from the tracker must NOT raise in the
        sweep, and must drain the risk-manager slot via ``close_position``.

        Prior to the fix, ``update_position(size=0.0)`` raised the size>0
        validator and the prior ``daily_risk_used`` allocation stayed
        inflated until the close ack popped the position.
        """
        engine = _make_engine()
        engine.risk_manager = MagicMock()

        position = Position(
            symbol="TEST",
            side=PositionSide.LONG,
            size=0.10,
            entry_price=100.0,
            entry_time=datetime.now(UTC),
            order_id="drained-1",
            original_size=0.10,
            current_size=0.0,  # 100% partial-exit drained, close ack pending
        )
        engine.live_position_tracker.track_recovered_position(position, db_id=None)

        engine._ensure_positions_registered_with_risk_manager()

        # Must not call update_position at all — that path raises ValueError
        # for size=0 and would leave daily_risk_used stale.
        engine.risk_manager.update_position.assert_not_called()
        # Must drain the slot via close_position so the next entry sees the
        # right remaining daily-risk budget.
        engine.risk_manager.close_position.assert_called_once_with("TEST")

    def test_sweep_uses_current_size_after_partial_exit(self) -> None:
        """After a partial exit shrinks ``current_size``, the next sweep
        must report the post-exit fraction — passing the original ``size``
        would silently re-inflate ``risk_manager.daily_risk_used`` by the
        full original allocation, undoing
        ``risk_manager.adjust_position_after_partial_exit``.
        """
        engine = _make_engine()
        engine.risk_manager = MagicMock()

        position = Position(
            symbol="TEST",
            side=PositionSide.LONG,
            size=0.10,  # original entry fraction
            entry_price=100.0,
            entry_time=datetime.now(UTC),
            order_id="recovered-123",
            original_size=0.10,
            current_size=0.04,  # 60% partial exit already taken
        )
        engine.live_position_tracker.track_recovered_position(position, db_id=None)

        engine._ensure_positions_registered_with_risk_manager()

        kwargs = engine.risk_manager.update_position.call_args.kwargs
        assert kwargs["size"] == pytest.approx(0.04), (
            "sweep must use current_size, not the original size — otherwise "
            "daily_risk_used re-inflates after partial exits"
        )

    def test_sweep_is_safe_when_no_positions(self) -> None:
        engine = _make_engine()
        engine.risk_manager = MagicMock()
        engine._ensure_positions_registered_with_risk_manager()
        engine.risk_manager.update_position.assert_not_called()

    def test_sweep_no_op_when_risk_manager_missing(self) -> None:
        """Defensive: must not raise when risk_manager is None."""
        engine = _make_engine()
        engine.risk_manager = None
        # Should silently no-op, not crash.
        engine._ensure_positions_registered_with_risk_manager()

    def test_sweep_swallows_per_position_errors(self) -> None:
        """One bad position must not block registration of the others."""
        engine = _make_engine()
        engine.risk_manager = MagicMock()
        engine.risk_manager.update_position.side_effect = [
            RuntimeError("boom"),
            None,
        ]

        for i, sym in enumerate(("BAD", "GOOD")):
            position = Position(
                symbol=sym,
                side=PositionSide.LONG,
                size=0.1,
                entry_price=100.0 + i,
                entry_time=datetime.now(UTC),
                order_id=f"order-{sym}",
                original_size=0.1,
                current_size=0.1,
            )
            engine.live_position_tracker.track_recovered_position(position, db_id=None)

        engine._ensure_positions_registered_with_risk_manager()

        # Both positions attempted, error on first didn't block second.
        assert engine.risk_manager.update_position.call_count == 2


@pytest.mark.fast
class TestProcessCandlePositionsParity:
    """Live's direct ComponentStrategy.process_candle path must pass populated
    current_positions, not ``None``.

    Backtest passes the positions list via RuntimeContext (engine.py:695-720).
    Live's direct path previously hardcoded ``None``, so any strategy
    consulting current_positions saw an empty list in live but populated in
    backtest.
    """

    def test_build_component_positions_returns_tracked_positions(self) -> None:
        engine = _make_engine()
        # Pre-track a position
        position = Position(
            symbol="TEST",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=100.0,
            entry_time=datetime.now(UTC),
            order_id="order-1",
            original_size=0.1,
            current_size=0.1,
            entry_balance=10_000.0,
        )
        engine.live_position_tracker.track_recovered_position(position, db_id=None)

        positions = engine._build_component_positions(current_price=110.0)

        assert len(positions) == 1
        assert positions[0].symbol == "TEST"
        assert positions[0].side == "long"
        assert positions[0].entry_price == pytest.approx(100.0)
        assert positions[0].current_price == pytest.approx(110.0)

    def test_build_component_positions_empty_when_flat(self) -> None:
        engine = _make_engine()
        positions = engine._build_component_positions(current_price=100.0)
        assert positions == []

    def test_runtime_context_uses_same_helper(self) -> None:
        """RuntimeContext path must surface positions via the shared helper.

        Locks in the refactor that both code paths (StrategyRuntime and
        direct ComponentStrategy.process_candle) feed off the same builder.
        """
        engine = _make_engine()
        position = Position(
            symbol="TEST",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=100.0,
            entry_time=datetime.now(UTC),
            order_id="order-1",
            original_size=0.1,
            current_size=0.1,
            entry_balance=10_000.0,
        )
        engine.live_position_tracker.track_recovered_position(position, db_id=None)

        ctx = engine._build_runtime_context(
            balance=10_000.0,
            current_price=110.0,
            current_time=datetime.now(UTC),
        )
        assert ctx.current_positions is not None
        assert len(ctx.current_positions) == 1
        assert ctx.current_positions[0].symbol == "TEST"
