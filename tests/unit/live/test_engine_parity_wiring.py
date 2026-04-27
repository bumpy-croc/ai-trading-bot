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
