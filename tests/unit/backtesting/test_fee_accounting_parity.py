"""Parity tests for backtest fee accounting against live's record_trade contract.

Two parity invariants are locked in here:

1. **PerformanceTracker.record_trade(fee=...) sums entry + exit fees.**
   Backtest used to pass only ``exit_fee`` to ``record_trade``, while live
   passes ``total_fee = entry_fee + exit_fee`` (plus interest_cost). That
   silently undercounted ``total_fees_paid`` in backtest reports by ~50%
   for any strategy with non-zero fee_rate. The fix stashes ``entry_fee``
   on ``ActiveTrade.metadata`` at entry, copies it onto the completed
   ``Trade`` at close, and the engine reads it back when calling
   ``record_trade``. Live already does the same via
   ``position.metadata["entry_fee"]``.

2. **log_trade persists margin_interest_cost.** Live writes the
   ``margin_interest_cost`` column to the trades table; backtest used to
   drop it from the audit trail entirely. The fix stashes the computed
   interest on the trade metadata and reads it back in
   ``EventLogger.log_completed_trade``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from src.database.models import TradeSource
from src.engines.backtest.execution.execution_engine import ExecutionEngine
from src.engines.backtest.execution.exit_handler import ExitHandler
from src.engines.backtest.execution.position_tracker import PositionTracker
from src.engines.backtest.logging.event_logger import EventLogger
from src.engines.backtest.models import ActiveTrade, Trade
from src.engines.shared.execution.execution_model import ExecutionModel
from src.engines.shared.execution.fill_policy import default_fill_policy
from src.engines.shared.models import PositionSide


@pytest.mark.fast
class TestEntryFeeOnMetadata:
    """ExecutionEngine must stash entry_fee/entry_slippage on the new trade.

    Without this, the exit path can't reconstruct total fees, so the
    PerformanceTracker.record_trade fee sum stays ~50% short.
    """

    def test_immediate_entry_records_entry_fee_metadata(self) -> None:
        engine = ExecutionEngine(fee_rate=0.001, slippage_rate=0.0)
        result = engine.execute_immediate_entry(
            symbol="TEST",
            side="long",
            size_fraction=0.1,
            base_price=100.0,
            current_time=datetime(2024, 1, 1, tzinfo=UTC),
            balance=10_000.0,
            stop_loss=95.0,
            take_profit=110.0,
        )

        assert result.executed is True
        assert result.trade is not None
        assert result.trade.metadata["entry_fee"] == pytest.approx(result.entry_fee)
        assert "entry_slippage_cost" in result.trade.metadata

    def test_pending_entry_records_entry_fee_metadata(self) -> None:
        engine = ExecutionEngine(fee_rate=0.001, slippage_rate=0.0005)
        engine.queue_entry(
            side="long",
            size_fraction=0.1,
            sl_pct=0.05,
            tp_pct=0.10,
            signal_price=100.0,
            signal_time=datetime(2024, 1, 1, tzinfo=UTC),
        )
        result = engine.execute_pending_entry(
            symbol="TEST",
            open_price=100.5,
            current_time=datetime(2024, 1, 1, 1, tzinfo=UTC),
            balance=10_000.0,
        )

        assert result.executed is True
        assert result.trade is not None
        assert result.trade.metadata["entry_fee"] == pytest.approx(result.entry_fee)
        assert result.trade.metadata["entry_slippage_cost"] == pytest.approx(result.slippage_cost)


@pytest.mark.fast
class TestPositionTrackerCarriesEntryMetadata:
    """``close_position`` must copy entry_fee/entry_slippage onto completed Trade.

    Otherwise the engine has no way to reconstruct total fees at exit.
    """

    def test_completed_trade_inherits_entry_fee_metadata(self) -> None:
        tracker = PositionTracker()
        active = ActiveTrade(
            symbol="TEST",
            side=PositionSide.LONG,
            entry_price=100.0,
            entry_time=datetime(2024, 1, 1, tzinfo=UTC),
            size=0.1,
            entry_balance=10_000.0,
        )
        active.metadata["entry_fee"] = 1.0
        active.metadata["entry_slippage_cost"] = 0.5
        tracker.open_position(active)

        result = tracker.close_position(
            exit_price=110.0,
            exit_time=datetime(2024, 1, 1, 1, tzinfo=UTC),
            exit_reason="test",
            basis_balance=10_000.0,
        )

        assert result.trade.metadata.get("entry_fee") == pytest.approx(1.0)
        assert result.trade.metadata.get("entry_slippage_cost") == pytest.approx(0.5)


@pytest.mark.fast
class TestExitHandlerStashesMarginInterestCost:
    """When margin interest accrues, ``execute_exit`` must surface the cost on
    the completed trade so ``EventLogger`` can persist it to the DB."""

    def _build_handler(self, rate: float) -> tuple[ExitHandler, PositionTracker]:
        position_tracker = PositionTracker()
        execution_engine = ExecutionEngine(fee_rate=0.0, slippage_rate=0.0)
        risk_manager = Mock()
        risk_manager.close_position = Mock()
        handler = ExitHandler(
            execution_engine=execution_engine,
            position_tracker=position_tracker,
            risk_manager=risk_manager,
            execution_model=ExecutionModel(default_fill_policy()),
            annual_margin_interest_rate=rate,
        )
        return handler, position_tracker

    def test_interest_recorded_on_trade_metadata_when_rate_set(self) -> None:
        handler, tracker = self._build_handler(rate=0.10)
        entry = datetime(2024, 1, 1, tzinfo=UTC)
        exit_t = entry + timedelta(hours=24)
        tracker.open_position(
            ActiveTrade(
                symbol="TEST",
                side=PositionSide.LONG,
                entry_price=100.0,
                entry_time=entry,
                size=0.5,
                entry_balance=10_000.0,
            )
        )

        completed_trade, _net_pnl, _fee, _slip = handler.execute_exit(
            exit_price=100.0,
            exit_reason="test",
            current_time=exit_t,
            current_price=100.0,
            balance=10_000.0,
            symbol="TEST",
        )

        assert "margin_interest_cost" in completed_trade.metadata
        assert completed_trade.metadata["margin_interest_cost"] > 0

    def test_no_interest_metadata_when_rate_zero(self) -> None:
        handler, tracker = self._build_handler(rate=0.0)
        entry = datetime(2024, 1, 1, tzinfo=UTC)
        exit_t = entry + timedelta(hours=24)
        tracker.open_position(
            ActiveTrade(
                symbol="TEST",
                side=PositionSide.LONG,
                entry_price=100.0,
                entry_time=entry,
                size=0.5,
                entry_balance=10_000.0,
            )
        )

        completed_trade, _net_pnl, _fee, _slip = handler.execute_exit(
            exit_price=100.0,
            exit_reason="test",
            current_time=exit_t,
            current_price=100.0,
            balance=10_000.0,
            symbol="TEST",
        )

        assert "margin_interest_cost" not in completed_trade.metadata


@pytest.mark.fast
class TestEventLoggerPersistsMarginInterest:
    """``EventLogger.log_completed_trade`` must forward margin_interest_cost
    to ``db_manager.log_trade`` when present on the trade metadata."""

    def _make_trade(self, with_interest: bool) -> Trade:
        trade = Trade(
            symbol="TEST",
            side=PositionSide.LONG,
            entry_price=100.0,
            exit_price=110.0,
            entry_time=datetime(2024, 1, 1, tzinfo=UTC),
            exit_time=datetime(2024, 1, 1, 1, tzinfo=UTC),
            size=0.1,
            pnl=10.0,
            pnl_percent=0.01,
            exit_reason="test",
        )
        if with_interest:
            trade.metadata["margin_interest_cost"] = 1.23
        return trade

    def test_persists_margin_interest_when_present(self) -> None:
        db_manager = Mock()
        db_manager.log_trade = Mock(return_value=1)
        logger = EventLogger(db_manager=db_manager, log_to_database=True, session_id=42)

        logger.log_completed_trade(
            trade=self._make_trade(with_interest=True),
            symbol="TEST",
            strategy_name="test_strat",
            source=TradeSource.BACKTEST,
        )

        kwargs = db_manager.log_trade.call_args.kwargs
        assert kwargs["margin_interest_cost"] == pytest.approx(1.23)

    def test_passes_none_when_no_interest_recorded(self) -> None:
        db_manager = Mock()
        db_manager.log_trade = Mock(return_value=1)
        logger = EventLogger(db_manager=db_manager, log_to_database=True, session_id=42)

        logger.log_completed_trade(
            trade=self._make_trade(with_interest=False),
            symbol="TEST",
            strategy_name="test_strat",
            source=TradeSource.BACKTEST,
        )

        kwargs = db_manager.log_trade.call_args.kwargs
        assert kwargs["margin_interest_cost"] is None
