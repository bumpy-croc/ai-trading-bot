"""Tests for pending entry execution behavior."""

from datetime import datetime
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

from src.engines.backtest.execution.entry_handler import EntryHandler
from src.engines.backtest.execution.execution_engine import ExecutionEngine
from src.engines.shared.execution.execution_decision import ExecutionDecision


def test_pending_entry_no_fill_does_not_execute_trade() -> None:
    """Pending entries should not execute when the execution model declines the fill."""
    execution_engine = ExecutionEngine()
    execution_engine.queue_entry(
        side="long",
        size_fraction=0.1,
        sl_pct=0.05,
        tp_pct=0.04,
        signal_price=100.0,
        signal_time=datetime(2024, 1, 1),
    )
    execution_engine.execute_pending_entry = MagicMock()

    position_tracker = Mock()
    risk_manager = Mock()
    execution_model = Mock()
    execution_model.decide_fill.return_value = ExecutionDecision.no_fill("market price unavailable")

    handler = EntryHandler(
        execution_engine=execution_engine,
        position_tracker=position_tracker,
        risk_manager=risk_manager,
        execution_model=execution_model,
    )

    candle = pd.Series(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
        }
    )

    result = handler.process_pending_entry(
        symbol="BTCUSDT",
        open_price=100.0,
        current_time=datetime(2024, 1, 1, 1),
        balance=10000.0,
        candle=candle,
    )

    assert result.executed is False
    assert result.pending is True
    execution_engine.execute_pending_entry.assert_not_called()
    assert execution_engine.has_pending_entry is True
    position_tracker.open_position.assert_not_called()
    risk_manager.update_position.assert_not_called()


def test_pending_entry_updates_risk_tracking() -> None:
    """Regression for #757: a filled pending (next-bar) entry must register in
    the REAL risk manager. The handler previously passed the PositionSide enum
    to update_position, whose `side in VALID_SIDES` (strings) check raised
    ValueError on every call — swallowed by the except — so daily risk and
    position tracking silently skipped all next-bar entries."""
    from src.risk.risk_manager import RiskManager

    execution_engine = ExecutionEngine()
    execution_engine.queue_entry(
        side="long",
        size_fraction=0.1,
        sl_pct=0.05,
        tp_pct=0.04,
        signal_price=100.0,
        signal_time=datetime(2024, 1, 1),
    )

    position_tracker = Mock()
    risk_manager = RiskManager()
    execution_model = Mock()
    execution_model.decide_fill.return_value = ExecutionDecision(
        should_fill=True,
        fill_price=100.0,
        filled_quantity=10.0,
        liquidity="taker",
        reason="test fill",
    )

    handler = EntryHandler(
        execution_engine=execution_engine,
        position_tracker=position_tracker,
        risk_manager=risk_manager,
        execution_model=execution_model,
    )

    candle = pd.Series(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0}
    )

    result = handler.process_pending_entry(
        symbol="BTCUSDT",
        open_price=100.0,
        current_time=datetime(2024, 1, 1, 1),
        balance=10000.0,
        candle=candle,
    )

    assert result.executed is True
    assert "BTCUSDT" in risk_manager.positions
    assert risk_manager.positions["BTCUSDT"]["side"] == "long"
    assert risk_manager.daily_risk_used == pytest.approx(0.1)
