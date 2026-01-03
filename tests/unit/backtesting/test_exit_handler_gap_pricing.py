"""Tests for backtest stop-loss gap pricing behavior."""

from datetime import datetime
from unittest.mock import Mock

import pandas as pd
import pytest

from src.engines.backtest.execution.exit_handler import ExitHandler
from src.engines.backtest.execution.position_tracker import PositionTracker
from src.engines.backtest.models import ActiveTrade
from src.engines.shared.execution.execution_decision import ExecutionDecision
from src.engines.shared.execution.execution_model import ExecutionModel
from src.engines.shared.execution.fill_policy import default_fill_policy
from src.engines.shared.models import PositionSide


def test_long_stop_loss_uses_candle_low_on_gap() -> None:
    """Long stop-loss exits should use the candle low when price gaps through."""
    position_tracker = PositionTracker()
    trade = ActiveTrade(
        symbol="TEST",
        side=PositionSide.LONG,
        entry_price=100.0,
        entry_time=datetime(2024, 1, 1),
        size=0.1,
        stop_loss=95.0,
    )
    position_tracker.open_position(trade)

    exit_handler = ExitHandler(
        execution_engine=Mock(),
        position_tracker=position_tracker,
        risk_manager=Mock(),
        execution_model=ExecutionModel(default_fill_policy()),
        use_high_low_for_stops=True,
    )

    candle = pd.Series(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 94.0,
            "close": 98.0,
            "volume": 1000.0,
        }
    )

    result = exit_handler.check_exit_conditions(
        runtime_decision=None,
        candle=candle,
        current_price=98.0,
        symbol="TEST",
    )

    assert result.is_stop_loss is True
    assert result.exit_reason == "Stop loss"
    assert result.exit_price == pytest.approx(94.0)


def test_short_stop_loss_uses_candle_high_on_gap() -> None:
    """Short stop-loss exits should use the candle high when price gaps through."""
    position_tracker = PositionTracker()
    trade = ActiveTrade(
        symbol="TEST",
        side=PositionSide.SHORT,
        entry_price=100.0,
        entry_time=datetime(2024, 1, 1),
        size=0.1,
        stop_loss=105.0,
    )
    position_tracker.open_position(trade)

    exit_handler = ExitHandler(
        execution_engine=Mock(),
        position_tracker=position_tracker,
        risk_manager=Mock(),
        execution_model=ExecutionModel(default_fill_policy()),
        use_high_low_for_stops=True,
    )

    candle = pd.Series(
        {
            "open": 100.0,
            "high": 110.0,
            "low": 96.0,
            "close": 102.0,
            "volume": 1000.0,
        }
    )

    result = exit_handler.check_exit_conditions(
        runtime_decision=None,
        candle=candle,
        current_price=102.0,
        symbol="TEST",
    )

    assert result.is_stop_loss is True
    assert result.exit_reason == "Stop loss"
    assert result.exit_price == pytest.approx(110.0)


def test_execute_exit_preserves_gap_price_for_stop_loss() -> None:
    """Stop-loss exits should use execution-model gap pricing and keep slippage enabled."""
    position_tracker = PositionTracker()
    trade = ActiveTrade(
        symbol="TEST",
        side=PositionSide.LONG,
        entry_price=100.0,
        entry_time=datetime(2024, 1, 1),
        size=0.1,
        stop_loss=95.0,
    )
    position_tracker.open_position(trade)

    execution_engine = Mock()
    execution_engine.calculate_exit_costs.return_value = (94.0, 0.0, 0.0)
    execution_model = Mock()
    execution_model.decide_fill.return_value = ExecutionDecision(
        should_fill=True,
        fill_price=94.0,
        filled_quantity=trade.size,
        liquidity="taker",
        reason="stop order triggered",
    )

    exit_handler = ExitHandler(
        execution_engine=execution_engine,
        position_tracker=position_tracker,
        risk_manager=Mock(),
        execution_model=execution_model,
        use_high_low_for_stops=True,
    )

    candle = pd.Series(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 94.0,
            "close": 98.0,
            "volume": 1000.0,
        }
    )

    exit_handler.execute_exit(
        exit_price=95.0,
        exit_reason="Stop loss",
        current_time=datetime(2024, 1, 1, 1),
        current_price=98.0,
        balance=10000.0,
        symbol="TEST",
        candle=candle,
    )

    call_kwargs = execution_engine.calculate_exit_costs.call_args.kwargs
    base_price = call_kwargs["base_price"]
    apply_slippage = call_kwargs["apply_slippage"]
    assert base_price == pytest.approx(94.0)
    assert apply_slippage is True
