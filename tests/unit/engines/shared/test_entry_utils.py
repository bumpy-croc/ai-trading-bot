"""Tests for shared entry utilities."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.config.constants import DEFAULT_TAKE_PROFIT_PCT
from src.engines.shared.entry_utils import (
    EntryPlan,
    extract_entry_plan,
    resolve_stop_loss_take_profit_pct,
)
from src.engines.shared.models import PositionSide
from src.strategies.components.signal_generator import Signal, SignalDirection
from src.strategies.components.strategy import TradingDecision


class StubStrategy:
    """Strategy stub providing stop-loss and optional take-profit guidance."""

    def __init__(
        self,
        stop_loss_price: float,
        take_profit_pct: float | None = None,
        raise_error: bool = False,
    ) -> None:
        self.stop_loss_price = stop_loss_price
        self.take_profit_pct = take_profit_pct
        self.raise_error = raise_error

    def get_stop_loss_price(
        self,
        current_price: float,
        signal: object | None,
        regime: object | None,
    ) -> float:
        if self.raise_error:
            raise ValueError("stop loss unavailable")
        return self.stop_loss_price


def _decision(
    direction: SignalDirection,
    position_size: float,
    metadata: dict[str, object] | None = None,
) -> TradingDecision:
    signal = Signal(direction=direction, strength=0.5, confidence=0.5, metadata={})
    return TradingDecision(
        timestamp=datetime.now(UTC),
        signal=signal,
        position_size=position_size,
        regime=None,
        risk_metrics={},
        execution_time_ms=1.0,
        metadata=metadata or {},
    )


def test_extract_entry_plan_hold_returns_none() -> None:
    """Hold signals do not produce entry plans."""
    # Arrange
    decision = _decision(SignalDirection.HOLD, position_size=100.0)

    # Act
    plan = extract_entry_plan(decision, balance=1000.0)

    # Assert
    assert plan is None


def test_extract_entry_plan_sell_requires_short_flag() -> None:
    """Short entries require enter_short metadata."""
    # Arrange
    decision = _decision(SignalDirection.SELL, position_size=100.0)

    # Act
    plan = extract_entry_plan(decision, balance=1000.0)

    # Assert
    assert plan is None


def test_extract_entry_plan_buy_returns_long() -> None:
    """Buy signals map to long entry plans."""
    # Arrange
    decision = _decision(SignalDirection.BUY, position_size=100.0)

    # Act
    plan = extract_entry_plan(decision, balance=1000.0)

    # Assert
    assert isinstance(plan, EntryPlan)
    assert plan.side == PositionSide.LONG
    assert plan.size_fraction == pytest.approx(0.1)


def test_extract_entry_plan_clamps_fraction() -> None:
    """Entry plan size fraction clamps to 1.0."""
    # Arrange
    decision = _decision(SignalDirection.BUY, position_size=2000.0)

    # Act
    plan = extract_entry_plan(decision, balance=1000.0)

    # Assert
    assert plan is not None
    assert plan.size_fraction == pytest.approx(1.0)


def test_resolve_stop_loss_take_profit_pct_uses_strategy_values() -> None:
    """Strategy stop loss and take profit override defaults when enabled."""
    # Arrange
    decision = _decision(SignalDirection.BUY, position_size=100.0)
    strategy = StubStrategy(stop_loss_price=95.0, take_profit_pct=0.04)

    # Act
    sl_pct, tp_pct = resolve_stop_loss_take_profit_pct(
        current_price=100.0,
        entry_side=PositionSide.LONG,
        runtime_decision=decision,
        component_strategy=strategy,
        default_take_profit_pct=None,
        use_strategy_take_profit=True,
    )

    # Assert
    assert sl_pct == pytest.approx(0.05)
    assert tp_pct == pytest.approx(0.04)


def test_resolve_stop_loss_take_profit_pct_falls_back_on_error() -> None:
    """Stop loss errors fall back to default percentages."""
    # Arrange
    decision = _decision(SignalDirection.BUY, position_size=100.0)
    strategy = StubStrategy(stop_loss_price=0.0, raise_error=True)

    # Act
    sl_pct, tp_pct = resolve_stop_loss_take_profit_pct(
        current_price=100.0,
        entry_side=PositionSide.LONG,
        runtime_decision=decision,
        component_strategy=strategy,
        default_stop_loss_pct=0.03,
        default_take_profit_pct=None,
        use_strategy_take_profit=False,
    )

    # Assert
    assert sl_pct == pytest.approx(0.03)
    assert tp_pct == pytest.approx(DEFAULT_TAKE_PROFIT_PCT)
