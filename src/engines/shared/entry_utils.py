"""Shared entry utilities for backtest and live trading engines.

This module provides shared helpers for extracting entry plans and resolving
stop-loss/take-profit percentages to keep entry logic consistent across engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from src.config.constants import (
    DEFAULT_TAKE_PROFIT_PCT,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_MIN_STOP_LOSS_PCT,
    DEFAULT_MAX_STOP_LOSS_PCT,
)
from src.engines.shared.models import PositionSide
from src.engines.shared.side_utils import to_side_string
from src.strategies.components import SignalDirection
from src.utils.bounds import clamp_fraction, clamp_stop_loss_pct


class SignalLike(Protocol):
    """Protocol for signal-like objects used in entry decisions."""

    direction: SignalDirection


class EntryDecisionLike(Protocol):
    """Protocol for entry decision objects used by entry handlers."""

    signal: SignalLike
    position_size: float
    metadata: Mapping[str, object] | None
    regime: object | None


class StopLossStrategyLike(Protocol):
    """Protocol for strategies that provide stop-loss and take-profit guidance."""

    take_profit_pct: float | None

    def get_stop_loss_price(
        self,
        current_price: float,
        signal: object | None,
        regime: object | None,
    ) -> float:
        """Return stop-loss price for the current decision."""


@dataclass(frozen=True)
class EntryPlan:
    """Normalized entry plan containing side and size fraction."""

    side: PositionSide
    size_fraction: float

    @property
    def side_str(self) -> str:
        """Return position side as a lowercase string."""
        return to_side_string(self.side)


def extract_entry_plan(
    decision: EntryDecisionLike | None,
    balance: float,
) -> EntryPlan | None:
    """Extract entry side and size fraction from a runtime decision.

    Args:
        decision: Runtime decision from strategy.
        balance: Current account balance.

    Returns:
        EntryPlan with side and size fraction, or None if no entry is allowed.
    """
    if decision is None or balance <= 0:
        return None

    if decision.signal.direction == SignalDirection.HOLD or decision.position_size <= 0:
        return None

    metadata = getattr(decision, "metadata", None) or {}
    if decision.signal.direction == SignalDirection.SELL and not bool(
        metadata.get("enter_short")
    ):
        return None

    side = (
        PositionSide.LONG
        if decision.signal.direction == SignalDirection.BUY
        else PositionSide.SHORT
    )
    size_fraction = float(decision.position_size) / float(balance)
    size_fraction = clamp_fraction(size_fraction)

    if size_fraction <= 0:
        return None

    return EntryPlan(side=side, size_fraction=size_fraction)


def resolve_stop_loss_take_profit_pct(
    current_price: float,
    entry_side: PositionSide,
    runtime_decision: EntryDecisionLike | None,
    component_strategy: StopLossStrategyLike | None,
    *,
    default_stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    default_take_profit_pct: float | None = None,
    min_stop_loss_pct: float = DEFAULT_MIN_STOP_LOSS_PCT,
    max_stop_loss_pct: float = DEFAULT_MAX_STOP_LOSS_PCT,
    use_strategy_take_profit: bool = False,
    stop_loss_exceptions: tuple[type[Exception], ...] = (
        AttributeError,
        ValueError,
        TypeError,
        ZeroDivisionError,
    ),
) -> tuple[float, float]:
    """Resolve stop-loss and take-profit percentages for an entry.

    Args:
        current_price: Current market price.
        entry_side: Entry side (LONG or SHORT).
        runtime_decision: Runtime decision for signal/regime context.
        component_strategy: Strategy providing stop-loss and take-profit guidance.
        default_stop_loss_pct: Default stop-loss percent if strategy is unavailable.
        default_take_profit_pct: Optional take-profit percent override.
        min_stop_loss_pct: Minimum stop-loss percent.
        max_stop_loss_pct: Maximum stop-loss percent.
        use_strategy_take_profit: Whether to use strategy take-profit when default is None.
        stop_loss_exceptions: Exception types to ignore when calculating stop loss.

    Returns:
        Tuple of (stop_loss_pct, take_profit_pct).
    """
    sl_pct = default_stop_loss_pct
    if component_strategy is not None:
        try:
            signal = runtime_decision.signal if runtime_decision else None
            regime = runtime_decision.regime if runtime_decision else None
            stop_loss_price = component_strategy.get_stop_loss_price(
                current_price, signal, regime
            )
            sl_pct = _calculate_stop_loss_pct(current_price, stop_loss_price, entry_side)
            sl_pct = clamp_stop_loss_pct(
                sl_pct,
                min_pct=min_stop_loss_pct,
                max_pct=max_stop_loss_pct,
            )
        except stop_loss_exceptions:
            pass

    tp_pct = default_take_profit_pct
    if use_strategy_take_profit and tp_pct is None and component_strategy is not None:
        tp_pct = getattr(component_strategy, "take_profit_pct", None)
    if tp_pct is None:
        tp_pct = DEFAULT_TAKE_PROFIT_PCT

    return sl_pct, tp_pct


def _calculate_stop_loss_pct(
    current_price: float,
    stop_loss_price: float,
    entry_side: PositionSide,
) -> float:
    """Calculate stop-loss percentage from a stop-loss price."""
    if entry_side == PositionSide.LONG:
        return (current_price - stop_loss_price) / current_price
    return (stop_loss_price - current_price) / current_price


__all__ = [
    "EntryPlan",
    "extract_entry_plan",
    "resolve_stop_loss_take_profit_pct",
]
