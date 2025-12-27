"""Shared exit condition checking for backtest and live trading engines.

This module provides unified logic for checking stop loss, take profit,
and calculating P&L percentages consistently across both engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.engines.shared.models import BasePosition

from src.engines.shared.models import PositionSide


@dataclass
class ExitConditionResult:
    """Result of checking an exit condition.

    Attributes:
        triggered: Whether the exit condition was triggered.
        exit_price: Recommended exit price (for SL/TP).
    """

    triggered: bool
    exit_price: float | None = None


def check_stop_loss(
    position: BasePosition,
    current_price: float,
    candle_high: float | None = None,
    candle_low: float | None = None,
    use_high_low: bool = True,
) -> ExitConditionResult:
    """Check if stop loss should be triggered.

    Uses candle high/low for realistic worst-case execution detection
    when available and enabled.

    For long positions, checks if candle_low breached the stop level.
    For short positions, checks if candle_high breached the stop level.

    Args:
        position: Position to check.
        current_price: Current (close) price.
        candle_high: Candle high price (optional).
        candle_low: Candle low price (optional).
        use_high_low: Whether to use high/low for detection.

    Returns:
        ExitConditionResult with trigger status and exit price.
    """
    if position.stop_loss is None:
        return ExitConditionResult(triggered=False)

    stop_loss_val = float(position.stop_loss)
    is_long = position.side == PositionSide.LONG or (
        hasattr(position.side, "value") and position.side.value == "long"
    )

    # Use high/low for more realistic detection
    if use_high_low and candle_low is not None and candle_high is not None:
        if is_long:
            # For long SL, check if candle_low breached the stop
            triggered = candle_low <= stop_loss_val
            if triggered:
                # Use max(stop_loss, candle_low) for realistic worst-case execution
                exit_price = max(stop_loss_val, candle_low)
            else:
                exit_price = None
        else:
            # For short SL, check if candle_high breached the stop
            triggered = candle_high >= stop_loss_val
            if triggered:
                # Use min(stop_loss, candle_high) for realistic worst-case execution
                exit_price = min(stop_loss_val, candle_high)
            else:
                exit_price = None
    else:
        # Fallback to close price only
        if is_long:
            triggered = current_price <= stop_loss_val
        else:
            triggered = current_price >= stop_loss_val
        exit_price = stop_loss_val if triggered else None

    return ExitConditionResult(triggered=triggered, exit_price=exit_price)


def check_take_profit(
    position: BasePosition,
    current_price: float,
    candle_high: float | None = None,
    candle_low: float | None = None,
    use_high_low: bool = True,
) -> ExitConditionResult:
    """Check if take profit should be triggered.

    Uses candle high/low for realistic detection when available and enabled.

    For long positions, checks if candle_high reached the take profit level.
    For short positions, checks if candle_low reached the take profit level.

    Args:
        position: Position to check.
        current_price: Current (close) price.
        candle_high: Candle high price (optional).
        candle_low: Candle low price (optional).
        use_high_low: Whether to use high/low for detection.

    Returns:
        ExitConditionResult with trigger status and exit price.
    """
    if position.take_profit is None:
        return ExitConditionResult(triggered=False)

    take_profit_val = float(position.take_profit)
    is_long = position.side == PositionSide.LONG or (
        hasattr(position.side, "value") and position.side.value == "long"
    )

    # Use high/low for more realistic detection
    if use_high_low and candle_high is not None and candle_low is not None:
        if is_long:
            triggered = candle_high >= take_profit_val
        else:
            triggered = candle_low <= take_profit_val
        exit_price = take_profit_val if triggered else None
    else:
        # Fallback to close price only
        if is_long:
            triggered = current_price >= take_profit_val
        else:
            triggered = current_price <= take_profit_val
        exit_price = take_profit_val if triggered else None

    return ExitConditionResult(triggered=triggered, exit_price=exit_price)


def calculate_pnl_percent(
    entry_price: float,
    current_price: float,
    side: PositionSide | str,
) -> float:
    """Calculate unrealized P&L as a percentage.

    This provides a unified P&L calculation that works consistently
    across both backtest and live trading engines.

    Args:
        entry_price: Entry price of the position.
        current_price: Current market price.
        side: Position side (LONG or SHORT).

    Returns:
        P&L as a decimal percentage (e.g., 0.05 = +5%).
    """
    if entry_price <= 0:
        return 0.0

    pnl_pct = (current_price - entry_price) / entry_price

    # Determine if position is long
    is_long = side == PositionSide.LONG or (
        isinstance(side, str) and side.lower() == "long"
    ) or (hasattr(side, "value") and side.value == "long")

    # Invert for short positions
    if not is_long:
        pnl_pct = -pnl_pct

    return pnl_pct


def calculate_sized_pnl_percent(
    entry_price: float,
    current_price: float,
    side: PositionSide | str,
    position_size: float,
) -> float:
    """Calculate sized P&L percentage (for sized return metrics).

    This multiplies the raw P&L percentage by position size to get
    the return relative to total portfolio.

    Args:
        entry_price: Entry price of the position.
        current_price: Current market price.
        side: Position side (LONG or SHORT).
        position_size: Position size as fraction of balance (0-1).

    Returns:
        Sized P&L as a decimal percentage.
    """
    raw_pnl = calculate_pnl_percent(entry_price, current_price, side)
    return raw_pnl * position_size


__all__ = [
    "ExitConditionResult",
    "check_stop_loss",
    "check_take_profit",
    "calculate_pnl_percent",
    "calculate_sized_pnl_percent",
]
