"""Shared utilities for building MarketSnapshots and mapping order sides.

This module consolidates duplicated logic across entry and exit handlers
to ensure consistent snapshot construction and order side mapping.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd

from src.data_providers.exchange_interface import OrderSide
from src.engines.shared.execution.market_snapshot import MarketSnapshot

if TYPE_CHECKING:
    from src.engines.backtest.models import ActiveTrade
    from src.engines.live.execution.position_tracker import LivePosition, PositionSide

DEFAULT_VOLUME = 0.0


def coerce_float(value: object, fallback: float) -> float:
    """Coerce a value to float or return fallback on failure.

    Returns the fallback for None, non-coercible values, NaN, or infinity
    to prevent invalid data from corrupting position state.

    Args:
        value: Value to coerce (may be None, int, float, str, etc.)
        fallback: Value to return if coercion fails or value is non-finite.

    Returns:
        The coerced float value, or fallback if coercion fails or value is NaN/infinity.
    """
    if value is None:
        return fallback
    try:
        result = float(value)
        # Return fallback for NaN/infinity to prevent state corruption
        if not math.isfinite(result):
            return fallback
        return result
    except (TypeError, ValueError):
        return fallback


def build_snapshot_from_price(
    symbol: str,
    current_price: float,
    volume: float = DEFAULT_VOLUME,
) -> MarketSnapshot:
    """Build a MarketSnapshot using only the current price.

    Used when only a single price point is available (e.g., live entry signals).
    All OHLC values are set to the current price.

    Args:
        symbol: Trading symbol.
        current_price: Current market price.
        volume: Trading volume (defaults to 0).

    Returns:
        MarketSnapshot with current_price for all price fields.
    """
    return MarketSnapshot(
        symbol=symbol,
        timestamp=datetime.now(UTC),
        last_price=current_price,
        high=current_price,
        low=current_price,
        close=current_price,
        volume=volume,
    )


def build_snapshot_from_ohlc(
    symbol: str,
    current_price: float,
    candle_high: float | None,
    candle_low: float | None,
    volume: float = DEFAULT_VOLUME,
) -> MarketSnapshot:
    """Build a MarketSnapshot from explicit high/low values.

    Used when high/low extremes are provided separately (e.g., live exit checks).
    Uses coerce_float for consistency with build_snapshot_from_candle, ensuring
    NaN/infinity values fall back to current_price.

    Args:
        symbol: Trading symbol.
        current_price: Current market price (used as close).
        candle_high: Candle high price (defaults to current_price if None/NaN/inf).
        candle_low: Candle low price (defaults to current_price if None/NaN/inf).
        volume: Trading volume (defaults to 0).

    Returns:
        MarketSnapshot with the provided OHLC values.
    """
    high = coerce_float(candle_high, current_price)
    low = coerce_float(candle_low, current_price)
    return MarketSnapshot(
        symbol=symbol,
        timestamp=datetime.now(UTC),
        last_price=current_price,
        high=high,
        low=low,
        close=current_price,
        volume=volume,
    )


def build_snapshot_from_candle(
    symbol: str,
    current_time: datetime,
    current_price: float,
    candle: pd.Series | None,
    default_volume: float = DEFAULT_VOLUME,
) -> MarketSnapshot:
    """Build a MarketSnapshot from a pandas Series candle.

    Used in backtesting where candle data is available as a pandas Series.

    Args:
        symbol: Trading symbol.
        current_time: Current simulation time.
        current_price: Current price (used as fallback for missing values).
        candle: Pandas Series with OHLCV data, or None.
        default_volume: Default volume if not available in candle.

    Returns:
        MarketSnapshot populated from the candle data.
    """
    if candle is not None and hasattr(candle, "get"):
        high = coerce_float(candle.get("high"), current_price)
        low = coerce_float(candle.get("low"), current_price)
        close = coerce_float(candle.get("close"), current_price)
        volume = coerce_float(candle.get("volume"), default_volume)
    else:
        high = current_price
        low = current_price
        close = current_price
        volume = default_volume

    return MarketSnapshot(
        symbol=symbol,
        timestamp=current_time,
        last_price=current_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def map_entry_order_side_from_string(side: str) -> OrderSide:
    """Map a position side string to an entry order side.

    Args:
        side: Position side as string ('long' or 'short').

    Returns:
        OrderSide.BUY for long, OrderSide.SELL for short.

    Raises:
        ValueError: If the side string is not recognized.
    """
    side_lower = side.lower()
    if side_lower == "long":
        return OrderSide.BUY
    if side_lower == "short":
        return OrderSide.SELL
    raise ValueError(f"Unsupported entry side: {side}")


def map_entry_order_side_from_enum(side: PositionSide) -> OrderSide:
    """Map a PositionSide enum to an entry order side.

    Args:
        side: Position side enum.

    Returns:
        OrderSide.BUY for LONG, OrderSide.SELL for SHORT.

    Raises:
        ValueError: If the side is not recognized.
    """
    # Import here to avoid circular imports
    from src.engines.live.execution.position_tracker import PositionSide

    if side == PositionSide.LONG:
        return OrderSide.BUY
    if side == PositionSide.SHORT:
        return OrderSide.SELL
    raise ValueError(f"Unsupported entry side: {side}")


def map_exit_order_side_from_position(position: LivePosition) -> OrderSide:
    """Map a live position's side to an exit order side.

    Args:
        position: Live position to exit.

    Returns:
        OrderSide.SELL for LONG positions, OrderSide.BUY for SHORT.

    Raises:
        ValueError: If the position side is not recognized.
    """
    # Import here to avoid circular imports
    from src.engines.live.execution.position_tracker import PositionSide

    if position.side == PositionSide.LONG:
        return OrderSide.SELL
    if position.side == PositionSide.SHORT:
        return OrderSide.BUY
    raise ValueError(f"Unsupported position side: {position.side}")


def map_exit_order_side_from_trade(trade: ActiveTrade) -> OrderSide:
    """Map a backtest trade's side to an exit order side.

    Args:
        trade: Active trade to exit.

    Returns:
        OrderSide.SELL for long trades, OrderSide.BUY for short.

    Raises:
        ValueError: If the trade side is not recognized.
    """
    side_str = trade.side.value if hasattr(trade.side, "value") else str(trade.side)
    if side_str == "long":
        return OrderSide.SELL
    if side_str == "short":
        return OrderSide.BUY
    raise ValueError(f"Unsupported trade side: {side_str}")
