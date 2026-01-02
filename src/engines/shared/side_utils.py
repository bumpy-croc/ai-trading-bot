"""Utility functions for consistent side handling across engines.

This module provides helper functions for converting between different
representations of position sides (enum, string) and checking side properties.
These utilities ensure consistent side handling across backtest and live engines.

Usage:
    from src.engines.shared.side_utils import to_side_string, to_position_side, is_long

    # Convert any side representation to string
    side_str = to_side_string(position.side)  # "long" or "short"

    # Convert any side representation to PositionSide enum
    side_enum = to_position_side(side_str)  # PositionSide.LONG or PositionSide.SHORT

    # Check if side is long/short
    if is_long(position.side):
        ...
"""

from __future__ import annotations

from src.engines.shared.models import PositionSide

# Type alias for any acceptable side representation
SideType = PositionSide | str


def to_side_string(side: SideType) -> str:
    """Convert any side representation to lowercase string.

    Handles PositionSide enum, strings (case-insensitive), and objects with .value attribute.

    Args:
        side: Side in any format (PositionSide, str, or object with .value)

    Returns:
        "long" or "short"

    Raises:
        ValueError: If side cannot be converted to a valid side string

    Examples:
        >>> to_side_string(PositionSide.LONG)
        'long'
        >>> to_side_string("LONG")
        'long'
        >>> to_side_string("short")
        'short'
    """
    # Handle PositionSide enum
    if isinstance(side, PositionSide):
        return side.value

    # Handle objects with .value attribute (other enums)
    if hasattr(side, "value"):
        side = side.value

    # Handle string
    if isinstance(side, str):
        side_lower = side.lower()
        if side_lower in ("long", "short"):
            return side_lower
        # Map order sides (BUY/SELL) to position sides (long/short)
        # BUY typically opens a long, SELL typically opens a short
        if side_lower == "buy":
            return "long"
        if side_lower == "sell":
            return "short"
        raise ValueError(
            f"Invalid side string: '{side}'. Expected 'long', 'short', 'buy', or 'sell'."
        )

    raise ValueError(f"Cannot convert {type(side).__name__} to side string: {side}")


def to_position_side(side: SideType) -> PositionSide:
    """Convert any side representation to PositionSide enum.

    Args:
        side: Side in any format (PositionSide, str, or object with .value)

    Returns:
        PositionSide.LONG or PositionSide.SHORT

    Raises:
        ValueError: If side cannot be converted to PositionSide

    Examples:
        >>> to_position_side("long")
        <PositionSide.LONG: 'long'>
        >>> to_position_side(PositionSide.SHORT)
        <PositionSide.SHORT: 'short'>
    """
    # Already a PositionSide
    if isinstance(side, PositionSide):
        return side

    # Convert to string first, then to enum
    side_str = to_side_string(side)
    return PositionSide.LONG if side_str == "long" else PositionSide.SHORT


def is_long(side: SideType) -> bool:
    """Check if side represents a long position.

    Args:
        side: Side in any format

    Returns:
        True if long, False if short

    Examples:
        >>> is_long(PositionSide.LONG)
        True
        >>> is_long("short")
        False
    """
    return to_side_string(side) == "long"


def is_short(side: SideType) -> bool:
    """Check if side represents a short position.

    Args:
        side: Side in any format

    Returns:
        True if short, False if long

    Examples:
        >>> is_short(PositionSide.SHORT)
        True
        >>> is_short("long")
        False
    """
    return to_side_string(side) == "short"


def opposite_side(side: SideType) -> PositionSide:
    """Get the opposite side.

    Args:
        side: Side in any format

    Returns:
        PositionSide.SHORT if input is long, PositionSide.LONG if input is short

    Examples:
        >>> opposite_side(PositionSide.LONG)
        <PositionSide.SHORT: 'short'>
        >>> opposite_side("short")
        <PositionSide.LONG: 'long'>
    """
    return PositionSide.SHORT if is_long(side) else PositionSide.LONG


def opposite_side_string(side: SideType) -> str:
    """Get the opposite side as a string.

    Args:
        side: Side in any format

    Returns:
        "short" if input is long, "long" if input is short

    Examples:
        >>> opposite_side_string(PositionSide.LONG)
        'short'
        >>> opposite_side_string("short")
        'long'
    """
    return "short" if is_long(side) else "long"


def get_position_side(position: object, default: str = "long") -> str:
    """Extract side from a position-like object as a lowercase string.

    Safely extracts the 'side' attribute from any position object and converts
    it to a lowercase string. Returns a default value if the position is None
    or has no side attribute.

    Args:
        position: Any object with a 'side' attribute (or None).
        default: Default side to return if position is None or has no side.

    Returns:
        Side as lowercase string ('long' or 'short').

    Examples:
        >>> get_position_side(trade)  # trade.side = PositionSide.LONG
        'long'
        >>> get_position_side(None)
        'long'
        >>> get_position_side(position_without_side)
        'long'
    """
    if position is None:
        return default

    side = getattr(position, "side", None)
    if side is None:
        return default

    return to_side_string(side)
