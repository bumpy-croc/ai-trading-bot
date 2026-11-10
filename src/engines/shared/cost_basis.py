"""Quantity-weighted cost basis for positions built from several fills.

Position sizes are cost notional divided by the entry balance, so a slice of
size ``f`` bought at price ``p`` holds ``f / p`` units per unit of balance. The
average cost per unit of a position made of such slices is therefore the
harmonic, notional-weighted mean of the slice prices. Measuring a pro-rata exit
from that average gives the same P&L as exiting each slice from its own cost.
"""

from __future__ import annotations

import math


def blend_cost_basis(
    held_cost_price: float,
    held_size: float,
    added_price: float,
    added_size: float,
) -> float:
    """Return the average cost per unit after adding a slice to a position.

    Args:
        held_cost_price: Average cost per unit of the position before the add.
        held_size: Remaining cost notional of the position, in balance-fraction
            units (``current_size``).
        added_price: Fill price of the added slice.
        added_size: Cost notional of the added slice, in the same units.

    Returns:
        The blended average cost per unit.

    Raises:
        ValueError: If a price is not positive and finite, or a size is negative
            or not finite.
    """
    for name, price in (("held_cost_price", held_cost_price), ("added_price", added_price)):
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"{name} must be positive and finite, got {price}")
    for name, size in (("held_size", held_size), ("added_size", added_size)):
        if not math.isfinite(size) or size < 0:
            raise ValueError(f"{name} must be non-negative and finite, got {size}")

    if added_size == 0:
        return held_cost_price
    if held_size == 0:
        return added_price
    return (held_size + added_size) / (held_size / held_cost_price + added_size / added_price)
