from __future__ import annotations

from typing import Literal


def normalize_position_size(
    strategy_returned_size: float,
    balance: float,
    mode: Literal["fraction", "notional"] = "fraction",
) -> float:
    """Normalize a strategy-returned size into a fraction of balance.

    - fraction: size is already fraction (0..1)
    - notional: size is dollar notion; convert to fraction by dividing by balance
    """
    if balance <= 0:
        return 0.0
    if strategy_returned_size <= 0:
        return 0.0
    if mode == "notional":
        return max(0.0, min(1.0, strategy_returned_size / balance))
    # fraction
    return max(0.0, min(1.0, strategy_returned_size))
