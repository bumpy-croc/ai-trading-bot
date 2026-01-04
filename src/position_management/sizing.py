from __future__ import annotations

import logging
import math
from typing import Literal

logger = logging.getLogger(__name__)


def normalize_position_size(
    strategy_returned_size: float,
    balance: float,
    mode: Literal["fraction", "notional"] = "fraction",
) -> float:
    """Normalize a strategy-returned size into a fraction of balance.

    - fraction: size is already fraction (0..1)
    - notional: size is dollar notional; convert to fraction by dividing by balance
    """
    # Epsilon threshold to prevent division by near-zero balances
    EPSILON = 1e-8

    # Validate inputs are finite to prevent NaN/Infinity propagation
    if not math.isfinite(balance) or not math.isfinite(strategy_returned_size):
        logger.warning(
            f"Invalid input to normalize_position_size: balance={balance}, size={strategy_returned_size}"
        )
        return 0.0
    if balance <= EPSILON:
        logger.warning(f"Balance too small in normalize_position_size: {balance}")
        return 0.0
    if strategy_returned_size <= 0:
        return 0.0
    if mode == "notional":
        return max(0.0, min(1.0, strategy_returned_size / balance))
    # fraction
    return max(0.0, min(1.0, strategy_returned_size))
