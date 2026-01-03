"""Execution decision output for execution models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LiquidityType = Literal["maker", "taker", "unknown"]

ZERO_QUANTITY = 0.0
UNKNOWN_LIQUIDITY: LiquidityType = "unknown"


@dataclass(frozen=True)
class ExecutionDecision:
    """Represents the fill decision for an order intent.

    Downstream engines should treat fill_price and filled_quantity as
    the simulated execution result when should_fill is True.
    """

    should_fill: bool
    fill_price: float | None
    filled_quantity: float
    liquidity: LiquidityType
    reason: str

    @classmethod
    def no_fill(cls, reason: str) -> ExecutionDecision:
        """Create a no-fill decision with a reason."""
        return cls(
            should_fill=False,
            fill_price=None,
            filled_quantity=ZERO_QUANTITY,
            liquidity=UNKNOWN_LIQUIDITY,
            reason=reason,
        )
