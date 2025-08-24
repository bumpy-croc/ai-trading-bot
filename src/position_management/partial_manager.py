from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple


@dataclass
class PositionState:
    """Lightweight state used by PartialExitPolicy.

    This is intentionally decoupled from DB/engine models to enable reuse in
    both live and backtesting without import cycles.
    """

    entry_price: float
    side: str  # 'long' | 'short'
    original_size: float  # fraction of balance at entry
    current_size: float   # fraction of balance remaining
    partial_exits_taken: int = 0
    scale_ins_taken: int = 0
    last_partial_exit_price: Optional[float] = None
    last_scale_in_price: Optional[float] = None


@dataclass
class PartialExitPolicy:
    """Policy for partial exits (scale-out) and scale-ins.

    Targets and thresholds are expressed as decimal returns from entry, e.g., 0.03 = +3 %.
    Sizes are expressed as fractions of the original position size.
    """

    exit_targets: List[float]
    exit_sizes: List[float]
    scale_in_thresholds: List[float] = field(default_factory=list)
    scale_in_sizes: List[float] = field(default_factory=list)
    max_scale_ins: int = 0

    def __post_init__(self):
        if len(self.exit_targets) != len(self.exit_sizes):
            raise ValueError("exit_targets and exit_sizes must have equal length")
        if any(t <= 0 for t in self.exit_targets):
            raise ValueError("exit_targets must be positive")
        if any(s <= 0 or s > 1 for s in self.exit_sizes):
            raise ValueError("exit_sizes must be in (0, 1]")
        if len(self.scale_in_thresholds) != len(self.scale_in_sizes):
            raise ValueError("scale_in_thresholds and scale_in_sizes must have equal length")
        if any(t <= 0 for t in self.scale_in_thresholds):
            raise ValueError("scale_in_thresholds must be positive")
        if any(s <= 0 or s > 1 for s in self.scale_in_sizes):
            raise ValueError("scale_in_sizes must be in (0, 1]")
        if self.max_scale_ins < 0:
            raise ValueError("max_scale_ins must be >= 0")

    def _pnl_pct(self, position: PositionState, current_price: float) -> float:
        if position.entry_price <= 0:
            return 0.0
        if position.side == "long":
            return (current_price - position.entry_price) / position.entry_price
        else:
            return (position.entry_price - current_price) / position.entry_price

    def check_partial_exits(self, position: PositionState, current_price: float) -> list[dict]:
        """Return a list of partial exit actions to perform at this price.

        Each action dict contains: {'type': 'partial_exit', 'size': float, 'target_level': int}
        Size is a fraction of ORIGINAL size; callers should translate to current_size quantity.
        """
        actions: list[dict] = []
        pnl = self._pnl_pct(position, current_price)

        # Determine next target index to consider based on partial_exits_taken
        next_idx = position.partial_exits_taken
        while next_idx < len(self.exit_targets):
            target = self.exit_targets[next_idx]
            if pnl >= target:  # works for both long and short via pnl sign logic
                actions.append(
                    {
                        "type": "partial_exit",
                        "size": self.exit_sizes[next_idx],
                        "target_level": next_idx,
                    }
                )
                next_idx += 1
            else:
                break

        return actions

    def check_scale_in_opportunity(
        self, position: PositionState, current_price: float, market_data: Optional[dict] = None
    ) -> Optional[dict]:
        """Return a scale-in action dict or None.

        Action dict: {'type': 'scale_in', 'size': float, 'threshold_level': int}
        """
        if position.scale_ins_taken >= self.max_scale_ins:
            return None

        pnl = self._pnl_pct(position, current_price)
        # Find the first threshold not yet used
        for idx in range(position.scale_ins_taken, len(self.scale_in_thresholds)):
            if pnl >= self.scale_in_thresholds[idx]:
                return {
                    "type": "scale_in",
                    "size": self.scale_in_sizes[idx],
                    "threshold_level": idx,
                }
        return None

    # Helpers to update state post execution
    def apply_partial_exit(self, position: PositionState, executed_size_fraction_of_original: float, price: float) -> PositionState:
        """Apply partial exit to position state and return the updated state for clarity."""
        delta = executed_size_fraction_of_original * float(position.original_size)
        position.current_size = max(0.0, float(position.current_size) - delta)
        position.partial_exits_taken = min(position.partial_exits_taken + 1, len(self.exit_targets))
        position.last_partial_exit_price = price
        return position

    def apply_scale_in(self, position: PositionState, add_size_fraction_of_original: float, price: float) -> PositionState:
        """Apply scale-in to position state and return the updated state for clarity."""
        delta = add_size_fraction_of_original * float(position.original_size)
        position.current_size = min(1.0, float(position.current_size) + delta)
        position.scale_ins_taken = min(position.scale_ins_taken + 1, self.max_scale_ins)
        position.last_scale_in_price = price
        return position