from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PositionState:
    """Lightweight state used by PartialExitPolicy.

    This is intentionally decoupled from DB/engine models to enable reuse in
    both live and backtesting without import cycles.
    """

    entry_price: float
    side: str  # 'long' | 'short'
    original_size: float  # fraction of balance at entry
    current_size: float  # fraction of balance remaining
    partial_exits_taken: int = 0
    scale_ins_taken: int = 0
    last_partial_exit_price: float | None = None
    last_scale_in_price: float | None = None

    def __post_init__(self):
        """Validate position state to prevent financial calculation errors."""
        if self.original_size <= 0:
            raise ValueError(f"original_size must be positive, got {self.original_size}")
        if self.current_size < 0:
            raise ValueError(f"current_size cannot be negative, got {self.current_size}")
        if self.entry_price <= 0 or not math.isfinite(self.entry_price):
            raise ValueError(f"entry_price must be finite and positive, got {self.entry_price}")
        if self.side not in ("long", "short"):
            raise ValueError(f"side must be 'long' or 'short', got {self.side}")
        # Validate counter fields to prevent index errors in check methods
        if self.partial_exits_taken < 0:
            raise ValueError(
                f"partial_exits_taken cannot be negative, got {self.partial_exits_taken}"
            )
        if self.scale_ins_taken < 0:
            raise ValueError(f"scale_ins_taken cannot be negative, got {self.scale_ins_taken}")


@dataclass
class PartialExitPolicy:
    """Policy for partial exits (scale-out) and scale-ins.

    Targets and thresholds are expressed as decimal returns from entry, e.g., 0.03 = +3 %.
    Sizes are expressed as fractions of the original position size.

    Limits:
        exit_targets: Maximum of 10 exit targets allowed (typical configs use 2-3 exits).
        This limit prevents infinite loops from malformed configurations.
    """

    exit_targets: list[float]
    exit_sizes: list[float]
    scale_in_thresholds: list[float] = field(default_factory=list)
    scale_in_sizes: list[float] = field(default_factory=list)
    max_scale_ins: int = 0

    def __post_init__(self):
        if len(self.exit_targets) != len(self.exit_sizes):
            raise ValueError("exit_targets and exit_sizes must have equal length")
        if any(t <= 0 for t in self.exit_targets):
            raise ValueError("exit_targets must be positive")
        if any(s <= 0 or s > 1 for s in self.exit_sizes):
            raise ValueError("exit_sizes must be in (0, 1]")
        if len(self.exit_targets) > 10:
            raise ValueError(f"exit_targets limited to 10 entries, got {len(self.exit_targets)}")
        if len(self.scale_in_thresholds) != len(self.scale_in_sizes):
            raise ValueError("scale_in_thresholds and scale_in_sizes must have equal length")
        if any(t <= 0 for t in self.scale_in_thresholds):
            raise ValueError("scale_in_thresholds must be positive")
        if any(s <= 0 or s > 1 for s in self.scale_in_sizes):
            raise ValueError("scale_in_sizes must be in (0, 1]")
        if self.max_scale_ins < 0:
            raise ValueError("max_scale_ins must be >= 0")

    def _pnl_pct(self, position: PositionState, current_price: float) -> float:
        """Calculate PnL percentage for position, validating price inputs."""
        if not math.isfinite(current_price) or current_price <= 0:
            return 0.0
        if position.entry_price <= 0 or not math.isfinite(position.entry_price):
            return 0.0
        if position.side == "long":
            return (current_price - position.entry_price) / position.entry_price
        else:
            return (position.entry_price - current_price) / position.entry_price

    def _validate_fraction(self, fraction: float, param_name: str = "fraction") -> bool:
        """Validate fraction parameter for partial exit/scale-in operations.

        Returns True if fraction is valid, False otherwise. Logs warnings for invalid values.

        Args:
            fraction: The fraction value to validate
            param_name: Name of the parameter for logging purposes

        Returns:
            True if fraction is numeric, finite, non-negative, and <= 1.0
        """
        if not isinstance(fraction, int | float):
            logger.warning(f"{param_name} must be numeric, got {type(fraction)}")
            return False
        if not math.isfinite(fraction):
            logger.warning(f"{param_name} must be finite, got {fraction}")
            return False
        if fraction < 0:
            logger.warning(f"{param_name} cannot be negative, got {fraction}")
            return False
        if fraction > 1.0:
            logger.warning(f"{param_name} cannot exceed 1.0, got {fraction}")
            return False
        return True

    def check_partial_exits(
        self, position: PositionState, current_price: float
    ) -> list[dict[str, Any]]:
        """Return a list of partial exit actions to perform at this price.

        Each action dict contains: {'type': 'partial_exit', 'size': float, 'target_level': int}
        Size is a fraction of ORIGINAL size; callers should translate to current_size quantity.
        """
        actions: list[dict[str, Any]] = []
        pnl = self._pnl_pct(position, current_price)

        # Determine next target index to consider based on partial_exits_taken
        next_idx = position.partial_exits_taken

        # Add iteration guard to prevent infinite loops from malformed configurations
        # Capped at 10 since exit_targets is limited to 10 entries by __post_init__
        max_iterations = min(len(self.exit_targets), 10)
        iteration = 0

        while next_idx < len(self.exit_targets) and iteration < max_iterations:
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
            iteration += 1

        return actions

    def check_scale_in_opportunity(
        self,
        position: PositionState,
        current_price: float,
        market_data: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
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
    def apply_partial_exit(
        self, position: PositionState, executed_size_fraction_of_original: float, price: float
    ):
        """Apply a partial exit to the position state, validating inputs."""
        if not self._validate_fraction(
            executed_size_fraction_of_original, "executed_size_fraction"
        ):
            return

        if position.original_size <= 0:
            logger.warning(
                f"Cannot apply partial exit to position with original_size={position.original_size}"
            )
            return
        if not math.isfinite(price) or price <= 0:
            logger.warning(f"Cannot apply partial exit with invalid price={price}")
            return
        delta = executed_size_fraction_of_original * position.original_size
        position.current_size = max(0.0, position.current_size - delta)
        position.partial_exits_taken = min(position.partial_exits_taken + 1, len(self.exit_targets))
        position.last_partial_exit_price = price

    def apply_scale_in(
        self, position: PositionState, add_size_fraction_of_original: float, price: float
    ):
        """Apply a scale-in to the position state, validating inputs."""
        if not self._validate_fraction(add_size_fraction_of_original, "add_size_fraction"):
            return

        if position.original_size <= 0:
            logger.warning(
                f"Cannot apply scale-in to position with original_size={position.original_size}"
            )
            return
        if not math.isfinite(price) or price <= 0:
            logger.warning(f"Cannot apply scale-in with invalid price={price}")
            return
        delta = add_size_fraction_of_original * position.original_size
        position.current_size = min(1.0, position.current_size + delta)
        position.scale_ins_taken = min(position.scale_ins_taken + 1, self.max_scale_ins)
        position.last_scale_in_price = price
