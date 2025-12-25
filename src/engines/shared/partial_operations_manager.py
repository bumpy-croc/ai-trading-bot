"""Unified partial exit and scale-in management for trading engines.

This module provides consistent partial operations logic for both
backtesting and live trading engines.

ARCHITECTURE:
- PartialExitPolicy: Configuration dataclass (exit_targets, exit_sizes, etc.)
- PartialOperationsManager: Stateful manager that applies policy logic
- Returns single next action (not lists) for clean control flow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.position_management.partial_manager import PartialExitPolicy

logger = logging.getLogger(__name__)

# Epsilon for floating-point comparisons in financial calculations
EPSILON = 1e-9


@dataclass
class PartialExitResult:
    """Result of a partial exit check.

    Attributes:
        should_exit: Whether a partial exit should be executed.
        exit_fraction: Fraction of ORIGINAL position to exit (0-1).
        target_index: Which target triggered the exit.
        reason: Description of why exit triggered.
    """

    should_exit: bool = False
    exit_fraction: float | None = None
    target_index: int | None = None
    reason: str | None = None


@dataclass
class ScaleInResult:
    """Result of a scale-in check.

    Attributes:
        should_scale: Whether a scale-in should be executed.
        scale_fraction: Fraction of ORIGINAL position to add.
        target_index: Which threshold triggered the scale-in.
        reason: Description of why scale-in triggered.
    """

    should_scale: bool = False
    scale_fraction: float | None = None
    target_index: int | None = None
    reason: str | None = None


class PartialOperationsManager:
    """Unified partial exit and scale-in management.

    This class provides consistent logic for partial position operations
    that is used by both backtesting and live trading engines.

    The manager works with PartialExitPolicy which has:
    - exit_targets: list[float] - PnL thresholds (e.g., [0.05, 0.10])
    - exit_sizes: list[float] - Fractions to exit (e.g., [0.5, 0.5])
    - scale_in_thresholds: list[float] - PnL thresholds for scale-in
    - scale_in_sizes: list[float] - Fractions to add
    - max_scale_ins: int - Maximum scale-in operations

    Returns single next action (not lists) for clean control flow.

    Attributes:
        policy: The partial exit policy to apply.
        total_partial_exits: Count of partial exits executed.
        total_scale_ins: Count of scale-ins executed.
    """

    def __init__(self, policy: PartialExitPolicy | None = None) -> None:
        """Initialize the partial operations manager.

        Args:
            policy: Partial exit policy to apply, or None to disable.
        """
        self.policy = policy
        self.total_partial_exits: int = 0
        self.total_scale_ins: int = 0

    def set_policy(self, policy: PartialExitPolicy | None) -> None:
        """Update the partial operations policy.

        Args:
            policy: New policy to use, or None to disable.
        """
        self.policy = policy

    def check_partial_exit(
        self,
        position: Any,
        current_price: float,
        current_pnl_pct: float | None = None,
    ) -> PartialExitResult:
        """Check if the next partial exit should be triggered.

        Returns single next action to execute. Caller should loop if
        multiple exits can trigger simultaneously.

        Args:
            position: Current position with partial exit tracking.
                     Must have: entry_price, side, partial_exits_taken.
            current_price: Current market price.
            current_pnl_pct: Current PnL percentage (calculated if not provided).

        Returns:
            PartialExitResult with next action or should_exit=False.
        """
        if self.policy is None:
            return PartialExitResult()

        # Get position attributes
        entry_price = getattr(position, "entry_price", None)
        side = self._get_side_str(position)
        partial_exits_taken = getattr(position, "partial_exits_taken", 0)

        if entry_price is None or entry_price <= 0:
            return PartialExitResult()

        # Calculate PnL percentage if not provided
        if current_pnl_pct is None:
            if side == "long":
                current_pnl_pct = (current_price - entry_price) / entry_price
            else:
                current_pnl_pct = (entry_price - current_price) / entry_price

        # Get policy exit configuration
        exit_targets = getattr(self.policy, "exit_targets", [])
        exit_sizes = getattr(self.policy, "exit_sizes", [])

        if not exit_targets or not exit_sizes:
            return PartialExitResult()

        # Find next target to execute
        if partial_exits_taken < len(exit_targets):
            target_pct = exit_targets[partial_exits_taken]
            exit_fraction = exit_sizes[partial_exits_taken]

            if current_pnl_pct >= target_pct and exit_fraction > 0:
                return PartialExitResult(
                    should_exit=True,
                    exit_fraction=exit_fraction,
                    target_index=partial_exits_taken,
                    reason=f"Partial exit target {partial_exits_taken + 1}: profit {current_pnl_pct:.2%} >= {target_pct:.2%}",
                )

        return PartialExitResult()

    def check_scale_in(
        self,
        position: Any,
        current_price: float,
        balance: float,
        current_pnl_pct: float | None = None,
    ) -> ScaleInResult:
        """Check if the next scale-in should be triggered.

        Returns single next action to execute.

        Args:
            position: Current position with scale-in tracking.
                     Must have: entry_price, side, scale_ins_taken.
            current_price: Current market price.
            balance: Current account balance (unused but kept for compatibility).
            current_pnl_pct: Current PnL percentage (calculated if not provided).

        Returns:
            ScaleInResult with next action or should_scale=False.
        """
        if self.policy is None:
            return ScaleInResult()

        # Get position attributes
        entry_price = getattr(position, "entry_price", None)
        side = self._get_side_str(position)
        scale_ins_taken = getattr(position, "scale_ins_taken", 0)

        if entry_price is None or entry_price <= 0:
            return ScaleInResult()

        # Calculate PnL percentage if not provided
        if current_pnl_pct is None:
            if side == "long":
                current_pnl_pct = (current_price - entry_price) / entry_price
            else:
                current_pnl_pct = (entry_price - current_price) / entry_price

        # Get policy scale-in configuration
        scale_in_thresholds = getattr(self.policy, "scale_in_thresholds", [])
        scale_in_sizes = getattr(self.policy, "scale_in_sizes", [])
        max_scale_ins = getattr(self.policy, "max_scale_ins", 0)

        if not scale_in_thresholds or not scale_in_sizes:
            return ScaleInResult()

        # Check if max scale-ins reached
        if scale_ins_taken >= max_scale_ins:
            return ScaleInResult()

        # Find next threshold to execute
        if scale_ins_taken < len(scale_in_thresholds):
            threshold_pct = scale_in_thresholds[scale_ins_taken]
            scale_fraction = scale_in_sizes[scale_ins_taken]

            if current_pnl_pct >= threshold_pct and scale_fraction > 0:
                return ScaleInResult(
                    should_scale=True,
                    scale_fraction=scale_fraction,
                    target_index=scale_ins_taken,
                    reason=f"Scale-in threshold {scale_ins_taken + 1}: profit {current_pnl_pct:.2%} >= {threshold_pct:.2%}",
                )

        return ScaleInResult()

    def _get_side_str(self, position: Any) -> str:
        """Get the side as a lowercase string.

        Args:
            position: Position object with a 'side' attribute.

        Returns:
            Side as lowercase string ('long' or 'short').
        """
        side = getattr(position, "side", None)
        if side is None:
            return "long"
        if hasattr(side, "value"):
            return side.value.lower()
        return str(side).lower()


__all__ = [
    "PartialOperationsManager",
    "PartialExitResult",
    "ScaleInResult",
]
