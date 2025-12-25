"""Unified partial exit and scale-in management for trading engines.

This module provides consistent partial operations logic for both
backtesting and live trading engines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.position_management.partial_manager import PartialExitPolicy

logger = logging.getLogger(__name__)


@dataclass
class PartialExitResult:
    """Result of a partial exit check.

    Attributes:
        should_exit: Whether a partial exit should be executed.
        exit_fraction: Fraction of position to exit (0-1).
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
        scale_fraction: Fraction to add to position.
        target_index: Which target triggered the scale-in.
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
        """Check if a partial exit should be triggered.

        Args:
            position: Current position with partial exit tracking.
            current_price: Current market price.
            current_pnl_pct: Current PnL percentage (calculated if not provided).

        Returns:
            PartialExitResult indicating whether to exit.
        """
        if self.policy is None:
            return PartialExitResult()

        # Get position attributes
        entry_price = getattr(position, "entry_price", None)
        side = self._get_side_str(position)
        partial_exits_taken = getattr(position, "partial_exits_taken", 0)

        if entry_price is None:
            return PartialExitResult()

        # Calculate PnL percentage if not provided
        if current_pnl_pct is None:
            if side == "long":
                current_pnl_pct = (current_price - entry_price) / entry_price
            else:
                current_pnl_pct = (entry_price - current_price) / entry_price

        # Get policy targets
        targets = getattr(self.policy, "profit_targets", [])
        if not targets:
            return PartialExitResult()

        # Check each target
        for i, target in enumerate(targets):
            if i < partial_exits_taken:
                continue  # Already took this target

            target_pct = target.get("profit_pct", 0)
            exit_fraction = target.get("exit_fraction", 0)

            if current_pnl_pct >= target_pct and exit_fraction > 0:
                return PartialExitResult(
                    should_exit=True,
                    exit_fraction=exit_fraction,
                    target_index=i,
                    reason=f"Partial exit target {i+1}: profit {current_pnl_pct:.2%} >= {target_pct:.2%}",
                )

        return PartialExitResult()

    def check_scale_in(
        self,
        position: Any,
        current_price: float,
        balance: float,
        current_pnl_pct: float | None = None,
    ) -> ScaleInResult:
        """Check if a scale-in should be triggered.

        Args:
            position: Current position with scale-in tracking.
            current_price: Current market price.
            balance: Current account balance.
            current_pnl_pct: Current PnL percentage (calculated if not provided).

        Returns:
            ScaleInResult indicating whether to scale in.
        """
        if self.policy is None:
            return ScaleInResult()

        # Get position attributes
        entry_price = getattr(position, "entry_price", None)
        side = self._get_side_str(position)
        scale_ins_taken = getattr(position, "scale_ins_taken", 0)

        if entry_price is None:
            return ScaleInResult()

        # Calculate PnL percentage if not provided
        if current_pnl_pct is None:
            if side == "long":
                current_pnl_pct = (current_price - entry_price) / entry_price
            else:
                current_pnl_pct = (entry_price - current_price) / entry_price

        # Get scale-in targets
        scale_targets = getattr(self.policy, "scale_in_targets", [])
        if not scale_targets:
            return ScaleInResult()

        # Check each target
        for i, target in enumerate(scale_targets):
            if i < scale_ins_taken:
                continue  # Already took this target

            target_pct = target.get("profit_pct", 0)
            scale_fraction = target.get("scale_fraction", 0)

            if current_pnl_pct >= target_pct and scale_fraction > 0:
                return ScaleInResult(
                    should_scale=True,
                    scale_fraction=scale_fraction,
                    target_index=i,
                    reason=f"Scale-in target {i+1}: profit {current_pnl_pct:.2%} >= {target_pct:.2%}",
                )

        return ScaleInResult()

    def apply_partial_exit(
        self,
        position: Any,
        exit_fraction: float,
        current_price: float,
        entry_balance: float,
    ) -> float:
        """Execute a partial exit and calculate realized PnL.

        Args:
            position: Position to partially exit.
            exit_fraction: Fraction to exit (0-1).
            current_price: Current market price.
            entry_balance: Balance at position entry.

        Returns:
            Realized PnL from the partial exit.
        """
        entry_price = getattr(position, "entry_price", 0)
        side = self._get_side_str(position)
        current_size = getattr(position, "current_size", getattr(position, "size", 0))

        # Calculate exit size
        exit_size = current_size * exit_fraction
        exit_notional = exit_size * entry_balance

        # Calculate PnL
        if side == "long":
            pnl = exit_notional * (current_price - entry_price) / entry_price
        else:
            pnl = exit_notional * (entry_price - current_price) / entry_price

        # Update position tracking
        new_size = current_size - exit_size
        if hasattr(position, "current_size"):
            position.current_size = new_size
        if hasattr(position, "partial_exits_taken"):
            position.partial_exits_taken += 1

        self.total_partial_exits += 1

        logger.debug(
            "Partial exit: %.2f%% of position at %.2f, PnL=%.2f",
            exit_fraction * 100,
            current_price,
            pnl,
        )

        return pnl

    def apply_scale_in(
        self,
        position: Any,
        scale_fraction: float,
        current_price: float,
    ) -> None:
        """Execute a scale-in operation.

        Args:
            position: Position to scale into.
            scale_fraction: Fraction to add.
            current_price: Current market price.
        """
        original_size = getattr(position, "original_size", getattr(position, "size", 0))
        current_size = getattr(position, "current_size", getattr(position, "size", 0))

        # Calculate new size
        add_size = original_size * scale_fraction
        new_size = current_size + add_size

        # Update position tracking
        if hasattr(position, "current_size"):
            position.current_size = new_size
        if hasattr(position, "scale_ins_taken"):
            position.scale_ins_taken += 1

        self.total_scale_ins += 1

        logger.debug(
            "Scale-in: +%.2f%% at %.2f, new size=%.2f",
            scale_fraction * 100,
            current_price,
            new_size,
        )

    def _get_side_str(self, position: Any) -> str:
        """Get the side as a lowercase string."""
        side = getattr(position, "side", None)
        if side is None:
            return "long"
        if hasattr(side, "value"):
            return side.value.lower()
        return str(side).lower()

    def reset_stats(self) -> None:
        """Reset operation counters."""
        self.total_partial_exits = 0
        self.total_scale_ins = 0


__all__ = [
    "PartialOperationsManager",
    "PartialExitResult",
    "ScaleInResult",
]
