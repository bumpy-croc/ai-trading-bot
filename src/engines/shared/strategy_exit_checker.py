"""Unified strategy exit checking logic for trading engines.

This module provides consistent exit signal evaluation for both
backtesting and live trading engines, ensuring parity in strategy
exit decisions.

ARCHITECTURE:
- Single source of truth for strategy exit checking
- Used by both ExitHandler (backtest) and LiveExitHandler (live)
- Prevents divergence in exit decision logic
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.engines.shared.models import normalize_side

if TYPE_CHECKING:
    from src.engines.shared.models import BasePosition
    from src.strategies.components import Strategy as ComponentStrategy

logger = logging.getLogger(__name__)


@dataclass
class StrategyExitResult:
    """Result of checking strategy exit conditions.

    Attributes:
        should_exit: Whether the strategy signals an exit.
        exit_reason: Description of exit reason if should_exit is True.
    """

    should_exit: bool
    exit_reason: str = "Hold"


class StrategyExitChecker:
    """Unified strategy exit checking logic.

    This class provides consistent exit signal evaluation for both
    backtesting and live trading engines.

    The checker evaluates:
    1. Signal reversal (position direction vs signal direction)
    2. Component strategy's should_exit_position() method
    """

    def check_exit(
        self,
        position: BasePosition | Any,
        current_price: float,
        runtime_decision: Any,
        component_strategy: ComponentStrategy | None,
        volume: float = 0.0,
        timestamp: datetime | None = None,
    ) -> StrategyExitResult:
        """Check if strategy signals an exit for the position.

        This method provides unified exit checking logic used by both
        backtest and live engines to ensure identical behavior.

        Args:
            position: Current position to evaluate for exit.
            current_price: Current market price.
            runtime_decision: Decision from strategy runtime containing signal.
            component_strategy: Component strategy with exit logic.
            volume: Current volume (optional, for market data).
            timestamp: Current timestamp (optional, for market data).

        Returns:
            StrategyExitResult with exit decision and reason.
        """
        if runtime_decision is None or position is None:
            return StrategyExitResult(should_exit=False)

        # Check signal reversal first
        reversal_result = self._check_signal_reversal(position, runtime_decision)
        if reversal_result.should_exit:
            return reversal_result

        # Check component strategy exit
        if component_strategy is not None:
            strategy_result = self._check_component_exit(
                position=position,
                current_price=current_price,
                runtime_decision=runtime_decision,
                component_strategy=component_strategy,
                volume=volume,
                timestamp=timestamp,
            )
            if strategy_result.should_exit:
                return strategy_result

        return StrategyExitResult(should_exit=False)

    def _check_signal_reversal(
        self,
        position: BasePosition | Any,
        runtime_decision: Any,
    ) -> StrategyExitResult:
        """Check if signal direction indicates exit (reversal).

        Args:
            position: Current position.
            runtime_decision: Decision containing signal direction.

        Returns:
            StrategyExitResult indicating if reversal detected.
        """
        try:
            from src.strategies.components import SignalDirection

            side_str = normalize_side(getattr(position, "side", None))

            if (
                side_str == "long"
                and runtime_decision.signal.direction == SignalDirection.SELL
            ):
                return StrategyExitResult(
                    should_exit=True,
                    exit_reason="Signal reversal",
                )
            if (
                side_str == "short"
                and runtime_decision.signal.direction == SignalDirection.BUY
            ):
                return StrategyExitResult(
                    should_exit=True,
                    exit_reason="Signal reversal",
                )
        except (AttributeError, ImportError):
            pass

        return StrategyExitResult(should_exit=False)

    def _check_component_exit(
        self,
        position: BasePosition | Any,
        current_price: float,
        runtime_decision: Any,
        component_strategy: ComponentStrategy,
        volume: float,
        timestamp: datetime | None,
    ) -> StrategyExitResult:
        """Check component strategy's should_exit_position.

        Args:
            position: Current position.
            current_price: Current market price.
            runtime_decision: Decision with regime context.
            component_strategy: Strategy with exit logic.
            volume: Current volume.
            timestamp: Current timestamp.

        Returns:
            StrategyExitResult from component strategy evaluation.
        """
        try:
            from src.strategies.components import MarketData as ComponentMarketData
            from src.strategies.components import Position as ComponentPosition

            # Compute notional value from current position size and entry balance
            # This ensures consistent evaluation between backtest and live engines
            current_size = self._get_current_size(position)
            entry_balance = getattr(position, "entry_balance", None) or 0.0
            notional = current_size * float(entry_balance)

            side_str = normalize_side(getattr(position, "side", None))

            component_position = ComponentPosition(
                symbol=position.symbol,
                side=side_str,
                size=notional,
                entry_price=float(position.entry_price),
                current_price=float(current_price),
                entry_time=position.entry_time,
            )

            market_data = ComponentMarketData(
                symbol=position.symbol,
                price=float(current_price),
                volume=float(volume),
                timestamp=timestamp,
            )

            regime = getattr(runtime_decision, "regime", None)

            if component_strategy.should_exit_position(
                component_position, market_data, regime
            ):
                return StrategyExitResult(
                    should_exit=True,
                    exit_reason="Strategy signal",
                )

        except (AttributeError, ValueError, TypeError, ImportError) as e:
            logger.debug("Component exit check failed: %s", e)

        return StrategyExitResult(should_exit=False)

    def _get_current_size(self, position: Any) -> float:
        """Get current position size (accounting for partial exits).

        Args:
            position: Position with size attributes.

        Returns:
            Current size fraction.
        """
        current_size = getattr(position, "current_size", None)
        if current_size is not None:
            return float(current_size)
        return float(getattr(position, "size", 0.0))


__all__ = [
    "StrategyExitChecker",
    "StrategyExitResult",
]
