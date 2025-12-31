"""LiveExitHandler processes exit conditions for live trading.

Handles exit condition checking including stop loss, take profit,
trailing stops, time-based exits, and partial operations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.engines.live.execution.execution_engine import LiveExecutionEngine
from src.engines.live.execution.position_tracker import (
    LivePosition,
    LivePositionTracker,
    PositionSide,
)
from src.engines.shared.partial_operations_manager import (
    EPSILON,
    PartialOperationsManager,
)
from src.engines.shared.trailing_stop_manager import TrailingStopManager

if TYPE_CHECKING:
    from src.position_management.time_exits import TimeExitPolicy
    from src.position_management.trailing_stops import TrailingStopPolicy
    from src.risk.risk_manager import RiskManager
    from src.strategies.components import Strategy as ComponentStrategy

logger = logging.getLogger(__name__)

# Maximum partial exits to process per cycle (defense-in-depth against malformed policies).
# Ten caps worst-case overhead while still supporting typical multi-target exit ladders.
MAX_PARTIAL_EXITS_PER_CYCLE = 10
# Maximum acceptable filled-price deviation from entry price before logging a critical warning.
MAX_FILLED_PRICE_DEVIATION = 0.5


@dataclass
class LiveExitCheck:
    """Result of exit condition check."""

    should_exit: bool
    exit_reason: str = ""
    limit_price: float | None = None  # For SL/TP pricing


@dataclass
class LiveExitResult:
    """Result of executing an exit."""

    success: bool
    realized_pnl: float = 0.0
    realized_pnl_percent: float = 0.0
    exit_price: float = 0.0
    exit_fee: float = 0.0
    slippage_cost: float = 0.0
    exit_reason: str = ""
    error: str | None = None


class LiveExitHandler:
    """Processes exit conditions for live trading.

    This class encapsulates exit-related logic including:
    - Stop loss detection
    - Take profit detection
    - Trailing stop updates
    - Time-based exits
    - Partial exit operations
    - Strategy signal exits
    """

    def __init__(
        self,
        execution_engine: LiveExecutionEngine,
        position_tracker: LivePositionTracker,
        risk_manager: RiskManager | None = None,
        trailing_stop_policy: TrailingStopPolicy | None = None,
        time_exit_policy: TimeExitPolicy | None = None,
        partial_manager: PartialOperationsManager | None = None,
        use_high_low_for_stops: bool = True,
        max_position_size: float = 0.1,
        max_filled_price_deviation: float = MAX_FILLED_PRICE_DEVIATION,
    ) -> None:
        """Initialize exit handler.

        Args:
            execution_engine: Engine for executing exits.
            position_tracker: Tracker for position state.
            risk_manager: Risk manager for position updates.
            trailing_stop_policy: Policy for trailing stops.
            time_exit_policy: Policy for time-based exits.
            partial_manager: Unified partial operations manager.
            use_high_low_for_stops: Use candle high/low for SL/TP detection.
            max_position_size: Maximum position size for scale-ins.
            max_filled_price_deviation: Threshold for logging suspicious fill prices.
        """
        self.execution_engine = execution_engine
        self.position_tracker = position_tracker
        self.risk_manager = risk_manager
        self.trailing_stop_policy = trailing_stop_policy
        self.time_exit_policy = time_exit_policy
        self.partial_manager = partial_manager
        self.use_high_low_for_stops = use_high_low_for_stops
        self.max_position_size = max_position_size
        self.max_filled_price_deviation = max_filled_price_deviation
        # Use shared managers for consistent logic across engines
        self._trailing_stop_manager = TrailingStopManager(trailing_stop_policy)

    def check_exit_conditions(
        self,
        position: LivePosition,
        current_price: float,
        candle_high: float | None = None,
        candle_low: float | None = None,
        runtime_decision: Any = None,
        component_strategy: ComponentStrategy | None = None,
    ) -> LiveExitCheck:
        """Check all exit conditions for a position.

        Args:
            position: Position to check.
            current_price: Current market price.
            candle_high: Candle high price for realistic detection.
            candle_low: Candle low price for realistic detection.
            runtime_decision: Decision from strategy runtime.
            component_strategy: Component strategy for exit signals.

        Returns:
            LiveExitCheck with exit decision and reason.
        """
        # Check strategy exit signal first
        if runtime_decision is not None and component_strategy is not None:
            should_exit, reason = self._check_strategy_exit(
                position, current_price, runtime_decision, component_strategy
            )
            if should_exit:
                return LiveExitCheck(
                    should_exit=True,
                    exit_reason=reason,
                    limit_price=None,
                )

        # Check stop loss (priority over take profit)
        if position.stop_loss is not None:
            if self._check_stop_loss(position, current_price, candle_high, candle_low):
                return LiveExitCheck(
                    should_exit=True,
                    exit_reason="Stop loss",
                    limit_price=position.stop_loss,
                )

        # Check take profit
        if position.take_profit is not None:
            if self._check_take_profit(position, current_price, candle_high, candle_low):
                return LiveExitCheck(
                    should_exit=True,
                    exit_reason="Take profit",
                    limit_price=position.take_profit,
                )

        # Check time-based exit
        if self.time_exit_policy is not None:
            hit_time_exit, time_reason = self.time_exit_policy.check_time_exit_conditions(
                position.entry_time, datetime.now(UTC)
            )
            if hit_time_exit:
                return LiveExitCheck(
                    should_exit=True,
                    exit_reason=time_reason or "Time exit",
                    limit_price=None,
                )

        return LiveExitCheck(should_exit=False)

    def execute_exit(
        self,
        position: LivePosition,
        exit_reason: str,
        current_price: float,
        limit_price: float | None,
        current_balance: float,
        candle_high: float | None = None,
        candle_low: float | None = None,
        data_provider: Any = None,
    ) -> LiveExitResult:
        """Execute an exit for a position.

        Args:
            position: Position to close.
            exit_reason: Reason for exit.
            current_price: Current market price.
            limit_price: Limit price for SL/TP exits.
            current_balance: Current account balance.
            candle_high: Candle high for realistic execution modeling.
            candle_low: Candle low for realistic execution modeling.
            data_provider: Data provider for price fallback.

        Returns:
            LiveExitResult with execution details.
        """
        if position.order_id is None:
            return LiveExitResult(
                success=False,
                error="Position has no order_id",
            )

        # Determine base exit price with realistic execution modeling
        if limit_price is not None:
            # For SL/TP, use worst-case execution price from candle high/low
            if self.use_high_low_for_stops and candle_high is not None and candle_low is not None:
                if "Stop loss" in exit_reason:
                    # Stop losses execute at worst price
                    if position.side == PositionSide.LONG:
                        # Long SL: use max(stop_loss, candle_low) for realistic worst-case
                        base_exit_price = max(limit_price, candle_low)
                    else:
                        # Short SL: use min(stop_loss, candle_high) for realistic worst-case
                        base_exit_price = min(limit_price, candle_high)
                elif "Take profit" in exit_reason:
                    # Take profits execute at the limit price or worse (never better than limit)
                    base_exit_price = limit_price
                else:
                    base_exit_price = limit_price
            else:
                base_exit_price = limit_price
        else:
            base_exit_price = current_price

        # IMPORTANT: Use exit notional (accounting for price change) for accurate fee calculation.
        # This correctly models real exchange behavior where fees are charged on the actual
        # value at trade time:
        # - Winning positions: selling more valuable assets → higher fee
        # - Losing positions: selling less valuable assets → lower fee
        # This differs from entry fees which use the original notional value.
        position_notional = self._calculate_position_notional(
            position=position,
            current_balance=current_balance,
            exit_price=base_exit_price,
        )

        # Execute via execution engine
        execution_result = self.execution_engine.execute_exit(
            symbol=position.symbol,
            side=position.side,
            order_id=position.order_id,
            base_price=base_exit_price,
            position_notional=position_notional,
        )

        if not execution_result.success:
            return LiveExitResult(
                success=False,
                error=execution_result.error,
            )

        # Close position in tracker
        close_result = self.position_tracker.close_position(
            order_id=position.order_id,
            exit_price=execution_result.executed_price,
            exit_reason=exit_reason,
            basis_balance=current_balance,
        )

        if close_result is None:
            return LiveExitResult(
                success=False,
                error="Failed to close position in tracker",
            )

        # Update risk manager
        if self.risk_manager is not None:
            try:
                self.risk_manager.close_position(position.symbol)
            except (AttributeError, ValueError, KeyError) as e:
                logger.warning(
                    "Failed to update risk manager for closed position %s: %s",
                    position.symbol,
                    e,
                )

        return LiveExitResult(
            success=True,
            realized_pnl=close_result.realized_pnl,
            realized_pnl_percent=close_result.realized_pnl_percent,
            exit_price=close_result.exit_price,
            exit_fee=execution_result.exit_fee,
            slippage_cost=execution_result.slippage_cost,
            exit_reason=exit_reason,
        )

    def execute_filled_exit(
        self,
        position: LivePosition,
        exit_reason: str,
        filled_price: float,
        current_balance: float,
    ) -> LiveExitResult:
        """Finalize an exit where the exchange already filled the order.

        Args:
            position: Position to close.
            exit_reason: Reason for exit.
            filled_price: Exchange-reported fill price.
            current_balance: Current account balance.

        Returns:
            LiveExitResult with execution details.
        """
        if position.order_id is None:
            return LiveExitResult(
                success=False,
                error="Position has no order_id",
            )

        if not self.position_tracker.has_position(position.order_id):
            logger.warning(
                "Filled exit received for already closed position %s",
                position.order_id,
            )
            return LiveExitResult(
                success=False,
                error="Position already closed",
            )

        if filled_price <= 0:
            return LiveExitResult(
                success=False,
                error="Invalid filled price",
            )

        if position.entry_price > 0:
            price_change = abs(filled_price - position.entry_price) / position.entry_price
            if price_change > self.max_filled_price_deviation:
                logger.critical(
                    "Suspicious fill price for %s: entry=%.2f filled=%.2f (%.1f%% move)",
                    position.symbol,
                    position.entry_price,
                    filled_price,
                    price_change * 100,
                )

        # Filled exits use the exchange-reported price; slippage models adverse execution costs.
        executed_price = self.execution_engine.apply_exit_slippage(
            filled_price, position.side
        )

        position_notional = self._calculate_position_notional(
            position=position,
            current_balance=current_balance,
            exit_price=executed_price,
        )

        exit_fee = self.execution_engine.calculate_exit_fee(position_notional)
        slippage_cost = self.execution_engine.calculate_slippage_cost(position_notional)

        close_result = self.position_tracker.close_position(
            order_id=position.order_id,
            exit_price=executed_price,
            exit_reason=exit_reason,
            basis_balance=current_balance,
        )
        if close_result is None:
            return LiveExitResult(
                success=False,
                error="Failed to close position in tracker",
            )

        if self.risk_manager is not None:
            try:
                self.risk_manager.close_position(position.symbol)
            except (AttributeError, ValueError, KeyError) as e:
                logger.warning(
                    "Failed to update risk manager for closed position %s: %s",
                    position.symbol,
                    e,
                )

        return LiveExitResult(
            success=True,
            realized_pnl=close_result.realized_pnl,
            realized_pnl_percent=close_result.realized_pnl_percent,
            exit_price=close_result.exit_price,
            exit_fee=exit_fee,
            slippage_cost=slippage_cost,
            exit_reason=exit_reason,
        )

    def _check_strategy_exit(
        self,
        position: LivePosition,
        current_price: float,
        runtime_decision: Any,
        component_strategy: ComponentStrategy,
    ) -> tuple[bool, str]:
        """Check if strategy signals an exit.

        Args:
            position: Position to check.
            current_price: Current market price.
            runtime_decision: Decision from strategy runtime.
            component_strategy: Component strategy.

        Returns:
            Tuple of (should_exit, reason).
        """
        from src.strategies.components import MarketData as ComponentMarketData
        from src.strategies.components import Position as ComponentPosition
        from src.strategies.components import SignalDirection

        # Check for signal reversal
        if (
            position.side == PositionSide.LONG
            and runtime_decision.signal.direction == SignalDirection.SELL
        ):
            return True, "Signal reversal"
        if (
            position.side == PositionSide.SHORT
            and runtime_decision.signal.direction == SignalDirection.BUY
        ):
            return True, "Signal reversal"

        # Check component strategy exit
        try:
            component_position = ComponentPosition(
                symbol=position.symbol,
                side=position.side.value,
                size=float(position.size),
                entry_price=float(position.entry_price),
                current_price=float(current_price),
                entry_time=position.entry_time,
            )
            market_data = ComponentMarketData(
                symbol=position.symbol,
                price=float(current_price),
                volume=0.0,
            )
            regime = runtime_decision.regime if runtime_decision else None

            if component_strategy.should_exit_position(
                component_position, market_data, regime
            ):
                return True, "Strategy signal"
        except (AttributeError, ValueError, TypeError) as e:
            logger.debug("Component exit check failed: %s", e)

        return False, "Hold"

    def _check_stop_loss(
        self,
        position: LivePosition,
        current_price: float,
        candle_high: float | None = None,
        candle_low: float | None = None,
    ) -> bool:
        """Check if stop loss should be triggered.

        Uses candle high/low for realistic worst-case execution detection.
        For long positions, uses max(stop_loss, candle_low) to model realistic fill prices
        since stop losses typically execute at or worse than the stop level.

        Args:
            position: Position to check.
            current_price: Current (close) price.
            candle_high: Candle high price.
            candle_low: Candle low price.

        Returns:
            True if stop loss was triggered.
        """
        if position.stop_loss is None:
            return False

        # Use high/low for more realistic detection
        if (
            self.use_high_low_for_stops
            and candle_low is not None
            and candle_high is not None
        ):
            if position.side == PositionSide.LONG:
                # For long SL, check if candle_low breached the stop
                return candle_low <= position.stop_loss
            else:
                # For short SL, check if candle_high breached the stop
                return candle_high >= position.stop_loss
        else:
            # Fallback to close price only
            if position.side == PositionSide.LONG:
                return current_price <= position.stop_loss
            else:
                return current_price >= position.stop_loss

    def _check_take_profit(
        self,
        position: LivePosition,
        current_price: float,
        candle_high: float | None = None,
        candle_low: float | None = None,
    ) -> bool:
        """Check if take profit should be triggered.

        Args:
            position: Position to check.
            current_price: Current (close) price.
            candle_high: Candle high price.
            candle_low: Candle low price.

        Returns:
            True if take profit was triggered.
        """
        if position.take_profit is None:
            return False

        # Use high/low for more realistic detection
        if (
            self.use_high_low_for_stops
            and candle_high is not None
            and candle_low is not None
        ):
            if position.side == PositionSide.LONG:
                return candle_high >= position.take_profit
            else:
                return candle_low <= position.take_profit
        else:
            # Fallback to close price only
            if position.side == PositionSide.LONG:
                return current_price >= position.take_profit
            else:
                return current_price <= position.take_profit

    def _calculate_position_notional(
        self,
        position: LivePosition,
        current_balance: float,
        exit_price: float,
    ) -> float:
        """Calculate exit notional accounting for price movement."""
        fraction = float(
            position.current_size if position.current_size is not None else position.size
        )
        basis_balance = (
            float(position.entry_balance)
            if position.entry_balance is not None and position.entry_balance > 0
            else current_balance
        )
        price_adjustment = exit_price / position.entry_price if position.entry_price > 0 else 1.0
        return basis_balance * fraction * price_adjustment

    def update_trailing_stops(
        self,
        df: pd.DataFrame,
        current_index: int,
        current_price: float,
    ) -> None:
        """Update trailing stops for all positions.

        Uses shared TrailingStopManager for consistent logic across engines.

        Args:
            df: DataFrame with market data.
            current_index: Current candle index.
            current_price: Current market price.
        """
        if self._trailing_stop_manager.policy is None:
            return

        for order_id, position in self.position_tracker.positions.items():
            # Use shared trailing stop manager for calculation
            result = self._trailing_stop_manager.update(
                position=position,
                current_price=current_price,
                df=df,
                index=current_index,
            )

            if not result.updated:
                continue

            # Determine new activation states
            new_activated = result.trailing_activated or position.trailing_stop_activated
            new_breakeven = result.breakeven_triggered or position.breakeven_triggered

            # Update position via tracker
            changed = self.position_tracker.update_trailing_stop(
                order_id=order_id,
                new_stop_loss=result.new_stop_price,
                activated=new_activated,
                breakeven_triggered=new_breakeven,
            )

            if changed:
                logger.info(
                    "Trailing stop updated for %s %s: SL=%.4f (activated=%s, BE=%s)",
                    position.symbol,
                    position.side.value,
                    position.stop_loss or 0.0,
                    position.trailing_stop_activated,
                    position.breakeven_triggered,
                )

    def check_partial_operations(
        self,
        df: pd.DataFrame,
        current_index: int,
        current_price: float,
        current_balance: float,
    ) -> None:
        """Check and execute partial exits and scale-ins.

        Uses unified PartialOperationsManager for consistent logic.

        Threading Safety:
        - position_tracker.positions property returns a copy with internal locking
        - All position_tracker methods are protected by _positions_lock
        - Safe for concurrent access from OrderTracker callbacks and main trading loop
        - list() creates snapshot of the copy for iteration safety
        - No additional locking needed at this level

        Args:
            df: DataFrame with market data.
            current_index: Current candle index.
            current_price: Current market price.
            current_balance: Current account balance.
        """
        if self.partial_manager is None:
            return

        # Defensive iteration: list() creates snapshot to prevent concurrent modification errors
        for order_id, position in list(self.position_tracker.positions.items()):
            try:
                # Check for partial exits (loop to handle multiple exits in same cycle)
                iteration_count = 0
                while iteration_count < MAX_PARTIAL_EXITS_PER_CYCLE:
                    exit_result = self.partial_manager.check_partial_exit(
                        position=position,
                        current_price=current_price,
                    )

                    if not exit_result.should_exit:
                        break

                    # Convert from fraction of original to fraction of current
                    exit_size_of_original = exit_result.exit_fraction
                    current_size_fraction = position.current_size / position.original_size

                    # Protect against division by zero (position fully closed)
                    if abs(current_size_fraction) < EPSILON:
                        logger.debug(
                            "Position %s fully closed, skipping further partial exits",
                            position.symbol,
                        )
                        break

                    exit_size_of_current = exit_size_of_original / current_size_fraction

                    # Validate bounds and check for NaN/Infinity
                    if (
                        exit_size_of_current <= 0
                        or exit_size_of_current > 1.0
                        or not math.isfinite(exit_size_of_current)
                    ):
                        break

                    self._execute_partial_exit(
                        order_id=order_id,
                        position=position,
                        delta_fraction=exit_size_of_current,
                        price=current_price,
                        target_level=exit_result.target_index,
                        fraction_of_original=exit_size_of_original,
                        current_balance=current_balance,
                    )

                    iteration_count += 1

                # Check for scale-ins
                scale_result = self.partial_manager.check_scale_in(
                    position=position,
                    current_price=current_price,
                    balance=current_balance,
                )

                if scale_result.should_scale:
                    add_size_of_original = scale_result.scale_fraction

                    self._execute_scale_in(
                        order_id=order_id,
                        position=position,
                        delta_fraction=add_size_of_original,
                        price=current_price,
                        threshold_level=scale_result.target_index,
                        fraction_of_original=add_size_of_original,
                    )

            except (AttributeError, ValueError, KeyError, ZeroDivisionError) as e:
                logger.warning("Partial ops evaluation failed for %s: %s", position.symbol, e)
            except Exception as e:
                logger.warning("Unexpected error in partial ops for %s: %s", position.symbol, e)

    def _execute_partial_exit(
        self,
        order_id: str,
        position: LivePosition,
        delta_fraction: float,
        price: float,
        target_level: int,
        fraction_of_original: float,
        current_balance: float,
    ) -> None:
        """Execute a partial exit.

        Args:
            order_id: Order ID of position.
            position: Position to partially exit.
            delta_fraction: Fraction of current position to exit.
            price: Current market price.
            target_level: Profit target level.
            fraction_of_original: Fraction of original position.
            current_balance: Current account balance.
        """
        result = self.position_tracker.apply_partial_exit(
            order_id=order_id,
            delta_fraction=delta_fraction,
            price=price,
            target_level=target_level,
            fraction_of_original=fraction_of_original,
            basis_balance=current_balance,
            fee_rate=self.execution_engine.fee_rate,
            slippage_rate=self.execution_engine.slippage_rate,
        )

        if result is not None:
            # Update risk manager
            if self.risk_manager is not None:
                try:
                    self.risk_manager.adjust_position_after_partial_exit(
                        position.symbol, delta_fraction
                    )
                except (AttributeError, ValueError, KeyError) as e:
                    logger.debug("Risk manager partial-exit accounting failed: %s", e)

            # If fully closed by partials, close position
            if result.new_current_size <= 1e-9:
                self.execute_exit(
                    position=position,
                    exit_reason=f"Partial exits complete @ level {target_level}",
                    current_price=price,
                    limit_price=None,
                    current_balance=current_balance,
                )

    def _execute_scale_in(
        self,
        order_id: str,
        position: LivePosition,
        delta_fraction: float,
        price: float,
        threshold_level: int,
        fraction_of_original: float,
    ) -> None:
        """Execute a scale-in.

        Args:
            order_id: Order ID of position.
            position: Position to scale into.
            delta_fraction: Fraction to add.
            price: Current market price.
            threshold_level: Threshold level.
            fraction_of_original: Fraction of original position.
        """
        result = self.position_tracker.apply_scale_in(
            order_id=order_id,
            delta_fraction=delta_fraction,
            price=price,
            threshold_level=threshold_level,
            fraction_of_original=fraction_of_original,
            max_position_size=self.max_position_size,
        )

        if result is not None:
            # Update risk manager
            if self.risk_manager is not None:
                try:
                    self.risk_manager.adjust_position_after_scale_in(
                        position.symbol, delta_fraction
                    )
                except (AttributeError, ValueError, KeyError) as e:
                    logger.debug("Risk manager scale-in accounting failed: %s", e)
