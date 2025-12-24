"""LiveExitHandler processes exit conditions for live trading.

Handles exit condition checking including stop loss, take profit,
trailing stops, time-based exits, and partial operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.live.execution.execution_engine import LiveExecutionEngine
from src.live.execution.position_tracker import (
    LivePosition,
    LivePositionTracker,
    PositionSide,
)

if TYPE_CHECKING:
    from src.position_management.partial_manager import PartialExitPolicy
    from src.position_management.time_exits import TimeExitPolicy
    from src.position_management.trailing_stops import TrailingStopPolicy
    from src.risk.risk_manager import RiskManager
    from src.strategies.components import Strategy as ComponentStrategy

logger = logging.getLogger(__name__)


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
        partial_manager: PartialExitPolicy | None = None,
        use_high_low_for_stops: bool = True,
        max_position_size: float = 0.1,
    ) -> None:
        """Initialize exit handler.

        Args:
            execution_engine: Engine for executing exits.
            position_tracker: Tracker for position state.
            risk_manager: Risk manager for position updates.
            trailing_stop_policy: Policy for trailing stops.
            time_exit_policy: Policy for time-based exits.
            partial_manager: Manager for partial exits/scale-ins.
            use_high_low_for_stops: Use candle high/low for SL/TP detection.
            max_position_size: Maximum position size for scale-ins.
        """
        self.execution_engine = execution_engine
        self.position_tracker = position_tracker
        self.risk_manager = risk_manager
        self.trailing_stop_policy = trailing_stop_policy
        self.time_exit_policy = time_exit_policy
        self.partial_manager = partial_manager
        self.use_high_low_for_stops = use_high_low_for_stops
        self.max_position_size = max_position_size

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
                position.entry_time, datetime.utcnow()
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
        data_provider: Any = None,
    ) -> LiveExitResult:
        """Execute an exit for a position.

        Args:
            position: Position to close.
            exit_reason: Reason for exit.
            current_price: Current market price.
            limit_price: Limit price for SL/TP exits.
            current_balance: Current account balance.
            data_provider: Data provider for price fallback.

        Returns:
            LiveExitResult with execution details.
        """
        if position.order_id is None:
            return LiveExitResult(
                success=False,
                error="Position has no order_id",
            )

        # Determine base exit price
        if limit_price is not None:
            base_exit_price = limit_price
        else:
            base_exit_price = current_price

        # Calculate position notional for fee calculation
        fraction = float(
            position.current_size
            if position.current_size is not None
            else position.size
        )
        basis_balance = (
            float(position.entry_balance)
            if position.entry_balance is not None and position.entry_balance > 0
            else current_balance
        )
        position_notional = basis_balance * fraction

        # Execute via execution engine
        exec_result = self.execution_engine.execute_exit(
            symbol=position.symbol,
            side=position.side,
            order_id=position.order_id,
            base_price=base_exit_price,
            position_notional=position_notional,
        )

        if not exec_result.success:
            return LiveExitResult(
                success=False,
                error=exec_result.error,
            )

        # Close position in tracker
        close_result = self.position_tracker.close_position(
            order_id=position.order_id,
            exit_price=exec_result.executed_price,
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
            except Exception as e:
                logger.warning(
                    "Failed to update risk manager for closed position %s: %s",
                    position.symbol,
                    e,
                )

        return LiveExitResult(
            success=True,
            realized_pnl=close_result.realized_pnl,
            realized_pnl_percent=close_result.realized_pnl_percent,
            exit_fee=exec_result.exit_fee,
            slippage_cost=exec_result.slippage_cost,
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
        except Exception as e:
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
                return candle_low <= position.stop_loss
            else:
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

    def update_trailing_stops(
        self,
        df: pd.DataFrame,
        current_index: int,
        current_price: float,
    ) -> None:
        """Update trailing stops for all positions.

        Args:
            df: DataFrame with market data.
            current_index: Current candle index.
            current_price: Current market price.
        """
        if self.trailing_stop_policy is None:
            return

        # Determine ATR if available
        atr_value = None
        try:
            if "atr" in df.columns and current_index < len(df):
                val = df["atr"].iloc[current_index]
                atr_value = float(val) if val is not None and not pd.isna(val) else None
        except Exception:
            atr_value = None

        for order_id, position in self.position_tracker.positions.items():
            side_str = position.side.value
            existing_sl = position.stop_loss

            new_stop, activated, breakeven = self.trailing_stop_policy.update_trailing_stop(
                side=side_str,
                entry_price=float(position.entry_price),
                current_price=float(current_price),
                existing_stop=float(existing_sl) if existing_sl is not None else None,
                position_fraction=float(position.size),
                atr=atr_value,
                trailing_activated=bool(position.trailing_stop_activated),
                breakeven_triggered=bool(position.breakeven_triggered),
            )

            # Update position via tracker
            changed = self.position_tracker.update_trailing_stop(
                order_id=order_id,
                new_stop_loss=new_stop,
                activated=activated,
                breakeven_triggered=breakeven,
            )

            if changed:
                logger.info(
                    "Trailing stop updated for %s %s: SL=%.4f (activated=%s, BE=%s)",
                    position.symbol,
                    side_str,
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

        Args:
            df: DataFrame with market data.
            current_index: Current candle index.
            current_price: Current market price.
            current_balance: Current account balance.
        """
        if self.partial_manager is None:
            return

        for order_id, position in list(self.position_tracker.positions.items()):
            # Build position state
            state = self._build_position_state(position)

            # Calculate P&L percentage
            if position.side == PositionSide.LONG:
                pnl_pct = (current_price - position.entry_price) / position.entry_price
            else:
                pnl_pct = (position.entry_price - current_price) / position.entry_price

            # Check for partial exits
            exit_action = self.partial_manager.check_partial_exit(pnl_pct, state)
            if exit_action is not None:
                self._execute_partial_exit(
                    order_id=order_id,
                    position=position,
                    delta_fraction=exit_action.exit_fraction_of_current,
                    price=current_price,
                    target_level=exit_action.target_level,
                    fraction_of_original=exit_action.exit_fraction_of_original,
                    current_balance=current_balance,
                )

            # Check for scale-ins
            scale_action = self.partial_manager.check_scale_in(pnl_pct, state)
            if scale_action is not None:
                self._execute_scale_in(
                    order_id=order_id,
                    position=position,
                    delta_fraction=scale_action.add_fraction_of_original,
                    price=current_price,
                    threshold_level=scale_action.threshold_level,
                    fraction_of_original=scale_action.add_fraction_of_original,
                )

    def _build_position_state(self, position: LivePosition) -> Any:
        """Build PositionState for partial operations.

        Args:
            position: Position to build state for.

        Returns:
            PositionState for partial manager.
        """
        from src.position_management.partial_manager import PositionState

        return PositionState(
            original_size=float(position.original_size or position.size),
            current_size=float(position.current_size or position.size),
            partial_exits_taken=int(position.partial_exits_taken),
            scale_ins_taken=int(position.scale_ins_taken),
        )

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
        )

        if result is not None:
            # Update risk manager
            if self.risk_manager is not None:
                try:
                    self.risk_manager.adjust_position_after_partial_exit(
                        position.symbol, delta_fraction
                    )
                except Exception as e:
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
                except Exception as e:
                    logger.debug("Risk manager scale-in accounting failed: %s", e)
