"""EntryHandler processes entry signals and coordinates entry execution.

Handles entry signal processing, position sizing, risk adjustments,
and coordination with the execution engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.backtesting.models import ActiveTrade
from src.strategies.components import SignalDirection

if TYPE_CHECKING:
    from src.backtesting.execution.execution_engine import ExecutionEngine
    from src.backtesting.execution.position_tracker import PositionTracker
    from src.position_management.dynamic_risk import DynamicRiskManager
    from src.risk.risk_manager import RiskManager
    from src.strategies.components import Strategy as ComponentStrategy

logger = logging.getLogger(__name__)


@dataclass
class EntrySignalResult:
    """Result of processing an entry signal."""

    should_enter: bool
    side: str | None = None
    size_fraction: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    reasons: list[str] | None = None
    component_notional: float | None = None


@dataclass
class EntryExecutionResult:
    """Result of executing an entry."""

    trade: ActiveTrade | None = None
    entry_fee: float = 0.0
    slippage_cost: float = 0.0
    executed: bool = False
    pending: bool = False
    reasons: list[str] | None = None


class EntryHandler:
    """Processes entry signals and coordinates entry execution.

    This class encapsulates entry-related logic including:
    - Runtime entry signal processing
    - Position sizing with risk adjustments
    - Correlation control
    - Dynamic risk adjustments
    - Stop loss and take profit calculation
    """

    def __init__(
        self,
        execution_engine: ExecutionEngine,
        position_tracker: PositionTracker,
        risk_manager: RiskManager,
        component_strategy: ComponentStrategy | None = None,
        dynamic_risk_manager: DynamicRiskManager | None = None,
        correlation_handler: Any | None = None,
        default_take_profit_pct: float | None = None,
    ) -> None:
        """Initialize entry handler.

        Args:
            execution_engine: Engine for executing entries.
            position_tracker: Tracker for position state.
            risk_manager: Risk manager for position sizing.
            component_strategy: Component strategy for signals.
            dynamic_risk_manager: Manager for dynamic risk adjustments.
            correlation_handler: Handler for correlation-based sizing.
            default_take_profit_pct: Default take profit percentage.
        """
        self.execution_engine = execution_engine
        self.position_tracker = position_tracker
        self.risk_manager = risk_manager
        self.component_strategy = component_strategy
        self.dynamic_risk_manager = dynamic_risk_manager
        self.correlation_handler = correlation_handler
        self.default_take_profit_pct = default_take_profit_pct

    def set_component_strategy(self, strategy: ComponentStrategy | None) -> None:
        """Update the component strategy (for regime switching).

        Args:
            strategy: New component strategy.
        """
        self.component_strategy = strategy

    def process_runtime_decision(
        self,
        runtime_decision: Any,
        balance: float,
        current_price: float,
        current_time: datetime,
        df: pd.DataFrame,
        index: int,
        symbol: str,
        timeframe: str,
        peak_balance: float | None = None,
        trading_session_id: int | None = None,
    ) -> EntrySignalResult:
        """Process a runtime decision and determine entry parameters.

        Args:
            runtime_decision: Decision from strategy runtime.
            balance: Current account balance.
            current_price: Current market price.
            current_time: Current timestamp.
            df: DataFrame with market data.
            index: Current candle index.
            symbol: Trading symbol.
            timeframe: Candle timeframe.
            peak_balance: Peak balance for drawdown calculations.
            trading_session_id: Session ID for logging.

        Returns:
            EntrySignalResult with entry decision and parameters.
        """
        reasons = []

        # Extract entry side and size from decision
        entry_side, size_fraction = self._extract_entry_plan(runtime_decision, balance)

        if entry_side is None or size_fraction <= 0:
            reasons.append("runtime_hold")
            reasons.append(f"balance_{balance:.2f}")
            return EntrySignalResult(
                should_enter=False,
                reasons=reasons,
            )

        # Apply correlation control if available
        if self.correlation_handler is not None and size_fraction > 0:
            size_fraction = self.correlation_handler.apply_correlation_control(
                symbol=symbol,
                timeframe=timeframe,
                df=df,
                index=index,
                candidate_fraction=size_fraction,
            )

        # Apply dynamic risk adjustments
        if self.dynamic_risk_manager is not None and size_fraction > 0:
            size_fraction = self._apply_dynamic_risk(
                original_size=size_fraction,
                current_time=current_time,
                balance=balance,
                peak_balance=peak_balance or balance,
                trading_session_id=trading_session_id,
            )

        if size_fraction <= 0:
            reasons.append("size_reduced_to_zero")
            return EntrySignalResult(
                should_enter=False,
                reasons=reasons,
            )

        # Calculate stop loss and take profit
        sl_pct, tp_pct = self._calculate_sl_tp_pct(
            current_price=current_price,
            entry_side=entry_side,
            runtime_decision=runtime_decision,
        )

        # Build entry reasons
        reasons.append("runtime_entry")
        reasons.append(f"side_{entry_side}")
        reasons.append(f"position_size_{size_fraction:.4f}")
        reasons.append(f"balance_{balance:.2f}")

        metadata = getattr(runtime_decision, "metadata", {}) or {}
        reasons.append(f"enter_short_{bool(metadata.get('enter_short'))}")

        # Calculate component notional
        component_notional = (
            float(runtime_decision.position_size)
            if runtime_decision and hasattr(runtime_decision, "position_size")
            else size_fraction * balance
        )

        # Calculate SL/TP prices
        if entry_side == "long":
            stop_loss = current_price * (1 - sl_pct)
            take_profit = current_price * (1 + tp_pct)
        else:
            stop_loss = current_price * (1 + sl_pct)
            take_profit = current_price * (1 - tp_pct)

        return EntrySignalResult(
            should_enter=True,
            side=entry_side,
            size_fraction=size_fraction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasons=reasons,
            component_notional=component_notional,
        )

    def execute_entry(
        self,
        signal: EntrySignalResult,
        symbol: str,
        current_price: float,
        current_time: datetime,
        balance: float,
    ) -> EntryExecutionResult:
        """Execute an entry based on the signal result.

        Args:
            signal: Entry signal with parameters.
            symbol: Trading symbol.
            current_price: Current market price.
            current_time: Current timestamp.
            balance: Current account balance.

        Returns:
            EntryExecutionResult with execution details.
        """
        if not signal.should_enter or signal.side is None:
            return EntryExecutionResult(
                executed=False,
                reasons=signal.reasons,
            )

        # Calculate SL/TP percentages for pending order
        default_sl_pct = 0.05
        default_tp_pct = 0.04
        if signal.side == "long":
            sl_pct = (
                (current_price - signal.stop_loss) / current_price
                if signal.stop_loss
                else default_sl_pct
            )
            tp_pct = (
                (signal.take_profit - current_price) / current_price
                if signal.take_profit
                else default_tp_pct
            )
        else:
            sl_pct = (
                (signal.stop_loss - current_price) / current_price
                if signal.stop_loss
                else default_sl_pct
            )
            tp_pct = (
                (current_price - signal.take_profit) / current_price
                if signal.take_profit
                else default_tp_pct
            )

        # Use next-bar execution if enabled
        if self.execution_engine.use_next_bar_execution:
            self.execution_engine.queue_entry(
                side=signal.side,
                size_fraction=signal.size_fraction,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                signal_price=current_price,
                signal_time=current_time,
                component_notional=signal.component_notional,
            )
            return EntryExecutionResult(
                pending=True,
                reasons=signal.reasons,
            )

        # Immediate execution
        result = self.execution_engine.execute_immediate_entry(
            symbol=symbol,
            side=signal.side,
            size_fraction=signal.size_fraction,
            current_price=current_price,
            current_time=current_time,
            balance=balance,
            stop_loss=signal.stop_loss or current_price * 0.95,
            take_profit=signal.take_profit or current_price * 1.04,
            component_notional=signal.component_notional,
        )

        if result.executed and result.trade:
            # Open position in tracker
            self.position_tracker.open_position(result.trade)

            # Update risk manager
            try:
                self.risk_manager.update_position(
                    symbol=symbol,
                    side=signal.side,
                    size=signal.size_fraction,
                    entry_price=result.trade.entry_price,
                )
            except Exception as e:
                logger.warning(
                    "Failed to update risk manager for %s on %s: %s",
                    signal.side,
                    symbol,
                    e,
                )

        return EntryExecutionResult(
            trade=result.trade,
            entry_fee=result.entry_fee,
            slippage_cost=result.slippage_cost,
            executed=result.executed,
            reasons=signal.reasons,
        )

    def process_pending_entry(
        self,
        symbol: str,
        open_price: float,
        current_time: datetime,
        balance: float,
    ) -> EntryExecutionResult:
        """Process a pending entry on bar open.

        Args:
            symbol: Trading symbol.
            open_price: Opening price of current bar.
            current_time: Current timestamp.
            balance: Current account balance.

        Returns:
            EntryExecutionResult with execution details.
        """
        if not self.execution_engine.has_pending_entry:
            return EntryExecutionResult(executed=False)

        result = self.execution_engine.execute_pending_entry(
            symbol=symbol,
            open_price=open_price,
            current_time=current_time,
            balance=balance,
        )

        if result.executed and result.trade:
            # Open position in tracker
            self.position_tracker.open_position(result.trade)

            # Update risk manager
            try:
                self.risk_manager.update_position(
                    symbol=symbol,
                    side=result.trade.side,
                    size=result.trade.size,
                    entry_price=result.trade.entry_price,
                )
            except Exception as e:
                logger.warning(
                    "Failed to update risk manager for pending entry on %s: %s",
                    symbol,
                    e,
                )

        return EntryExecutionResult(
            trade=result.trade,
            entry_fee=result.entry_fee,
            slippage_cost=result.slippage_cost,
            executed=result.executed,
        )

    def _extract_entry_plan(
        self,
        decision: Any,
        balance: float,
    ) -> tuple[str | None, float]:
        """Extract entry side and size from runtime decision.

        Args:
            decision: Runtime decision from strategy.
            balance: Current account balance.

        Returns:
            Tuple of (side, size_fraction).
        """
        if decision is None:
            return None, 0.0

        if balance <= 0:
            return None, 0.0

        # Check signal direction
        if decision.signal.direction == SignalDirection.HOLD or decision.position_size <= 0:
            return None, 0.0

        # Check for short entry authorization
        metadata = getattr(decision, "metadata", {}) or {}
        if decision.signal.direction == SignalDirection.SELL and not bool(
            metadata.get("enter_short")
        ):
            return None, 0.0

        # Determine side
        side = "long" if decision.signal.direction == SignalDirection.BUY else "short"

        # Calculate size fraction
        size_fraction = float(decision.position_size) / float(balance)
        size_fraction = max(0.0, min(1.0, size_fraction))

        return side, size_fraction

    def _calculate_sl_tp_pct(
        self,
        current_price: float,
        entry_side: str,
        runtime_decision: Any,
    ) -> tuple[float, float]:
        """Calculate stop loss and take profit percentages.

        Args:
            current_price: Current market price.
            entry_side: 'long' or 'short'.
            runtime_decision: Runtime decision for regime context.

        Returns:
            Tuple of (sl_pct, tp_pct).
        """
        # Try to get stop loss from strategy
        sl_pct = 0.05  # Default 5%
        if self.component_strategy is not None:
            try:
                stop_loss_price = self.component_strategy.get_stop_loss_price(
                    current_price,
                    runtime_decision.signal if runtime_decision else None,
                    runtime_decision.regime if runtime_decision else None,
                )
                if entry_side == "long":
                    sl_pct = (current_price - stop_loss_price) / current_price
                else:
                    sl_pct = (stop_loss_price - current_price) / current_price
                sl_pct = max(0.01, min(0.20, sl_pct))  # Clamp 1-20%
            except Exception:
                pass

        # Get take profit
        tp_pct = self.default_take_profit_pct
        if tp_pct is None and self.component_strategy is not None:
            tp_pct = getattr(self.component_strategy, "take_profit_pct", 0.04)
        if tp_pct is None:
            tp_pct = 0.04

        return sl_pct, tp_pct

    def _apply_dynamic_risk(
        self,
        original_size: float,
        current_time: datetime,
        balance: float,
        peak_balance: float,
        trading_session_id: int | None,
    ) -> float:
        """Apply dynamic risk adjustments to position size.

        Args:
            original_size: Original position size fraction.
            current_time: Current timestamp.
            balance: Current account balance.
            peak_balance: Peak account balance.
            trading_session_id: Session ID for logging.

        Returns:
            Adjusted position size fraction.
        """
        if self.dynamic_risk_manager is None:
            return original_size

        try:
            adjustments = self.dynamic_risk_manager.calculate_dynamic_risk_adjustments(
                current_balance=balance,
                peak_balance=peak_balance,
                session_id=trading_session_id,
            )

            adjusted_size = original_size * adjustments.position_size_factor

            # Log significant adjustments
            if abs(adjustments.position_size_factor - 1.0) > 0.1:
                logger.debug(
                    "Dynamic risk adjustment at %s: size factor=%.2f, reason=%s",
                    current_time,
                    adjustments.position_size_factor,
                    adjustments.primary_reason,
                )

            return adjusted_size

        except Exception as e:
            logger.warning("Failed to apply dynamic risk adjustment: %s", e)
            return original_size
