"""LiveEntryHandler processes entry signals and coordinates entry execution.

Handles entry signal processing, position sizing, risk adjustments,
and coordination with the execution engine for live trading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.config.constants import (
    DEFAULT_MAX_POSITION_SIZE,
    DEFAULT_MAX_STOP_LOSS_PCT,
    DEFAULT_MIN_STOP_LOSS_PCT,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
)
from src.engines.live.execution.execution_engine import LiveExecutionEngine
from src.engines.live.execution.position_tracker import LivePosition, PositionSide
from src.engines.shared.dynamic_risk_handler import DynamicRiskHandler
from src.strategies.components import SignalDirection
from src.utils.bounds import clamp_fraction
from src.utils.price_targets import PriceTargetCalculator

if TYPE_CHECKING:
    from src.engines.shared.correlation_handler import CorrelationHandler
    from src.position_management.dynamic_risk import DynamicRiskManager
    from src.risk.risk_manager import RiskManager
    from src.strategies.components import Strategy as ComponentStrategy

logger = logging.getLogger(__name__)


@dataclass
class LiveEntrySignal:
    """Result of processing an entry signal."""

    should_enter: bool
    side: PositionSide | None = None
    size_fraction: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    reasons: list[str] | None = None
    signal_strength: float = 0.0
    signal_confidence: float = 0.0


@dataclass
class LiveEntryResult:
    """Result of executing an entry."""

    position: LivePosition | None = None
    entry_fee: float = 0.0
    slippage_cost: float = 0.0
    executed: bool = False
    reasons: list[str] | None = None
    error: str | None = None


class LiveEntryHandler:
    """Processes entry signals and coordinates entry execution for live trading.

    This class encapsulates entry-related logic including:
    - Runtime entry signal processing
    - Position sizing with risk adjustments
    - Dynamic risk adjustments
    - Stop loss and take profit calculation
    """

    def __init__(
        self,
        execution_engine: LiveExecutionEngine,
        risk_manager: RiskManager | None = None,
        component_strategy: ComponentStrategy | None = None,
        dynamic_risk_manager: DynamicRiskManager | None = None,
        correlation_handler: CorrelationHandler | None = None,
        max_position_size: float = DEFAULT_MAX_POSITION_SIZE,
        default_take_profit_pct: float | None = None,
    ) -> None:
        """Initialize entry handler.

        Args:
            execution_engine: Engine for executing entries.
            risk_manager: Risk manager for position sizing.
            component_strategy: Component strategy for signals.
            dynamic_risk_manager: Manager for dynamic risk adjustments.
            correlation_handler: Handler for correlation-based position sizing.
            max_position_size: Maximum position size as fraction.
            default_take_profit_pct: Default take profit percentage.
        """
        self.execution_engine = execution_engine
        self.risk_manager = risk_manager
        self.component_strategy = component_strategy
        self.dynamic_risk_manager = dynamic_risk_manager
        self.correlation_handler = correlation_handler
        self.max_position_size = max_position_size
        self.default_take_profit_pct = default_take_profit_pct
        # Use shared DynamicRiskHandler for consistent risk adjustment logic
        self._dynamic_risk_handler = DynamicRiskHandler(dynamic_risk_manager)

    def set_component_strategy(self, strategy: ComponentStrategy | None) -> None:
        """Update the component strategy (for hot-swapping).

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
        symbol: str | None = None,
        timeframe: str | None = None,
        df: pd.DataFrame | None = None,
        index: int | None = None,
        peak_balance: float | None = None,
        trading_session_id: int | None = None,
    ) -> LiveEntrySignal:
        """Process a runtime decision and determine entry parameters.

        Args:
            runtime_decision: Decision from strategy runtime.
            balance: Current account balance.
            current_price: Current market price.
            current_time: Current timestamp.
            symbol: Trading symbol (for correlation control).
            timeframe: Trading timeframe (for correlation control).
            df: Market data DataFrame (for correlation control).
            index: Current candle index (for correlation control).
            peak_balance: Peak balance for drawdown calculations.
            trading_session_id: Session ID for logging.

        Returns:
            LiveEntrySignal with entry decision and parameters.
        """
        reasons = []

        # Validate critical inputs before processing
        if balance <= 0:
            reasons.append(f"invalid_balance_{balance:.2f}")
            return LiveEntrySignal(should_enter=False, reasons=reasons)

        if current_price <= 0:
            reasons.append(f"invalid_price_{current_price:.8f}")
            return LiveEntrySignal(should_enter=False, reasons=reasons)

        # Extract entry side and size from decision
        entry_side, size_fraction = self._extract_entry_plan(runtime_decision, balance)

        if entry_side is None or size_fraction <= 0:
            reasons.append("runtime_hold")
            reasons.append(f"balance_{balance:.2f}")
            return LiveEntrySignal(
                should_enter=False,
                reasons=reasons,
            )

        # Enforce maximum position size to prevent over-concentration risk
        # (matches backtest engine behavior for backtest-live parity)
        size_fraction = min(size_fraction, self.max_position_size)

        # Apply correlation control if available
        # (matches backtest engine behavior for backtest-live parity)
        if (
            self.correlation_handler is not None
            and size_fraction > 0
            and symbol is not None
            and timeframe is not None
            and df is not None
            and index is not None
        ):
            size_fraction = self.correlation_handler.apply_correlation_control(
                symbol=symbol,
                timeframe=timeframe,
                df=df,
                index=index,
                candidate_fraction=size_fraction,
            )

        # Reduce position size during drawdown or adverse market conditions
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
            return LiveEntrySignal(
                should_enter=False,
                reasons=reasons,
            )

        # Compute protective stop loss and profit target for risk management
        sl_price, tp_price = self._calculate_sl_tp(
            current_price=current_price,
            entry_side=entry_side,
            runtime_decision=runtime_decision,
        )

        # Capture signal quality metrics for trade logging and analysis
        signal_strength = 0.0
        signal_confidence = 0.0
        if runtime_decision is not None and hasattr(runtime_decision, "signal"):
            signal_strength = runtime_decision.signal.strength
            signal_confidence = runtime_decision.signal.confidence

        # Record entry context for debugging and post-trade analysis
        reasons.append("runtime_entry")
        reasons.append(f"side_{entry_side.value}")
        reasons.append(f"position_size_{size_fraction:.4f}")
        reasons.append(f"balance_{balance:.2f}")

        metadata = getattr(runtime_decision, "metadata", {}) or {}
        reasons.append(f"enter_short_{bool(metadata.get('enter_short'))}")

        return LiveEntrySignal(
            should_enter=True,
            side=entry_side,
            size_fraction=size_fraction,
            stop_loss=sl_price,
            take_profit=tp_price,
            reasons=reasons,
            signal_strength=signal_strength,
            signal_confidence=signal_confidence,
        )

    def execute_entry(
        self,
        signal: LiveEntrySignal,
        symbol: str,
        current_price: float,
        balance: float,
    ) -> LiveEntryResult:
        """Execute an entry based on the signal result.

        Args:
            signal: Entry signal with parameters.
            symbol: Trading symbol.
            current_price: Current market price.
            balance: Current account balance.

        Returns:
            LiveEntryResult with execution details.
        """
        if not signal.should_enter or signal.side is None:
            return LiveEntryResult(
                executed=False,
                reasons=signal.reasons,
            )

        # Execute via execution engine
        exec_result = self.execution_engine.execute_entry(
            symbol=symbol,
            side=signal.side,
            size_fraction=signal.size_fraction,
            base_price=current_price,
            balance=balance,
        )

        if not exec_result.success:
            return LiveEntryResult(
                executed=False,
                reasons=signal.reasons,
                error=exec_result.error,
            )

        # CRITICAL: Subtract entry fee from entry_balance to match backtest behavior.
        # This ensures P&L calculations use the same basis in both engines.
        entry_balance = balance - exec_result.entry_fee
        # Create position with actual quantity from execution
        position = LivePosition(
            symbol=symbol,
            side=signal.side,
            size=signal.size_fraction,
            entry_price=exec_result.executed_price,
            entry_time=datetime.now(UTC),
            entry_balance=entry_balance,
            quantity=exec_result.quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            order_id=exec_result.order_id,
            original_size=signal.size_fraction,
            current_size=signal.size_fraction,
        )

        return LiveEntryResult(
            position=position,
            entry_fee=exec_result.entry_fee,
            slippage_cost=exec_result.slippage_cost,
            executed=True,
            reasons=signal.reasons,
        )

    def _extract_entry_plan(
        self,
        decision: Any,
        balance: float,
    ) -> tuple[PositionSide | None, float]:
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
        if decision.signal.direction == SignalDirection.BUY:
            side = PositionSide.LONG
        else:
            side = PositionSide.SHORT

        # Calculate size fraction using shared utility for parity with backtest
        size_fraction = float(decision.position_size) / float(balance)
        size_fraction = clamp_fraction(size_fraction)

        return side, size_fraction

    def _calculate_sl_tp(
        self,
        current_price: float,
        entry_side: PositionSide,
        runtime_decision: Any,
    ) -> tuple[float | None, float | None]:
        """Calculate stop loss and take profit prices.

        Args:
            current_price: Current market price.
            entry_side: Entry side (LONG or SHORT).
            runtime_decision: Runtime decision for regime context.

        Returns:
            Tuple of (stop_loss_price, take_profit_price).
        """
        # Validate current_price to prevent division by zero
        if current_price <= 0:
            logger.warning(
                "Invalid current_price (%.8f) in _calculate_sl_tp - cannot compute SL/TP",
                current_price,
            )
            return None, None

        # Default percentages
        default_sl_pct = DEFAULT_STOP_LOSS_PCT
        tp_pct = self.default_take_profit_pct or DEFAULT_TAKE_PROFIT_PCT

        # Try to get stop loss from strategy
        sl_pct = default_sl_pct
        if self.component_strategy is not None:
            try:
                signal = runtime_decision.signal if runtime_decision else None
                regime = runtime_decision.regime if runtime_decision else None
                stop_loss_price = self.component_strategy.get_stop_loss_price(
                    current_price, signal, regime
                )
                if entry_side == PositionSide.LONG:
                    sl_pct = (current_price - stop_loss_price) / current_price
                else:
                    sl_pct = (stop_loss_price - current_price) / current_price
                sl_pct = max(DEFAULT_MIN_STOP_LOSS_PCT, min(DEFAULT_MAX_STOP_LOSS_PCT, sl_pct))
            except (AttributeError, ValueError, TypeError, ZeroDivisionError):
                pass

        # Calculate prices using shared calculator
        side_str = "long" if entry_side == PositionSide.LONG else "short"
        stop_loss, take_profit = PriceTargetCalculator.sl_tp(
            entry_price=current_price,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            side=side_str,
        )

        return stop_loss, take_profit

    def _apply_dynamic_risk(
        self,
        original_size: float,
        current_time: datetime,
        balance: float,
        peak_balance: float,
        trading_session_id: int | None,
    ) -> float:
        """Apply dynamic risk adjustments to position size.

        Delegates to shared DynamicRiskHandler for consistent logic
        between backtest and live engines.

        Args:
            original_size: Original position size fraction.
            current_time: Current timestamp.
            balance: Current account balance.
            peak_balance: Peak account balance.
            trading_session_id: Session ID for logging.

        Returns:
            Adjusted position size fraction.
        """
        # Update handler's manager in case it changed
        self._dynamic_risk_handler.set_manager(self.dynamic_risk_manager)
        return self._dynamic_risk_handler.apply_dynamic_risk(
            original_size=original_size,
            current_time=current_time,
            balance=balance,
            peak_balance=peak_balance,
            trading_session_id=trading_session_id,
        )

    def get_dynamic_risk_adjustments(self) -> list[dict]:
        """Get and clear dynamic risk adjustments tracked by this handler.

        Returns:
            List of dynamic risk adjustment records.
        """
        return self._dynamic_risk_handler.get_adjustments(clear=True)
