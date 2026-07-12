"""ExitHandler processes exit signals and coordinates exit execution.

Handles all exit conditions including signal exits, stop loss, take profit,
trailing stops, time-based exits, and partial operations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from src.config.constants import (
    DEFAULT_BASIS_BALANCE_FALLBACK,
    DEFAULT_MAX_PARTIAL_EXITS_PER_CYCLE,
    DEFAULT_MAX_POSITION_SIZE,
)
from src.data_providers.exchange_interface import OrderSide, OrderType
from src.engines.backtest.models import ActiveTrade, Trade
from src.engines.shared.barrier_touch import check_barrier_touch
from src.engines.shared.execution.execution_model import ExecutionModel
from src.engines.shared.execution.market_snapshot import MarketSnapshot
from src.engines.shared.execution.order_intent import OrderIntent
from src.engines.shared.execution.snapshot_builder import (
    build_snapshot_from_candle,
    map_exit_order_side_from_trade,
)
from src.engines.shared.exposure import scale_in_gross_cap_headroom
from src.engines.shared.partial_operations_manager import (
    EPSILON,
    PartialOperationsManager,
)
from src.engines.shared.side_utils import to_side_string
from src.engines.shared.strategy_exit_checker import StrategyExitChecker
from src.engines.shared.trailing_stop_manager import TrailingStopManager
from src.engines.shared.validation import is_position_fully_closed
from src.performance.metrics import Side, pnl_percent

if TYPE_CHECKING:
    from src.engines.backtest.execution.execution_engine import ExecutionEngine
    from src.engines.backtest.execution.position_tracker import PositionTracker
    from src.position_management.early_cut import EarlyCutPolicy
    from src.position_management.time_exits import TimeExitPolicy
    from src.position_management.trailing_stops import TrailingStopPolicy
    from src.risk.risk_manager import RiskManager
    from src.strategies.components.exposure_governor import ExposureGovernor

logger = logging.getLogger(__name__)

# Use centralized constant for partial exits limit (defense-in-depth against malformed policies)
MAX_PARTIAL_EXITS_PER_CYCLE = DEFAULT_MAX_PARTIAL_EXITS_PER_CYCLE
ZERO_VALUE = 0.0


def _resolve_basis_balance(trade: ActiveTrade, balance: float | None) -> float:
    """Resolve the basis balance for P&L and notional calculations.

    Prefers entry_balance stored on the trade; falls back to the
    caller-supplied current balance so calculations scale correctly with
    the actual account size instead of a hard-coded constant.
    """
    entry_balance = getattr(trade, "entry_balance", None)
    if entry_balance is not None and entry_balance > 0:
        return float(entry_balance)
    if balance is not None and balance > 0:
        return float(balance)
    return DEFAULT_BASIS_BALANCE_FALLBACK


@dataclass
class ExitCheckResult:
    """Result of checking exit conditions."""

    should_exit: bool
    exit_reason: str
    exit_price: float
    is_stop_loss: bool = False
    is_take_profit: bool = False
    is_time_limit: bool = False
    is_early_cut: bool = False
    is_signal: bool = False


@dataclass
class PartialOpsResult:
    """Result of partial operations processing."""

    realized_pnl: float
    partial_exits: list[dict]
    scale_ins: list[dict]
    scale_in_fees: float = 0.0


class ExitHandler:
    """Processes exit signals and coordinates exit execution.

    This class encapsulates all exit-related logic including:
    - Runtime exit signal processing
    - Stop loss and take profit checks
    - Trailing stop updates
    - Time-based exit checks
    - Partial exit and scale-in processing
    """

    def __init__(
        self,
        execution_engine: ExecutionEngine,
        position_tracker: PositionTracker,
        risk_manager: RiskManager,
        execution_model: ExecutionModel,
        trailing_stop_policy: TrailingStopPolicy | None = None,
        time_exit_policy: TimeExitPolicy | None = None,
        early_cut_policy: EarlyCutPolicy | None = None,
        partial_manager: PartialOperationsManager | None = None,
        enable_engine_risk_exits: bool = True,
        use_high_low_for_stops: bool = True,
        annual_margin_interest_rate: float = 0.0,
        max_position_size: float = DEFAULT_MAX_POSITION_SIZE,
    ) -> None:
        """Initialize exit handler.

        Args:
            execution_engine: Engine for executing exits.
            position_tracker: Tracker for position state.
            risk_manager: Risk manager for position updates.
            execution_model: Execution model for fill decisions.
            trailing_stop_policy: Policy for trailing stops.
            time_exit_policy: Policy for time-based exits.
            early_cut_policy: MFE-conditioned early-cut policy (default OFF).
            partial_manager: Unified partial operations manager.
            enable_engine_risk_exits: Enable SL/TP checks.
            use_high_low_for_stops: Use high/low for SL/TP detection.
            max_position_size: Maximum position size (balance fraction) that
                scale-ins may grow a position to (parity with live).
            annual_margin_interest_rate: Annual borrow/funding rate as a
                decimal (e.g. ``0.05`` for 5% APR). Defaults to 0.0 (spot
                trading, no carry cost). When > 0, interest is accrued on
                the position notional for the holding period and deducted
                from realized PnL on close — mirroring the live engine's
                ``MarginInterestTracker`` behavior so margin-mode backtests
                do not silently overstate returns.
        """
        if annual_margin_interest_rate < 0 or not math.isfinite(annual_margin_interest_rate):
            raise ValueError(
                f"annual_margin_interest_rate must be non-negative and finite, "
                f"got {annual_margin_interest_rate}"
            )

        self.execution_engine = execution_engine
        self.position_tracker = position_tracker
        self.risk_manager = risk_manager
        self.execution_model = execution_model
        self.trailing_stop_policy = trailing_stop_policy
        self.time_exit_policy = time_exit_policy
        self.early_cut_policy = early_cut_policy
        self.partial_manager = partial_manager
        self.enable_engine_risk_exits = enable_engine_risk_exits
        self.use_high_low_for_stops = use_high_low_for_stops
        self.annual_margin_interest_rate = float(annual_margin_interest_rate)
        self.max_position_size = float(max_position_size)
        # Use shared managers for consistent logic across engines
        self._trailing_stop_manager = TrailingStopManager(trailing_stop_policy)
        self._strategy_exit_checker = StrategyExitChecker()
        # #802 follow-up P3: optional exposure governor to cap scale-in exposure
        # (set by the engine; None => inert). Mirrors the live exit handler.
        self._exposure_governor: ExposureGovernor | None = None

    def configure_exposure_gate(self, exposure_governor: ExposureGovernor | None) -> None:
        """Wire the #802 exposure governor so scale-ins respect the gross cap."""
        self._exposure_governor = exposure_governor

    def _calculate_margin_interest(
        self,
        position_notional: float,
        entry_time: datetime,
        exit_time: datetime,
    ) -> float:
        """Compute margin interest cost for the holding period.

        Mirrors live's ``MarginInterestTracker`` semantics: interest accrues
        on the position notional from entry to exit at the configured annual
        rate. Returns 0.0 when the rate is disabled or inputs are invalid,
        so spot-mode backtests are unaffected.
        """
        if self.annual_margin_interest_rate <= 0:
            return 0.0
        if position_notional <= 0 or not math.isfinite(position_notional):
            return 0.0
        try:
            seconds_held = (exit_time - entry_time).total_seconds()
        except (TypeError, ValueError):
            return 0.0
        if seconds_held <= 0:
            return 0.0
        # 365 calendar days, matching live MarginInterestTracker's actual/365
        # convention. Change here only if the live engine ever switches basis.
        seconds_per_year = 365.0 * 24.0 * 3600.0
        interest = (
            position_notional * self.annual_margin_interest_rate * (seconds_held / seconds_per_year)
        )
        # Defense-in-depth: extreme rate × notional × duration could overflow
        # to inf/NaN on degenerate inputs, which would propagate through
        # net_pnl into balance corruption. Treat non-finite as zero.
        if not math.isfinite(interest) or interest < 0:
            return 0.0
        return interest

    def _build_snapshot(
        self,
        symbol: str,
        current_time: datetime,
        current_price: float,
        candle: pd.Series | None,
    ) -> MarketSnapshot:
        """Build a MarketSnapshot from the available candle data."""
        return build_snapshot_from_candle(
            symbol=symbol,
            current_time=current_time,
            current_price=current_price,
            candle=candle,
            default_volume=ZERO_VALUE,
        )

    def _map_exit_order_side(self, trade: ActiveTrade) -> OrderSide:
        """Map a position side to an exit order side."""
        return map_exit_order_side_from_trade(trade)

    def _build_exit_intent(
        self,
        trade: ActiveTrade,
        exit_reason: str,
        order_side: OrderSide,
    ) -> OrderIntent:
        """Build an OrderIntent for the exit based on the exit reason."""
        order_type = OrderType.MARKET
        limit_price = None
        stop_price = None

        if "Stop loss" in exit_reason:
            order_type = OrderType.STOP_LOSS
            stop_price = trade.stop_loss
        elif "Take profit" in exit_reason:
            order_type = OrderType.TAKE_PROFIT
            limit_price = trade.take_profit

        quantity = trade.current_size if trade.current_size is not None else trade.size
        return OrderIntent(
            symbol=trade.symbol,
            side=order_side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            exit_reason=exit_reason,
        )

    def update_trailing_stop(
        self,
        current_price: float,
        df: pd.DataFrame,
        index: int,
    ) -> tuple[bool, str | None]:
        """Update trailing stop for active position.

        Uses shared TrailingStopManager for consistent logic across engines.

        Args:
            current_price: Current market price.
            df: DataFrame with market data.
            index: Current candle index.

        Returns:
            Tuple of (was_updated, log_message).
        """
        trade = self.position_tracker.current_trade
        if trade is None:
            return False, None

        # Use shared trailing stop manager for calculation
        result = self._trailing_stop_manager.update(
            position=trade,
            current_price=current_price,
            df=df,
            index=index,
        )

        if not result.updated:
            return False, None

        # Determine new activation states
        # If breakeven just triggered, set both flags
        new_activated = result.trailing_activated or trade.trailing_stop_activated
        new_breakeven = result.breakeven_triggered or trade.breakeven_triggered

        # Apply update via position tracker (a None stop price means trailing
        # just activated without a stop improvement; flag updates still apply).
        changed = self.position_tracker.update_trailing_stop(
            new_stop_loss=result.new_stop_price,
            activated=new_activated,
            breakeven_triggered=new_breakeven,
        )

        if changed:
            log_msg = (
                f"sl_updated={trade.stop_loss}, "
                f"activated={trade.trailing_stop_activated}, "
                f"breakeven={trade.breakeven_triggered}"
            )
            return True, log_msg

        return False, None

    def check_partial_operations(
        self,
        current_price: float,
        df: pd.DataFrame,
        index: int,
        indicators: dict | None = None,
        balance: float | None = None,
    ) -> PartialOpsResult:
        """Check and execute partial exits and scale-ins.

        Uses unified PartialOperationsManager for consistent logic.

        Args:
            current_price: Current market price.
            df: DataFrame with market data.
            index: Current candle index.
            indicators: Extracted indicators (unused, kept for compatibility).
            balance: Current account balance, used as fallback for basis_balance
                when entry_balance is unavailable.

        Returns:
            PartialOpsResult with realized PnL and actions taken.
        """
        if self.partial_manager is None:
            return PartialOpsResult(realized_pnl=0.0, partial_exits=[], scale_ins=[])

        trade = self.position_tracker.current_trade
        if trade is None:
            return PartialOpsResult(realized_pnl=0.0, partial_exits=[], scale_ins=[])

        realized_pnl = 0.0
        partial_exits = []
        scale_ins = []
        total_scale_in_fees = 0.0

        try:
            # Check partial exits (loop until no more triggers)
            iteration_count = 0
            while iteration_count < MAX_PARTIAL_EXITS_PER_CYCLE:
                result = self.partial_manager.check_partial_exit(
                    position=trade,
                    current_price=current_price,
                )

                if not result.should_exit:
                    break

                # Policy sizes are fractions of the ORIGINAL position.
                # cast: exit_fraction is always set when should_exit is True
                exit_size_of_original = cast(float, result.exit_fraction)
                # cast: BasePosition.__post_init__ auto-initializes current/original
                # size from size, so they are never None after construction.
                current_size = cast(float, trade.current_size)
                original_size = cast(float, trade.original_size)
                if is_position_fully_closed(
                    current_size,
                    original_size,
                    epsilon=EPSILON,
                ):
                    logger.debug("Position fully closed, skipping further partial exits")
                    break

                # Convert to balance-fraction units (the PartialExitExecutor
                # contract, identical to current_size units), capped at what
                # remains of the position.
                exit_size_of_balance = min(
                    exit_size_of_original * original_size,
                    current_size,
                )
                if exit_size_of_balance <= EPSILON:
                    break

                basis_balance = _resolve_basis_balance(trade, balance)

                # Execute partial exit via position tracker
                pnl = self.position_tracker.apply_partial_exit(
                    exit_fraction=exit_size_of_balance,
                    current_price=current_price,
                    basis_balance=basis_balance,
                )
                realized_pnl += pnl

                partial_exits.append(
                    {
                        "size": exit_size_of_balance,
                        "price": current_price,
                        "pnl": pnl,
                        "target_level": result.target_index,
                    }
                )

                # Update risk manager (exposure is tracked in balance-fraction
                # units, same as the size decrement above)
                try:
                    self.risk_manager.adjust_position_after_partial_exit(
                        trade.symbol, exit_size_of_balance
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to adjust risk manager after partial exit for %s: %s",
                        trade.symbol,
                        exc,
                    )

                # Increment after execution to match live engine behavior
                iteration_count += 1

            # A position fully consumed by partial exits must not be revived
            # by a scale-in (zombie position). Re-read sizes: the partial loop
            # above may have just consumed the remainder.
            fully_closed = is_position_fully_closed(
                cast(float, trade.current_size),
                cast(float, trade.original_size),
                epsilon=EPSILON,
            )

            # Check scale-in opportunity
            scale_result = None
            if not fully_closed:
                scale_result = self.partial_manager.check_scale_in(
                    position=trade,
                    current_price=current_price,
                    balance=0.0,  # Unused by manager but required
                )

            if scale_result is not None and scale_result.should_scale:
                # cast: scale_fraction is always set when should_scale is True
                add_size_of_original = cast(float, scale_result.scale_fraction)

                if add_size_of_original > 0:
                    # Calculate effective size respecting risk limits.
                    # cast: BasePosition.__post_init__ auto-initializes original_size
                    # from size, so it is never None after construction.
                    delta_add = add_size_of_original * cast(float, trade.original_size)
                    remaining_daily = max(
                        0.0,
                        self.risk_manager.params.max_daily_risk - self.risk_manager.daily_risk_used,
                    )
                    add_effective = min(delta_add, remaining_daily)

                    # Enforce the engine-wide max-position cap on scale-ins
                    # (parity with live): the daily-risk budget alone would
                    # let a scale-in grow an at-cap position past the cap.
                    if add_effective > 0:
                        current_exposure = cast(float, trade.current_size)
                        headroom = max(0.0, self.max_position_size - current_exposure)
                        if add_effective > headroom:
                            logger.info(
                                "Clamping %s scale-in to max-position headroom: "
                                "requested=%.4f, headroom=%.4f (current=%.4f, cap=%.4f)",
                                trade.symbol,
                                add_effective,
                                headroom,
                                current_exposure,
                                self.max_position_size,
                            )
                            add_effective = headroom

                    # #802 follow-up P3: respect the regime gross exposure cap on
                    # scale-ins too (parity with live; inert unless governor on).
                    if add_effective > 0:
                        gross_headroom = scale_in_gross_cap_headroom(
                            self._exposure_governor, [trade], balance
                        )
                        if gross_headroom is not None and add_effective > gross_headroom:
                            logger.info(
                                "Clamping %s scale-in to gross-exposure cap headroom: "
                                "requested=%.4f, headroom=%.4f",
                                trade.symbol,
                                add_effective,
                                gross_headroom,
                            )
                            add_effective = gross_headroom

                    if add_effective > 0:
                        # Calculate scale-in fees (same as initial entry fees)
                        scale_basis = _resolve_basis_balance(trade, balance)
                        scale_notional = scale_basis * add_effective
                        side_str = to_side_string(trade.side)
                        scale_fee, scale_slippage = self.execution_engine.calculate_scale_in_costs(
                            price=current_price,
                            notional=scale_notional,
                            side=side_str,
                        )
                        total_scale_in_fees += scale_fee

                        self.position_tracker.apply_scale_in(
                            add_effective,
                            max_position_size=self.max_position_size,
                        )

                        scale_ins.append(
                            {
                                "size": add_effective,
                                "price": current_price,
                                "fee": scale_fee,
                                "slippage": scale_slippage,
                            }
                        )

                        try:
                            self.risk_manager.adjust_position_after_scale_in(
                                trade.symbol, add_effective
                            )
                        except Exception as exc:
                            logger.warning(
                                "Failed to adjust risk manager after scale-in for %s: %s",
                                trade.symbol,
                                exc,
                            )

        except (AttributeError, ValueError, KeyError, ZeroDivisionError) as e:
            logger.warning("Partial ops evaluation failed: %s", e)
        except Exception as e:
            logger.warning("Unexpected error in partial ops: %s", e)

        return PartialOpsResult(
            realized_pnl=realized_pnl,
            partial_exits=partial_exits,
            scale_ins=scale_ins,
            scale_in_fees=total_scale_in_fees,
        )

    def check_exit_conditions(
        self,
        runtime_decision: Any,
        candle: pd.Series,
        current_price: float,
        symbol: str,
        component_strategy: Any | None = None,
        df: pd.DataFrame | None = None,
    ) -> ExitCheckResult:
        """Check all exit conditions for the current position.

        Checks in priority order:
        1. Runtime/signal exit
        2. Stop loss (using high/low)
        3. Take profit (using high/low)
        4. Time limit exit
        5. MFE early-cut (only when a policy and candle history are provided)

        Args:
            runtime_decision: Current runtime decision from strategy.
            candle: Current candle data.
            current_price: Current close price.
            symbol: Trading symbol.
            component_strategy: Strategy for exit signal checks.
            df: OHLCV history covering the position's entry; required for the
                MFE early-cut check (the check holds without it).

        Returns:
            ExitCheckResult with exit decision and details.
        """
        trade = self.position_tracker.current_trade
        if trade is None:
            return ExitCheckResult(
                should_exit=False,
                exit_reason="Hold",
                exit_price=current_price,
            )

        # Check runtime exit signal
        exit_signal, runtime_reason = self._check_runtime_exit(
            runtime_decision, symbol, candle, current_price, component_strategy
        )

        # Get high/low for stop checks. Fall back to close price if high/low
        # are NaN (e.g., missing OHLCV data) to prevent NaN comparisons that
        # silently disable stop loss / take profit checks.
        if self.use_high_low_for_stops:
            candle_high = float(candle["high"])
            candle_low = float(candle["low"])
            if not math.isfinite(candle_high):
                candle_high = current_price
            if not math.isfinite(candle_low):
                candle_low = current_price
        else:
            candle_high = current_price
            candle_low = current_price

        # Convert PositionSide enum to string for comparisons
        side_str = to_side_string(trade.side)

        # Check stop loss / take profit using the shared OHLC barrier-touch
        # helper (single source of truth also reused by the training
        # pipeline's triple-barrier label simulator — see
        # src/engines/shared/barrier_touch.py).
        hit_stop_loss = False
        sl_exit_price = current_price
        hit_take_profit = False
        tp_exit_price = current_price
        if self.enable_engine_risk_exits:
            barrier_result = check_barrier_touch(
                side=side_str,
                candle_high=candle_high,
                candle_low=candle_low,
                stop_loss_price=(float(trade.stop_loss) if trade.stop_loss is not None else None),
                take_profit_price=(
                    float(trade.take_profit) if trade.take_profit is not None else None
                ),
            )
            hit_stop_loss = barrier_result.hit_stop_loss
            if hit_stop_loss:
                # When price gaps through the stop, the position could have exited
                # anywhere between the stop price and the candle extreme. Use the
                # worst case (candle low for longs, candle high for shorts) for
                # conservative backtest assumptions.
                sl_exit_price = barrier_result.stop_loss_exit_price
            hit_take_profit = barrier_result.hit_take_profit
            if hit_take_profit:
                tp_exit_price = barrier_result.take_profit_exit_price

        # Check time limit. Capture the policy-specific reason (e.g.
        # "Max holding period", "Weekend flat", "End of day flat") so it
        # propagates into Trade.exit_reason — matching the live engine,
        # which already does `time_reason or "Time exit"`.
        hit_time_limit = False
        time_reason: str | None = None
        if self.time_exit_policy is not None:
            try:
                current_time = candle.name if hasattr(candle, "name") else datetime.now(UTC)
                # Ensure consistent timezone handling - localize naive timestamps to UTC
                # to prevent TypeError when comparing with UTC-aware entry_time
                if hasattr(current_time, "tzinfo") and current_time.tzinfo is None:
                    current_time = current_time.replace(tzinfo=UTC)
                should_time_exit, time_reason = self.time_exit_policy.check_time_exit_conditions(
                    trade.entry_time, current_time
                )
                hit_time_limit = should_time_exit
            except (TypeError, ValueError, AttributeError) as e:
                logger.warning("Time exit check failed: %s", e)

        # Check MFE early-cut (shared policy — identical logic in live).
        hit_early_cut = False
        early_cut_reason: str | None = None
        if self.early_cut_policy is not None and df is not None:
            try:
                current_time = candle.name if hasattr(candle, "name") else datetime.now(UTC)
                if hasattr(current_time, "tzinfo") and current_time.tzinfo is None:
                    current_time = current_time.replace(tzinfo=UTC)
                early_cut_decision = self.early_cut_policy.check_early_cut_conditions(
                    side=side_str,
                    entry_price=float(trade.entry_price),
                    entry_time=trade.entry_time,
                    now_time=current_time,
                    df=df,
                )
                hit_early_cut = early_cut_decision.should_exit
                early_cut_reason = early_cut_decision.reason
            except (TypeError, ValueError, AttributeError) as e:
                logger.warning("Early-cut check failed: %s", e)

        # Determine final exit decision
        should_exit = (
            exit_signal or hit_stop_loss or hit_take_profit or hit_time_limit or hit_early_cut
        )

        # Determine exit reason and price (priority: SL > TP > Time > Early cut > Signal)
        if hit_stop_loss:
            exit_reason = "Stop loss"
            exit_price = sl_exit_price
        elif hit_take_profit:
            exit_reason = "Take profit"
            exit_price = tp_exit_price
        elif hit_time_limit:
            # Use the policy-specific reason for parity with the live engine.
            # Default fallback string also matches live ("Time exit").
            exit_reason = time_reason or "Time exit"
            exit_price = current_price
        elif hit_early_cut:
            exit_reason = early_cut_reason or "Early cut"
            exit_price = current_price
        elif exit_signal:
            exit_reason = runtime_reason
            exit_price = current_price
        else:
            exit_reason = "Hold"
            exit_price = current_price

        return ExitCheckResult(
            should_exit=should_exit,
            exit_reason=exit_reason,
            exit_price=exit_price,
            is_stop_loss=hit_stop_loss,
            is_take_profit=hit_take_profit,
            is_time_limit=hit_time_limit,
            is_early_cut=hit_early_cut and not (hit_stop_loss or hit_take_profit or hit_time_limit),
            is_signal=exit_signal
            and not (hit_stop_loss or hit_take_profit or hit_time_limit or hit_early_cut),
        )

    def _check_runtime_exit(
        self,
        decision: Any,
        symbol: str,
        candle: pd.Series,
        current_price: float,
        component_strategy: Any | None,
    ) -> tuple[bool, str]:
        """Check if runtime decision indicates exit.

        Uses shared StrategyExitChecker for consistent logic across engines.

        Args:
            decision: Runtime decision from strategy.
            symbol: Trading symbol.
            candle: Current candle data.
            current_price: Current close price.
            component_strategy: Strategy for should_exit_position check.

        Returns:
            Tuple of (should_exit, reason).
        """
        trade = self.position_tracker.current_trade
        if decision is None or trade is None:
            return False, "Hold"

        # Extract volume and timestamp from candle for the shared checker
        volume = float(candle.get("volume", 0.0) if hasattr(candle, "get") else 0.0)
        timestamp = candle.name if hasattr(candle, "name") else None

        # Use shared strategy exit checker for consistent logic
        result = self._strategy_exit_checker.check_exit(
            position=trade,
            current_price=current_price,
            runtime_decision=decision,
            component_strategy=component_strategy,
            volume=volume,
            timestamp=timestamp,
        )

        return result.should_exit, result.exit_reason

    def execute_exit(
        self,
        exit_price: float,
        exit_reason: str,
        current_time: datetime,
        current_price: float,
        balance: float,
        symbol: str,
        candle: pd.Series | None = None,
    ) -> tuple[Trade, float, float, float]:
        """Execute exit and close position.

        Args:
            exit_price: Base exit price (before slippage).
            exit_reason: Reason for exit.
            current_time: Current timestamp.
            current_price: Current market price.
            balance: Current account balance.
            symbol: Trading symbol.
            candle: Optional candle data for execution modeling.

        Returns:
            Tuple of (completed_trade, pnl_cash, exit_fee, slippage_cost).
        """
        trade = self.position_tracker.current_trade
        if trade is None:
            raise ValueError("No active position to exit")
        if trade.entry_price <= 0:
            raise ValueError(
                f"Invalid entry_price {trade.entry_price} - cannot calculate exit fees"
            )
        # Validate exit prices to prevent NaN/Infinity propagation into P&L
        if exit_price <= 0 or not math.isfinite(exit_price):
            raise ValueError(f"Invalid exit_price: {exit_price}")
        if current_price <= 0 or not math.isfinite(current_price):
            raise ValueError(f"Invalid current_price: {current_price}")

        order_side = self._map_exit_order_side(trade)
        snapshot = self._build_snapshot(
            symbol=symbol,
            current_time=current_time,
            current_price=current_price,
            candle=candle,
        )
        order_intent = self._build_exit_intent(trade, exit_reason, order_side)
        decision = self.execution_model.decide_fill(order_intent, snapshot)

        base_exit_price = exit_price
        liquidity = None
        if decision.should_fill and decision.fill_price is not None:
            liquidity = decision.liquidity
            base_exit_price = decision.fill_price
        else:
            logger.warning(
                "Exit fill decision fallback for %s: %s",
                symbol,
                decision.reason,
            )

        # Get position notional for fee calculation
        # IMPORTANT: Use exit notional (accounting for price change) for accurate fee calculation.
        # This correctly models real exchange behavior where fees are charged on the actual
        # value at trade time:
        # - Winning positions: selling more valuable assets → higher fee
        # - Losing positions: selling less valuable assets → lower fee
        # This differs from entry fees which use the original notional value.
        entry_balance = getattr(trade, "entry_balance", None)
        basis_balance = (
            float(entry_balance)
            if entry_balance is not None and entry_balance > 0
            else float(balance)
        )
        fraction = float(getattr(trade, "current_size", trade.size))
        entry_notional = basis_balance * fraction
        # Scale by price change to get exit notional (this is intentional and correct)
        position_notional = entry_notional * (base_exit_price / trade.entry_price)

        # Convert PositionSide enum to string for cost calculation
        side_str = to_side_string(trade.side)
        apply_slippage = True

        # Calculate exit costs
        final_exit_price, exit_fee, slippage_cost = self.execution_engine.calculate_exit_costs(
            base_price=base_exit_price,
            side=side_str,
            position_notional=position_notional,
            liquidity=liquidity,
            apply_slippage=apply_slippage,
        )

        # Close position and get completed trade
        close_result = self.position_tracker.close_position(
            exit_price=final_exit_price,
            exit_time=current_time,
            exit_reason=exit_reason,
            basis_balance=basis_balance,
        )

        # Margin/borrow interest accrual over the holding period. Folded into
        # exit_fee so PerformanceTracker.record_trade reports it as a cost,
        # matching live's `record_trade(fee=total_fee + interest_cost, ...)`.
        # Trade.pnl stays gross (price movement only) — same convention as live.
        interest_cost = self._calculate_margin_interest(
            position_notional=position_notional,
            entry_time=trade.entry_time,
            exit_time=current_time,
        )
        if interest_cost > 0:
            logger.debug(
                "Deducted margin interest %.4f from %s PnL (rate=%.4f, held=%.2fh)",
                interest_cost,
                symbol,
                self.annual_margin_interest_rate,
                (current_time - trade.entry_time).total_seconds() / 3600.0,
            )

        # Stash interest_cost on the completed trade's metadata so it can be
        # persisted to the trades.margin_interest_cost DB column via
        # event_logger.log_completed_trade — parity with live, which passes
        # margin_interest_cost=interest_cost directly to db_manager.log_trade
        # (src/engines/live/trading_engine.py:3397-3420).
        if interest_cost > 0 and close_result.trade is not None:
            try:
                close_result.trade.metadata["margin_interest_cost"] = float(interest_cost)
            except (AttributeError, TypeError, ValueError):
                logger.debug("Could not stash margin_interest_cost on completed trade metadata")

        # Subtract exit fee and margin interest from PnL
        net_pnl = close_result.pnl_cash - exit_fee - interest_cost

        # Close position in risk manager
        try:
            self.risk_manager.close_position(symbol)
        except Exception as e:
            logger.warning("Failed to update risk manager on close for %s: %s", symbol, e)

        return close_result.trade, net_pnl, exit_fee + interest_cost, slippage_cost

    def calculate_current_pnl_pct(self, current_price: float) -> float:
        """Calculate current unrealized PnL percentage.

        Uses shared pnl_percent function for consistency with live engine.
        Note: Returns raw P&L percentage (unsized, fraction=1.0) for logging.

        Args:
            current_price: Current market price.

        Returns:
            PnL as percentage of entry price.
        """
        trade = self.position_tracker.current_trade
        if trade is None:
            return 0.0

        # Validate prices before calling pnl_percent to prevent ValueError
        if trade.entry_price <= 0 or current_price <= 0:
            return 0.0

        side_str = to_side_string(trade.side)
        side_enum = Side.LONG if side_str == "long" else Side.SHORT
        return pnl_percent(trade.entry_price, current_price, side_enum, 1.0)
