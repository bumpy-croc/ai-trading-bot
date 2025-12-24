"""LivePositionTracker manages active position state and MFE/MAE tracking.

Centralizes multi-position lifecycle management including partial exits,
scale-ins, and performance metric tracking for live trading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from src.config.constants import (
    DEFAULT_MFE_MAE_PRECISION_DECIMALS,
    DEFAULT_MFE_MAE_UPDATE_FREQUENCY_SECONDS,
)
from src.performance.metrics import Side, cash_pnl, pnl_percent
from src.position_management.mfe_mae_tracker import MFEMAETracker, MFEMetrics

if TYPE_CHECKING:
    from src.database.manager import DatabaseManager

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position direction."""

    LONG = "long"
    SHORT = "short"


@dataclass
class LivePosition:
    """Represents an active trading position in live trading.

    Mirrors the structure needed for live trading with all fields
    for tracking position state, P&L, and partial operations.
    """

    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    entry_time: datetime
    entry_balance: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    order_id: str | None = None
    # Partial operations runtime state
    original_size: float | None = None
    current_size: float | None = None
    partial_exits_taken: int = 0
    scale_ins_taken: int = 0
    last_partial_exit_price: float | None = None
    last_scale_in_price: float | None = None
    # Trailing stop state
    trailing_stop_activated: bool = False
    trailing_stop_price: float | None = None
    breakeven_triggered: bool = False


@dataclass
class PositionCloseResult:
    """Result of closing a position."""

    realized_pnl: float
    realized_pnl_percent: float
    exit_price: float
    exit_time: datetime
    mfe_mae_metrics: MFEMetrics | None = None


@dataclass
class PartialExitResult:
    """Result of a partial exit operation."""

    realized_pnl: float
    new_current_size: float
    partial_exits_taken: int


@dataclass
class ScaleInResult:
    """Result of a scale-in operation."""

    new_size: float
    new_current_size: float
    scale_ins_taken: int


class LivePositionTracker:
    """Tracks active position state, partial operations, and MFE/MAE metrics.

    This class manages the lifecycle of multiple active positions including:
    - Position state (entry, current size, trailing stop status)
    - Partial exit and scale-in tracking
    - Maximum Favorable/Adverse Excursion (MFE/MAE) metrics
    - Unrealized P&L updates
    """

    def __init__(
        self,
        db_manager: DatabaseManager | None = None,
        mfe_mae_precision: int = DEFAULT_MFE_MAE_PRECISION_DECIMALS,
        mfe_mae_update_frequency: float = DEFAULT_MFE_MAE_UPDATE_FREQUENCY_SECONDS,
    ) -> None:
        """Initialize position tracker.

        Args:
            db_manager: Database manager for persistence (optional).
            mfe_mae_precision: Decimal precision for MFE/MAE calculations.
            mfe_mae_update_frequency: Seconds between MFE/MAE DB persists.
        """
        self._positions: dict[str, LivePosition] = {}
        self._position_db_ids: dict[str, int | None] = {}
        self.db_manager = db_manager
        self.mfe_mae_tracker = MFEMAETracker(precision_decimals=mfe_mae_precision)
        self._mfe_mae_update_frequency = mfe_mae_update_frequency
        self._last_mfe_mae_persist: datetime | None = None

    @property
    def positions(self) -> dict[str, LivePosition]:
        """Get all active positions."""
        return self._positions

    @property
    def position_count(self) -> int:
        """Get count of active positions."""
        return len(self._positions)

    @property
    def position_db_ids(self) -> dict[str, int | None]:
        """Get mapping of order_id to database position ID."""
        return self._position_db_ids

    def has_position(self, order_id: str) -> bool:
        """Check if a position exists by order_id."""
        return order_id in self._positions

    def get_position(self, order_id: str) -> LivePosition | None:
        """Get a position by order_id."""
        return self._positions.get(order_id)

    def reset(self) -> None:
        """Reset tracker state for a new trading session."""
        for order_id in list(self._positions.keys()):
            self.mfe_mae_tracker.clear(order_id)
        self._positions.clear()
        self._position_db_ids.clear()
        self._last_mfe_mae_persist = None

    def open_position(
        self,
        position: LivePosition,
        session_id: int | None = None,
        strategy_name: str | None = None,
    ) -> int | None:
        """Start tracking a new position.

        Args:
            position: The position to track.
            session_id: Trading session ID for database logging.
            strategy_name: Strategy name for database logging.

        Returns:
            Database position ID if logged, None otherwise.
        """
        if position.order_id is None:
            logger.warning("Position has no order_id, cannot track")
            return None

        order_id = position.order_id
        self._positions[order_id] = position
        self.mfe_mae_tracker.clear(order_id)

        logger.debug(
            "Opened %s position at %.2f, size=%.4f, order_id=%s",
            position.side.value,
            position.entry_price,
            position.size,
            order_id,
        )

        # Log to database if available
        db_id = None
        if self.db_manager is not None and session_id is not None:
            try:
                quantity = (
                    (position.size * (position.entry_balance or 0)) / position.entry_price
                    if position.entry_price > 0
                    else 0.0
                )
                db_id = self.db_manager.log_position(
                    symbol=position.symbol,
                    side=position.side.value,
                    entry_price=position.entry_price,
                    size=position.size,
                    entry_balance=position.entry_balance,
                    strategy_name=strategy_name or "unknown",
                    entry_order_id=order_id,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    quantity=quantity,
                    session_id=session_id,
                    trailing_stop_activated=False,
                    trailing_stop_price=None,
                    breakeven_triggered=False,
                )
                # Initialize partial fields in DB
                self.db_manager.update_position(
                    position_id=db_id,
                    original_size=position.size,
                    current_size=position.size,
                    partial_exits_taken=0,
                    scale_ins_taken=0,
                )
            except Exception as e:
                logger.warning("Failed to log position to database: %s", e)

        self._position_db_ids[order_id] = db_id
        return db_id

    def close_position(
        self,
        order_id: str,
        exit_price: float,
        exit_reason: str,
        basis_balance: float,
    ) -> PositionCloseResult | None:
        """Close a position and compute final trade record.

        Args:
            order_id: Order ID of position to close.
            exit_price: Exit price (after slippage).
            exit_reason: Reason for exit.
            basis_balance: Balance basis for P&L calculation.

        Returns:
            PositionCloseResult with realized P&L and metrics, or None if not found.
        """
        position = self._positions.get(order_id)
        if position is None:
            logger.warning("No position found with order_id: %s", order_id)
            return None

        exit_time = datetime.utcnow()
        fraction = float(
            position.current_size if position.current_size is not None else position.size
        )

        # Calculate P&L
        if position.side == PositionSide.LONG:
            trade_pnl_pct = pnl_percent(
                position.entry_price, exit_price, Side.LONG, fraction
            )
        else:
            trade_pnl_pct = pnl_percent(
                position.entry_price, exit_price, Side.SHORT, fraction
            )

        # Determine balance basis
        entry_balance = position.entry_balance
        if entry_balance is not None and entry_balance > 0:
            actual_basis = float(entry_balance)
        else:
            actual_basis = basis_balance

        realized_pnl = cash_pnl(trade_pnl_pct, actual_basis)

        # Get MFE/MAE metrics before clearing
        metrics = self.mfe_mae_tracker.get_position_metrics(order_id)

        logger.info(
            "Closed %s at %.2f, PnL=%.2f (%.2f%%), reason=%s",
            position.side.value,
            exit_price,
            realized_pnl,
            trade_pnl_pct * 100,
            exit_reason,
        )

        # Clear tracker state
        self.mfe_mae_tracker.clear(order_id)
        del self._positions[order_id]
        self._position_db_ids.pop(order_id, None)

        return PositionCloseResult(
            realized_pnl=realized_pnl,
            realized_pnl_percent=trade_pnl_pct,
            exit_price=exit_price,
            exit_time=exit_time,
            mfe_mae_metrics=metrics,
        )

    def update_pnl(self, current_price: float, fallback_balance: float) -> None:
        """Update unrealized P&L for all positions.

        Args:
            current_price: Current market price.
            fallback_balance: Balance to use if position has no entry_balance.
        """
        for position in self._positions.values():
            fraction = float(
                position.current_size
                if position.current_size is not None
                else position.size
            )
            if fraction <= 0:
                position.unrealized_pnl = 0.0
                position.unrealized_pnl_percent = 0.0
                continue

            basis_balance = (
                float(position.entry_balance)
                if position.entry_balance is not None and position.entry_balance > 0
                else float(fallback_balance)
            )

            if position.side == PositionSide.LONG:
                pnl_pct = pnl_percent(
                    position.entry_price, current_price, Side.LONG, fraction
                )
            else:
                pnl_pct = pnl_percent(
                    position.entry_price, current_price, Side.SHORT, fraction
                )

            position.unrealized_pnl = cash_pnl(pnl_pct, basis_balance)
            position.unrealized_pnl_percent = pnl_pct * 100.0

    def update_mfe_mae(self, current_price: float, persist_to_db: bool = True) -> None:
        """Compute and persist rolling MFE/MAE for all active positions.

        Args:
            current_price: Current market price.
            persist_to_db: Whether to persist to database (throttled).
        """
        now = datetime.utcnow()

        for order_id, position in self._positions.items():
            self.mfe_mae_tracker.update_position_metrics(
                position_key=order_id,
                entry_price=float(position.entry_price),
                current_price=float(current_price),
                side=position.side.value,
                position_fraction=float(position.size),
                current_time=now,
            )

        # Throttle DB persistence to avoid overhead
        if not persist_to_db or self.db_manager is None:
            return

        should_persist = False
        if self._last_mfe_mae_persist is None:
            should_persist = True
        else:
            delta = (now - self._last_mfe_mae_persist).total_seconds()
            should_persist = delta >= self._mfe_mae_update_frequency

        if not should_persist:
            return

        self._last_mfe_mae_persist = now
        for order_id, position in self._positions.items():
            db_id = self._position_db_ids.get(order_id)
            if db_id is None:
                continue
            try:
                metrics = self.mfe_mae_tracker.get_position_metrics(order_id)
                if not metrics:
                    continue
                self.db_manager.update_position(
                    position_id=db_id,
                    current_price=float(current_price),
                    unrealized_pnl=float(position.unrealized_pnl),
                    unrealized_pnl_percent=float(position.unrealized_pnl_percent),
                    mfe=float(metrics.mfe),
                    mae=float(metrics.mae),
                    mfe_price=float(metrics.mfe_price) if metrics.mfe_price else None,
                    mae_price=float(metrics.mae_price) if metrics.mae_price else None,
                    mfe_time=metrics.mfe_time,
                    mae_time=metrics.mae_time,
                )
            except Exception as e:
                logger.debug("MFE/MAE DB update failed for %s: %s", order_id, e)

    def apply_partial_exit(
        self,
        order_id: str,
        delta_fraction: float,
        price: float,
        target_level: int,
        fraction_of_original: float,
        basis_balance: float,
    ) -> PartialExitResult | None:
        """Reduce position size via partial exit.

        Args:
            order_id: Order ID of position.
            delta_fraction: Fraction of current position to exit.
            price: Current market price for P&L calculation.
            target_level: Which profit target level triggered this exit.
            fraction_of_original: Fraction of original position being exited.
            basis_balance: Fallback balance for P&L calculation.

        Returns:
            PartialExitResult with realized P&L and new size.
        """
        position = self._positions.get(order_id)
        if position is None:
            logger.warning("No position found for partial exit: %s", order_id)
            return None

        # Adjust runtime position sizes
        if position.original_size is None:
            position.original_size = position.size
        if position.current_size is None:
            position.current_size = position.size

        position.current_size = max(0.0, float(position.current_size) - float(delta_fraction))
        position.partial_exits_taken += 1
        position.last_partial_exit_price = price

        # Calculate realized P&L on the exited fraction
        if position.side == PositionSide.LONG:
            pnl_pct = pnl_percent(position.entry_price, price, Side.LONG, delta_fraction)
        else:
            pnl_pct = pnl_percent(position.entry_price, price, Side.SHORT, delta_fraction)

        actual_basis = (
            float(position.entry_balance)
            if position.entry_balance is not None and position.entry_balance > 0
            else basis_balance
        )
        realized_pnl = cash_pnl(pnl_pct, actual_basis)

        logger.debug(
            "Partial exit: %.4f of position %s, realized PnL=%.2f",
            delta_fraction,
            order_id,
            realized_pnl,
        )

        # Persist to DB
        db_id = self._position_db_ids.get(order_id)
        if self.db_manager is not None and db_id is not None:
            try:
                self.db_manager.apply_partial_exit_update(
                    position_id=db_id,
                    executed_fraction_of_original=float(fraction_of_original),
                    price=float(price),
                    target_level=int(target_level),
                )
            except Exception as e:
                logger.debug("DB partial-exit update failed: %s", e)

        return PartialExitResult(
            realized_pnl=realized_pnl,
            new_current_size=position.current_size,
            partial_exits_taken=position.partial_exits_taken,
        )

    def apply_scale_in(
        self,
        order_id: str,
        delta_fraction: float,
        price: float,
        threshold_level: int,
        fraction_of_original: float,
        max_position_size: float = 1.0,
    ) -> ScaleInResult | None:
        """Increase position size via scale-in.

        Args:
            order_id: Order ID of position.
            delta_fraction: Additional size fraction to add.
            price: Current market price.
            threshold_level: Which threshold level triggered this scale-in.
            fraction_of_original: Fraction of original being added.
            max_position_size: Maximum allowed position size.

        Returns:
            ScaleInResult with new sizes.
        """
        position = self._positions.get(order_id)
        if position is None:
            logger.warning("No position found for scale-in: %s", order_id)
            return None

        # Adjust runtime position sizes
        if position.original_size is None:
            position.original_size = position.size
        if position.current_size is None:
            position.current_size = position.size

        position.current_size = min(1.0, float(position.current_size) + float(delta_fraction))
        position.size = min(max_position_size, float(position.size) + float(delta_fraction))
        position.scale_ins_taken += 1
        position.last_scale_in_price = price

        logger.debug(
            "Scale-in: +%.4f to position %s, new size=%.4f",
            delta_fraction,
            order_id,
            position.current_size,
        )

        # Persist to DB
        db_id = self._position_db_ids.get(order_id)
        if self.db_manager is not None and db_id is not None:
            try:
                self.db_manager.apply_scale_in_update(
                    position_id=db_id,
                    added_fraction_of_original=float(fraction_of_original),
                    price=float(price),
                    threshold_level=int(threshold_level),
                )
            except Exception as e:
                logger.debug("DB scale-in update failed: %s", e)

        return ScaleInResult(
            new_size=position.size,
            new_current_size=position.current_size,
            scale_ins_taken=position.scale_ins_taken,
        )

    def update_trailing_stop(
        self,
        order_id: str,
        new_stop_loss: float | None,
        activated: bool,
        breakeven_triggered: bool,
    ) -> bool:
        """Update trailing stop state for a position.

        Args:
            order_id: Order ID of position.
            new_stop_loss: New stop loss price.
            activated: Whether trailing stop is now activated.
            breakeven_triggered: Whether breakeven has been triggered.

        Returns:
            True if stop loss was updated.
        """
        position = self._positions.get(order_id)
        if position is None:
            return False

        changed = False

        if new_stop_loss is not None:
            current_sl = position.stop_loss
            # Only update if new stop is better
            if position.side == PositionSide.LONG:
                should_update = current_sl is None or new_stop_loss > float(current_sl)
            else:
                should_update = current_sl is None or new_stop_loss < float(current_sl)

            if should_update:
                position.stop_loss = new_stop_loss
                position.trailing_stop_price = new_stop_loss
                changed = True

        if activated != position.trailing_stop_activated:
            position.trailing_stop_activated = activated
            changed = True

        if breakeven_triggered != position.breakeven_triggered:
            position.breakeven_triggered = breakeven_triggered
            changed = True

        # Persist trailing stop state to DB
        if changed:
            db_id = self._position_db_ids.get(order_id)
            if self.db_manager is not None and db_id is not None:
                try:
                    self.db_manager.update_position(
                        position_id=db_id,
                        trailing_stop_activated=position.trailing_stop_activated,
                        trailing_stop_price=position.trailing_stop_price,
                        breakeven_triggered=position.breakeven_triggered,
                        stop_loss=position.stop_loss,
                    )
                except Exception as e:
                    logger.debug("Failed to persist trailing stop update: %s", e)

        return changed

    def get_position_state(self, order_id: str) -> dict | None:
        """Get current position state for external use.

        Args:
            order_id: Order ID of position.

        Returns:
            Dictionary with position details, or None if not found.
        """
        position = self._positions.get(order_id)
        if position is None:
            return None

        return {
            "order_id": order_id,
            "symbol": position.symbol,
            "side": position.side.value,
            "entry_price": position.entry_price,
            "entry_time": position.entry_time,
            "size": position.size,
            "current_size": position.current_size,
            "original_size": position.original_size,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "unrealized_pnl": position.unrealized_pnl,
            "unrealized_pnl_percent": position.unrealized_pnl_percent,
            "trailing_stop_activated": position.trailing_stop_activated,
            "trailing_stop_price": position.trailing_stop_price,
            "breakeven_triggered": position.breakeven_triggered,
            "partial_exits_taken": position.partial_exits_taken,
            "scale_ins_taken": position.scale_ins_taken,
        }

    def recover_positions(
        self,
        session_id: int,
    ) -> list[LivePosition]:
        """Recover positions from database on restart.

        Args:
            session_id: Trading session ID to recover from.

        Returns:
            List of recovered positions.
        """
        if self.db_manager is None:
            logger.warning("Cannot recover positions without database manager")
            return []

        recovered = []
        try:
            db_positions = self.db_manager.get_open_positions(session_id)
            for db_pos in db_positions:
                position = LivePosition(
                    symbol=db_pos.symbol,
                    side=PositionSide(db_pos.side),
                    size=float(db_pos.size),
                    entry_price=float(db_pos.entry_price),
                    entry_time=db_pos.entry_time,
                    entry_balance=float(db_pos.entry_balance) if db_pos.entry_balance else None,
                    stop_loss=float(db_pos.stop_loss) if db_pos.stop_loss else None,
                    take_profit=float(db_pos.take_profit) if db_pos.take_profit else None,
                    order_id=db_pos.entry_order_id,
                    original_size=float(db_pos.original_size) if db_pos.original_size else None,
                    current_size=float(db_pos.current_size) if db_pos.current_size else None,
                    partial_exits_taken=int(db_pos.partial_exits_taken or 0),
                    scale_ins_taken=int(db_pos.scale_ins_taken or 0),
                    trailing_stop_activated=bool(db_pos.trailing_stop_activated),
                    trailing_stop_price=(
                        float(db_pos.trailing_stop_price) if db_pos.trailing_stop_price else None
                    ),
                    breakeven_triggered=bool(db_pos.breakeven_triggered),
                )
                self._positions[position.order_id] = position
                self._position_db_ids[position.order_id] = db_pos.id
                recovered.append(position)
                logger.info(
                    "Recovered position: %s %s @ %.2f",
                    position.side.value,
                    position.symbol,
                    position.entry_price,
                )
        except Exception as e:
            logger.error("Failed to recover positions: %s", e)

        return recovered
