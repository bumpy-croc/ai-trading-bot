"""LiveExecutionEngine handles order execution with fees and slippage.

Encapsulates the mechanics of trade execution including:
- Fee calculations
- Slippage modeling
- Live order execution (when enabled)
- Paper trading simulation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.data_providers.exchange_interface import OrderSide, OrderType
from src.engines.shared.cost_calculator import CostCalculator
from src.engines.shared.models import PositionSide

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class LiveExecutionResult:
    """Result of a live trade execution."""

    success: bool
    order_id: str | None = None
    executed_price: float | None = None
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    slippage_cost: float = 0.0
    error: str | None = None


@dataclass
class EntryExecutionResult:
    """Result of executing an entry order."""

    success: bool
    order_id: str | None = None
    executed_price: float = 0.0
    position_value: float = 0.0
    quantity: float = 0.0
    entry_fee: float = 0.0
    slippage_cost: float = 0.0
    error: str | None = None


@dataclass
class ExitExecutionResult:
    """Result of executing an exit order."""

    success: bool
    executed_price: float = 0.0
    exit_fee: float = 0.0
    slippage_cost: float = 0.0
    error: str | None = None


class LiveExecutionEngine:
    """Handles order execution with fees and slippage for live trading.

    This class encapsulates execution mechanics including:
    - Fee rate calculations (entry and exit)
    - Slippage modeling (adverse price movement)
    - Live order execution via exchange interface
    - Paper trading simulation mode
    """

    def __init__(
        self,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        enable_live_trading: bool = False,
        exchange_interface: Any = None,
    ) -> None:
        """Initialize execution engine.

        Args:
            fee_rate: Fee rate per trade (0.001 = 0.1%).
            slippage_rate: Slippage rate per trade (0.0005 = 0.05%).
            enable_live_trading: Whether to execute real orders.
            exchange_interface: Exchange provider for live orders.
        """
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.enable_live_trading = enable_live_trading
        self.exchange_interface = exchange_interface

        # Validate that live trading has required exchange interface
        if enable_live_trading and exchange_interface is None:
            raise ValueError(
                "Cannot enable live trading without exchange interface. "
                "Provide exchange_interface or set enable_live_trading=False."
            )

        # Use shared cost calculator for all fee and slippage calculations
        self._cost_calculator = CostCalculator(
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )

    @staticmethod
    def _position_side_to_str(side: PositionSide) -> str:
        """Convert PositionSide enum to string for cost calculations.

        Args:
            side: Position side enum.

        Returns:
            'long' or 'short'.
        """
        return "long" if side == PositionSide.LONG else "short"

    @property
    def total_fees_paid(self) -> float:
        """Get total fees paid across all trades."""
        return self._cost_calculator.total_fees_paid

    @property
    def total_slippage_cost(self) -> float:
        """Get total slippage cost across all trades."""
        return self._cost_calculator.total_slippage_cost

    def reset_tracking(self) -> None:
        """Reset fee and slippage tracking."""
        self._cost_calculator.reset_totals()

    def apply_entry_slippage(self, price: float, side: PositionSide) -> float:
        """Apply slippage to entry price (price moves against us).

        Slippage models the cost of market impact and adverse selection that occurs
        when entering a position, ensuring realistic backtest and live trading results.

        This method is kept for backward compatibility. New code should use
        the shared CostCalculator via calculate_entry_costs.

        Args:
            price: Base price before slippage.
            side: Position side (LONG or SHORT).

        Returns:
            Price after slippage applied.
        """
        # Use simple calculation to preserve backward compatibility
        if side == PositionSide.LONG:
            return price * (1 + self.slippage_rate)
        else:
            return price * (1 - self.slippage_rate)

    def apply_exit_slippage(self, price: float, side: PositionSide) -> float:
        """Apply slippage to exit price (price moves against us).

        Exit slippage accounts for market impact costs when closing positions,
        ensuring P&L calculations reflect realistic execution conditions.

        This method is kept for backward compatibility. New code should use
        the shared CostCalculator via calculate_exit_costs.

        Args:
            price: Base price before slippage.
            side: Position side (LONG or SHORT).

        Returns:
            Price after slippage applied.
        """
        # Use simple calculation to preserve backward compatibility
        if side == PositionSide.LONG:
            return price * (1 - self.slippage_rate)
        else:
            return price * (1 + self.slippage_rate)

    def calculate_entry_fee(self, position_value: float) -> float:
        """Calculate entry fee for a position.

        Args:
            position_value: Notional value of position.

        Returns:
            Fee amount.
        """
        return self._cost_calculator.calculate_fee(position_value)

    def calculate_exit_fee(self, position_notional: float) -> float:
        """Calculate exit fee for a position.

        Args:
            position_notional: Notional value of position at exit.

        Returns:
            Fee amount.
        """
        return self._cost_calculator.calculate_fee(position_notional)

    def calculate_slippage_cost(self, position_value: float) -> float:
        """Calculate slippage cost for a trade.

        Args:
            position_value: Notional value of position.

        Returns:
            Slippage cost amount.
        """
        return abs(position_value * self.slippage_rate)

    def execute_entry(
        self,
        symbol: str,
        side: PositionSide,
        size_fraction: float,
        base_price: float,
        balance: float,
    ) -> EntryExecutionResult:
        """Execute an entry order with fees and slippage.

        Args:
            symbol: Trading symbol.
            side: Position side (LONG or SHORT).
            size_fraction: Position size as fraction of balance.
            base_price: Current market price.
            balance: Account balance.

        Returns:
            EntryExecutionResult with execution details.
        """
        try:
            # Calculate position value and costs using shared cost calculator
            position_value = size_fraction * balance
            side_str = self._position_side_to_str(side)

            cost_result = self._cost_calculator.calculate_entry_costs(
                price=base_price,
                notional=position_value,
                side=side_str,
            )

            executed_price = cost_result.executed_price
            entry_fee = cost_result.fee
            slippage_cost = cost_result.slippage_cost
            quantity = position_value / executed_price if executed_price > 0 else 0.0

            # Execute real order if enabled
            if self.enable_live_trading:
                order_id = self._execute_live_order(symbol, side, position_value, executed_price)
                if not order_id:
                    return EntryExecutionResult(
                        success=False,
                        error="Failed to execute live order",
                    )
            else:
                order_id = f"paper_{int(time.time() * 1000)}"
                logger.info("PAPER TRADE - Would open %s position on %s", side.value, symbol)

            return EntryExecutionResult(
                success=True,
                order_id=order_id,
                executed_price=executed_price,
                position_value=position_value,
                quantity=quantity,
                entry_fee=entry_fee,
                slippage_cost=slippage_cost,
            )

        except (ValueError, ArithmeticError, TypeError) as e:
            logger.error("Failed to execute entry: %s", e, exc_info=True)
            return EntryExecutionResult(
                success=False,
                error=str(e),
            )

    def execute_exit(
        self,
        symbol: str,
        side: PositionSide,
        order_id: str,
        base_price: float,
        position_notional: float,
    ) -> ExitExecutionResult:
        """Execute an exit order with fees and slippage.

        Args:
            symbol: Trading symbol.
            side: Position side (LONG or SHORT).
            order_id: Order ID of position to close.
            base_price: Exit price before slippage.
            position_notional: Notional value of position.

        Returns:
            ExitExecutionResult with execution details.
        """
        try:
            # Calculate costs using shared cost calculator
            side_str = self._position_side_to_str(side)

            cost_result = self._cost_calculator.calculate_exit_costs(
                price=base_price,
                notional=position_notional,
                side=side_str,
            )

            executed_price = cost_result.executed_price
            exit_fee = cost_result.fee
            slippage_cost = cost_result.slippage_cost

            # Execute real order if enabled
            if self.enable_live_trading:
                quantity = position_notional / base_price if base_price > 0 else 0.0
                success = self._close_live_order(
                    symbol,
                    side,
                    quantity,
                    position_notional=position_notional,
                    order_id=order_id,
                )
                if not success:
                    return ExitExecutionResult(
                        success=False,
                        error="Failed to close live order",
                    )
            else:
                logger.info("PAPER TRADE - Would close %s position on %s", side.value, symbol)

            return ExitExecutionResult(
                success=True,
                executed_price=executed_price,
                exit_fee=exit_fee,
                slippage_cost=slippage_cost,
            )

        except (ValueError, ArithmeticError, TypeError) as e:
            logger.error("Failed to execute exit: %s", e, exc_info=True)
            return ExitExecutionResult(
                success=False,
                error=str(e),
            )

    def _execute_live_order(
        self,
        symbol: str,
        side: PositionSide,
        value: float,
        price: float,
    ) -> str | None:
        """Execute a real market order via exchange.

        Args:
            symbol: Trading symbol.
            side: Position side.
            value: Order value.
            price: Expected price.

        Returns:
            Order ID if successful, None otherwise.
        """
        if self.exchange_interface is None:
            logger.warning("No exchange interface configured - using paper order ID")
            return f"real_{int(time.time() * 1000)}"

        try:
            order_side = OrderSide.BUY if side == PositionSide.LONG else OrderSide.SELL
            quantity = value / price if price > 0 else 0.0
            quantity = self._normalize_quantity(symbol, quantity, value)
            if quantity <= 0:
                return None

            return self.exchange_interface.place_order(
                symbol=symbol,
                side=order_side,
                order_type=OrderType.MARKET,
                quantity=quantity,
            )
        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error("Live order execution failed: %s", e)
            return None

    def _close_live_order(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        position_notional: float,
        order_id: str | None = None,
    ) -> bool:
        """Close a real market order via exchange.

        Args:
            symbol: Trading symbol.
            side: Position side to close.
            quantity: Quantity to close.
            order_id: Order ID to close.

        Returns:
            True if successful, False otherwise.
        """
        if self.exchange_interface is None:
            logger.warning("No exchange interface configured - simulating order close")
            return True

        try:
            if quantity <= 0:
                logger.error("Invalid close quantity %.8f for %s", quantity, symbol)
                return False

            order_side = OrderSide.SELL if side == PositionSide.LONG else OrderSide.BUY
            quantity = self._normalize_quantity(symbol, quantity, position_notional)
            if quantity <= 0:
                return False

            close_order_id = self.exchange_interface.place_order(
                symbol=symbol,
                side=order_side,
                order_type=OrderType.MARKET,
                quantity=quantity,
            )
            if close_order_id:
                logger.info(
                    "Live close order placed: %s %s qty=%.8f order_id=%s",
                    symbol,
                    order_side.value,
                    quantity,
                    close_order_id,
                )
                return True
            logger.error("Failed to close live order for %s (order_id=%s)", symbol, order_id)
            return False
        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error("Live order close failed: %s", e)
            return False

    def _normalize_quantity(self, symbol: str, quantity: float, value: float) -> float:
        """Normalize quantity based on exchange symbol info."""
        if quantity <= 0 or self.exchange_interface is None:
            return 0.0

        symbol_info = self.exchange_interface.get_symbol_info(symbol)
        if not symbol_info:
            return quantity

        step_size = symbol_info.get("step_size", 0.00001)
        if step_size > 0:
            quantity = round(quantity / step_size) * step_size

        min_qty = symbol_info.get("min_qty", 0)
        if quantity < min_qty:
            logger.error(
                "Calculated quantity %.8f below minimum %.8f for %s",
                quantity,
                min_qty,
                symbol,
            )
            return 0.0

        min_notional = symbol_info.get("min_notional", 0)
        if value < min_notional:
            logger.error(
                "Order value %.2f below minimum notional %.2f for %s",
                value,
                min_notional,
                symbol,
            )
            return 0.0

        return quantity

    def get_execution_stats(self) -> dict:
        """Get execution statistics.

        Returns:
            Dictionary with fee and slippage totals.
        """
        return {
            "total_fees_paid": self.total_fees_paid,
            "total_slippage_cost": self.total_slippage_cost,
            "fee_rate": self.fee_rate,
            "slippage_rate": self.slippage_rate,
            "enable_live_trading": self.enable_live_trading,
        }
