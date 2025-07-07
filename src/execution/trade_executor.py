"""
Trade Execution Engine

This module contains the core TradeExecutor class, which is responsible for
managing trade execution, position tracking, and P&L calculation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Protocol, TYPE_CHECKING

from database.manager import DatabaseManager
from database.models import PositionSide, TradeSource
from performance.metrics import Side, cash_pnl, pnl_percent
from execution.order_executors import OrderResult

# Forward reference for OrderResult
if TYPE_CHECKING:
    from .order_executors import OrderResult

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution environment mode"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


@dataclass
class TradeRequest:
    """Request to open a trade"""
    symbol: str
    side: Side
    size: float  # Position size as fraction of balance
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_name: str = "unknown"
    confidence: Optional[float] = None


@dataclass
class TradeResult:
    """Result of trade execution"""
    success: bool
    trade_id: Optional[str] = None
    position_id: Optional[int] = None
    error_message: Optional[str] = None
    executed_price: Optional[float] = None
    executed_size: Optional[float] = None
    pnl: Optional[float] = None


@dataclass
class CloseRequest:
    """Request to close a position"""
    position_id: str
    reason: str
    price: Optional[float] = None  # If None, use current market price


class OrderExecutor(Protocol):
    """Protocol for order execution (live vs simulated)"""
    
    def execute_buy_order(self, symbol: str, quantity: float, price: Optional[float] = None) -> 'OrderResult':
        """Execute buy order, return OrderResult"""
        ...
    
    def execute_sell_order(self, symbol: str, quantity: float, price: Optional[float] = None) -> 'OrderResult':
        """Execute sell order, return OrderResult"""
        ...
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        ...


class TradeExecutor:
    """
    Trade execution engine for backtesting and live trading.
    
    This class standardizes how trades are opened, closed, and tracked,
    regardless of the execution environment.
    """
    
    def __init__(
        self,
        mode: ExecutionMode,
        order_executor: OrderExecutor,
        db_manager: DatabaseManager,
        session_id: Optional[int] = None,
        initial_balance: float = 10000.0
    ):
        self.mode = mode
        self.order_executor = order_executor
        self.db_manager = db_manager
        self.session_id = session_id
        
        # Account state
        self.current_balance = initial_balance
        self.initial_balance = initial_balance
        
        # Position tracking
        self.active_positions: Dict[str, Position] = {}
        self.position_counter = 0
        
        # Trade history
        self.completed_trades: List[TradeResult] = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
    def open_position(self, request: TradeRequest) -> TradeResult:
        """Open a new trading position"""
        try:
            # Validate request
            if request.size <= 0 or request.size > 1.0:
                return TradeResult(
                    success=False, 
                    error_message=f"Invalid position size: {request.size}"
                )
            
            # Calculate position value
            position_value = request.size * self.current_balance
            quantity = position_value / request.price
            
            # Execute order
            order_result = self.order_executor.execute_buy_order(
                request.symbol, quantity, request.price
            )
            
            if not isinstance(order_result, OrderResult) or not order_result.success:
                error_message = order_result.error_message if hasattr(order_result, 'error_message') else "Invalid order result"
                return TradeResult(
                    success=False,
                    error_message=error_message
                )
            
            order_id = order_result.order_id or f"{self.mode.value}_{self.position_counter}"
            if not order_result.order_id:
                self.position_counter += 1
            
            # Create position
            position = Position(
                id=order_id,
                symbol=request.symbol,
                side=request.side,
                entry_price=request.price,
                size=request.size,
                quantity=quantity,
                entry_time=datetime.now(),
                stop_loss=request.stop_loss,
                take_profit=request.take_profit,
                strategy_name=request.strategy_name,
                confidence=request.confidence
            )
            
            self.active_positions[order_id] = position
            
            # Log to database (if available)
            position_db_id = None
            if self.db_manager:
                position_db_id = self.db_manager.log_position(
                    symbol=request.symbol,
                    side=PositionSide(request.side.value),
                    entry_price=request.price,
                    size=request.size,
                    strategy_name=request.strategy_name,
                    order_id=order_id,
                    stop_loss=request.stop_loss,
                    take_profit=request.take_profit,
                    quantity=quantity,
                    session_id=self.session_id
                )
            
            logger.info(
                f"📈 Opened {request.side.value} position: {request.symbol} "
                f"@ ${request.price:.2f} (size: {request.size:.1%})"
            )
            
            return TradeResult(
                success=True,
                trade_id=order_id,
                position_id=position_db_id,
                executed_price=request.price,
                executed_size=request.size
            )
            
        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            return TradeResult(
                success=False,
                error_message=str(e)
            )
    
    def close_position(self, request: CloseRequest) -> TradeResult:
        """Close an existing position"""
        try:
            position = self.active_positions.get(request.position_id)
            if not position:
                return TradeResult(
                    success=False,
                    error_message=f"Position {request.position_id} not found"
                )
            
            # Get exit price
            exit_price = request.price
            if exit_price is None:
                exit_price = self.order_executor.get_current_price(position.symbol)
            
            # Execute closing order
            close_order_result = self.order_executor.execute_sell_order(
                position.symbol, position.quantity, exit_price
            )
            
            if not isinstance(close_order_result, OrderResult) or not close_order_result.success:
                error_message = close_order_result.error_message if hasattr(close_order_result, 'error_message') else "Invalid order result"
                return TradeResult(
                    success=False,
                    error_message=error_message
                )
            
            # Calculate P&L
            pnl_pct = pnl_percent(
                position.entry_price, 
                exit_price, 
                position.side, 
                position.size
            )
            pnl_cash = cash_pnl(pnl_pct, self.current_balance)
            
            # Update balance
            self.current_balance += pnl_cash
            self.total_pnl += pnl_cash
            self.total_trades += 1
            if pnl_cash > 0:
                self.winning_trades += 1
            
            # Log trade to database (if available)
            trade_id = None
            if self.db_manager:
                trade_source = TradeSource.LIVE if self.mode == ExecutionMode.LIVE else TradeSource.PAPER
                if self.mode == ExecutionMode.BACKTEST:
                    trade_source = TradeSource.BACKTEST
                
                trade_id = self.db_manager.log_trade(
                    symbol=position.symbol,
                    side=PositionSide(position.side.value),
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    size=position.size,
                    entry_time=position.entry_time,
                    exit_time=datetime.now(),
                    pnl=pnl_cash,
                    exit_reason=request.reason,
                    strategy_name=position.strategy_name,
                    source=trade_source,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    order_id=position.id,
                    confidence_score=position.confidence,
                    session_id=self.session_id
                )
                
                # Update balance in database
                self.db_manager.update_balance(
                    self.current_balance,
                    f'trade_pnl_{request.reason.lower()}',
                    'system',
                    self.session_id
                )
            
            # Remove from active positions
            del self.active_positions[request.position_id]
            
            # Create trade result
            trade_result = TradeResult(
                success=True,
                trade_id=str(trade_id) if trade_id else None,
                executed_price=exit_price,
                executed_size=position.size,
                pnl=pnl_cash
            )
            
            # Add to completed trades
            self.completed_trades.append(trade_result)
            
            pnl_str = f"+${pnl_cash:.2f}" if pnl_cash > 0 else f"${pnl_cash:.2f}"
            logger.info(
                f"🏁 Closed {position.side.value} position: {position.symbol} "
                f"@ ${exit_price:.2f} ({request.reason}) - P&L: {pnl_str}"
            )
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return TradeResult(
                success=False,
                error_message=str(e)
            )
    
    def update_position_pnl(self, symbol: str) -> None:
        """Update unrealized P&L for all positions of a symbol"""
        current_price = self.order_executor.get_current_price(symbol)
        
        for position in self.active_positions.values():
            if position.symbol == symbol:
                position.unrealized_pnl = pnl_percent(
                    position.entry_price,
                    current_price,
                    position.side,
                    position.size
                )
    
    def get_account_summary(self) -> Dict:
        """Get current account summary"""
        unrealized_pnl = sum(
            pos.unrealized_pnl or 0.0 for pos in self.active_positions.values()
        )
        
        return {
            'balance': self.current_balance,
            'equity': self.current_balance + unrealized_pnl,
            'total_pnl': self.total_pnl,
            'active_positions': len(self.active_positions),
            'total_trades': self.total_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        }


@dataclass
class Position:
    """Position data structure"""
    id: str
    symbol: str
    side: Side
    entry_price: float
    size: float  # Position size as fraction of balance
    quantity: float  # Actual quantity
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_name: str = "unknown"
    confidence: Optional[float] = None
    unrealized_pnl: Optional[float] = None 