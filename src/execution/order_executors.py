"""
Order execution classes for different trading modes.

These classes handle the actual order execution for the TradeExecutor,
providing different implementations for backtesting, paper trading, and live trading.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderResult:
    """Result of an order execution"""
    success: bool
    order_id: Optional[str] = None
    executed_price: Optional[float] = None
    executed_quantity: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: Optional[pd.Timestamp] = None


class BaseOrderExecutor(ABC):
    """Base class for order executors"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def execute_buy_order(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET
    ) -> OrderResult:
        """Execute a buy order"""
        pass
    
    @abstractmethod
    def execute_sell_order(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET
    ) -> OrderResult:
        """Execute a sell order"""
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        pass


class BacktestOrderExecutor(BaseOrderExecutor):
    """Order executor for backtesting - simulates perfect execution"""
    
    def __init__(self, market_data: pd.DataFrame):
        super().__init__()
        self.market_data = market_data
        self.current_index = 0
        self.order_counter = 0
        self.slippage = 0.001  # 0.1% slippage simulation
        self.commission = 0.001  # 0.1% commission
    
    async def initialize(self):
        """Dummy method for compatibility"""
        pass

    async def get_account_info(self):
        """Dummy method for compatibility"""
        return {}

    async def cleanup(self):
        """Dummy method for compatibility"""
        pass

    def set_current_index(self, index: int):
        """Set the current market data index"""
        self.current_index = min(index, len(self.market_data) - 1)
    
    def execute_buy_order(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET
    ) -> OrderResult:
        """Simulate buy order execution in backtest"""
        try:
            if self.current_index >= len(self.market_data):
                return OrderResult(
                    success=False,
                    error_message="No market data available"
                )
            
            current_candle = self.market_data.iloc[self.current_index]
            
            if order_type == OrderType.MARKET:
                # Use current close price with slippage
                executed_price = current_candle['close'] * (1 + self.slippage)
            elif order_type == OrderType.LIMIT and price:
                # For limit orders, check if price would have been hit
                if price >= current_candle['low']:
                    executed_price = price
                else:
                    return OrderResult(
                        success=False,
                        error_message="Limit price not reached"
                    )
            else:
                executed_price = current_candle['close']
            
            # Apply commission
            executed_price *= (1 + self.commission)
            
            self.order_counter += 1
            order_id = f"backtest_buy_{self.order_counter}"
            
            self.logger.debug(f"Simulated buy order: {quantity} {symbol} at {executed_price}")
            
            return OrderResult(
                success=True,
                order_id=order_id,
                executed_price=executed_price,
                executed_quantity=quantity,
                timestamp=current_candle.name
            )
            
        except Exception as e:
            self.logger.error(f"Error executing buy order: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    def execute_sell_order(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET
    ) -> OrderResult:
        """Simulate sell order execution in backtest"""
        try:
            if self.current_index >= len(self.market_data):
                return OrderResult(
                    success=False,
                    error_message="No market data available"
                )
            
            current_candle = self.market_data.iloc[self.current_index]
            
            if order_type == OrderType.MARKET:
                # Use current close price with slippage (negative for sell)
                executed_price = current_candle['close'] * (1 - self.slippage)
            elif order_type == OrderType.LIMIT and price:
                # For limit orders, check if price would have been hit
                if price <= current_candle['high']:
                    executed_price = price
                else:
                    return OrderResult(
                        success=False,
                        error_message="Limit price not reached"
                    )
            else:
                executed_price = current_candle['close']
            
            # Apply commission
            executed_price *= (1 - self.commission)
            
            self.order_counter += 1
            order_id = f"backtest_sell_{self.order_counter}"
            
            self.logger.debug(f"Simulated sell order: {quantity} {symbol} at {executed_price}")
            
            return OrderResult(
                success=True,
                order_id=order_id,
                executed_price=executed_price,
                executed_quantity=quantity,
                timestamp=current_candle.name
            )
            
        except Exception as e:
            self.logger.error(f"Error executing sell order: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price from market data"""
        if self.current_index >= len(self.market_data):
            return 0.0
        return self.market_data.iloc[self.current_index]['close']
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order (always succeeds in backtest)"""
        self.logger.debug(f"Cancelled order {order_id}")
        return True


class PaperOrderExecutor(BaseOrderExecutor):
    """Order executor for paper trading - simulates execution with real market prices"""
    
    def __init__(self, data_provider):
        super().__init__()
        self.data_provider = data_provider
        self.order_counter = 0
        self.slippage = 0.002  # 0.2% slippage for paper trading
        self.commission = 0.001  # 0.1% commission
        self.pending_orders = {}
    
    async def initialize(self):
        """Dummy method for compatibility"""
        pass

    async def get_account_info(self):
        """Dummy method for compatibility"""
        return {'balances': {'USDT': 10000}}

    async def cleanup(self):
        """Dummy method for compatibility"""
        pass

    def execute_buy_order(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET
    ) -> OrderResult:
        """Execute buy order in paper trading mode"""
        try:
            current_price = self.data_provider.get_current_price(symbol)
            
            if order_type == OrderType.MARKET:
                executed_price = current_price * (1 + self.slippage + self.commission)
            elif order_type == OrderType.LIMIT and price:
                if price >= current_price:
                    executed_price = min(price, current_price * (1 + self.slippage))
                else:
                    # Store as pending order
                    self.order_counter += 1
                    order_id = f"paper_buy_pending_{self.order_counter}"
                    self.pending_orders[order_id] = {
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': price,
                        'side': 'buy',
                        'type': order_type
                    }
                    return OrderResult(
                        success=True,
                        order_id=order_id,
                        error_message="Order pending"
                    )
            else:
                executed_price = current_price * (1 + self.slippage + self.commission)
            
            self.order_counter += 1
            order_id = f"paper_buy_{self.order_counter}"
            
            self.logger.info(f"Paper buy order: {quantity} {symbol} at {executed_price}")
            
            return OrderResult(
                success=True,
                order_id=order_id,
                executed_price=executed_price,
                executed_quantity=quantity,
                timestamp=pd.Timestamp.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error executing paper buy order: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    def execute_sell_order(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET
    ) -> OrderResult:
        """Execute sell order in paper trading mode"""
        try:
            current_price = self.data_provider.get_current_price(symbol)
            
            if order_type == OrderType.MARKET:
                executed_price = current_price * (1 - self.slippage - self.commission)
            elif order_type == OrderType.LIMIT and price:
                if price <= current_price:
                    executed_price = max(price, current_price * (1 - self.slippage))
                else:
                    # Store as pending order
                    self.order_counter += 1
                    order_id = f"paper_sell_pending_{self.order_counter}"
                    self.pending_orders[order_id] = {
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': price,
                        'side': 'sell',
                        'type': order_type
                    }
                    return OrderResult(
                        success=True,
                        order_id=order_id,
                        error_message="Order pending"
                    )
            else:
                executed_price = current_price * (1 - self.slippage - self.commission)
            
            self.order_counter += 1
            order_id = f"paper_sell_{self.order_counter}"
            
            self.logger.info(f"Paper sell order: {quantity} {symbol} at {executed_price}")
            
            return OrderResult(
                success=True,
                order_id=order_id,
                executed_price=executed_price,
                executed_quantity=quantity,
                timestamp=pd.Timestamp.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error executing paper sell order: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price from data provider"""
        try:
            return self.data_provider.get_current_price(symbol)
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return 0.0
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
            self.logger.info(f"Cancelled order {order_id}")
            return True
        return False


class LiveOrderExecutor(BaseOrderExecutor):
    """Order executor for live trading - executes real orders"""
    
    def __init__(self, exchange_client):
        super().__init__()
        self.exchange_client = exchange_client
        self.order_counter = 0
    
    def execute_buy_order(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET
    ) -> OrderResult:
        """Execute real buy order on exchange"""
        try:
            if order_type == OrderType.MARKET:
                # Execute market buy order
                result = self.exchange_client.create_market_buy_order(symbol, quantity)
            elif order_type == OrderType.LIMIT and price:
                # Execute limit buy order
                result = self.exchange_client.create_limit_buy_order(symbol, quantity, price)
            else:
                return OrderResult(
                    success=False,
                    error_message="Invalid order type or missing price"
                )
            
            if result and result.get('id'):
                self.logger.info(f"Live buy order executed: {quantity} {symbol}")
                return OrderResult(
                    success=True,
                    order_id=result['id'],
                    executed_price=result.get('price'),
                    executed_quantity=result.get('amount'),
                    timestamp=pd.Timestamp.now()
                )
            else:
                return OrderResult(
                    success=False,
                    error_message="Order execution failed"
                )
                
        except Exception as e:
            self.logger.error(f"Error executing live buy order: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    def execute_sell_order(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET
    ) -> OrderResult:
        """Execute real sell order on exchange"""
        try:
            if order_type == OrderType.MARKET:
                # Execute market sell order
                result = self.exchange_client.create_market_sell_order(symbol, quantity)
            elif order_type == OrderType.LIMIT and price:
                # Execute limit sell order
                result = self.exchange_client.create_limit_sell_order(symbol, quantity, price)
            else:
                return OrderResult(
                    success=False,
                    error_message="Invalid order type or missing price"
                )
            
            if result and result.get('id'):
                self.logger.info(f"Live sell order executed: {quantity} {symbol}")
                return OrderResult(
                    success=True,
                    order_id=result['id'],
                    executed_price=result.get('price'),
                    executed_quantity=result.get('amount'),
                    timestamp=pd.Timestamp.now()
                )
            else:
                return OrderResult(
                    success=False,
                    error_message="Order execution failed"
                )
                
        except Exception as e:
            self.logger.error(f"Error executing live sell order: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    async def get_account_info(self):
        """Get live account information from the exchange"""
        if hasattr(self.exchange_client, "fetch_balance"):
            try:
                balance = await self.exchange_client.fetch_balance()
                return {'balances': {'USDT': balance.get('USDT', {}).get('free', 0)}}
            except Exception as e:
                self.logger.error(f"Error fetching account balance: {e}")
        return {}

    async def get_current_price(self, symbol: str) -> float:
        """Get current price from exchange"""
        try:
            ticker = self.exchange_client.fetch_ticker(symbol)
            return ticker['last'] if ticker else 0.0
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return 0.0
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on exchange"""
        try:
            result = self.exchange_client.cancel_order(order_id)
            if result:
                self.logger.info(f"Cancelled live order {order_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False

    async def cleanup(self):
        """Clean up exchange client resources."""
        if hasattr(self.exchange_client, 'close'):
            await self.exchange_client.close()
            self.logger.info("Exchange client connection closed.") 