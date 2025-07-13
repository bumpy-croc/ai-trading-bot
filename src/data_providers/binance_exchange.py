"""
Binance Exchange Implementation

This module implements the ExchangeInterface for Binance, providing
real order execution, account synchronization, and position management.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceOrderException
    from binance.enums import SIDE_BUY, SIDE_SELL
    BINANCE_AVAILABLE = True
except ImportError:
    # Fallback for environments without binance library
    logger.warning("Binance library not available - using mock implementation")
    Client = None
    BinanceAPIException = Exception
    BinanceOrderException = Exception
    SIDE_BUY = "BUY"
    SIDE_SELL = "SELL"
    BINANCE_AVAILABLE = False

from .exchange_interface import (
    ExchangeInterface, OrderSide, OrderType, OrderStatus,
    AccountBalance, Position, Order, Trade
)

class BinanceExchange(ExchangeInterface):
    """Binance exchange implementation"""
    
    def _initialize_client(self):
        """Initialize Binance client"""
        if not BINANCE_AVAILABLE:
            logger.warning("Binance library not available - using mock client")
            self._client = None
            return
            
        try:
            self._client = Client(self.api_key, self.api_secret, testnet=self.testnet)
            logger.info(f"Binance client initialized (testnet: {self.testnet})")
        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test connection to Binance"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - connection test skipped")
            return False
            
        try:
            # Test server time
            server_time = self._client.get_server_time()
            logger.info(f"Binance connection test successful - server time: {server_time}")
            return True
        except Exception as e:
            logger.error(f"Binance connection test failed: {e}")
            return False
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get Binance account information"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning empty account info")
            return {}
            
        try:
            account_info = self._client.get_account()
            return {
                'maker_commission': account_info.get('makerCommission'),
                'taker_commission': account_info.get('takerCommission'),
                'buyer_commission': account_info.get('buyerCommission'),
                'seller_commission': account_info.get('sellerCommission'),
                'can_trade': account_info.get('canTrade'),
                'can_withdraw': account_info.get('canWithdraw'),
                'can_deposit': account_info.get('canDeposit'),
                'update_time': account_info.get('updateTime')
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}
    
    def get_balances(self) -> List[AccountBalance]:
        """Get all account balances"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning empty balances")
            return []
            
        try:
            account_info = self._client.get_account()
            balances = []
            
            for balance_data in account_info.get('balances', []):
                free = float(balance_data.get('free', 0))
                locked = float(balance_data.get('locked', 0))
                total = free + locked
                
                if total > 0:  # Only include non-zero balances
                    balance = AccountBalance(
                        asset=balance_data['asset'],
                        free=free,
                        locked=locked,
                        total=total,
                        last_updated=datetime.utcnow()
                    )
                    balances.append(balance)
            
            return balances
            
        except Exception as e:
            logger.error(f"Failed to get balances: {e}")
            return []
    
    def get_balance(self, asset: str) -> Optional[AccountBalance]:
        """Get balance for a specific asset"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning None for balance")
            return None
            
        try:
            account_info = self._client.get_account()
            
            for balance_data in account_info.get('balances', []):
                if balance_data['asset'] == asset:
                    free = float(balance_data.get('free', 0))
                    locked = float(balance_data.get('locked', 0))
                    total = free + locked
                    
                    return AccountBalance(
                        asset=asset,
                        free=free,
                        locked=locked,
                        total=total,
                        last_updated=datetime.utcnow()
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get balance for {asset}: {e}")
            return None
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open positions (for futures trading)"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning empty positions")
            return []
            
        try:
            # Note: This is for futures trading. For spot trading, positions are just holdings
            # For now, we'll return an empty list for spot trading
            # In the future, this could be extended for futures/margin trading
            
            # For spot trading, we can consider holdings as "positions"
            balances = self.get_balances()
            positions = []
            
            for balance in balances:
                if balance.asset != 'USDT' and balance.total > 0:
                    # Get current price for the asset
                    try:
                        ticker = self._client.get_symbol_ticker(symbol=f"{balance.asset}USDT")
                        current_price = float(ticker['price'])
                        
                        position = Position(
                            symbol=f"{balance.asset}USDT",
                            side="long",
                            size=balance.total,
                            entry_price=current_price,  # Simplified - we don't track entry price for holdings
                            current_price=current_price,
                            unrealized_pnl=0.0,  # Simplified
                            margin_type="spot",
                            leverage=1.0,
                            order_id="",  # No order ID for holdings
                            open_time=datetime.utcnow(),  # Simplified
                            last_update_time=datetime.utcnow()
                        )
                        positions.append(position)
                        
                    except Exception as e:
                        logger.debug(f"Could not get price for {balance.asset}: {e}")
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning empty orders")
            return []
            
        try:
            if symbol:
                orders_data = self._client.get_open_orders(symbol=symbol)
            else:
                orders_data = self._client.get_open_orders()
            
            orders = []
            for order_data in orders_data:
                order = Order(
                    order_id=order_data['orderId'],
                    symbol=order_data['symbol'],
                    side=OrderSide.BUY if order_data['side'] == SIDE_BUY else OrderSide.SELL,
                    order_type=self._convert_order_type(order_data['type']),
                    quantity=float(order_data['origQty']),
                    price=float(order_data['price']) if order_data['price'] != '0' else None,
                    status=self._convert_order_status(order_data['status']),
                    filled_quantity=float(order_data['executedQty']),
                    average_price=float(order_data['avgPrice']) if order_data['avgPrice'] != '0' else None,
                    commission=0.0,  # Will be updated from trade history
                    commission_asset="",
                    create_time=datetime.fromtimestamp(order_data['time'] / 1000),
                    update_time=datetime.fromtimestamp(order_data['updateTime'] / 1000),
                    stop_price=float(order_data['stopPrice']) if order_data.get('stopPrice') else None,
                    time_in_force=order_data.get('timeInForce', 'GTC')
                )
                orders.append(order)
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        """Get specific order by ID"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning None for order")
            return None
            
        try:
            order_data = self._client.get_order(symbol=symbol, orderId=order_id)
            
            order = Order(
                order_id=order_data['orderId'],
                symbol=order_data['symbol'],
                side=OrderSide.BUY if order_data['side'] == SIDE_BUY else OrderSide.SELL,
                order_type=self._convert_order_type(order_data['type']),
                quantity=float(order_data['origQty']),
                price=float(order_data['price']) if order_data['price'] != '0' else None,
                status=self._convert_order_status(order_data['status']),
                filled_quantity=float(order_data['executedQty']),
                average_price=float(order_data['avgPrice']) if order_data['avgPrice'] != '0' else None,
                commission=0.0,
                commission_asset="",
                create_time=datetime.fromtimestamp(order_data['time'] / 1000),
                update_time=datetime.fromtimestamp(order_data['updateTime'] / 1000),
                stop_price=float(order_data['stopPrice']) if order_data.get('stopPrice') else None,
                time_in_force=order_data.get('timeInForce', 'GTC')
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Get recent trades for a symbol"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning empty trades")
            return []
            
        try:
            trades_data = self._client.get_my_trades(symbol=symbol, limit=limit)
            
            trades = []
            for trade_data in trades_data:
                trade = Trade(
                    trade_id=trade_data['id'],
                    order_id=trade_data['orderId'],
                    symbol=trade_data['symbol'],
                    side=OrderSide.BUY if trade_data['isBuyer'] else OrderSide.SELL,
                    quantity=float(trade_data['qty']),
                    price=float(trade_data['price']),
                    commission=float(trade_data['commission']),
                    commission_asset=trade_data['commissionAsset'],
                    time=datetime.fromtimestamp(trade_data['time'] / 1000)
                )
                trades.append(trade)
            
            return trades
            
        except Exception as e:
            logger.error(f"Failed to get recent trades for {symbol}: {e}")
            return []
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC"
    ) -> Optional[str]:
        """Place a new order and return order ID"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - cannot place order")
            return None
            
        try:
            # Validate parameters first
            is_valid, error_msg = self.validate_order_parameters(symbol, side, order_type, quantity, price)
            if not is_valid:
                logger.error(f"Order validation failed: {error_msg}")
                return None
            
            # Convert to Binance parameters
            binance_side = SIDE_BUY if side == OrderSide.BUY else SIDE_SELL
            binance_type = self._convert_to_binance_order_type(order_type)
            
            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'side': binance_side,
                'type': binance_type,
                'quantity': quantity
            }
            
            if price is not None:
                order_params['price'] = price
            
            if stop_price is not None:
                order_params['stopPrice'] = stop_price
            
            if time_in_force != "GTC":
                order_params['timeInForce'] = time_in_force
            
            # Place the order
            result = self._client.create_order(**order_params)
            
            order_id = result['orderId']
            logger.info(f"Order placed successfully: {order_id} - {symbol} {side.value} {quantity}")
            
            return str(order_id)
            
        except BinanceOrderException as e:
            logger.error(f"Binance order error: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an existing order"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - cannot cancel order")
            return False
            
        try:
            result = self._client.cancel_order(symbol=symbol, orderId=order_id)
            logger.info(f"Order cancelled successfully: {order_id}")
            return True
            
        except BinanceOrderException as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> bool:
        """Cancel all open orders"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - cannot cancel orders")
            return False
            
        try:
            if symbol:
                result = self._client.cancel_all_orders(symbol=symbol)
            else:
                # Cancel all orders for all symbols
                open_orders = self.get_open_orders()
                for order in open_orders:
                    self.cancel_order(order.order_id, order.symbol)
            
            logger.info(f"All orders cancelled successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get trading symbol information"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning None for symbol info")
            return None
            
        try:
            exchange_info = self._client.get_exchange_info()
            
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == symbol:
                    # Extract relevant information
                    filters = {f['filterType']: f for f in symbol_info['filters']}
                    
                    return {
                        'symbol': symbol,
                        'base_asset': symbol_info['baseAsset'],
                        'quote_asset': symbol_info['quoteAsset'],
                        'status': symbol_info['status'],
                        'min_qty': float(filters.get('LOT_SIZE', {}).get('minQty', 0)),
                        'max_qty': float(filters.get('LOT_SIZE', {}).get('maxQty', float('inf'))),
                        'step_size': float(filters.get('LOT_SIZE', {}).get('stepSize', 0)),
                        'min_price': float(filters.get('PRICE_FILTER', {}).get('minPrice', 0)),
                        'max_price': float(filters.get('PRICE_FILTER', {}).get('maxPrice', float('inf'))),
                        'tick_size': float(filters.get('PRICE_FILTER', {}).get('tickSize', 0)),
                        'min_notional': float(filters.get('MIN_NOTIONAL', {}).get('minNotional', 0))
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            return None
    
    def _convert_order_type(self, binance_type: str) -> OrderType:
        """Convert Binance order type to our enum"""
        mapping = {
            'MARKET': OrderType.MARKET,
            'LIMIT': OrderType.LIMIT,
            'STOP_LOSS': OrderType.STOP_LOSS,
            'STOP_LOSS_LIMIT': OrderType.STOP_LOSS,
            'TAKE_PROFIT': OrderType.TAKE_PROFIT,
            'TAKE_PROFIT_LIMIT': OrderType.TAKE_PROFIT
        }
        return mapping.get(binance_type, OrderType.MARKET)
    
    def _convert_to_binance_order_type(self, order_type: OrderType) -> str:
        """Convert our order type enum to Binance format"""
        mapping = {
            OrderType.MARKET: 'MARKET',
            OrderType.LIMIT: 'LIMIT',
            OrderType.STOP_LOSS: 'STOP_LOSS',
            OrderType.TAKE_PROFIT: 'TAKE_PROFIT'
        }
        return mapping.get(order_type, 'MARKET')
    
    def _convert_order_status(self, binance_status: str) -> OrderStatus:
        """Convert Binance order status to our enum"""
        mapping = {
            'NEW': OrderStatus.PENDING,
            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
            'FILLED': OrderStatus.FILLED,
            'CANCELED': OrderStatus.CANCELLED,
            'REJECTED': OrderStatus.REJECTED,
            'EXPIRED': OrderStatus.EXPIRED
        }
        return mapping.get(binance_status, OrderStatus.PENDING)