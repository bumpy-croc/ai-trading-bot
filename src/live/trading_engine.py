"""
Live Trading Engine

This engine integrates the new components for live trading:
- TradeExecutor for consistent trade management
- SignalGenerator for standardized signal generation
- TradingDataRepository for data access
- LiveOrderExecutor for real order execution
"""

import asyncio
import logging
import time
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime, timedelta
import pandas as pd

from execution.trade_executor import TradeExecutor, TradeRequest, CloseRequest, ExecutionMode
from execution.signal_generator import SignalGenerator, MarketContext
from execution.order_executors import LiveOrderExecutor
from data.repository import TradingDataRepository
from strategies.base import BaseStrategy
from database.manager import DatabaseManager
from database.models import TradeSource
from config.constants import DEFAULT_INITIAL_BALANCE
from performance.metrics import Side
from data_providers.binance_data_provider import BinanceDataProvider

logger = logging.getLogger(__name__)


class LiveTradingEngine:
    """
    Live trading engine using the new component architecture.
    
    This replaces the original TradingEngine with a cleaner implementation
    that uses our TradeExecutor, SignalGenerator, and TradingDataRepository.
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        data_provider: Any, # For compatibility
        api_key: str,
        api_secret: str,
        symbol: str = "BTCUSDT",
        timeframe: str = "1d",
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
        database_url: Optional[str] = None,
        paper_trading: bool = True,
        max_position_size: float = 0.1,  # 10% of balance
        stop_loss_pct: float = 0.02,     # 2% stop loss
        take_profit_pct: float = 0.04    # 4% take profit
    ):
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.paper_trading = paper_trading
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Handle both DataProvider and TradingDataRepository
        if hasattr(data_provider, 'get_market_data'):
            # It's a TradingDataRepository
            self.data_repository = data_provider
        else:
            # It's a DataProvider, create a TradingDataRepository
            db_manager = DatabaseManager(database_url)
            self.data_repository = TradingDataRepository(db_manager, data_provider)
        
        # Initialize database manager
        self.db_manager = DatabaseManager(database_url)
        self.trading_session_id = None
        
        # Initialize components
        self.signal_generator = SignalGenerator(strategy)
        
        # Initialize order executor based on mode
        if paper_trading:
            from data_providers.binance_data_provider import BinanceDataProvider
            from execution.order_executors import PaperOrderExecutor
            data_provider = BinanceDataProvider()
            self.order_executor = PaperOrderExecutor(data_provider)
        else:
            from execution.order_executors import LiveOrderExecutor
            # Create exchange client (this would need proper implementation)
            exchange_client = None  # TODO: Initialize proper exchange client
            self.order_executor = LiveOrderExecutor(exchange_client)
        self.trade_executor = TradeExecutor(
            mode=ExecutionMode.LIVE if not paper_trading else ExecutionMode.PAPER,
            order_executor=self.order_executor,
            db_manager=self.db_manager,
            session_id=None,  # Will be set when session starts
            initial_balance=initial_balance
        )
        
        # Trading state
        self.is_running = False
        self.last_signal_time = None
        self.performance_history = []
        
        # Event callbacks
        self.on_trade_opened: Optional[Callable] = None
        self.on_trade_closed: Optional[Callable] = None
        self.on_signal_generated: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
    async def start(self) -> None:
        """Start the live trading engine"""
        try:
            logger.info("Starting live trading engine...")
            
            # Create trading session
            mode = TradeSource.PAPER if self.paper_trading else TradeSource.LIVE
            self.trading_session_id = self.db_manager.create_trading_session(
                strategy_name=self.strategy.__class__.__name__,
                symbol=self.symbol,
                timeframe=self.timeframe,
                mode=mode,
                initial_balance=self.initial_balance,
                strategy_config=getattr(self.strategy, 'get_parameters', lambda: {})(),
                session_name=f"Live_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Update trade executor with session ID
            self.trade_executor.session_id = self.trading_session_id
            
            # Initialize order executor
            await self.order_executor.initialize()
            
            # Validate account and get initial balance
            account_info = await self.order_executor.get_account_info()
            if account_info and not self.paper_trading:
                # Use real account balance for live trading
                usdt_balance = account_info.get('balances', {}).get('USDT', 0)
                if usdt_balance > 0:
                    self.trade_executor.current_balance = float(usdt_balance)
                    logger.info(f"Using real account balance: ${usdt_balance:.2f}")
            
            self.is_running = True
            logger.info(f"Live trading engine started (Paper: {self.paper_trading})")
            
            # Start main trading loop
            await self._trading_loop()
            
        except Exception as e:
            logger.error(f"Error starting live trading engine: {e}")
            if self.on_error:
                await self.on_error(e)
            raise
    
    async def stop(self) -> None:
        """Stop the live trading engine"""
        if not self.is_running:
            return

        logger.info("Stopping live trading engine...")
        self.is_running = False
        
        # Close any open positions
        await self._close_all_positions("engine_stop")
        
        # End trading session
        if self.trading_session_id and self.db_manager:
            self.db_manager.end_trading_session(
                session_id=self.trading_session_id,
                final_balance=self.trade_executor.current_balance
            )
        
        # Cleanup order executor
        await self.order_executor.cleanup()
        
        logger.info("Live trading engine stopped")
    
    async def _trading_loop(self) -> None:
        """Main trading loop"""
        while self.is_running:
            try:
                # Get current market data
                current_price = await self.order_executor.get_current_price(self.symbol)
                if not current_price:
                    logger.warning("Could not get current price, skipping iteration")
                    await asyncio.sleep(60)  # Wait 1 minute before retry
                    continue
                
                # Get historical data for analysis
                end_time = datetime.now()
                start_time = end_time - timedelta(days=100)  # Get last 100 days
                
                df = self.data_repository.get_market_data(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    start_date=start_time,
                    end_date=end_time,
                    include_indicators=True
                )
                
                if df.empty:
                    logger.warning("No historical data available, skipping iteration")
                    await asyncio.sleep(300)  # Wait 5 minutes before retry
                    continue
                
                # Create market context
                context = MarketContext(
                    symbol=self.symbol,
                    current_price=current_price,
                    timestamp=datetime.now(),
                    timeframe=self.timeframe,
                    data=df,
                    index=len(df) - 1  # Current index is the last data point
                )
                
                # Generate signal
                signal = self.signal_generator.generate_signal(
                    context=context,
                    current_balance=self.trade_executor.current_balance,
                    active_positions=self.trade_executor.active_positions
                )
                
                # Log signal generation
                if self.on_signal_generated:
                    await self.on_signal_generated(signal)
                
                # Process signal
                await self._process_signal(signal, current_price)
                
                # Check stop losses and take profits
                await self._check_position_exits(current_price)
                
                # Update performance tracking
                await self._update_performance_tracking(current_price)
                
                # Log current status
                active_positions = len(self.trade_executor.active_positions)
                logger.info(f"Trading loop: Price=${current_price:.2f}, "
                           f"Balance=${self.trade_executor.current_balance:.2f}, "
                           f"Positions={active_positions}, Signal={signal.action}")
                
                # Wait before next iteration (adjust based on timeframe)
                sleep_time = self._get_sleep_time()
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                if self.on_error:
                    await self.on_error(e)
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _process_signal(self, signal, current_price: float) -> None:
        """Process a trading signal"""
        try:
            if signal.action == "enter" and len(self.trade_executor.active_positions) == 0:
                # Open new position
                position_size = min(signal.position_size, 
                                  self.trade_executor.current_balance * self.max_position_size)
                
                # Calculate stop loss and take profit
                if signal.side == Side.LONG:
                    stop_loss = current_price * (1 - self.stop_loss_pct)
                    take_profit = current_price * (1 + self.take_profit_pct)
                else:  # SHORT
                    stop_loss = current_price * (1 + self.stop_loss_pct)
                    take_profit = current_price * (1 - self.take_profit_pct)
                
                trade_request = TradeRequest(
                    symbol=self.symbol,
                    side=signal.side,
                    size=position_size,
                    price=signal.price or current_price,
                    stop_loss=signal.stop_loss or stop_loss,
                    take_profit=signal.take_profit or take_profit,
                    strategy_name=self.strategy.__class__.__name__,
                    confidence=signal.confidence
                )
                
                result = self.trade_executor.open_position(trade_request)
                if result.success:
                    logger.info(f"Opened {signal.side.value} position: "
                               f"Size=${position_size:.2f}, Price=${result.executed_price:.2f}")
                    if self.on_trade_opened:
                        await self.on_trade_opened(result)
                else:
                    logger.warning(f"Failed to open position: {result.error_message}")
            
            elif signal.action == "exit" and len(self.trade_executor.active_positions) > 0:
                # Close existing positions
                await self._close_all_positions("strategy_exit")
                
        except Exception as e:
            logger.error(f"Error processing signal: {str(e)}")
            if self.on_error:
                await self.on_error(e)
    
    async def _check_position_exits(self, current_price: float) -> None:
        """Check if any positions should be closed due to stop loss or take profit"""
        positions_to_close = []
        
        for position_id, position in self.trade_executor.active_positions.items():
            should_close = False
            close_reason = ""
            
            # Check stop loss
            if position.stop_loss:
                if ((position.side == Side.LONG and current_price <= position.stop_loss) or
                    (position.side == Side.SHORT and current_price >= position.stop_loss)):
                    should_close = True
                    close_reason = "stop_loss"
            
            # Check take profit
            if not should_close and position.take_profit:
                if ((position.side == Side.LONG and current_price >= position.take_profit) or
                    (position.side == Side.SHORT and current_price <= position.take_profit)):
                    should_close = True
                    close_reason = "take_profit"
            
            if should_close:
                positions_to_close.append((position_id, close_reason))
        
        # Close positions
        for position_id, reason in positions_to_close:
            close_request = CloseRequest(
                position_id=position_id,
                reason=reason,
                price=current_price
            )
            result = self.trade_executor.close_position(close_request)
            if result.success:
                logger.info(f"Closed position ({reason}): P&L=${result.pnl:.2f}")
                if self.on_trade_closed:
                    await self.on_trade_closed(result)
            else:
                logger.warning(f"Failed to close position: {result.error_message}")
    
    async def _close_all_positions(self, reason: str) -> None:
        """Close all open positions"""
        current_price = await self.order_executor.get_current_price(self.symbol)
        if not current_price:
            logger.warning("Could not get current price for closing positions")
            return
        
        for position_id in list(self.trade_executor.active_positions.keys()):
            close_request = CloseRequest(
                position_id=position_id,
                reason=reason,
                price=current_price
            )
            result = self.trade_executor.close_position(close_request)
            if result.success:
                logger.info(f"Closed position ({reason}): P&L=${result.pnl:.2f}")
                if self.on_trade_closed:
                    await self.on_trade_closed(result)
    
    async def _update_performance_tracking(self, current_price: float) -> None:
        """Update performance tracking"""
        # Calculate current equity
        equity = self.trade_executor.current_balance
        
        # Add unrealized P&L
        for position in self.trade_executor.active_positions.values():
            if position.side == Side.LONG:
                unrealized_pnl = (current_price - position.entry_price) / position.entry_price * position.size
            else:  # SHORT
                unrealized_pnl = (position.entry_price - current_price) / position.entry_price * position.size
            
            equity += unrealized_pnl
        
        # Store performance point
        self.performance_history.append({
            'timestamp': datetime.now(),
            'balance': self.trade_executor.current_balance,
            'equity': equity,
            'price': current_price,
            'active_positions': len(self.trade_executor.active_positions)
        })
        
        # Keep only last 1000 points to avoid memory issues
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def _get_sleep_time(self) -> int:
        """Get sleep time based on timeframe"""
        timeframe_sleep = {
            '1m': 60,      # 1 minute
            '5m': 300,     # 5 minutes
            '15m': 900,    # 15 minutes
            '1h': 3600,    # 1 hour
            '4h': 14400,   # 4 hours
            '1d': 86400,   # 1 day
        }
        return timeframe_sleep.get(self.timeframe, 300)  # Default 5 minutes
    
    def get_current_status(self) -> Dict:
        """Get current trading status"""
        return {
            'is_running': self.is_running,
            'strategy': self.strategy.__class__.__name__,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'paper_trading': self.paper_trading,
            'current_balance': self.trade_executor.current_balance,
            'active_positions': len(self.trade_executor.active_positions),
            'session_id': self.trading_session_id,
            'last_signal_time': self.last_signal_time
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        if not self.trading_session_id or not self.data_repository:
            return {}
        
        return self.data_repository.calculate_session_metrics(
            session_id=self.trading_session_id,
            initial_balance=self.initial_balance
        )
    
    def get_trade_history(self) -> List[Dict]:
        """Get trade history"""
        if not self.trading_session_id or not self.data_repository:
            return []
        
        return self.data_repository.get_trades(session_id=self.trading_session_id)
    
    def get_signal_history(self) -> List[Dict]:
        """Get signal history"""
        if not self.signal_generator:
            return []
        
        return [signal.__dict__ for signal in self.signal_generator.get_signal_history()]
    
    def get_performance_history(self) -> List[Dict]:
        """Get performance tracking history"""
        return self.performance_history.copy()
    
    # Event handler setters
    def set_on_trade_opened(self, callback: Callable) -> None:
        """Set callback for when a trade is opened"""
        self.on_trade_opened = callback
    
    def set_on_trade_closed(self, callback: Callable) -> None:
        """Set callback for when a trade is closed"""
        self.on_trade_closed = callback
    
    def set_on_signal_generated(self, callback: Callable) -> None:
        """Set callback for when a signal is generated"""
        self.on_signal_generated = callback
    
    def set_on_error(self, callback: Callable) -> None:
        """Set callback for when an error occurs"""
        self.on_error = callback 