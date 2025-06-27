import logging
import time
import json
import os
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import pandas as pd
import threading
from dataclasses import dataclass
from enum import Enum
import signal
import sys

from core.data_providers.data_provider import DataProvider
from core.data_providers.binance_data_provider import BinanceDataProvider
from core.data_providers.sentiment_provider import SentimentDataProvider
from strategies.base import BaseStrategy
from core.risk.risk_manager import RiskManager, RiskParameters
from live.strategy_manager import StrategyManager
from core.database.manager import DatabaseManager
from core.database.models import TradeSource

logger = logging.getLogger(__name__)

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"

@dataclass
class Position:
    """Represents an active trading position"""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    order_id: Optional[str] = None

@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    exit_reason: str
    
class LiveTradingEngine:
    """
    Advanced live trading engine that executes strategies in real-time.
    
    Features:
    - Real-time data streaming
    - Actual order execution
    - Position management
    - Risk management
    - Sentiment integration
    - Error handling & recovery
    - Performance monitoring
    - Graceful shutdown
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        data_provider: DataProvider,
        sentiment_provider: Optional[SentimentDataProvider] = None,
        risk_parameters: Optional[RiskParameters] = None,
        check_interval: int = 60,  # seconds
        initial_balance: float = 10000,
        max_position_size: float = 0.1,  # 10% of balance per position
        enable_live_trading: bool = False,  # Safety flag - must be explicitly enabled
        log_trades: bool = True,
        alert_webhook_url: Optional[str] = None,
        enable_hot_swapping: bool = True,  # Enable strategy hot-swapping
        database_url: Optional[str] = None  # Database connection URL
    ):
        self.strategy = strategy
        self.data_provider = data_provider
        self.sentiment_provider = sentiment_provider
        self.risk_manager = RiskManager(risk_parameters)
        self.check_interval = check_interval
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_position_size = max_position_size
        self.enable_live_trading = enable_live_trading
        self.log_trades = log_trades
        self.alert_webhook_url = alert_webhook_url
        self.enable_hot_swapping = enable_hot_swapping
        
        # Initialize database manager
        self.db_manager = DatabaseManager(database_url)
        self.trading_session_id: Optional[int] = None
        
        # Initialize strategy manager for hot-swapping
        self.strategy_manager = None
        if enable_hot_swapping:
            self.strategy_manager = StrategyManager()
            self.strategy_manager.current_strategy = strategy
            self.strategy_manager.on_strategy_change = self._handle_strategy_change
            self.strategy_manager.on_model_update = self._handle_model_update
        
        # Trading state
        self.is_running = False
        self.positions: Dict[str, Position] = {}
        self.position_db_ids: Dict[str, int] = {}  # Map order_id to database position ID
        self.completed_trades: List[Trade] = []
        self.last_data_update = None
        self.last_account_snapshot = None  # Track when we last logged account state
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
        
        # Error handling
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10
        self.error_cooldown = 300  # 5 minutes
        
        # Threading
        self.main_thread = None
        self.stop_event = threading.Event()
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"LiveTradingEngine initialized - Live Trading: {'ENABLED' if enable_live_trading else 'DISABLED'}")
        
    def start(self, symbol: str, timeframe: str = "1h"):
        """Start the live trading engine"""
        if self.is_running:
            logger.warning("Trading engine is already running")
            return
            
        self.is_running = True
        logger.info(f"🚀 Starting live trading for {symbol} on {timeframe} timeframe")
        logger.info(f"Initial balance: ${self.current_balance:,.2f}")
        logger.info(f"Max position size: {self.max_position_size*100:.1f}% of balance")
        logger.info(f"Check interval: {self.check_interval}s")
        
        if not self.enable_live_trading:
            logger.warning("⚠️  PAPER TRADING MODE - No real orders will be executed")
        
        # Create trading session in database
        mode = TradeSource.LIVE if self.enable_live_trading else TradeSource.PAPER
        self.trading_session_id = self.db_manager.create_trading_session(
            strategy_name=self.strategy.__class__.__name__,
            symbol=symbol,
            timeframe=timeframe,
            mode=mode,
            initial_balance=self.initial_balance,
            strategy_config=getattr(self.strategy, 'config', {})
        )
        
        # Start main trading loop in separate thread
        self.main_thread = threading.Thread(target=self._trading_loop, args=(symbol, timeframe))
        self.main_thread.daemon = True
        self.main_thread.start()
        
        try:
            # Keep main thread alive
            while self.is_running and self.main_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()
            
    def stop(self):
        """Stop the trading engine gracefully"""
        if not self.is_running:
            return
            
        logger.info("🛑 Stopping trading engine...")
        self.is_running = False
        self.stop_event.set()
        
        # Close all open positions
        if self.positions:
            logger.info(f"Closing {len(self.positions)} open positions...")
            for position in list(self.positions.values()):
                self._close_position(position, "Engine shutdown")
        
        # Wait for main thread to finish
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=30)
            
        # Print final statistics
        self._print_final_stats()
        
        # End the trading session in database
        if self.trading_session_id:
            self.db_manager.end_trading_session(
                session_id=self.trading_session_id,
                final_balance=self.current_balance
            )
        
        logger.info("Trading engine stopped")
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)
        
    def _trading_loop(self, symbol: str, timeframe: str):
        """Main trading loop"""
        logger.info("Trading loop started")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # Fetch latest market data
                df = self._get_latest_data(symbol, timeframe)
                if df is None or df.empty:
                    logger.warning("No market data received")
                    self._sleep_with_interrupt(self.check_interval)
                    continue
                
                # Add sentiment data if available
                if self.sentiment_provider:
                    df = self._add_sentiment_data(df, symbol)
                
                # Check for pending strategy/model updates
                if self.strategy_manager and self.strategy_manager.has_pending_update():
                    logger.info("🔄 Applying pending strategy/model update...")
                    success = self.strategy_manager.apply_pending_update()
                    if success:
                        self.strategy = self.strategy_manager.current_strategy
                        logger.info("✅ Strategy/model update applied successfully")
                        self._send_alert("Strategy/Model updated in live trading")
                    else:
                        logger.error("❌ Failed to apply strategy/model update")
                
                # Calculate indicators
                df = self.strategy.calculate_indicators(df)
                
                # Remove warmup period and ensure we have enough data
                df = df.dropna()
                if len(df) < 2:
                    logger.warning("Insufficient data for analysis")
                    self._sleep_with_interrupt(self.check_interval)
                    continue
                
                current_index = len(df) - 1
                current_candle = df.iloc[current_index]
                current_price = current_candle['close']
                
                # Update position PnL
                self._update_position_pnl(current_price)
                
                # Check exit conditions for existing positions
                self._check_exit_conditions(df, current_index, current_price)
                
                # Check entry conditions if not at maximum positions
                if len(self.positions) < self._get_max_positions():
                    self._check_entry_conditions(df, current_index, symbol, current_price)
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Log account snapshot to database periodically (every 5 minutes)
                now = datetime.now()
                if self.last_account_snapshot is None or (now - self.last_account_snapshot).seconds >= 300:
                    self._log_account_snapshot()
                    self.last_account_snapshot = now
                
                # Log status periodically
                if self.total_trades % 10 == 0 or len(self.positions) > 0:
                    self._log_status(symbol, current_price)
                
                # Reset error counter on successful iteration
                self.consecutive_errors = 0
                
            except Exception as e:
                self.consecutive_errors += 1
                logger.error(f"Error in trading loop (#{self.consecutive_errors}): {e}")
                
                if self.consecutive_errors >= self.max_consecutive_errors:
                    logger.critical(f"Too many consecutive errors ({self.consecutive_errors}). Stopping engine.")
                    self.stop()
                    break
                
                # Sleep longer after errors
                sleep_time = min(self.error_cooldown, self.check_interval * self.consecutive_errors)
                self._sleep_with_interrupt(sleep_time)
                continue
            
            # Normal sleep between checks
            self._sleep_with_interrupt(self.check_interval)
        
        logger.info("Trading loop ended")
        
    def _get_latest_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch latest market data with error handling"""
        try:
            df = self.data_provider.get_live_data(symbol, timeframe, limit=200)
            self.last_data_update = datetime.now()
            return df
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            return None
            
    def _add_sentiment_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add sentiment data to price data"""
        try:
            if hasattr(self.sentiment_provider, 'get_live_sentiment'):
                # Get live sentiment for recent data
                live_sentiment = self.sentiment_provider.get_live_sentiment()
                
                # Apply to recent candles (last 4 hours)
                recent_mask = df.index >= (df.index.max() - pd.Timedelta(hours=4))
                for feature, value in live_sentiment.items():
                    if feature not in df.columns:
                        df[feature] = 0.0
                    df.loc[recent_mask, feature] = value
                
                # Mark sentiment freshness
                df['sentiment_freshness'] = 0
                df.loc[recent_mask, 'sentiment_freshness'] = 1
                
                logger.debug(f"Applied live sentiment to {recent_mask.sum()} recent candles")
            else:
                # Fallback to historical sentiment
                logger.debug("Using historical sentiment data")
                
        except Exception as e:
            logger.error(f"Failed to add sentiment data: {e}")
            
        return df
        
    def _update_position_pnl(self, current_price: float):
        """Update unrealized PnL for all positions"""
        for position in self.positions.values():
            if position.side == PositionSide.LONG:
                position.unrealized_pnl = (current_price - position.entry_price) / position.entry_price * position.size
            else:  # SHORT
                position.unrealized_pnl = (position.entry_price - current_price) / position.entry_price * position.size
                
    def _check_exit_conditions(self, df: pd.DataFrame, current_index: int, current_price: float):
        """Check if any positions should be closed"""
        positions_to_close = []
        
        for position in self.positions.values():
            should_exit = False
            exit_reason = ""
            
            # Check strategy exit conditions
            if self.strategy.check_exit_conditions(df, current_index, position.entry_price):
                should_exit = True
                exit_reason = "Strategy signal"
            
            # Check stop loss
            elif position.stop_loss and self._check_stop_loss(position, current_price):
                should_exit = True
                exit_reason = "Stop loss"
                
            # Check take profit
            elif position.take_profit and self._check_take_profit(position, current_price):
                should_exit = True
                exit_reason = "Take profit"
                
            # Check maximum position time (24 hours)
            elif (datetime.now() - position.entry_time).total_seconds() > 86400:
                should_exit = True
                exit_reason = "Time limit"
            
            if should_exit:
                positions_to_close.append((position, exit_reason))
        
        # Close positions
        for position, reason in positions_to_close:
            self._close_position(position, reason)
            
    def _check_entry_conditions(self, df: pd.DataFrame, current_index: int, symbol: str, current_price: float):
        """Check if new positions should be opened"""
        if not self.strategy.check_entry_conditions(df, current_index):
            return
            
        # Calculate position size
        position_size = self.strategy.calculate_position_size(df, current_index, self.current_balance)
        position_size = min(position_size, self.max_position_size)  # Cap at max position size
        
        if position_size <= 0:
            return
            
        # Calculate risk management levels
        stop_loss = self.strategy.calculate_stop_loss(df, current_index, current_price, 'long')
        take_profit = current_price * 1.04  # 4% take profit target
        
        # Open new position
        self._open_position(symbol, PositionSide.LONG, position_size, current_price, stop_loss, take_profit)
        
    def _open_position(self, symbol: str, side: PositionSide, size: float, price: float, 
                      stop_loss: Optional[float] = None, take_profit: Optional[float] = None):
        """Open a new trading position"""
        try:
            position_value = size * self.current_balance
            
            if self.enable_live_trading:
                # Execute real order
                order_id = self._execute_order(symbol, side, position_value, price)
                if not order_id:
                    logger.error("Failed to execute order")
                    return
            else:
                order_id = f"paper_{int(time.time())}"
                logger.info(f"📄 PAPER TRADE - Would open {side.value} position")
            
            # Create position
            position = Position(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=price,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                order_id=order_id
            )
            
            self.positions[order_id] = position
            
            # Log position to database
            position_db_id = self.db_manager.log_position(
                symbol=symbol,
                side=side.value,
                entry_price=price,
                size=size,
                strategy_name=self.strategy.__class__.__name__,
                order_id=order_id,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=position_value / price,  # Calculate actual quantity
                session_id=self.trading_session_id
            )
            self.position_db_ids[order_id] = position_db_id
            
            logger.info(f"🚀 Opened {side.value} position: {symbol} @ ${price:.2f} (Size: ${position_value:.2f})")
            
            # Send alert if configured
            self._send_alert(f"Position Opened: {symbol} {side.value} @ ${price:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            self.db_manager.log_event(
                "ERROR",
                f"Failed to open position: {str(e)}",
                severity="error",
                component="LiveTradingEngine",
                stack_trace=str(e),
                session_id=self.trading_session_id
            )
            
    def _close_position(self, position: Position, reason: str):
        """Close an existing position"""
        try:
            # Get current price for closing
            current_data = self.data_provider.get_live_data(position.symbol, "1h", limit=1)
            if current_data.empty:
                logger.error("Cannot get current price for position close")
                return
                
            exit_price = current_data.iloc[-1]['close']
            
            if self.enable_live_trading:
                # Execute real closing order
                success = self._close_order(position.symbol, position.order_id)
                if not success:
                    logger.error("Failed to close order")
                    return
            else:
                logger.info(f"📄 PAPER TRADE - Would close {position.side.value} position")
            
            # Calculate PnL
            if position.side == PositionSide.LONG:
                pnl_pct = (exit_price - position.entry_price) / position.entry_price
            else:
                pnl_pct = (position.entry_price - exit_price) / position.entry_price
                
            pnl_dollar = pnl_pct * position.size * self.current_balance
            
            # Update balance
            self.current_balance += pnl_dollar
            self.total_pnl += pnl_dollar
            
            # Create trade record
            trade = Trade(
                symbol=position.symbol,
                side=position.side,
                size=position.size,
                entry_price=position.entry_price,
                exit_price=exit_price,
                entry_time=position.entry_time,
                exit_time=datetime.now(),
                pnl=pnl_dollar,
                exit_reason=reason
            )
            
            self.completed_trades.append(trade)
            self.total_trades += 1
            if pnl_dollar > 0:
                self.winning_trades += 1
            
            # Log trade to database
            source = TradeSource.LIVE if self.enable_live_trading else TradeSource.PAPER
            trade_db_id = self.db_manager.log_trade(
                symbol=position.symbol,
                side=position.side.value,
                entry_price=position.entry_price,
                exit_price=exit_price,
                size=position.size,
                entry_time=position.entry_time,
                exit_time=trade.exit_time,
                pnl=pnl_dollar,
                exit_reason=reason,
                strategy_name=self.strategy.__class__.__name__,
                source=source,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                order_id=position.order_id,
                session_id=self.trading_session_id
            )
            
            # Update position status in database
            if position.order_id in self.position_db_ids:
                self.db_manager.close_position(self.position_db_ids[position.order_id])
                del self.position_db_ids[position.order_id]
                
            # Remove from active positions
            if position.order_id in self.positions:
                del self.positions[position.order_id]
            
            # Log trade
            pnl_str = f"+${pnl_dollar:.2f}" if pnl_dollar > 0 else f"${pnl_dollar:.2f}"
            logger.info(f"🏁 Closed {position.side.value} position: {position.symbol} @ ${exit_price:.2f} ({reason}) - PnL: {pnl_str}")
            
            # Save trade log (file-based for backward compatibility)
            if self.log_trades:
                self._log_trade(trade)
            
            # Send alert
            self._send_alert(f"Position Closed: {position.symbol} {reason} - PnL: {pnl_str}")
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            self.db_manager.log_event(
                "ERROR",
                f"Failed to close position: {str(e)}",
                severity="error",
                component="LiveTradingEngine",
                stack_trace=str(e),
                session_id=self.trading_session_id
            )
            
    def _execute_order(self, symbol: str, side: PositionSide, value: float, price: float) -> Optional[str]:
        """Execute a real market order (implement based on your exchange)"""
        # This is a placeholder - implement actual order execution
        logger.warning("Real order execution not implemented - using paper trading")
        return f"real_{int(time.time())}"
        
    def _close_order(self, symbol: str, order_id: str) -> bool:
        """Close a real market order (implement based on your exchange)"""
        # This is a placeholder - implement actual order closing
        logger.warning("Real order closing not implemented - using paper trading")
        return True
        
    def _check_stop_loss(self, position: Position, current_price: float) -> bool:
        """Check if stop loss should be triggered"""
        if not position.stop_loss:
            return False
            
        if position.side == PositionSide.LONG:
            return current_price <= position.stop_loss
        else:
            return current_price >= position.stop_loss
            
    def _check_take_profit(self, position: Position, current_price: float) -> bool:
        """Check if take profit should be triggered"""
        if not position.take_profit:
            return False
            
        if position.side == PositionSide.LONG:
            return current_price >= position.take_profit
        else:
            return current_price <= position.take_profit
            
    def _get_max_positions(self) -> int:
        """Get maximum number of concurrent positions"""
        return 3  # Limit concurrent positions for risk management
        
    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
            
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def _log_account_snapshot(self):
        """Log current account state to database"""
        try:
            # Calculate total exposure
            total_exposure = sum(pos.size * self.current_balance for pos in self.positions.values())
            
            # Calculate equity (balance + unrealized P&L)
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            equity = self.current_balance + unrealized_pnl
            
            # Calculate current drawdown percentage
            current_drawdown = 0
            if self.peak_balance > 0:
                current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance * 100
            
            # TODO: Calculate daily P&L (requires tracking of day start balance)
            daily_pnl = 0  # Placeholder
            
            # Log snapshot to database
            self.db_manager.log_account_snapshot(
                balance=self.current_balance,
                equity=equity,
                total_pnl=self.total_pnl,
                open_positions=len(self.positions),
                total_exposure=total_exposure,
                drawdown=current_drawdown,
                daily_pnl=daily_pnl,
                session_id=self.trading_session_id
            )
            
        except Exception as e:
            logger.error(f"Failed to log account snapshot: {e}")
        
    def _log_status(self, symbol: str, current_price: float):
        """Log current trading status"""
        total_unrealized = sum(pos.unrealized_pnl * self.current_balance for pos in self.positions.values())
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        logger.info(f"📊 Status: {symbol} @ ${current_price:.2f} | "
                   f"Balance: ${self.current_balance:.2f} | "
                   f"Positions: {len(self.positions)} | "
                   f"Unrealized: ${total_unrealized:.2f} | "
                   f"Trades: {self.total_trades} ({win_rate:.1f}% win)")
                   
    def _log_trade(self, trade: Trade):
        """Log trade to file"""
        try:
            log_file = f"trades_{datetime.now().strftime('%Y%m')}.json"
            trade_data = {
                'timestamp': trade.exit_time.isoformat(),
                'symbol': trade.symbol,
                'side': trade.side.value,
                'size': trade.size,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl,
                'exit_reason': trade.exit_reason,
                'duration_minutes': (trade.exit_time - trade.entry_time).total_seconds() / 60
            }
            
            # Append to log file
            with open(log_file, 'a') as f:
                f.write(json.dumps(trade_data) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
            
    def _send_alert(self, message: str):
        """Send trading alert (webhook, email, etc.)"""
        if not self.alert_webhook_url:
            return
            
        try:
            import requests
            payload = {
                'text': f"🤖 Trading Bot: {message}",
                'timestamp': datetime.now().isoformat()
            }
            requests.post(self.alert_webhook_url, json=payload, timeout=10)
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            
    def _sleep_with_interrupt(self, seconds: int):
        """Sleep with ability to interrupt"""
        for _ in range(seconds):
            if self.stop_event.is_set() or not self.is_running:
                break
            time.sleep(1)
            
    def _print_final_stats(self):
        """Print final trading statistics"""
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        print("\n" + "="*60)
        print("🏁 FINAL TRADING STATISTICS")
        print("="*60)
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${self.current_balance:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Total PnL: ${self.total_pnl:+,.2f}")
        print(f"Max Drawdown: {self.max_drawdown*100:.2f}%")
        print(f"Total Trades: {self.total_trades}")
        print(f"Winning Trades: {self.winning_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Active Positions: {len(self.positions)}")
        
        if self.completed_trades:
            avg_trade = sum(trade.pnl for trade in self.completed_trades) / len(self.completed_trades)
            print(f"Average Trade: ${avg_trade:.2f}")
            
        print("="*60)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'total_return_pct': total_return,
            'total_pnl': self.total_pnl,
            'max_drawdown_pct': self.max_drawdown * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate_pct': win_rate,
            'active_positions': len(self.positions),
            'last_update': self.last_data_update.isoformat() if self.last_data_update else None,
            'is_running': self.is_running
        }
    
    def _handle_strategy_change(self, swap_data: Dict[str, Any]):
        """Handle strategy change callback"""
        logger.info(f"🔄 Strategy change requested: {swap_data}")
        
        # If requested to close positions, close them now
        if swap_data.get('close_positions', False):
            logger.info("🚪 Closing all positions before strategy swap")
            for position in list(self.positions.values()):
                self._close_position(position, "Strategy change - close requested")
        else:
            logger.info("📊 Keeping existing positions during strategy swap")
    
    def _handle_model_update(self, update_data: Dict[str, Any]):
        """Handle model update callback"""
        logger.info(f"🤖 Model update requested: {update_data}")
        # Model update logic is handled in strategy_manager.apply_pending_update()
    
    def hot_swap_strategy(self, new_strategy_name: str, 
                         close_positions: bool = False,
                         new_config: Optional[Dict] = None) -> bool:
        """
        Hot-swap to a new strategy during live trading
        
        Args:
            new_strategy_name: Name of new strategy
            close_positions: Whether to close existing positions
            new_config: Configuration for new strategy
            
        Returns:
            True if swap was initiated successfully
        """
        
        if not self.strategy_manager:
            logger.error("Strategy manager not initialized - hot swapping disabled")
            return False
        
        logger.info(f"🔄 Initiating hot-swap to strategy: {new_strategy_name}")
        
        success = self.strategy_manager.hot_swap_strategy(
            new_strategy_name=new_strategy_name,
            new_config=new_config,
            close_existing_positions=close_positions
        )
        
        if success:
            logger.info(f"✅ Hot-swap initiated successfully - will apply on next cycle")
            self._send_alert(f"Strategy hot-swap initiated: {self.strategy.name} → {new_strategy_name}")
        else:
            logger.error(f"❌ Hot-swap initiation failed")
        
        return success
    
    def update_model(self, new_model_path: str) -> bool:
        """
        Update ML models during live trading
        
        Args:
            new_model_path: Path to new model file
            
        Returns:
            True if update was initiated successfully
        """
        
        if not self.strategy_manager:
            logger.error("Strategy manager not initialized - model updates disabled")
            return False
        
        strategy_name = self.strategy.name.lower()
        
        logger.info(f"🤖 Initiating model update for strategy: {strategy_name}")
        
        success = self.strategy_manager.update_model(
            strategy_name=strategy_name,
            new_model_path=new_model_path,
            validate_model=True
        )
        
        if success:
            logger.info(f"✅ Model update initiated successfully - will apply on next cycle")
            self._send_alert(f"Model update initiated for {strategy_name}")
        else:
            logger.error(f"❌ Model update initiation failed")
        
        return success 