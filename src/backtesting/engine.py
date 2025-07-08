"""
Backtesting Engine

This engine integrates the following components:
- TradeExecutor for consistent trade management
- SignalGenerator for standardized signal generation
- TradingDataRepository for data access
- BacktestOrderExecutor for simulated order execution
"""

from typing import Optional, Dict, List, Any
import pandas as pd
import logging
from datetime import datetime

from execution.trade_executor import TradeExecutor, TradeRequest, CloseRequest, ExecutionMode
from execution.signal_generator import SignalGenerator, MarketContext
from execution.order_executors import BacktestOrderExecutor
from data.repository import TradingDataRepository
from strategies.base import BaseStrategy
from database.manager import DatabaseManager
from database.models import TradeSource
from config.constants import DEFAULT_INITIAL_BALANCE
from performance.metrics import Side

logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtesting engine using the new component architecture.
    
    This replaces the original Backtester with a cleaner implementation
    that uses our TradeExecutor, SignalGenerator, and TradingDataRepository.
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        data_provider: Any,  # Can be DataProvider or TradingDataRepository
        sentiment_provider: Optional[Any] = None,  # Ignored for compatibility
        risk_parameters: Optional[Any] = None,  # Ignored for compatibility
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
        database_url: Optional[str] = None,
        log_to_database: bool = True
    ):
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.log_to_database = log_to_database
        
        # Handle both DataProvider and TradingDataRepository
        if hasattr(data_provider, 'get_market_data'):
            # It's a TradingDataRepository
            self.data_repository = data_provider
            self.data_provider = None
        else:
            # It's a DataProvider, create a TradingDataRepository
            self.data_provider = data_provider
            # Create database manager if needed
            db_manager = DatabaseManager(database_url) if log_to_database else None
            self.data_repository = TradingDataRepository(db_manager, data_provider)
        
        # Initialize database manager
        self.db_manager = None
        self.trading_session_id = None
        if log_to_database:
            self.db_manager = DatabaseManager(database_url)
        
        # Initialize components (will be set up during run)
        self.signal_generator = None
        self.trade_executor = None
        self.order_executor = None
        
    def run(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> Dict:
        """Run backtest"""
        try:
            # Create trading session in database if enabled
            if self.log_to_database and self.db_manager:
                self.trading_session_id = self.db_manager.create_trading_session(
                    strategy_name=self.strategy.__class__.__name__,
                    symbol=symbol,
                    timeframe=timeframe,
                    mode=TradeSource.BACKTEST,
                    initial_balance=self.initial_balance,
                    strategy_config=getattr(self.strategy, 'get_parameters', lambda: {})(),
                    session_name=f"Backtest_{symbol}_{start.strftime('%Y%m%d')}"
                )
            
            # Get market data with indicators
            df = self.data_repository.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start,
                end_date=end,
                include_indicators=True
            )
            
            if df.empty:
                raise ValueError("No market data available for the specified period")
            
            logger.info(f"Starting backtest with {len(df)} candles")
            
            # Initialize components
            self.signal_generator = SignalGenerator(self.strategy)
            self.order_executor = BacktestOrderExecutor(df)
            self.trade_executor = TradeExecutor(
                mode=ExecutionMode.BACKTEST,
                order_executor=self.order_executor,
                db_manager=self.db_manager,
                session_id=self.trading_session_id,
                initial_balance=self.initial_balance
            )
            
            # Track performance over time
            equity_curve = []
            
            # Performance tracking
            log_interval = max(1, len(df) // 100)  # Log every 1% of progress
            
            # Main backtesting loop
            for i in range(len(df)):
                candle = df.iloc[i]
                
                # Update order executor's current index
                self.order_executor.set_current_index(i)
                
                # Record equity point (limit memory usage)
                equity_point = {
                    'timestamp': candle.name,
                    'balance': self.trade_executor.current_balance,
                    'equity': self._calculate_current_equity(candle['close'])
                }
                equity_curve.append(equity_point)
                
                # Limit equity curve size to prevent memory issues
                if len(equity_curve) > 10000:  # Keep last 10k points
                    equity_curve = equity_curve[::2]  # Downsample by 2x
                
                # Create market context for signal generation
                context = MarketContext(
                    symbol=symbol,
                    current_price=candle['close'],
                    timestamp=candle.name,
                    timeframe=timeframe,
                    data=df.iloc[:i+1],  # Historical data up to current point
                    index=i
                )
                
                # Generate signal
                signal = self.signal_generator.generate_signal(
                    context=context,
                    current_balance=self.trade_executor.current_balance,
                    active_positions=self.trade_executor.active_positions
                )
                
                # Process signal
                if signal.action == "enter" and len(self.trade_executor.active_positions) == 0:
                    # Open new position
                    trade_request = TradeRequest(
                        symbol=symbol,
                        side=signal.side,
                        size=signal.position_size,
                        price=signal.price or candle['close'],
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        strategy_name=self.strategy.__class__.__name__,
                        confidence=signal.confidence
                    )
                    
                    result = self.trade_executor.open_position(trade_request)
                    if result.success and i % log_interval == 0:
                        logger.info(f"Opened position: {signal.side.value} at {result.executed_price}")
                    elif not result.success:
                        logger.warning(f"Failed to open position: {result.error_message}")
                
                elif signal.action == "exit" and len(self.trade_executor.active_positions) > 0:
                    # Close existing positions
                    for position_id in list(self.trade_executor.active_positions.keys()):
                        close_request = CloseRequest(
                            position_id=position_id,
                            reason="strategy_exit",
                            price=candle['close']
                        )
                        
                        result = self.trade_executor.close_position(close_request)
                        if result.success and i % log_interval == 0:
                            logger.info(f"Closed position at {result.executed_price}, P&L: {result.pnl:.2f}")
                        elif not result.success:
                            logger.warning(f"Failed to close position: {result.error_message}")
                
                # Check for stop losses and take profits
                current_price = candle['close']
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
                
                # Close positions that hit stops
                for position_id, reason in positions_to_close:
                    close_request = CloseRequest(
                        position_id=position_id,
                        reason=reason,
                        price=current_price
                    )
                    result = self.trade_executor.close_position(close_request)
                    if result.success and i % log_interval == 0:
                        logger.info(f"Closed position ({reason}) at {result.executed_price}")
                
                # Progress logging
                if i % log_interval == 0:
                    progress = (i + 1) / len(df) * 100
                    logger.info(f"Backtest progress: {progress:.1f}% ({i+1}/{len(df)} candles)")
                
                # Check for maximum drawdown (safety stop)
                if self._calculate_current_drawdown(candle['close']) > 0.5:  # 50% max drawdown
                    logger.warning("Maximum drawdown exceeded. Stopping backtest.")
                    break
            
            # Close any remaining positions at the end
            final_price = df.iloc[-1]['close']
            for position_id in list(self.trade_executor.active_positions.keys()):
                close_request = CloseRequest(
                    position_id=position_id,
                    reason="backtest_end",
                    price=final_price
                )
                self.trade_executor.close_position(close_request)
            
            # Flush any pending batch operations
            if hasattr(self.trade_executor, 'flush_batch_operations'):
                self.trade_executor.flush_batch_operations()
            
            # Calculate final performance metrics
            performance_metrics = self._calculate_performance_metrics(equity_curve, start, end)
            
            # End trading session in database if enabled
            if self.log_to_database and self.db_manager and self.trading_session_id:
                self.db_manager.end_trading_session(
                    session_id=self.trading_session_id,
                    final_balance=self.trade_executor.current_balance
                )
            
            # Add session info to results
            performance_metrics['session_id'] = self.trading_session_id if self.log_to_database else None
            performance_metrics['strategy_name'] = self.strategy.__class__.__name__
            performance_metrics['symbol'] = symbol
            performance_metrics['timeframe'] = timeframe
            performance_metrics['start_date'] = start
            performance_metrics['end_date'] = end or datetime.now()
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise
    
    def _calculate_current_equity(self, current_price: float) -> float:
        """Calculate current equity including unrealized P&L"""
        equity = self.trade_executor.current_balance
        
        # Add unrealized P&L from open positions
        for position in self.trade_executor.active_positions.values():
            if position.side == Side.LONG:
                unrealized_pnl = (current_price - position.entry_price) / position.entry_price * position.size
            else:  # SHORT
                unrealized_pnl = (position.entry_price - current_price) / position.entry_price * position.size
            
            equity += unrealized_pnl * self.trade_executor.current_balance
        
        return equity
    
    def _calculate_current_drawdown(self, current_price: float) -> float:
        """Calculate current drawdown"""
        if not hasattr(self, '_peak_equity'):
            self._peak_equity = self.initial_balance

        current_equity = self._calculate_current_equity(current_price)

        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        return (self._peak_equity - current_equity) / self._peak_equity if self._peak_equity > 0 else 0
    
    def _calculate_performance_metrics(
        self,
        equity_curve: List[Dict],
        start_date: datetime,
        end_date: Optional[datetime]
    ) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        # Use database metrics if available, otherwise calculate from trade executor
        if self.trading_session_id and self.data_repository:
            try:
                trade_stats = self.data_repository.get_trade_performance(
                    session_id=self.trading_session_id
                )
                session_metrics = self.data_repository.calculate_session_metrics(
                    session_id=self.trading_session_id,
                    initial_balance=self.initial_balance
                )
            except Exception as e:
                logger.warning(f"Could not get database metrics: {e}")
                trade_stats = self._calculate_trade_stats_from_executor()
                session_metrics = self._calculate_session_metrics_from_equity(equity_curve)
        else:
            # Calculate from trade executor and equity curve
            trade_stats = self._calculate_trade_stats_from_executor()
            session_metrics = self._calculate_session_metrics_from_equity(equity_curve)
        
        # Calculate yearly returns from equity curve
        yearly_returns = self._calculate_yearly_returns(equity_curve)
        
        # Combine all metrics
        return {
            'total_trades': trade_stats.get('total_trades', 0),
            'win_rate': trade_stats.get('win_rate', 0),
            'total_return': session_metrics.get('total_return_pct', 0),
            'max_drawdown': session_metrics.get('max_drawdown_pct', 0),
            'sharpe_ratio': session_metrics.get('sharpe_ratio', 0),
            'initial_balance': self.initial_balance,
            'final_balance': self.trade_executor.current_balance,
            'annualized_return': session_metrics.get('cagr', 0),
            'yearly_returns': yearly_returns,
            'total_pnl': trade_stats.get('total_pnl', 0),
            'avg_trade_pnl': trade_stats.get('avg_pnl', 0),
            'max_win': trade_stats.get('max_win', 0),
            'max_loss': trade_stats.get('max_loss', 0),
            'profit_factor': trade_stats.get('profit_factor', 0)
        }
    
    def _calculate_trade_stats_from_executor(self) -> Dict:
        """Calculate trade statistics from trade executor history"""
        completed_trades = self.trade_executor.completed_trades
        
        if not completed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'max_win': 0,
                'max_loss': 0,
                'profit_factor': 0
            }
        
        total_trades = len(completed_trades)
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        losing_trades = [t for t in completed_trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        total_pnl = sum(t.pnl for t in completed_trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        max_win = max((t.pnl for t in winning_trades), default=0)
        max_loss = min((t.pnl for t in losing_trades), default=0)
        
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_win': max_win,
            'max_loss': max_loss,
            'profit_factor': profit_factor
        }
    
    def _calculate_yearly_returns(self, equity_curve: List[Dict]) -> Dict:
        """Calculate yearly returns from equity curve"""
        if not equity_curve:
            return {}
        
        try:
            equity_df = pd.DataFrame(equity_curve).set_index('timestamp')
            equity_df['year'] = equity_df.index.year
            yearly_returns = {}
            
            for year in equity_df['year'].unique():
                year_data = equity_df[equity_df['year'] == year]
                if len(year_data) > 1:
                    start_balance = year_data['balance'].iloc[0]
                    end_balance = year_data['balance'].iloc[-1]
                    if start_balance > 0:
                        yearly_returns[str(year)] = (end_balance / start_balance - 1) * 100
            
            return yearly_returns
        except Exception as e:
            logger.warning(f"Could not calculate yearly returns: {e}")
            return {}
    
    def _calculate_session_metrics_from_equity(self, equity_curve: List[Dict]) -> Dict:
        """Calculate session metrics from equity curve"""
        if not equity_curve:
            return {
                'total_return_pct': 0,
                'max_drawdown_pct': 0,
                'sharpe_ratio': 0,
                'cagr': 0
            }
        
        # Convert to DataFrame for analysis
        equity_df = pd.DataFrame(equity_curve).set_index('timestamp')
        
        # Calculate total return
        start_balance = equity_df['balance'].iloc[0]
        end_balance = equity_df['balance'].iloc[-1]
        total_return_pct = (end_balance / start_balance - 1) * 100 if start_balance > 0 else 0
        
        # Calculate maximum drawdown
        peak = equity_df['balance'].expanding().max()
        drawdown = (peak - equity_df['balance']) / peak * 100
        max_drawdown_pct = drawdown.max()
        
        # Calculate CAGR (Compound Annual Growth Rate)
        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = days / 365.25
        cagr = ((end_balance / start_balance) ** (1 / years) - 1) * 100 if years > 0 and start_balance > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        returns = equity_df['balance'].pct_change().dropna()
        if len(returns) > 1:
            sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_return_pct': total_return_pct,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'cagr': cagr
        }
    
    def get_trade_history(self) -> List[Dict]:
        """Get detailed trade history"""
        if not self.trading_session_id or not self.data_repository:
            return []
        
        return self.data_repository.get_trades(session_id=self.trading_session_id)
    
    def get_signal_history(self) -> List[Dict]:
        """Get signal generation history"""
        if not self.signal_generator:
            return []
        
        return [signal.__dict__ for signal in self.signal_generator.get_signal_history()] 