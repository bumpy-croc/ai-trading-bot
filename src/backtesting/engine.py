from typing import Optional, Dict, List
import pandas as pd
import logging
from datetime import datetime
from data_providers.data_provider import DataProvider
from strategies.base import BaseStrategy
from risk.risk_manager import RiskManager, RiskParameters
import numpy as np
from data_providers.sentiment_provider import SentimentDataProvider
from database.manager import DatabaseManager
from database.models import TradeSource, PositionSide

logger = logging.getLogger(__name__)

class Trade:
    """Represents a single trade"""
    def __init__(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        entry_time: datetime,
        size: float,
        stop_loss: float,
        take_profit: Optional[float] = None
    ):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.size = min(size, 1.0)  # Limit position size to 100% of balance
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.pnl: Optional[float] = None
        self.exit_reason: Optional[str] = None
        
    def close(self, price: float, time: datetime, reason: str):
        """Close the trade and calculate PnL"""
        self.exit_price = price
        self.exit_time = time
        self.exit_reason = reason
        
        # Calculate percentage return
        if self.side == 'long':
            self.pnl = ((self.exit_price - self.entry_price) / self.entry_price) * self.size
        else:  # short
            self.pnl = ((self.entry_price - self.exit_price) / self.entry_price) * self.size

class Backtester:
    """Backtesting engine for trading strategies"""
    
    def __init__(
        self,
        strategy: BaseStrategy,
        data_provider: DataProvider,
        sentiment_provider: Optional[SentimentDataProvider] = None,
        risk_parameters: Optional[RiskParameters] = None,
        initial_balance: float = 10000,
        database_url: Optional[str] = None,
        log_to_database: bool = True
    ):
        self.strategy = strategy
        self.data_provider = data_provider
        self.sentiment_provider = sentiment_provider
        self.risk_parameters = risk_parameters
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        
        # Database logging
        self.log_to_database = log_to_database
        self.db_manager = None
        self.trading_session_id = None
        if log_to_database:
            self.db_manager = DatabaseManager(database_url)
        
    def run(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> Dict:
        """Run backtest with sentiment data if available"""
        try:
            # Create trading session in database if enabled
            if self.log_to_database and self.db_manager:
                self.trading_session_id = self.db_manager.create_trading_session(
                    strategy_name=self.strategy.__class__.__name__,
                    symbol=symbol,
                    timeframe=timeframe,
                    mode=TradeSource.BACKTEST,
                    initial_balance=self.initial_balance,
                    strategy_config=getattr(self.strategy, 'config', {}),
                    session_name=f"Backtest_{symbol}_{start.strftime('%Y%m%d')}"
                )
            
            # Fetch price data
            df = self.data_provider.get_historical_data(symbol, timeframe, start, end)
            if df.empty:
                raise ValueError("No price data available for the specified period")
                
            # Fetch sentiment data if provider is available
            if self.sentiment_provider:
                sentiment_df = self.sentiment_provider.get_historical_sentiment(
                    symbol, start, end
                )
                if not sentiment_df.empty:
                    # Aggregate sentiment data to match price timeframe
                    sentiment_df = self.sentiment_provider.aggregate_sentiment(
                        sentiment_df, window=timeframe
                    )
                    # Merge sentiment data with price data
                    df = df.join(sentiment_df, how='left')
                    # Forward fill sentiment scores
                    df['sentiment_score'] = df['sentiment_score'].fillna(method='ffill')
                    # Fill any remaining NaN values with 0
                    df['sentiment_score'] = df['sentiment_score'].fillna(0)
            
            # Calculate indicators
            df = self.strategy.calculate_indicators(df)
            
            # Remove warmup period - only drop rows where essential price data is missing
            # Don't drop rows just because ML predictions or sentiment data is missing
            essential_columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.dropna(subset=essential_columns)
            
            logger.info(f"Starting backtest with {len(df)} candles")
            
            # Initialize metrics
            total_trades = 0
            winning_trades = 0
            returns = []
            max_drawdown = 0
            
            # Iterate through candles
            for i in range(len(df)):
                candle = df.iloc[i]
                
                # Update max drawdown
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance
                current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
                max_drawdown = max(max_drawdown, current_drawdown)
                
                # Check for exit if in position
                if self.current_trade is not None:
                    if self.strategy.check_exit_conditions(df, i, self.current_trade.entry_price):
                        # Close the trade
                        self.current_trade.close(candle['close'], candle.name, "Strategy exit")
                        
                        # Update balance
                        trade_pnl = self.current_trade.pnl * self.balance
                        self.balance += trade_pnl
                        
                        # Update metrics
                        total_trades += 1
                        if trade_pnl > 0:
                            winning_trades += 1
                        
                        # Calculate return for this period
                        trade_return = trade_pnl / (self.current_trade.entry_price * self.current_trade.size)
                        returns.append(trade_return)
                        
                        # Log trade
                        logger.info(f"Exited position at {candle['close']}, Balance: {self.balance:.2f}")
                        
                        # Log to database if enabled
                        if self.log_to_database and self.db_manager:
                            self.db_manager.log_trade(
                                symbol=symbol,
                                side="long",  # Backtester only does long trades currently
                                entry_price=self.current_trade.entry_price,
                                exit_price=self.current_trade.exit_price,
                                size=self.current_trade.size,
                                entry_time=self.current_trade.entry_time,
                                exit_time=self.current_trade.exit_time,
                                pnl=trade_pnl,
                                exit_reason=self.current_trade.exit_reason,
                                strategy_name=self.strategy.__class__.__name__,
                                source=TradeSource.BACKTEST,
                                stop_loss=self.current_trade.stop_loss,
                                take_profit=self.current_trade.take_profit,
                                session_id=self.trading_session_id
                            )
                        
                        # Store trade
                        self.trades.append(self.current_trade)
                        self.current_trade = None
                        
                        # Check if maximum drawdown exceeded
                        if current_drawdown > 0.5:  # 50% max drawdown
                            logger.warning("Maximum drawdown exceeded. Stopping backtest.")
                            break
                
                # Check for entry if not in position
                elif self.strategy.check_entry_conditions(df, i):
                    # Calculate position size
                    size = self.strategy.calculate_position_size(df, i, self.balance)
                    
                    if size > 0:
                        # Enter new trade
                        # Assuming df and index are available in this context
                        stop_loss = self.strategy.calculate_stop_loss(df, len(df) - 1, candle['close'], 'long')
                        self.current_trade = Trade(
                            symbol=symbol,
                            side='long',
                            entry_price=candle['close'],
                            entry_time=candle.name,
                            size=size,
                            stop_loss=stop_loss
                        )
                        logger.info(f"Entered long position at {candle['close']}")
            
            # Calculate final metrics
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
            
            # Calculate Sharpe ratio
            if len(returns) > 0:
                returns_array = np.array(returns)
                sharpe_ratio = np.sqrt(252) * (np.mean(returns_array) / np.std(returns_array)) if np.std(returns_array) != 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calculate annualized return
            days = (end - start).days if end else (datetime.now() - start).days
            annualized_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100

            # --- Yearly returns calculation ---
            yearly_returns = {}
            if not df.empty and hasattr(df.index, 'year'):
                df_years = df.copy()
                df_years['year'] = df_years.index.year
                for year, group in df_years.groupby('year'):
                    first_close = group['close'].iloc[0]
                    last_close = group['close'].iloc[-1]
                    if first_close > 0:
                        yearly_return = (last_close / first_close - 1) * 100
                        yearly_returns[str(year)] = yearly_return
            # --- End yearly returns ---
            
            # End trading session in database if enabled
            if self.log_to_database and self.db_manager and self.trading_session_id:
                self.db_manager.end_trading_session(
                    session_id=self.trading_session_id,
                    final_balance=self.balance
                )

            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'max_drawdown': max_drawdown * 100,
                'sharpe_ratio': sharpe_ratio,
                'final_balance': self.balance,
                'annualized_return': annualized_return,
                'yearly_returns': yearly_returns,
                'session_id': self.trading_session_id if self.log_to_database else None
            }
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise 