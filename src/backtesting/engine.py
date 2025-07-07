from typing import Optional, Dict, List, Any, Iterator
from pandas import DataFrame  # type: ignore
import pandas as pd  # type: ignore
import logging
from datetime import datetime
from data_providers.data_provider import DataProvider
from strategies.base import BaseStrategy
from risk.risk_manager import RiskManager, RiskParameters
import numpy as np  # type: ignore
from data_providers.sentiment_provider import SentimentDataProvider
from database.manager import DatabaseManager
from database.models import TradeSource, PositionSide
from config.constants import DEFAULT_INITIAL_BALANCE

# Shared performance metrics
from performance.metrics import (
    Side,
    cash_pnl,
    total_return as perf_total_return,
    cagr as perf_cagr,
    sharpe as perf_sharpe,
    max_drawdown as perf_max_drawdown,
)

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
        risk_parameters: Optional[Any] = None,
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
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
            df: DataFrame = self.data_provider.get_historical_data(symbol, timeframe, start, end)
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
            
            # -----------------------------
            # Metrics & tracking variables
            # -----------------------------
            total_trades = 0
            winning_trades = 0
            max_drawdown_running = 0  # interim tracker (still used for intra-loop stopping)

            # Track balance over time to enable robust performance stats
            balance_history = []  # (timestamp, balance)

            # Helper dict to track first/last balance of each calendar year
            yearly_balance = {}
            
            # Iterate through candles
            for i in range(len(df)):
                candle = df.iloc[i]
                
                # Record current balance for time-series analytics
                balance_history.append((candle.name, self.balance))

                # Track yearly start/end balances for return calc
                yr = candle.name.year
                if yr not in yearly_balance:
                    yearly_balance[yr] = {
                        'start': self.balance,
                        'end': self.balance
                    }
                else:
                    yearly_balance[yr]['end'] = self.balance
                
                # Update max drawdown
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance
                current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
                max_drawdown_running = max(max_drawdown_running, current_drawdown)
                
                # Check for exit if in position
                if self.current_trade is not None:
                    if self.strategy.check_exit_conditions(df, i, self.current_trade.entry_price):
                        # Close the trade
                        self.current_trade.close(candle['close'], candle.name, "Strategy exit")
                        
                        # Update balance (convert percentage PnL to absolute currency)
                        trade_pnl_percent: float = float(self.current_trade.pnl or 0.0)  # e.g. 0.02 for +2%
                        # Convert to absolute profit/loss based on current balance BEFORE applying PnL
                        trade_pnl: float = cash_pnl(trade_pnl_percent, self.balance)

                        self.balance += trade_pnl

                        # Update metrics
                        total_trades += 1
                        if trade_pnl > 0:
                            winning_trades += 1
                        
                        # Log trade
                        logger.info(f"Exited position at {candle['close']}, Balance: {self.balance:.2f}")
                        
                        # Log to database if enabled
                        if (self.log_to_database and self.db_manager and
                                self.current_trade.exit_price is not None and
                                self.current_trade.exit_time is not None and
                                self.current_trade.exit_reason is not None):
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
            total_return = perf_total_return(self.initial_balance, self.balance)
            
            # ----------------------------------------------
            # Sharpe ratio ‑ use *daily* returns of balance
            # ----------------------------------------------
            if balance_history:
                bh_df = pd.DataFrame(balance_history, columns=['timestamp', 'balance']).set_index('timestamp')
                # Resample to 1-day frequency for stability
                daily_balance = bh_df['balance'].resample('1D').last().ffill()
                daily_returns = daily_balance.pct_change().dropna()
                if not daily_returns.empty and daily_returns.std() != 0:
                    sharpe_ratio = perf_sharpe(daily_balance)
                else:
                    sharpe_ratio = 0
                # Re-calculate max drawdown from full equity curve
                max_drawdown_pct = perf_max_drawdown(daily_balance)
            else:
                sharpe_ratio = 0
                max_drawdown_pct = 0
            
            # Calculate annualized return
            days = (end - start).days if end else (datetime.now() - start).days
            annualized_return = perf_cagr(self.initial_balance, self.balance, days)

            # ---------------------------------------------
            # Yearly returns based on account balance
            # ---------------------------------------------
            yearly_returns = {}
            for yr, bal in yearly_balance.items():
                start_bal = bal['start']
                end_bal = bal['end']
                if start_bal > 0:
                    yearly_returns[str(yr)] = (end_bal / start_bal - 1) * 100
            
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
                'max_drawdown': max_drawdown_pct,
                'sharpe_ratio': sharpe_ratio,
                'final_balance': self.balance,
                'annualized_return': annualized_return,
                'yearly_returns': yearly_returns,
                'session_id': self.trading_session_id if self.log_to_database else None
            }
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise 