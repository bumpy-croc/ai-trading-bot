from typing import Optional, Dict, List
import pandas as pd
import logging
from datetime import datetime
from core.data.data_provider import DataProvider
from strategies.base import BaseStrategy
from core.risk.risk_manager import RiskManager, RiskParameters
import numpy as np

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
        risk_parameters: Optional[RiskParameters] = None,
        initial_balance: float = 10000
    ):
        self.strategy = strategy
        self.data_provider = data_provider
        self.risk_manager = RiskManager(risk_parameters)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        
    def run(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> Dict:
        """Run backtest over specified period"""
        # Fetch historical data
        df = self.data_provider.get_historical_data(symbol, timeframe, start, end)
        if df.empty:
            raise ValueError("No data available for backtesting")
            
        # Prepare data with indicators
        df = self.strategy.prepare_data(df)
        
        # Remove warmup period
        df = df.dropna()
        
        logger.info(f"Starting backtest with {len(df)} candles")
        
        # Initialize metrics
        total_trades = 0
        winning_trades = 0
        returns = []
        max_drawdown = 0
        
        # Iterate through candles
        for i in range(len(df)):
            candle = df.iloc[i]
            
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
                    
                    # Calculate drawdown
                    if self.balance > self.peak_balance:
                        self.peak_balance = self.balance
                    drawdown = (self.peak_balance - self.balance) / self.peak_balance
                    max_drawdown = max(max_drawdown, drawdown)
                    
                    # Log trade
                    logger.info(f"Exited position at {candle['close']}, Balance: {self.balance:.2f}")
                    
                    # Store trade
                    self.trades.append(self.current_trade)
                    self.current_trade = None
                    
                    # Check if maximum drawdown exceeded
                    if drawdown > 0.5:  # 50% max drawdown
                        logger.warning("Maximum drawdown exceeded. Stopping backtest.")
                        break
                    
                    # Calculate return for this period
                    returns.append(trade_pnl / self.initial_balance)
            
            # Check for entry if not in position
            elif self.strategy.check_entry_conditions(df, i):
                # Calculate position size
                size = self.strategy.calculate_position_size(df, i, self.balance)
                
                if size > 0:
                    # Enter new trade
                    stop_loss = self.strategy.calculate_stop_loss(candle['close'], 'long')
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
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'max_drawdown': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': self.balance
        } 