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
        
        # Early stop tracking
        self.early_stop_reason: Optional[str] = None
        self.early_stop_date: Optional[datetime] = None
        self.early_stop_candle_index: Optional[int] = None
        
        # Database logging
        self.log_to_database = log_to_database
        self.db_manager = None
        self.trading_session_id = None
        if log_to_database:
            self.db_manager = DatabaseManager(database_url)
            # Set up strategy logging
            if self.db_manager:
                self.strategy.set_database_manager(self.db_manager)
        
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
                
                # Set session ID on strategy for logging
                if hasattr(self.strategy, 'session_id'):
                    self.strategy.session_id = self.trading_session_id
            
            # Fetch price data
            df: DataFrame = self.data_provider.get_historical_data(symbol, timeframe, start, end)
            if df.empty:
                # Return empty results for empty data
                return {
                    'total_trades': 0,
                    'final_balance': self.initial_balance,
                    'total_return': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'win_rate': 0.0,
                    'avg_trade_duration': 0.0,
                    'total_fees': 0.0,
                    'trades': []
                }
                
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Validate index type - must be datetime-like for time-series analysis
            if not isinstance(df.index, pd.DatetimeIndex):
                # Try to convert to datetime index if possible
                try:
                    df.index = pd.to_datetime(df.index)
                except (ValueError, TypeError):
                    # If conversion fails, create a dummy datetime index
                    df.index = pd.date_range(start=start, periods=len(df), freq='H')
                
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
                    exit_signal = self.strategy.check_exit_conditions(df, i, self.current_trade.entry_price)
                    
                    # Log exit decision
                    if self.log_to_database and self.db_manager:
                        indicators = self._extract_indicators(df, i)
                        sentiment_data = self._extract_sentiment_data(df, i)
                        
                        # Calculate current P&L for context
                        current_pnl = (candle['close'] - self.current_trade.entry_price) / self.current_trade.entry_price
                        
                        self.db_manager.log_strategy_execution(
                            strategy_name=self.strategy.__class__.__name__,
                            symbol=symbol,
                            signal_type='exit',
                            action_taken='closed_position' if exit_signal else 'hold_position',
                            price=candle['close'],
                            timeframe=timeframe,
                            signal_strength=1.0 if exit_signal else 0.0,
                            confidence_score=indicators.get('prediction_confidence', 0.5),
                            indicators=indicators,
                            sentiment_data=sentiment_data if sentiment_data else None,
                            position_size=self.current_trade.size,
                            reasons=[
                                'exit_signal' if exit_signal else 'holding_position',
                                f'current_pnl_{current_pnl:.4f}',
                                f'position_age_{(candle.name - self.current_trade.entry_time).total_seconds():.0f}s',
                                f'entry_price_{self.current_trade.entry_price:.2f}'
                            ],
                            volume=indicators.get('volume'),
                            volatility=indicators.get('volatility'),
                            session_id=self.trading_session_id
                        )
                    
                    if exit_signal:
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
                        
                        # After updating self.balance, update yearly_balance for the exit year
                        exit_year = candle.name.year
                        if exit_year in yearly_balance:
                            yearly_balance[exit_year]['end'] = self.balance
                        
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
                            self.early_stop_reason = f"Maximum drawdown exceeded ({current_drawdown:.1%})"
                            self.early_stop_date = candle.name
                            self.early_stop_candle_index = i
                            logger.warning(f"Maximum drawdown exceeded ({current_drawdown:.1%}). Stopping backtest.")
                            break
                
                # Check for entry if not in position
                elif self.strategy.check_entry_conditions(df, i):
                    # Calculate position size
                    size = self.strategy.calculate_position_size(df, i, self.balance)
                    
                    # Log entry decision
                    if self.log_to_database and self.db_manager:
                        indicators = self._extract_indicators(df, i)
                        sentiment_data = self._extract_sentiment_data(df, i)
                        
                        self.db_manager.log_strategy_execution(
                            strategy_name=self.strategy.__class__.__name__,
                            symbol=symbol,
                            signal_type='entry',
                            action_taken='opened_long' if size > 0 else 'no_action',
                            price=candle['close'],
                            timeframe=timeframe,
                            signal_strength=1.0 if size > 0 else 0.0,
                            confidence_score=indicators.get('prediction_confidence', 0.5),
                            indicators=indicators,
                            sentiment_data=sentiment_data if sentiment_data else None,
                            position_size=size if size > 0 else None,
                            reasons=[
                                'entry_conditions_met',
                                f'position_size_{size:.4f}' if size > 0 else 'no_position_size',
                                f'balance_{self.balance:.2f}'
                            ],
                            volume=indicators.get('volume'),
                            volatility=indicators.get('volatility'),
                            session_id=self.trading_session_id
                        )
                    
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
                
                # Log no-action cases (when no position and no entry signal)
                else:
                    # Only log every 10th candle to avoid spam, but capture key decision points
                    if i % 10 == 0 and self.log_to_database and self.db_manager:
                        indicators = self._extract_indicators(df, i)
                        sentiment_data = self._extract_sentiment_data(df, i)
                        
                        self.db_manager.log_strategy_execution(
                            strategy_name=self.strategy.__class__.__name__,
                            symbol=symbol,
                            signal_type='entry',
                            action_taken='no_action',
                            price=candle['close'],
                            timeframe=timeframe,
                            signal_strength=0.0,
                            confidence_score=indicators.get('prediction_confidence', 0.5),
                            indicators=indicators,
                            sentiment_data=sentiment_data if sentiment_data else None,
                            reasons=[
                                'no_entry_conditions',
                                f'balance_{self.balance:.2f}',
                                f'candle_{i}_of_{len(df)}'
                            ],
                            volume=indicators.get('volume'),
                            volatility=indicators.get('volatility'),
                            session_id=self.trading_session_id
                        )
            
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
                'session_id': self.trading_session_id if self.log_to_database else None,
                'early_stop_reason': self.early_stop_reason,
                'early_stop_date': self.early_stop_date,
                'early_stop_candle_index': self.early_stop_candle_index
            }
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise 
    
    def _extract_indicators(self, df: pd.DataFrame, index: int) -> Dict:
        """Extract indicator values from dataframe for logging"""
        if index >= len(df):
            return {}
            
        indicators = {}
        current_row = df.iloc[index]
        
        # Common indicators to extract
        indicator_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_hist', 'atr', 'volatility',
            'trend_ma', 'short_ma', 'long_ma', 'volume_ma', 'trend_strength',
            'regime', 'body_size', 'upper_wick', 'lower_wick', 'onnx_pred',
            'ml_prediction', 'prediction_confidence'
        ]
        
        for col in indicator_columns:
            if col in df.columns and not pd.isna(current_row[col]):
                # Only convert to float if the value is numeric (int or float)
                if col == 'regime':
                    indicators[col] = current_row[col]  # Keep as string
                else:
                    try:
                        indicators[col] = float(current_row[col])
                    except (ValueError, TypeError):
                        # Skip non-numeric values
                        continue
        
        # Add basic OHLCV data
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                indicators[col] = float(current_row[col])
        
        return indicators
    
    def _extract_sentiment_data(self, df: pd.DataFrame, index: int) -> Dict:
        """Extract sentiment data from dataframe for logging"""
        if index >= len(df):
            return {}
            
        sentiment_data = {}
        current_row = df.iloc[index]
        
        # Sentiment columns to extract
        sentiment_columns = [
            'sentiment_score', 'sentiment_primary', 'sentiment_momentum', 
            'sentiment_volatility', 'sentiment_confidence'
        ]
        
        for col in sentiment_columns:
            if col in df.columns and not pd.isna(current_row[col]):
                sentiment_data[col] = float(current_row[col])
        
        return sentiment_data 