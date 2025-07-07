"""
Data Repository

This module provides a centralized repository for all trading-related data,
including market data, sentiment data, and trade history.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from database.manager import DatabaseManager
from database.models import PositionSide, TradeSource
from data_providers.data_provider import DataProvider
from performance.metrics import total_return, sharpe, max_drawdown, cagr

logger = logging.getLogger(__name__)


class TradingDataRepository:
    """
    Repository for all trading-related data access.
    
    This class provides a single interface for fetching:
    - Market data (price, volume, indicators)
    - Trade history and performance
    - Position tracking
    - Account balance history
    """
    
    def __init__(self, db_manager: DatabaseManager, data_provider: DataProvider):
        self.db = db_manager
        self.data_provider = data_provider
        
    # ==================== Market Data ====================
    
    def get_market_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: Optional[datetime] = None,
        include_indicators: bool = True
    ) -> pd.DataFrame:
        """
        Get market data with optional technical indicators.
        
        Returns standardized DataFrame with OHLCV + indicators.
        """
        try:
            # Get raw price data
            df = self.data_provider.get_historical_data(symbol, timeframe, start_date, end_date)
            
            if df.empty:
                logger.warning(f"No market data found for {symbol} {timeframe}")
                return df
            
            # Add technical indicators if requested
            if include_indicators:
                df = self._add_technical_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        return self.data_provider.get_current_price(symbol)
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add standard technical indicators to price data"""
        from indicators.technical import calculate_rsi, calculate_ema, calculate_atr
        
        try:
            # RSI
            df['rsi'] = calculate_rsi(df['close'], period=14)
            
            # EMAs
            df['ema_9'] = calculate_ema(df['close'], period=9)
            df['ema_21'] = calculate_ema(df['close'], period=21)
            df['ema_50'] = calculate_ema(df['close'], period=50)
            
            # ATR for volatility (only if we have enough data)
            if len(df) > 14:
                atr_df = calculate_atr(df, period=14)
                df['atr'] = atr_df['atr']
            else:
                df['atr'] = df['close'] * 0.02  # Mock 2% ATR for small datasets
            
            # Trend strength (EMA slope)
            df['trend_strength'] = df['ema_21'].pct_change(periods=5)
            
            # Volume trend
            if 'volume' in df.columns:
                df['volume_ma'] = df['volume'].rolling(20).mean()
                df['volume_trend'] = df['volume'] / df['volume_ma'] - 1
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
            return df
    
    # ==================== Trade Data ====================
    
    def get_trades(
        self, 
        session_id: Optional[int] = None,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get trade history with flexible filtering"""
        
        conditions = []
        params = []
        
        if session_id is not None:
            conditions.append("session_id = %s")
            params.append(session_id)
            
        if strategy_name:
            conditions.append("strategy_name = %s")
            params.append(strategy_name)
            
        if symbol:
            conditions.append("symbol = %s")
            params.append(symbol)
            
        if start_date:
            conditions.append("entry_time >= %s")
            params.append(start_date)
            
        if end_date:
            conditions.append("exit_time <= %s")
            params.append(end_date)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        limit_clause = f" LIMIT {limit}" if limit else ""
        
        query = f"""
        SELECT * FROM trades 
        WHERE {where_clause}
        ORDER BY exit_time DESC
        {limit_clause}
        """
        
        return self.db.execute_query(query, params)
    
    def get_trade_performance(
        self, 
        session_id: Optional[int] = None,
        strategy_name: Optional[str] = None
    ) -> Dict:
        """Get aggregated trade performance metrics"""
        
        conditions = ["exit_time IS NOT NULL"]  # Only completed trades
        params = []
        
        if session_id is not None:
            conditions.append("session_id = %s")
            params.append(session_id)
            
        if strategy_name:
            conditions.append("strategy_name = %s")
            params.append(strategy_name)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
            SUM(pnl) as total_pnl,
            AVG(pnl) as avg_pnl,
            MAX(pnl) as max_win,
            MIN(pnl) as max_loss,
            STDDEV(pnl) as pnl_std
        FROM trades 
        WHERE {where_clause}
        """
        
        result = self.db.execute_query(query, params)
        if not result:
            return {}
            
        data = result[0]
        return {
            'total_trades': data['total_trades'] or 0,
            'winning_trades': data['winning_trades'] or 0,
            'win_rate': (data['winning_trades'] / data['total_trades'] * 100) if data['total_trades'] > 0 else 0,
            'total_pnl': float(data['total_pnl'] or 0),
            'avg_pnl': float(data['avg_pnl'] or 0),
            'max_win': float(data['max_win'] or 0),
            'max_loss': float(data['max_loss'] or 0),
            'profit_factor': abs(data['max_win'] / data['max_loss']) if data['max_loss'] and data['max_loss'] < 0 else 0
        }
    
    # ==================== Position Data ====================
    
    def get_active_positions(self, session_id: Optional[int] = None) -> List[Dict]:
        """Get currently active positions"""
        
        conditions = ["exit_time IS NULL"]
        params = []
        
        if session_id is not None:
            conditions.append("session_id = %s")
            params.append(session_id)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
        SELECT * FROM positions 
        WHERE {where_clause}
        ORDER BY entry_time DESC
        """
        
        return self.db.execute_query(query, params)
    
    def get_position_history(
        self, 
        session_id: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get position history"""
        
        conditions = []
        params = []
        
        if session_id is not None:
            conditions.append("session_id = %s")
            params.append(session_id)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        limit_clause = f" LIMIT {limit}" if limit else ""
        
        query = f"""
        SELECT * FROM positions 
        WHERE {where_clause}
        ORDER BY entry_time DESC
        {limit_clause}
        """
        
        return self.db.execute_query(query, params)
    
    # ==================== Account Data ====================
    
    def get_balance_history(
        self, 
        session_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """Get account balance history"""
        
        conditions = []
        params = []
        
        if session_id is not None:
            conditions.append("session_id = %s")
            params.append(session_id)
            
        if start_date:
            conditions.append("timestamp >= %s")
            params.append(start_date)
            
        if end_date:
            conditions.append("timestamp <= %s")
            params.append(end_date)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        SELECT timestamp, balance 
        FROM account_history 
        WHERE {where_clause}
        ORDER BY timestamp ASC
        """
        
        return self.db.execute_query(query, params)
    
    def get_current_balance(self, session_id: Optional[int] = None) -> float:
        """Get most recent account balance"""
        
        conditions = []
        params = []
        
        if session_id is not None:
            conditions.append("session_id = %s")
            params.append(session_id)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        SELECT balance 
        FROM account_history 
        WHERE {where_clause}
        ORDER BY timestamp DESC 
        LIMIT 1
        """
        
        result = self.db.execute_query(query, params)
        return float(result[0]['balance']) if result else 0.0
    
    # ==================== Performance Analytics ====================
    
    def calculate_session_metrics(
        self, 
        session_id: int,
        initial_balance: Optional[float] = None
    ) -> Dict:
        """Calculate comprehensive performance metrics for a session"""
        
        # Get balance history
        balance_history = self.get_balance_history(session_id=session_id)
        if not balance_history:
            return {}
        
        # Convert to pandas for calculations
        df = pd.DataFrame(balance_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Get initial balance
        if initial_balance is None:
            initial_balance = df['balance'].iloc[0]
        
        current_balance = df['balance'].iloc[-1]
        
        # Calculate metrics using shared functions
        total_ret = total_return(initial_balance, current_balance)
        
        # Calculate time-based metrics
        start_date = df.index[0]
        end_date = df.index[-1]
        duration_days = (end_date - start_date).days
        
        # CAGR calculation
        cagr_value = cagr(initial_balance, current_balance, duration_days)
        
        # Sharpe ratio
        sharpe_value = sharpe(df['balance'])
        
        # Max drawdown
        max_dd = max_drawdown(df['balance'])
        
        # Get trade statistics
        trade_stats = self.get_trade_performance(session_id=session_id)
        
        return {
            'session_id': session_id,
            'initial_balance': initial_balance,
            'final_balance': current_balance,
            'total_return_pct': total_ret,
            'cagr_pct': cagr_value,
            'sharpe_ratio': sharpe_value,
            'max_drawdown_pct': max_dd,
            'duration_days': duration_days,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            **trade_stats
        }
    
    def get_strategy_comparison(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """Compare performance across different strategies"""
        
        conditions = ["exit_time IS NOT NULL"]
        params = []
        
        if start_date:
            conditions.append("exit_time >= %s")
            params.append(start_date)
            
        if end_date:
            conditions.append("exit_time <= %s")
            params.append(end_date)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
        SELECT 
            strategy_name,
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
            SUM(pnl) as total_pnl,
            AVG(pnl) as avg_pnl,
            STDDEV(pnl) as pnl_std,
            MAX(pnl) as max_win,
            MIN(pnl) as max_loss
        FROM trades 
        WHERE {where_clause}
        GROUP BY strategy_name
        ORDER BY total_pnl DESC
        """
        
        results = self.db.execute_query(query, params)
        
        # Calculate additional metrics
        for result in results:
            total_trades = result['total_trades']
            winning_trades = result['winning_trades']
            
            result['win_rate_pct'] = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            result['total_pnl'] = float(result['total_pnl'] or 0)
            result['avg_pnl'] = float(result['avg_pnl'] or 0)
            result['profit_factor'] = abs(result['max_win'] / result['max_loss']) if result['max_loss'] and result['max_loss'] < 0 else 0
        
        return results
    
    # ==================== Session Management ====================
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict]:
        """Get recent trading sessions"""
        
        query = """
        SELECT 
            s.*,
            COUNT(t.id) as trade_count,
            SUM(t.pnl) as total_pnl
        FROM trading_sessions s
        LEFT JOIN trades t ON s.id = t.session_id
        GROUP BY s.id
        ORDER BY s.start_time DESC
        LIMIT %s
        """
        
        return self.db.execute_query(query, [limit])
    
    def get_session_details(self, session_id: int) -> Optional[Dict]:
        """Get detailed information about a specific session"""
        
        query = """
        SELECT 
            s.*,
            COUNT(t.id) as trade_count,
            SUM(t.pnl) as total_pnl,
            AVG(t.pnl) as avg_pnl,
            SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) as winning_trades
        FROM trading_sessions s
        LEFT JOIN trades t ON s.id = t.session_id
        WHERE s.id = %s
        GROUP BY s.id
        """
        
        result = self.db.execute_query(query, [session_id])
        return result[0] if result else None 