"""
PostgreSQL database manager for handling all database operations
"""

import logging
import os
from typing import Optional, Dict, List, Any, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
import json

from sqlalchemy import create_engine, func, and_, or_, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool

from .models import (
    Base, Trade, Position, AccountHistory, PerformanceMetrics,
    TradingSession, SystemEvent, StrategyExecution,
    PositionSide, OrderStatus, TradeSource, EventType
)
from config.config_manager import get_config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages all PostgreSQL database operations for the trading system.
    
    Features:
    - PostgreSQL connection pooling and session management
    - Trade and position logging
    - Performance metrics calculation
    - Account history tracking
    - System event logging
    - Centralized database for shared access across services
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize PostgreSQL database manager.
        
        Args:
            database_url: Optional PostgreSQL database URL. If None, uses DATABASE_URL from environment.
        """
        self.database_url = database_url
        self.engine = None
        self.session_factory = None
        self._current_session_id: Optional[int] = None
        
        # Initialize database connection
        self._init_database()
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _init_database(self):
        """Initialize PostgreSQL database connection and session factory"""
        
        # Get database URL from configuration if not provided
        if self.database_url is None:
            config = get_config()
            self.database_url = config.get('DATABASE_URL')
            
            if self.database_url is None:
                raise ValueError(
                    "DATABASE_URL environment variable is required for PostgreSQL connection. "
                    "Please set DATABASE_URL in your environment or Railway configuration."
                )
        
        # Validate PostgreSQL URL
        if not self.database_url.startswith('postgresql'):
            raise ValueError(
                f"Only PostgreSQL databases are supported. "
                f"Expected URL to start with 'postgresql://', got: {self.database_url[:20]}..."
            )
        
        # Create engine with PostgreSQL configuration
        engine_kwargs = self._get_engine_config()
        
        try:
            self.engine = create_engine(self.database_url, **engine_kwargs)
            self.session_factory = sessionmaker(bind=self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
                logger.info("✅ PostgreSQL database connection established")
                
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL database: {e}")
            raise
    
    def _get_engine_config(self) -> Dict[str, Any]:
        """Get PostgreSQL engine configuration"""
        
        return {
            'poolclass': QueuePool,
            'pool_size': 5,
            'max_overflow': 10,
            'pool_pre_ping': True,
            'pool_recycle': 3600,  # 1 hour
            'echo': False,  # Set to True for SQL debugging
            'connect_args': {
                'sslmode': 'prefer',
                'connect_timeout': 10,
                'application_name': 'ai-trading-bot'
            }
        }
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        try:
            if self.engine is None:
                raise ValueError("Database engine not initialized")
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """
        Get a database session with automatic cleanup.
        
        Yields:
            SQLAlchemy session
        """
        if self.session_factory is None:
            raise ValueError("Session factory not initialized")
        
        session = self.session_factory()
        try:
            yield session
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """
        Test PostgreSQL database connection.
        
        Returns:
            True if connection is successful
        """
        try:
            with self.get_session() as session:
                # Use SQLAlchemy text() for security
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get PostgreSQL database information.
        
        Returns:
            Dictionary with database connection details
        """
        info = {
            'database_url': self.database_url,
            'database_type': 'postgresql',
            'connection_pool_size': getattr(self.engine.pool, 'size', 0) if self.engine else 0,
            'checked_in_connections': getattr(self.engine.pool, 'checkedin', 0) if self.engine else 0,
            'checked_out_connections': getattr(self.engine.pool, 'checkedout', 0) if self.engine else 0,
        }
        
        return info
    
    def create_trading_session(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        mode: Union[str, TradeSource],
        initial_balance: float,
        strategy_config: Optional[Dict] = None,
        session_name: Optional[str] = None
    ) -> int:
        """
        Create a new trading session.
        
        Returns:
            Trading session ID
        """
        with self.get_session() as session:
            # Convert string enum if necessary
            if isinstance(mode, str):
                mode = TradeSource[mode.upper()]
            
            # Generate session name if not provided - use UTC for consistency
            if session_name is None:
                session_name = f"{strategy_name}_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            trading_session = TradingSession(
                session_name=session_name,
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                mode=mode,
                initial_balance=initial_balance,
                strategy_config=strategy_config,
                start_time=datetime.utcnow()
            )
            
            session.add(trading_session)
            session.commit()
            
            # Set as current session
            self._current_session_id = trading_session.id
            
            # Log session creation
            self.log_event(
                event_type=EventType.ENGINE_START,
                message=f"Trading session created: {session_name}",
                details={
                    'session_id': trading_session.id,
                    'strategy': strategy_name,
                    'symbol': symbol,
                    'mode': mode.value,
                    'initial_balance': initial_balance
                }
            )
            
            logger.info(f"Created trading session #{trading_session.id}: {session_name}")
            return trading_session.id
    
    def end_trading_session(self, session_id: Optional[int] = None, final_balance: Optional[float] = None):
        """End a trading session and calculate final metrics."""
        session_id = session_id or self._current_session_id
        if not session_id:
            logger.warning("No active session to end")
            return
        
        with self.get_session() as db:
            trading_session = db.query(TradingSession).filter_by(id=session_id).first()
            if not trading_session:
                logger.error(f"Trading session {session_id} not found")
                return
            
            # Calculate final metrics
            trades = db.query(Trade).filter_by(session_id=session_id).all()
            
            trading_session.end_time = datetime.utcnow()
            trading_session.is_active = False
            trading_session.final_balance = final_balance
            trading_session.total_trades = len(trades)
            
            if trades:
                winning_trades = [t for t in trades if t.pnl > 0]
                # Protect against division by zero
                trading_session.win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
                trading_session.total_pnl = sum(t.pnl for t in trades)
                
                # Calculate max drawdown from account history
                account_history = db.query(AccountHistory).filter_by(
                    session_id=session_id
                ).order_by(AccountHistory.timestamp).all()
                
                if account_history:
                    peak_balance = trading_session.initial_balance
                    max_drawdown = 0
                    
                    for record in account_history:
                        if record.balance > peak_balance:
                            peak_balance = record.balance
                        # Protect against division by zero
                        if peak_balance > 0:
                            drawdown = (peak_balance - record.balance) / peak_balance
                            max_drawdown = max(max_drawdown, drawdown)
                    
                    trading_session.max_drawdown = max_drawdown * 100
            
            db.commit()
            
            # Log session end event
            self.log_event(
                event_type=EventType.ENGINE_STOP,
                message=f"Trading session ended: {trading_session.session_name}",
                details={
                    'duration_hours': (trading_session.end_time - trading_session.start_time).total_seconds() / 3600,
                    'total_trades': trading_session.total_trades,
                    'final_pnl': trading_session.total_pnl,
                    'win_rate': trading_session.win_rate
                }
            )
            
            logger.info(f"Ended trading session #{session_id}")
    
    def log_trade(
        self,
        symbol: str,
        side: Union[str, PositionSide],
        entry_price: float,
        exit_price: float,
        size: float,
        entry_time: datetime,
        exit_time: datetime,
        pnl: float,
        exit_reason: str,
        strategy_name: str,
        source: Union[str, TradeSource] = TradeSource.LIVE,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        order_id: Optional[str] = None,
        confidence_score: Optional[float] = None,
        strategy_config: Optional[Dict] = None,
        session_id: Optional[int] = None
    ) -> int:
        """
        Log a completed trade to the database.
        
        Returns:
            Trade ID
        """
        with self.get_session() as session:
            # Convert string enums if necessary
            if isinstance(side, str):
                side = PositionSide[side.upper()]
            if isinstance(source, str):
                source = TradeSource[source.upper()]
            
            # Calculate percentage P&L with division by zero protection
            if entry_price > 0:
                if side == PositionSide.LONG:
                    pnl_percent = ((exit_price - entry_price) / entry_price) * 100
                else:
                    pnl_percent = ((entry_price - exit_price) / entry_price) * 100
            else:
                logger.warning(f"Invalid entry price {entry_price} for trade calculation")
                pnl_percent = 0.0
            
            trade = Trade(
                symbol=symbol,
                side=side,
                source=source,
                entry_price=entry_price,
                exit_price=exit_price,
                size=size,
                entry_time=entry_time,
                exit_time=exit_time,
                pnl=pnl,
                pnl_percent=pnl_percent,
                exit_reason=exit_reason,
                strategy_name=strategy_name,
                stop_loss=stop_loss,
                take_profit=take_profit,
                order_id=order_id,
                confidence_score=confidence_score,
                strategy_config=strategy_config,
                session_id=session_id or self._current_session_id
            )
            
            session.add(trade)
            session.commit()
            
            logger.info(f"Logged trade #{trade.id}: {symbol} {side.value} P&L: ${pnl:.2f} ({pnl_percent:.2f}%)")
            
            # Update performance metrics - handle None session_id
            effective_session_id = session_id or self._current_session_id
            if effective_session_id:
                self._update_performance_metrics(effective_session_id)
            
            return trade.id
    
    def log_position(
        self,
        symbol: str,
        side: Union[str, PositionSide],
        entry_price: float,
        size: float,
        strategy_name: str,
        order_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        confidence_score: Optional[float] = None,
        quantity: Optional[float] = None,
        session_id: Optional[int] = None
    ) -> int:
        """
        Log a new position to the database.
        
        Returns:
            Position ID
        """
        with self.get_session() as session:
            # Convert string enum if necessary
            if isinstance(side, str):
                side = PositionSide[side.upper()]
            
            position = Position(
                symbol=symbol,
                side=side,
                status=OrderStatus.FILLED,
                entry_price=entry_price,
                size=size,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_time=datetime.utcnow(),
                strategy_name=strategy_name,
                confidence_score=confidence_score,
                order_id=order_id,
                session_id=session_id or self._current_session_id
            )
            
            session.add(position)
            session.commit()
            
            logger.info(f"Logged position #{position.id}: {symbol} {side.value} @ ${entry_price:.2f}")
            return position.id
    
    def update_position(
        self,
        position_id: int,
        current_price: float,
        unrealized_pnl: Optional[float] = None,
        unrealized_pnl_percent: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        """Update an existing position with current market data."""
        with self.get_session() as session:
            position = session.query(Position).filter_by(id=position_id).first()
            if not position:
                logger.error(f"Position {position_id} not found")
                return
            
            position.current_price = current_price
            position.last_update = datetime.utcnow()
            
            # Calculate unrealized P&L if not provided - with division by zero protection
            if unrealized_pnl is None and position.entry_price > 0:
                if position.side == PositionSide.LONG:
                    unrealized_pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
                else:
                    unrealized_pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100
                
                position.unrealized_pnl_percent = unrealized_pnl_percent
            else:
                position.unrealized_pnl = unrealized_pnl
                if unrealized_pnl_percent is not None:
                    position.unrealized_pnl_percent = unrealized_pnl_percent
            
            # Update stop loss / take profit if provided
            if stop_loss is not None:
                position.stop_loss = stop_loss
            if take_profit is not None:
                position.take_profit = take_profit
            
            session.commit()
    
    def close_position(self, position_id: int) -> bool:
        """Mark a position as closed."""
        with self.get_session() as session:
            position = session.query(Position).filter_by(id=position_id).first()
            if not position:
                logger.error(f"Position {position_id} not found")
                return False
            
            position.status = OrderStatus.FILLED
            position.last_update = datetime.utcnow()
            session.commit()
            
            logger.info(f"Closed position #{position_id}")
            return True
    
    def log_account_snapshot(
        self,
        balance: float,
        equity: float,
        total_pnl: float,
        open_positions: int,
        total_exposure: float,
        drawdown: float,
        daily_pnl: Optional[float] = None,
        margin_used: Optional[float] = None,
        session_id: Optional[int] = None
    ):
        """Log a snapshot of account state."""
        with self.get_session() as session:
            snapshot = AccountHistory(
                timestamp=datetime.utcnow(),
                balance=balance,
                equity=equity,
                total_pnl=total_pnl,
                daily_pnl=daily_pnl or 0,
                drawdown=drawdown,
                open_positions=open_positions,
                total_exposure=total_exposure,
                margin_used=margin_used or 0,
                margin_available=balance - (margin_used or 0),
                session_id=session_id or self._current_session_id
            )
            
            session.add(snapshot)
            session.commit()
    
    def log_event(
        self,
        event_type: Union[str, EventType],
        message: str,
        severity: str = 'info',
        component: Optional[str] = None,
        details: Optional[Dict] = None,
        session_id: Optional[int] = None
    ) -> int:
        """
        Log a system event.
        
        Returns:
            Event ID
        """
        with self.get_session() as session:
            # Convert string enum if necessary
            if isinstance(event_type, str):
                event_type = EventType[event_type.upper()]
            
            event = SystemEvent(
                event_type=event_type,
                message=message,
                severity=severity,
                component=component,
                details=details,
                session_id=session_id or self._current_session_id,
                timestamp=datetime.utcnow()
            )
            
            session.add(event)
            session.commit()
            
            return event.id
    
    def log_strategy_execution(
        self,
        strategy_name: str,
        symbol: str,
        signal_type: str,
        action_taken: str,
        price: float,
        timeframe: Optional[str] = None,
        signal_strength: Optional[float] = None,
        confidence_score: Optional[float] = None,
        indicators: Optional[Dict] = None,
        sentiment_data: Optional[Dict] = None,
        ml_predictions: Optional[Dict] = None,
        position_size: Optional[float] = None,
        reasons: Optional[List[str]] = None,
        volume: Optional[float] = None,
        volatility: Optional[float] = None,
        trade_id: Optional[int] = None,
        session_id: Optional[int] = None
    ):
        """Log detailed strategy execution information."""
        with self.get_session() as session:
            execution = StrategyExecution(
                timestamp=datetime.utcnow(),
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                signal_type=signal_type,
                signal_strength=signal_strength,
                confidence_score=confidence_score,
                indicators=indicators,
                sentiment_data=sentiment_data,
                ml_predictions=ml_predictions,
                action_taken=action_taken,
                position_size=position_size,
                reasons=reasons,
                price=price,
                volume=volume,
                volatility=volatility,
                trade_id=trade_id,
                session_id=session_id or self._current_session_id
            )
            
            session.add(execution)
            session.commit()
    
    def get_active_positions(self, session_id: Optional[int] = None) -> List[Dict]:
        """Get all active positions."""
        with self.get_session() as session:
            query = session.query(Position).filter(
                Position.status != OrderStatus.FILLED
            )
            
            if session_id:
                query = query.filter(Position.session_id == session_id)
            elif self._current_session_id:
                query = query.filter(Position.session_id == self._current_session_id)
            
            positions = query.all()
            
            return [
                {
                    'id': p.id,
                    'symbol': p.symbol,
                    'side': p.side.value,
                    'entry_price': p.entry_price,
                    'current_price': p.current_price,
                    'size': p.size,
                    'unrealized_pnl': p.unrealized_pnl,
                    'unrealized_pnl_percent': p.unrealized_pnl_percent,
                    'stop_loss': p.stop_loss,
                    'take_profit': p.take_profit,
                    'entry_time': p.entry_time,
                    'strategy': p.strategy_name
                }
                for p in positions
            ]
    
    def get_recent_trades(self, limit: int = 50, session_id: Optional[int] = None) -> List[Dict]:
        """Get recent trades."""
        with self.get_session() as session:
            query = session.query(Trade).order_by(Trade.exit_time.desc())
            
            if session_id:
                query = query.filter(Trade.session_id == session_id)
            elif self._current_session_id:
                query = query.filter(Trade.session_id == self._current_session_id)
            
            trades = query.limit(limit).all()
            
            return [
                {
                    'id': t.id,
                    'symbol': t.symbol,
                    'side': t.side.value,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'size': t.size,
                    'pnl': t.pnl,
                    'pnl_percent': t.pnl_percent,
                    'entry_time': t.entry_time,
                    'exit_time': t.exit_time,
                    'exit_reason': t.exit_reason,
                    'strategy': t.strategy_name
                }
                for t in trades
            ]
    
    def get_performance_metrics(
        self,
        period: str = 'all-time',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        session_id: Optional[int] = None
    ) -> Dict:
        """Get performance metrics for a specific period."""
        with self.get_session() as session:
            # Query trades within the period
            query = session.query(Trade)
            
            if session_id:
                query = query.filter(Trade.session_id == session_id)
            elif self._current_session_id:
                query = query.filter(Trade.session_id == self._current_session_id)
            
            if start_date:
                query = query.filter(Trade.exit_time >= start_date)
            if end_date:
                query = query.filter(Trade.exit_time <= end_date)
            
            trades = query.all()
            
            if not trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0
                }
            
            # Calculate metrics with division by zero protection
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            
            total_pnl = sum(t.pnl for t in trades)
            win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
            
            avg_win = (sum(t.pnl for t in winning_trades) / len(winning_trades)) if winning_trades else 0
            avg_loss = (sum(t.pnl for t in losing_trades) / len(losing_trades)) if losing_trades else 0
            
            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = abs(sum(t.pnl for t in losing_trades))
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
            
            # Get account history for drawdown calculation
            history_query = session.query(AccountHistory)
            if session_id:
                history_query = history_query.filter(AccountHistory.session_id == session_id)
            elif self._current_session_id:
                history_query = history_query.filter(AccountHistory.session_id == self._current_session_id)
            
            if start_date:
                history_query = history_query.filter(AccountHistory.timestamp >= start_date)
            if end_date:
                history_query = history_query.filter(AccountHistory.timestamp <= end_date)
            
            account_history = history_query.order_by(AccountHistory.timestamp).all()
            
            max_drawdown = 0
            if account_history:
                peak_balance = account_history[0].balance
                for record in account_history:
                    if record.balance > peak_balance:
                        peak_balance = record.balance
                    # Protect against division by zero
                    if peak_balance > 0:
                        drawdown = (peak_balance - record.balance) / peak_balance * 100
                        max_drawdown = max(max_drawdown, drawdown)
            
            return {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'best_trade': max((t.pnl for t in trades), default=0),
                'worst_trade': min((t.pnl for t in trades), default=0),
                'avg_trade_duration': sum(
                    (t.exit_time - t.entry_time).total_seconds() / 3600
                    for t in trades
                ) / len(trades) if trades else 0
            }
    
    def _update_performance_metrics(self, session_id: int):
        """Update performance metrics after each trade."""
        # Guard against None session_id
        if not session_id:
            logger.warning("Cannot update performance metrics: session_id is None")
            return
        
        try:
            # Calculate daily metrics
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            daily_metrics = self.get_performance_metrics(
                period='daily',
                start_date=today_start,
                session_id=session_id
            )
            
            with self.get_session() as db:
                # Check if today's metrics already exist
                existing = db.query(PerformanceMetrics).filter(
                    and_(
                        PerformanceMetrics.period == 'daily',
                        PerformanceMetrics.period_start == today_start,
                        PerformanceMetrics.session_id == session_id
                    )
                ).first()
                
                if existing:
                    # Update existing metrics
                    for key, value in daily_metrics.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                else:
                    # Create new metrics record
                    metrics = PerformanceMetrics(
                        period='daily',
                        period_start=today_start,
                        period_end=datetime.utcnow(),
                        session_id=session_id,
                        total_trades=daily_metrics.get('total_trades', 0),
                        winning_trades=daily_metrics.get('winning_trades', 0),
                        losing_trades=daily_metrics.get('losing_trades', 0),
                        win_rate=daily_metrics.get('win_rate', 0),
                        total_return=daily_metrics.get('total_pnl', 0),  # Map total_pnl to total_return
                        max_drawdown=daily_metrics.get('max_drawdown', 0),
                        avg_win=daily_metrics.get('avg_win', 0),
                        avg_loss=daily_metrics.get('avg_loss', 0),
                        profit_factor=daily_metrics.get('profit_factor', 0),
                        best_trade_pnl=daily_metrics.get('best_trade', 0),
                        worst_trade_pnl=daily_metrics.get('worst_trade', 0)
                    )
                    db.add(metrics)
                
                db.commit()
                
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to prevent database bloat."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        with self.get_session() as session:
            # Delete old inactive sessions and their related data
            old_sessions = session.query(TradingSession).filter(
                and_(
                    TradingSession.end_time < cutoff_date,
                    TradingSession.is_active == False
                )
            ).all()
            
            for old_session in old_sessions:
                # SQLAlchemy will cascade delete related records
                session.delete(old_session)
            
            session.commit()
            
            logger.info(f"Cleaned up {len(old_sessions)} old trading sessions")
    
    def execute_query(self, query: str, params=None):  # type: ignore[override]
        """Run a raw SQL query and return list of dict rows.

        Uses SQLAlchemy 2.x ``exec_driver_sql`` API so plain SQL strings work
        without needing to wrap them in ``text()``. Falls back gracefully for
        older versions.
        """
        params = params or ()
        try:
            with self.engine.connect() as connection:
                # SQLAlchemy 2.0
                try:
                    result = connection.exec_driver_sql(query, params)
                except AttributeError:
                    # Older SQLAlchemy (<1.4) fallback
                    from sqlalchemy import text
                    result = connection.execute(text(query), params)
                # Convert to list of dictionaries
                try:
                    rows = [dict(row) for row in result.mappings()]
                except AttributeError:
                    rows = [dict(row.items()) for row in result]
                return rows
        except SQLAlchemyError as e:
            logger.error(f"Raw query error: {e}")
            return []
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get PostgreSQL connection pool statistics"""
        if self.engine and hasattr(self.engine.pool, 'status'):
            return {
                'pool_status': self.engine.pool.status(),
                'checked_in': getattr(self.engine.pool, 'checkedin', 0),
                'checked_out': getattr(self.engine.pool, 'checkedout', 0),
                'overflow': getattr(self.engine.pool, 'overflow', 0),
                'invalid': getattr(self.engine.pool, 'invalid', 0)
            }
        return {'status': 'Connection pool statistics not available'}
    
    def cleanup_connection_pool(self):
        """Cleanup PostgreSQL connection pool"""
        if self.engine and hasattr(self.engine.pool, 'dispose'):
            self.engine.pool.dispose()
            logger.info("PostgreSQL connection pool disposed") 