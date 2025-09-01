"""
Mock Database Manager for Unit Tests

Provides an in-memory implementation of DatabaseManager that doesn't require
a real database connection. This significantly speeds up unit tests.
"""

import json
from datetime import datetime
from typing import Any, Optional
from unittest.mock import Mock

from src.database.models import EventType


class MockDatabaseManager:
    """Mock implementation of DatabaseManager for unit tests"""

    def __init__(self, database_url: Optional[str] = None):
        """Initialize mock database with in-memory storage"""
        self.database_url = database_url or "mock://memory"
        self._sessions = {}
        self._trades = {}
        self._positions = {}
        self._events = {}
        self._account_snapshots = []
        self._strategy_executions = []
        self._next_id = 1
        self._current_session_id = None

        # Mock connection stats
        self._connection_stats = {
            "pool_size": 5,
            "checked_in": 3,
            "checked_out": 2,
            "overflow": 0,
            "invalid": 0,
            "total": 5,
        }

    def _get_next_id(self) -> int:
        """Get next available ID"""
        id_val = self._next_id
        self._next_id += 1
        return id_val

    def test_connection(self) -> bool:
        """Test database connection - always succeeds for mock"""
        return True

    def get_database_info(self) -> dict[str, Any]:
        """Get mock database information"""
        return {
            "type": "Mock Database",
            "url": self.database_url,
            "version": "1.0.0",
            "size": f"{len(self._trades) + len(self._positions)} records",
            "tables": {
                "sessions": len(self._sessions),
                "trades": len(self._trades),
                "positions": len(self._positions),
                "events": len(self._events),
                "account_snapshots": len(self._account_snapshots),
                "strategy_executions": len(self._strategy_executions),
            },
        }

    def get_connection_stats(self) -> dict[str, Any]:
        """Get mock connection pool statistics"""
        return self._connection_stats

    def cleanup_connection_pool(self):
        """Mock cleanup - does nothing"""
        pass

    def create_trading_session(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        initial_balance: float,
        strategy_config: Optional[dict] = None,
        mode: str = "backtest",
        time_exit_config: Optional[dict] = None,
        market_timezone: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        """Create a new trading session"""
        session_id = self._get_next_id()
        self._current_session_id = session_id

        self._sessions[session_id] = {
            "id": session_id,
            "strategy_name": strategy_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "initial_balance": initial_balance,
            "strategy_config": json.dumps(strategy_config or {}),
            "mode": mode,
            "start_time": datetime.now(),
            "end_time": None,
            "final_balance": None,
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "time_exit_config": time_exit_config,
            "market_timezone": market_timezone,
        }

        return session_id

    def end_trading_session(self, session_id: Optional[int] = None, final_balance: Optional[float] = None):
        """End a trading session"""
        if session_id is None:
            session_id = self._current_session_id

        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session["end_time"] = datetime.now()
            if final_balance is not None:
                session["final_balance"] = final_balance

            # Calculate session stats
            session_trades = [t for t in self._trades.values() if t.get("session_id") == session_id]
            session["total_trades"] = len(session_trades)
            session["winning_trades"] = len([t for t in session_trades if t.get("pnl", 0) > 0])
            session["total_pnl"] = sum(t.get("pnl", 0) for t in session_trades)

            if session["total_trades"] > 0:
                session["final_balance"] = session["initial_balance"] + session["total_pnl"]
            else:
                session["final_balance"] = session["initial_balance"]

    def log_trade(
        self,
        symbol: str,
        side: Any,
        entry_price: float,
        exit_price: float,
        size: float,
        entry_time: datetime,
        exit_time: datetime,
        pnl: float,
        exit_reason: str,
        strategy_name: str,
        source: Any = None,
        order_id: Optional[str] = None,
        fees: float = 0.0,
        slippage: float = 0.0,
        session_id: Optional[int] = None,
        **kwargs: Any,
    ) -> int:
        """Log a completed trade"""
        trade_id = self._get_next_id()

        self._trades[trade_id] = {
            "id": trade_id,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "pnl": pnl,
            "exit_reason": exit_reason,
            "strategy_name": strategy_name,
            "source": source,
            "order_id": order_id,
            "fees": fees,
            "slippage": slippage,
            "session_id": session_id or self._current_session_id,
            # Optional MFE/MAE fields if provided
            "mfe": kwargs.get("mfe"),
            "mae": kwargs.get("mae"),
            "mfe_price": kwargs.get("mfe_price"),
            "mae_price": kwargs.get("mae_price"),
            "mfe_time": kwargs.get("mfe_time"),
            "mae_time": kwargs.get("mae_time"),
        }

        return trade_id

    def log_position(
        self,
        symbol: str,
        side: Any,
        entry_price: float,
        size: float,
        strategy_name: str,
        order_id: Optional[str] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        confidence_score: Optional[float] = None,
        quantity: Optional[float] = None,
        session_id: Optional[int] = None,
        **kwargs: Any,
    ) -> int:
        """Log a new position"""
        position_id = self._get_next_id()

        self._positions[position_id] = {
            "id": position_id,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "size": size,
            "quantity": quantity or size,
            "strategy_name": strategy_name,
            "order_id": order_id,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence_score": confidence_score,
            "status": Any,
            "entry_time": datetime.now(),
            "session_id": session_id or self._current_session_id,
            # Rolling MFE/MAE fields (optional)
            "mfe": kwargs.get("mfe"),
            "mae": kwargs.get("mae"),
            "mfe_price": kwargs.get("mfe_price"),
            "mae_price": kwargs.get("mae_price"),
            "mfe_time": kwargs.get("mfe_time"),
            "mae_time": kwargs.get("mae_time"),
        }

        return position_id

    def update_position(
        self,
        position_id: int,
        current_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        status: Optional[Any] = None,
        notes: Optional[str] = None,
        **kwargs: Any,
    ):
        """Update an existing position"""
        if position_id in self._positions:
            position = self._positions[position_id]
            if current_price is not None:
                position["current_price"] = current_price
            if stop_loss is not None:
                position["stop_loss"] = stop_loss
            if take_profit is not None:
                position["take_profit"] = take_profit
            if status is not None:
                position["status"] = status
            if notes is not None:
                position["notes"] = notes
            # Optional extended fields
            for k in ("mfe", "mae", "mfe_price", "mae_price", "mfe_time", "mae_time", "unrealized_pnl", "unrealized_pnl_percent", "size"):
                if k in kwargs and kwargs[k] is not None:
                    position[k] = kwargs[k]
            position["last_updated"] = datetime.now()

    def close_position(
        self,
        position_id: int,
        exit_price: Optional[float] = None,
        exit_time: Optional[datetime] = None,
        pnl: Optional[float] = None,
        exit_reason: Optional[str] = None,
    ) -> bool:
        """Close a position"""
        if position_id in self._positions:
            position = self._positions[position_id]
            position["status"] = Any
            position["exit_time"] = exit_time or datetime.now()
            if exit_price is not None:
                position["exit_price"] = exit_price
            if pnl is not None:
                position["pnl"] = pnl
            if exit_reason is not None:
                position["exit_reason"] = exit_reason
            return True
        return False

    def log_event(
        self,
        event_type: EventType | str,
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
        severity: str = "info",
        session_id: Optional[int] = None,
        **kwargs: Any,
    ) -> int:
        """Log an event"""
        event_id = self._get_next_id()

        # Support DatabaseManager-style parameters
        message = kwargs.get("message", description or "")
        details = kwargs.get("details", metadata)

        self._events[event_id] = {
            "id": event_id,
            "timestamp": datetime.now(),
            "event_type": event_type,
            "description": message,
            "metadata": json.dumps(details or {}),
            "severity": severity,
            "session_id": session_id or self._current_session_id,
        }

        return event_id

    def log_account_snapshot(
        self,
        balance: float,
        equity: float,
        margin_used: float = 0.0,
        free_margin: float = 0.0,
        margin_level: Optional[float] = None,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
        session_id: Optional[int] = None,
        **kwargs: Any,
    ):
        """Log account snapshot"""
        snapshot = {
            "timestamp": datetime.now(),
            "balance": balance,
            "equity": equity,
            "margin_used": margin_used,
            "free_margin": free_margin,
            "margin_level": margin_level,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "session_id": session_id or self._current_session_id,
        }
        self._account_snapshots.append(snapshot)

    # Simple balance API for compatibility
    def update_balance(
        self,
        new_balance: float,
        update_reason: str,
        updated_by: str,
        session_id: Optional[int] = None,
        **kwargs: Any,
    ) -> bool:
        self.log_account_snapshot(
            balance=new_balance,
            equity=new_balance,
            realized_pnl=kwargs.get("realized_pnl", 0.0),
            unrealized_pnl=kwargs.get("unrealized_pnl", 0.0),
            session_id=session_id or self._current_session_id,
        )
        return True

    def get_current_balance(self, session_id: Optional[int] = None) -> float:
        sess = session_id or self._current_session_id
        for snap in reversed(self._account_snapshots):
            if snap.get("session_id") == sess:
                return float(snap.get("balance", 0.0))
        return 0.0

    def log_strategy_execution(
        self,
        signal: str,
        confidence: float,
        indicators: dict[str, Any],
        market_conditions: dict[str, Any],
        metadata: Optional[dict] = None,
        session_id: Optional[int] = None,
    ):
        """Log strategy execution details"""
        execution = {
            "timestamp": datetime.now(),
            "signal": signal,
            "confidence": confidence,
            "indicators": json.dumps(indicators),
            "market_conditions": json.dumps(market_conditions),
            "metadata": json.dumps(metadata or {}),
            "session_id": session_id or self._current_session_id,
        }
        self._strategy_executions.append(execution)

    def get_active_positions(self, session_id: Optional[int] = None) -> list[dict]:
        """Get all active positions"""
        positions = []
        for pos in self._positions.values():
            if pos.get("status") == Any:
                if session_id is None or pos.get("session_id") == session_id:
                    positions.append(pos)
        return positions

    def get_recent_trades(self, limit: int = 100, session_id: Optional[int] = None) -> list[dict]:
        """Get recent trades"""
        trades = []
        for trade in self._trades.values():
            if session_id is None or trade.get("session_id") == session_id:
                trades.append(trade)

        # Sort by exit time descending
        trades.sort(key=lambda t: t.get("exit_time", datetime.min), reverse=True)
        return trades[:limit]

    def get_performance_metrics(self, session_id: Optional[int] = None) -> dict[str, Any]:
        """Get performance metrics"""
        if session_id is None:
            session_id = self._current_session_id

        if not session_id or session_id not in self._sessions:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "average_pnl": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
            }

        _session = self._sessions[session_id]
        trades = [t for t in self._trades.values() if t.get("session_id") == session_id]

        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "average_pnl": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
            }

        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in trades if t.get("pnl", 0) < 0]
        total_pnl = sum(t.get("pnl", 0) for t in trades)

        return {
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(trades) if trades else 0.0,
            "total_pnl": total_pnl,
            "average_pnl": total_pnl / len(trades) if trades else 0.0,
            "max_drawdown": 0.0,  # Simplified for mock
            "sharpe_ratio": 0.0,  # Simplified for mock
        }

    def cleanup_old_data(self, days_to_keep: int = 90):
        """Mock cleanup - does nothing"""
        pass

    def execute_query(self, query: str) -> Any:
        """Mock query execution"""
        return []

    def get_session(self):
        """Get a mock session context manager"""
        return MockSession()

    def set_database_manager(self, db_manager, session_id: Optional[int] = None):
        """Mock method for strategy compatibility"""
        self._current_session_id = session_id


class MockSession:
    """Mock database session for context manager compatibility"""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def query(self, *args, **kwargs):
        """Mock query method"""
        return Mock()

    def add(self, obj):
        """Mock add method"""
        pass

    def commit(self):
        """Mock commit method"""
        pass

    def rollback(self):
        """Mock rollback method"""
        pass

    def close(self):
        """Mock close method"""
        pass
