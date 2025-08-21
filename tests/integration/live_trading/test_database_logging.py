from datetime import datetime, timedelta

import pytest

pytestmark = pytest.mark.integration

try:
    from database.manager import DatabaseManager
    from database.models import (
        AccountBalance,
        AccountHistory,
        EventType,
        Position,
        PositionSide,
        StrategyExecution,
        SystemEvent,
        Trade,
        TradeSource,
        TradingSession,
    )

    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False


@pytest.mark.skipif(not DB_AVAILABLE, reason="Database components not available")
class TestDatabaseLogging:
    def test_trades_logged_to_database(self, mock_strategy, mock_data_provider):
        db_manager = DatabaseManager()
        session_id = db_manager.create_trading_session(
            strategy_name="TestStrategy",
            symbol="BTCUSDT",
            timeframe="1h",
            initial_balance=10000,
            strategy_config={"test": True},
            mode="live",
        )
        trade_data = {
            "symbol": "BTCUSDT",
            "side": PositionSide.LONG,
            "entry_price": 50000.0,
            "exit_price": 51000.0,
            "size": 0.1,
            "entry_time": datetime.now() - timedelta(hours=1),
            "exit_time": datetime.now(),
            "pnl": 100.0,
            "exit_reason": "take_profit",
            "strategy_name": "TestStrategy",
            "source": TradeSource.PAPER,
            "order_id": "test_order_001",
            "session_id": session_id,
        }
        trade_id = db_manager.log_trade(**trade_data)
        assert trade_id > 0
        with db_manager.get_session() as session:
            trade = session.query(Trade).filter_by(id=trade_id).first()
            assert trade is not None
            assert trade.symbol == "BTCUSDT"
            assert trade.side == PositionSide.LONG
            assert float(trade.entry_price) == 50000.0
            assert float(trade.exit_price) == 51000.0
            assert float(trade.pnl) == 100.0
            assert trade.exit_reason == "take_profit"
            assert trade.strategy_name == "TestStrategy"
            assert trade.session_id == session_id
        db_manager.end_trading_session(session_id)

    def test_events_logged_to_database(self, mock_strategy, mock_data_provider):
        db_manager = DatabaseManager()
        session_id = db_manager.create_trading_session(
            strategy_name="TestStrategy",
            symbol="BTCUSDT",
            timeframe="1h",
            initial_balance=10000,
            mode="live",
        )
        events = [
            {
                "event_type": EventType.ENGINE_START,
                "message": "Trading engine started",
                "severity": "info",
                "component": "trading_engine",
                "session_id": session_id,
            },
            {
                "event_type": EventType.STRATEGY_CHANGE,
                "message": "Strategy changed",
                "severity": "info",
                "component": "strategy_manager",
                "details": {"old_strategy": "BasicStrategy", "new_strategy": "MlBasic"},
                "session_id": session_id,
            },
            {
                "event_type": EventType.ERROR,
                "message": "API rate limit exceeded",
                "severity": "warning",
                "component": "data_provider",
                "session_id": session_id,
            },
        ]
        ids = [db_manager.log_event(**e) for e in events]
        for event_id in ids:
            assert event_id > 0
        with db_manager.get_session() as session:
            for i, event_id in enumerate(ids):
                event = session.query(SystemEvent).filter_by(id=event_id).first()
                assert event is not None
                assert event.event_type == events[i]["event_type"]
        db_manager.end_trading_session(session_id)

    def test_positions_logged_to_database(self, mock_strategy, mock_data_provider):
        db_manager = DatabaseManager()
        session_id = db_manager.create_trading_session(
            strategy_name="TestStrategy",
            symbol="BTCUSDT",
            timeframe="1h",
            initial_balance=10000,
            mode="live",
        )
        position_data = {
            "symbol": "BTCUSDT",
            "side": PositionSide.LONG,
            "entry_price": 50000.0,
            "size": 0.1,
            "strategy_name": "TestStrategy",
            "order_id": "test_position_001",
            "stop_loss": 49000.0,
            "take_profit": 52000.0,
            "confidence_score": 0.75,
            "quantity": 0.002,
            "session_id": session_id,
        }
        position_id = db_manager.log_position(**position_data)
        assert position_id > 0
        with db_manager.get_session() as session:
            position = session.query(Position).filter_by(id=position_id).first()
            assert position is not None
            assert position.symbol == "BTCUSDT"
            assert position.side == PositionSide.LONG
        db_manager.end_trading_session(session_id)

    def test_account_history_snapshots_logged(self, mock_strategy, mock_data_provider):
        db_manager = DatabaseManager()
        session_id = db_manager.create_trading_session(
            strategy_name="TestStrategy",
            symbol="BTCUSDT",
            timeframe="1h",
            initial_balance=10000,
            mode="live",
        )
        snapshot_data = {
            "balance": 10100.0,
            "equity": 10150.0,
            "total_pnl": 150.0,
            "open_positions": 2,
            "total_exposure": 5000.0,
            "drawdown": 0.05,
            "daily_pnl": 50.0,
            "margin_used": 2500.0,
            "session_id": session_id,
        }
        db_manager.log_account_snapshot(**snapshot_data)
        with db_manager.get_session() as session:
            snapshot = session.query(AccountHistory).filter_by(session_id=session_id).first()
            assert snapshot is not None
        db_manager.end_trading_session(session_id)

    def test_account_balance_logged(self, mock_strategy, mock_data_provider):
        db_manager = DatabaseManager()
        session_id = db_manager.create_trading_session(
            strategy_name="TestStrategy",
            symbol="BTCUSDT",
            timeframe="1h",
            initial_balance=10000,
            mode="live",
        )
        success = db_manager.update_balance(
            new_balance=10200.0,
            update_reason="trade_pnl",
            updated_by="system",
            session_id=session_id,
        )
        assert success
        current_balance = db_manager.get_current_balance(session_id)
        assert current_balance == 10200.0
        with db_manager.get_session() as session:
            balance_record = session.query(AccountBalance).filter_by(session_id=session_id).first()
            assert balance_record is not None
        db_manager.end_trading_session(session_id)

    def test_performance_metrics_logged(self, mock_strategy, mock_data_provider):
        db_manager = DatabaseManager()
        session_id = db_manager.create_trading_session(
            strategy_name="TestStrategy",
            symbol="BTCUSDT",
            timeframe="1h",
            initial_balance=10000,
            mode="live",
        )
        trade_data = {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "entry_price": 50000.0,
            "exit_price": 51000.0,
            "size": 0.1,
            "entry_time": datetime.now() - timedelta(hours=2),
            "exit_time": datetime.now() - timedelta(hours=1),
            "pnl": 100.0,
            "exit_reason": "take_profit",
            "strategy_name": "TestStrategy",
            "session_id": session_id,
        }
        db_manager.log_trade(**trade_data)
        metrics = db_manager.get_recent_performance_metrics(session_id=session_id)
        assert metrics["total_trades"] >= 1
        db_manager.end_trading_session(session_id)

    def test_strategy_execution_data_logged(self, mock_strategy, mock_data_provider):
        db_manager = DatabaseManager()
        session_id = db_manager.create_trading_session(
            strategy_name="TestStrategy",
            symbol="BTCUSDT",
            timeframe="1h",
            initial_balance=10000,
            mode="live",
        )
        execution_data = {
            "strategy_name": "TestStrategy",
            "symbol": "BTCUSDT",
            "signal_type": "entry",
            "action_taken": "opened_long",
            "price": 50000.0,
            "timeframe": "1h",
            "signal_strength": 0.8,
            "confidence_score": 0.75,
            "indicators": {"rsi": 45.5, "ema_20": 49800.0},
            "sentiment_data": {"sentiment_score": 0.6},
            "ml_predictions": {"price_prediction": 51000.0},
            "position_size": 0.1,
            "reasons": ["RSI oversold", "Price above EMA"],
            "volume": 1000.0,
            "volatility": 0.02,
            "session_id": session_id,
        }
        db_manager.log_strategy_execution(**execution_data)
        with db_manager.get_session() as session:
            exec_record = session.query(StrategyExecution).filter_by(session_id=session_id).first()
            assert exec_record is not None
        db_manager.end_trading_session(session_id)

    def test_trading_sessions_logged(self, mock_strategy, mock_data_provider):
        db_manager = DatabaseManager()
        session_name = "Test Session 2024"
        session_id = db_manager.create_trading_session(
            strategy_name="TestStrategy",
            symbol="BTCUSDT",
            timeframe="1h",
            initial_balance=10000,
            strategy_config={"test": True},
            session_name=session_name,
            mode="live",
        )
        assert session_id > 0
        with db_manager.get_session() as session:
            trading_session = session.query(TradingSession).filter_by(id=session_id).first()
            assert trading_session is not None
        db_manager.end_trading_session(session_id)
