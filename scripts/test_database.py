#!/usr/bin/env python3
"""
Test script for database logging functionality
"""

import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

import logging
from datetime import datetime, timedelta
from src.database.manager import DatabaseManager
from src.database.models import TradeSource, PositionSide

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_database_logging():
    """Test database logging functionality"""
    
    # Initialize database manager (will use SQLite by default)
    db_manager = DatabaseManager()
    logger.info("Database manager initialized")
    
    # Create a trading session
    session_id = db_manager.create_trading_session(
        strategy_name="TestStrategy",
        symbol="BTCUSDT",
        timeframe="1h",
        mode=TradeSource.PAPER,
        initial_balance=10000,
        strategy_config={"test_param": 123}
    )
    logger.info(f"Created trading session #{session_id}")
    
    # Log a position
    position_id = db_manager.log_position(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_price=45000,
        size=0.1,  # 10% of balance
        strategy_name="TestStrategy",
        order_id="test_order_123",
        stop_loss=43000,
        take_profit=47000,
        confidence_score=0.85,
        quantity=0.022,  # 0.022 BTC
        session_id=session_id
    )
    logger.info(f"Logged position #{position_id}")
    
    # Update position
    db_manager.update_position(
        position_id=position_id,
        current_price=45500,
        unrealized_pnl=50,
        unrealized_pnl_percent=1.11
    )
    logger.info("Updated position")
    
    # Log a trade
    trade_id = db_manager.log_trade(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_price=45000,
        exit_price=46000,
        size=0.1,
        entry_time=datetime.now() - timedelta(hours=2),
        exit_time=datetime.now(),
        pnl=100,
        exit_reason="Take profit",
        strategy_name="TestStrategy",
        source=TradeSource.PAPER,
        stop_loss=43000,
        take_profit=46000,
        order_id="test_order_123",
        confidence_score=0.85,
        session_id=session_id
    )
    logger.info(f"Logged trade #{trade_id}")
    
    # Log account snapshot
    db_manager.log_account_snapshot(
        balance=10100,
        equity=10150,
        total_pnl=100,
        open_positions=1,
        total_exposure=1000,
        drawdown=0.5,
        daily_pnl=100,
        margin_used=1000,
        session_id=session_id
    )
    logger.info("Logged account snapshot")
    
    # Log a system event
    db_manager.log_event(
        event_type="ALERT",
        message="Test alert message",
        severity="info",
        details={"test": "data"},
        component="TestScript",
        session_id=session_id
    )
    logger.info("Logged system event")
    
    # Get active positions
    positions = db_manager.get_active_positions(session_id)
    logger.info(f"Active positions: {len(positions)}")
    for pos in positions:
        logger.info(f"  - {pos['symbol']} {pos['side']} @ ${pos['entry_price']}")
    
    # Get recent trades
    trades = db_manager.get_recent_trades(limit=10, session_id=session_id)
    logger.info(f"Recent trades: {len(trades)}")
    for trade in trades:
        logger.info(f"  - {trade['symbol']} P&L: ${trade['pnl']:.2f} ({trade['pnl_percent']:.2f}%)")
    
    # Get performance metrics
    metrics = db_manager.get_performance_metrics(session_id=session_id)
    logger.info("Performance metrics:")
    logger.info(f"  - Total trades: {metrics['total_trades']}")
    logger.info(f"  - Win rate: {metrics['win_rate']:.1f}%")
    logger.info(f"  - Total P&L: ${metrics['total_pnl']:.2f}")
    logger.info(f"  - Max drawdown: {metrics['max_drawdown']:.1f}%")
    
    # End trading session
    db_manager.end_trading_session(session_id=session_id, final_balance=10100)
    logger.info("Ended trading session")
    
    logger.info("\n✅ Database logging test completed successfully!")
    
    # Show database file location
    logger.info(f"\nDatabase file created at: data/trading_bot.db")
    logger.info("You can use any SQLite viewer to inspect the data")


if __name__ == "__main__":
    test_database_logging() 