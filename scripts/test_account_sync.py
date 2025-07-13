#!/usr/bin/env python3
"""
Test Account Synchronization

This script tests the account synchronization functionality to ensure
data integrity between Binance and the bot's database.
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import get_config
from data_providers.binance_exchange import BinanceExchange
from database.manager import DatabaseManager
from live.account_sync import AccountSynchronizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_exchange_connection():
    """Test basic exchange connection"""
    logger.info("Testing exchange connection...")
    
    config = get_config()
    api_key = config.get('BINANCE_API_KEY')
    api_secret = config.get('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        logger.error("❌ Binance API credentials not found!")
        logger.error("Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        return False
    
    try:
        exchange = BinanceExchange(api_key, api_secret, testnet=False)
        
        # Test connection
        if exchange.test_connection():
            logger.info("✅ Exchange connection successful")
        else:
            logger.error("❌ Exchange connection failed")
            return False
        
        # Test account info
        account_info = exchange.get_account_info()
        if account_info:
            logger.info("✅ Account info retrieved successfully")
            logger.info(f"   Can trade: {account_info.get('can_trade', 'Unknown')}")
            logger.info(f"   Can withdraw: {account_info.get('can_withdraw', 'Unknown')}")
        else:
            logger.warning("⚠️ Could not retrieve account info")
        
        # Test balances
        balances = exchange.get_balances()
        logger.info(f"✅ Retrieved {len(balances)} balances")
        
        for balance in balances[:5]:  # Show first 5 balances
            logger.info(f"   {balance.asset}: {balance.total} (free: {balance.free}, locked: {balance.locked})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Exchange test failed: {e}")
        return False

def test_database_connection():
    """Test database connection"""
    logger.info("Testing database connection...")
    
    try:
        db_manager = DatabaseManager()
        
        if db_manager.test_connection():
            logger.info("✅ Database connection successful")
        else:
            logger.error("❌ Database connection failed")
            return False
        
        # Test database info
        db_info = db_manager.get_database_info()
        logger.info(f"✅ Database info: {db_info.get('database_type', 'Unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return False

def test_account_synchronization():
    """Test account synchronization"""
    logger.info("Testing account synchronization...")
    
    config = get_config()
    api_key = config.get('BINANCE_API_KEY')
    api_secret = config.get('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        logger.error("❌ Binance API credentials not found!")
        return False
    
    try:
        # Initialize components
        exchange = BinanceExchange(api_key, api_secret, testnet=False)
        db_manager = DatabaseManager()
        
        # Create a test session
        session_id = db_manager.create_trading_session(
            strategy_name="account_sync_test",
            symbol="BTCUSDT",
            timeframe="1h",
            mode="test",
            initial_balance=10000.0
        )
        
        logger.info(f"Created test session: {session_id}")
        
        # Initialize synchronizer
        synchronizer = AccountSynchronizer(exchange, db_manager, session_id)
        
        # Test basic sync
        logger.info("Performing basic account synchronization...")
        sync_result = synchronizer.sync_account_data(force=True)
        
        if sync_result.success:
            logger.info("✅ Account synchronization successful")
            
            # Log sync details
            data = sync_result.data
            logger.info(f"   Sync timestamp: {data.get('sync_timestamp', 'Unknown')}")
            
            # Balance sync results
            balance_sync = data.get('balance_sync', {})
            if balance_sync.get('synced', False):
                if balance_sync.get('corrected', False):
                    logger.info(f"   Balance corrected: ${balance_sync.get('old_balance', 0):.2f} -> ${balance_sync.get('new_balance', 0):.2f}")
                else:
                    logger.info(f"   Balance in sync: ${balance_sync.get('balance', 0):.2f}")
            
            # Position sync results
            position_sync = data.get('position_sync', {})
            if position_sync.get('synced', False):
                logger.info(f"   Positions synced: {position_sync.get('synced_positions', 0)}")
                logger.info(f"   New positions: {position_sync.get('new_positions', 0)}")
                logger.info(f"   Closed positions: {position_sync.get('closed_positions', 0)}")
            
            # Order sync results
            order_sync = data.get('order_sync', {})
            if order_sync.get('synced', False):
                logger.info(f"   Orders synced: {order_sync.get('synced_orders', 0)}")
                logger.info(f"   New orders: {order_sync.get('new_orders', 0)}")
                logger.info(f"   Cancelled orders: {order_sync.get('cancelled_orders', 0)}")
            
        else:
            logger.error(f"❌ Account synchronization failed: {sync_result.message}")
            return False
        
        # Test trade recovery
        logger.info("Testing trade recovery...")
        recovery_result = synchronizer.recover_missing_trades('BTCUSDT', days_back=7)
        
        if recovery_result.get('recovered', False):
            logger.info("✅ Trade recovery successful")
            logger.info(f"   Total exchange trades: {recovery_result.get('total_exchange_trades', 0)}")
            logger.info(f"   Total DB trades: {recovery_result.get('total_db_trades', 0)}")
            logger.info(f"   Missing trades: {recovery_result.get('missing_trades', 0)}")
            logger.info(f"   Recovered trades: {recovery_result.get('recovered_trades', 0)}")
        else:
            logger.warning(f"⚠️ Trade recovery failed: {recovery_result.get('error', 'Unknown error')}")
        
        # Clean up test session
        db_manager.end_trading_session(session_id)
        logger.info(f"Cleaned up test session: {session_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Account synchronization test failed: {e}")
        return False

def test_emergency_sync():
    """Test emergency synchronization"""
    logger.info("Testing emergency synchronization...")
    
    config = get_config()
    api_key = config.get('BINANCE_API_KEY')
    api_secret = config.get('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        logger.error("❌ Binance API credentials not found!")
        return False
    
    try:
        # Initialize components
        exchange = BinanceExchange(api_key, api_secret, testnet=False)
        db_manager = DatabaseManager()
        
        # Create a test session
        session_id = db_manager.create_trading_session(
            strategy_name="emergency_sync_test",
            symbol="BTCUSDT",
            timeframe="1h",
            mode="test",
            initial_balance=10000.0
        )
        
        logger.info(f"Created test session: {session_id}")
        
        # Initialize synchronizer
        synchronizer = AccountSynchronizer(exchange, db_manager, session_id)
        
        # Test emergency sync
        logger.info("Performing emergency synchronization...")
        sync_result = synchronizer.emergency_sync()
        
        if sync_result.success:
            logger.info("✅ Emergency synchronization successful")
            
            # Log emergency sync details
            data = sync_result.data
            emergency_recovery = data.get('emergency_trade_recovery', {})
            
            for symbol, result in emergency_recovery.items():
                if result.get('recovered', False):
                    logger.info(f"   {symbol}: {result.get('recovered_trades', 0)} trades recovered")
                else:
                    logger.warning(f"   {symbol}: {result.get('error', 'Unknown error')}")
            
        else:
            logger.error(f"❌ Emergency synchronization failed: {sync_result.message}")
            return False
        
        # Clean up test session
        db_manager.end_trading_session(session_id)
        logger.info(f"Cleaned up test session: {session_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Emergency synchronization test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🤖 Starting Account Synchronization Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Exchange Connection", test_exchange_connection),
        ("Database Connection", test_database_connection),
        ("Account Synchronization", test_account_synchronization),
        ("Emergency Synchronization", test_emergency_sync),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} test ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Results Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Account synchronization is working correctly.")
        return 0
    else:
        logger.error("⚠️ Some tests failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())