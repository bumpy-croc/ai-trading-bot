#!/usr/bin/env python3
"""
Verify database connection for both local and Railway deployments
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from database.manager import DatabaseManager
from config.config_manager import get_config


def verify_database_connection():
    """Verify database connection and display information"""
    
    print("🔍 Verifying Database Connection")
    print("=" * 50)
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Get database info
        db_info = db_manager.get_database_info()
        
        print(f"📊 Database Information:")
        print(f"  - Database URL: {db_info['database_url']}")
        print(f"  - Is PostgreSQL: {db_info['is_postgresql']}")
        print(f"  - Is SQLite: {db_info['is_sqlite']}")
        print(f"  - Connection Pool Size: {db_info['connection_pool_size']}")
        
        # Test connection
        print(f"\n🔗 Testing Connection...")
        if db_manager.test_connection():
            print("✅ Database connection successful!")
        else:
            print("❌ Database connection failed!")
            return False
        
        # Test session creation
        print(f"\n🎯 Testing Session Creation...")
        session_id = db_manager.create_trading_session(
            strategy_name="TestStrategy",
            symbol="BTCUSDT",
            timeframe="1h",
            mode="PAPER",
            initial_balance=10000.0,
            session_name="verification_test"
        )
        print(f"✅ Created test session #{session_id}")
        
        # Test logging
        print(f"\n📝 Testing Event Logging...")
        event_id = db_manager.log_event(
            event_type="TEST",
            message="Database verification test",
            severity="info",
            session_id=session_id
        )
        print(f"✅ Logged test event #{event_id}")
        
        # End session
        db_manager.end_trading_session(session_id, final_balance=10000.0)
        print(f"✅ Ended test session #{session_id}")
        
        # Display configuration
        print(f"\n⚙️  Configuration:")
        config = get_config()
        print(f"  - DATABASE_URL: {'Set' if config.get('DATABASE_URL') else 'Not set'}")
        print(f"  - Railway Environment: {'Yes' if config.get('RAILWAY_PROJECT_ID') else 'No'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    success = verify_database_connection()
    
    if success:
        print(f"\n✅ Database verification completed successfully!")
    else:
        print(f"\n❌ Database verification failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()