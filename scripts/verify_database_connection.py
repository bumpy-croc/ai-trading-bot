#!/usr/bin/env python3
"""
Verify PostgreSQL database connection for both local and Railway deployments
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from config.config_manager import get_config
from database.manager import DatabaseManager


def verify_database_connection():
    """Verify PostgreSQL database connection and display information"""

    print("🔍 Verifying PostgreSQL Database Connection")
    print("=" * 50)

    try:
        # Initialize database manager
        db_manager = DatabaseManager()

        # Get database info
        db_info = db_manager.get_database_info()

        print("📊 Database Information:")
        print(f"  - Database URL: {db_info['database_url']}")
        print(f"  - Database Type: {db_info['database_type']}")
        print(f"  - Connection Pool Size: {db_info['connection_pool_size']}")

        # Test connection
        print("\n🔗 Testing Connection...")
        if db_manager.test_connection():
            print("✅ PostgreSQL database connection successful!")
        else:
            print("❌ PostgreSQL database connection failed!")
            return False

        # Test session creation
        print("\n🎯 Testing Session Creation...")
        session_id = db_manager.create_trading_session(
            strategy_name="TestStrategy",
            symbol="BTCUSDT",
            timeframe="1h",
            mode="PAPER",
            initial_balance=10000.0,
            session_name="verification_test",
        )
        print(f"✅ Created test session #{session_id}")

        # Test logging
        print("\n📝 Testing Event Logging...")
        event_id = db_manager.log_event(
            event_type="TEST",
            message="PostgreSQL database verification test",
            severity="info",
            session_id=session_id,
        )
        print(f"✅ Logged test event #{event_id}")

        # End session
        db_manager.end_trading_session(session_id, final_balance=10000.0)
        print(f"✅ Ended test session #{session_id}")

        # Display configuration
        print("\n⚙️  Configuration:")
        config = get_config()
        print(f"  - DATABASE_URL: {'Set' if config.get('DATABASE_URL') else 'Not set'}")
        print(f"  - Railway Environment: {'Yes' if config.get('RAILWAY_PROJECT_ID') else 'No'}")

        # Connection pool statistics
        stats = db_manager.get_connection_stats()
        print("\n📊 Connection Pool Statistics:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure DATABASE_URL environment variable is set")
        print("  2. Verify PostgreSQL database is running and accessible")
        print("  3. Check that DATABASE_URL starts with 'postgresql://'")
        print("  4. For Railway: ensure PostgreSQL service is deployed")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function"""
    success = verify_database_connection()

    if success:
        print("\n✅ PostgreSQL database verification completed successfully!")
        print("\n📈 Your database is ready for:")
        print("  - Trading bot operations")
        print("  - Dashboard data sharing")
        print("  - Performance analytics")
        print("  - Historical data storage")
    else:
        print("\n❌ PostgreSQL database verification failed!")
        print("Please check the troubleshooting steps above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
