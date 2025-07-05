#!/usr/bin/env python3
"""
Railway Database Setup Script
Helps set up and configure the centralized PostgreSQL database on Railway
"""

import sys
import os
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from database.manager import DatabaseManager
from config.config_manager import get_config


def print_setup_instructions():
    """Print Railway database setup instructions"""
    
    print("🚀 Railway Database Centralization Setup")
    print("=" * 60)
    print()
    print("📋 Step-by-Step Instructions:")
    print()
    print("1. 🗄️  Create PostgreSQL Database Service")
    print("   - Go to your Railway project dashboard")
    print("   - Click '+ New' button")
    print("   - Select 'Database' > 'PostgreSQL'")
    print("   - Click 'Deploy' and wait for deployment")
    print()
    print("2. 🔗 Database Connection (Automatic)")
    print("   - Railway automatically provides these environment variables:")
    print("     • DATABASE_URL - Complete PostgreSQL connection string")
    print("     • PGHOST - Database host")
    print("     • PGPORT - Database port")
    print("     • PGUSER - Database username")
    print("     • PGPASSWORD - Database password")
    print("     • PGDATABASE - Database name")
    print()
    print("3. 🔧 Service Configuration")
    print("   - Both trading bot and dashboard services will automatically")
    print("     use the shared PostgreSQL database when DATABASE_URL is available")
    print("   - No code changes needed - configuration is automatic")
    print()
    print("4. 📊 Database Schema")
    print("   - Tables will be created automatically on first run")
    print("   - SQLAlchemy handles schema creation via models")
    print()
    print("5. 🔍 Verification")
    print("   - Run this script with --verify flag to test connection")
    print("   - Both services should connect to the same database")
    print()
    print("📈 Benefits:")
    print("  ✅ Shared data between trading bot and dashboard")
    print("  ✅ Better performance with connection pooling")
    print("  ✅ ACID transactions for data integrity")
    print("  ✅ Built-in backup and recovery")
    print("  ✅ Scalable for growing datasets")
    print()
    print("💡 Local Development:")
    print("  - Continues to use SQLite (no changes needed)")
    print("  - Railway deployment automatically uses PostgreSQL")
    print()


def verify_railway_setup():
    """Verify Railway database setup"""
    
    print("🔍 Verifying Railway Database Setup")
    print("=" * 50)
    
    try:
        # Check configuration
        config = get_config()
        
        print("📊 Environment Check:")
        railway_project = config.get('RAILWAY_PROJECT_ID')
        database_url = config.get('DATABASE_URL')
        
        if railway_project:
            print(f"✅ Railway Project ID: {railway_project}")
        else:
            print("⚠️  Not running on Railway (RAILWAY_PROJECT_ID not found)")
        
        if database_url:
            print(f"✅ Database URL: {database_url}")
            
            # Check if it's PostgreSQL
            if database_url.startswith('postgresql'):
                print("✅ PostgreSQL database detected")
            else:
                print("⚠️  Not using PostgreSQL database")
        else:
            print("⚠️  DATABASE_URL not found - using SQLite")
        
        print()
        
        # Test database connection
        print("🔗 Testing Database Connection...")
        db_manager = DatabaseManager()
        
        db_info = db_manager.get_database_info()
        print(f"  Database Type: {'PostgreSQL' if db_info['is_postgresql'] else 'SQLite'}")
        print(f"  Connection Pool Size: {db_info['connection_pool_size']}")
        
        if db_manager.test_connection():
            print("✅ Database connection successful!")
        else:
            print("❌ Database connection failed!")
            return False
        
        # Test basic operations
        print("\n🧪 Testing Database Operations...")
        
        # Create test session
        session_id = db_manager.create_trading_session(
            strategy_name="VerificationTest",
            symbol="BTCUSDT",
            timeframe="1h",
            mode="PAPER",
            initial_balance=10000.0,
            session_name="railway_verification"
        )
        print(f"✅ Created test session #{session_id}")
        
        # Log test event
        event_id = db_manager.log_event(
            event_type="TEST",
            message="Railway database verification test",
            severity="info",
            session_id=session_id
        )
        print(f"✅ Logged test event #{event_id}")
        
        # End session
        db_manager.end_trading_session(session_id, final_balance=10000.0)
        print(f"✅ Ended test session #{session_id}")
        
        # Connection stats
        stats = db_manager.get_connection_stats()
        print(f"\n📊 Connection Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print(f"\n✅ Railway database setup verification completed successfully!")
        
        if db_info['is_postgresql']:
            print("\n🎉 Congratulations! Your Railway database is properly configured.")
            print("   Both trading bot and dashboard services are now sharing the same PostgreSQL database.")
        else:
            print("\n⚠️  Note: Currently using SQLite (likely local development).")
            print("   On Railway, ensure PostgreSQL service is deployed and DATABASE_URL is set.")
        
        return True
        
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_migration_needed():
    """Check if data migration is needed"""
    
    print("🔍 Checking Migration Requirements")
    print("=" * 40)
    
    try:
        # Check if local SQLite database exists with data
        sqlite_path = Path("src/data/trading_bot.db")
        
        if sqlite_path.exists():
            print(f"✅ Local SQLite database found: {sqlite_path}")
            
            # Check if it has data
            import sqlite3
            conn = sqlite3.connect(sqlite_path)
            cursor = conn.cursor()
            
            # Count records in key tables
            tables = ['trading_sessions', 'trades', 'positions']
            total_records = 0
            
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    total_records += count
                    print(f"  {table}: {count} records")
                except sqlite3.OperationalError:
                    print(f"  {table}: Table not found")
            
            conn.close()
            
            if total_records > 0:
                print(f"\n📊 Total records found: {total_records}")
                print("\n📤 Data Migration Available:")
                print("  - Run: python scripts/export_sqlite_data.py")
                print("  - Then: python scripts/import_to_postgresql.py")
                print("  - This will migrate your existing data to PostgreSQL")
            else:
                print("\n✅ No data migration needed - database is empty")
        
        else:
            print("✅ No local SQLite database found - no migration needed")
        
    except Exception as e:
        print(f"❌ Error checking migration requirements: {e}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Railway Database Setup Helper")
    parser.add_argument("--verify", action="store_true", help="Verify database setup")
    parser.add_argument("--check-migration", action="store_true", help="Check migration requirements")
    
    args = parser.parse_args()
    
    if args.verify:
        success = verify_railway_setup()
        sys.exit(0 if success else 1)
    elif args.check_migration:
        check_migration_needed()
    else:
        print_setup_instructions()


if __name__ == "__main__":
    main()