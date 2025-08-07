#!/usr/bin/env python3
"""
Railway PostgreSQL Database Setup Script
Helps set up and configure the PostgreSQL database on Railway
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
    """Print Railway PostgreSQL database setup instructions"""
    
    print("🚀 Railway PostgreSQL Database Setup")
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
    print("     use the PostgreSQL database when DATABASE_URL is available")
    print("   - DatabaseManager requires PostgreSQL connection")
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
    print("  ✅ Connection pooling for better performance")
    print("  ✅ ACID transactions for data integrity")
    print("  ✅ Built-in backup and recovery")
    print("  ✅ Scalable for growing datasets")
    print()
    print("💡 Local Development:")
    print("  - Use Docker PostgreSQL or native installation")
    print("  - Set DATABASE_URL environment variable")
    print("  - See docs/RAILWAY_DATABASE_CENTRALIZATION_GUIDE.md")
    print()


def verify_railway_setup():
    """Verify Railway PostgreSQL database setup"""
    
    print("🔍 Verifying Railway PostgreSQL Database Setup")
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
                print("❌ Database URL does not start with 'postgresql://'")
                print("   Unsupported database URL. Railway verification requires 'postgresql://'.")
                return False
        else:
            print("❌ DATABASE_URL not found")
            print("   PostgreSQL connection string is required")
            return False
        
        print()
        
        # Test database connection
        print("🔗 Testing PostgreSQL Database Connection...")
        db_manager = DatabaseManager()
        
        db_info = db_manager.get_database_info()
        print(f"  Database Type: {db_info['database_type']}")
        print(f"  Connection Pool Size: {db_info['connection_pool_size']}")
        
        if db_manager.test_connection():
            print("✅ PostgreSQL database connection successful!")
        else:
            print("❌ PostgreSQL database connection failed!")
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
            message="Railway PostgreSQL database verification test",
            severity="info",
            session_id=session_id
        )
        print(f"✅ Logged test event #{event_id}")
        
        # End session
        db_manager.end_trading_session(session_id, final_balance=10000.0)
        print(f"✅ Ended test session #{session_id}")
        
        # Connection stats
        stats = db_manager.get_connection_stats()
        print(f"\n📊 Connection Pool Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print(f"\n✅ Railway PostgreSQL database setup verification completed successfully!")
        print("\n🎉 Congratulations! Your Railway PostgreSQL database is properly configured.")
        print("   Both trading bot and dashboard services can now share the same database.")
        
        return True
        
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure PostgreSQL service is deployed on Railway")
        print("  2. Check that DATABASE_URL environment variable is set")
        print("  3. Verify the DATABASE_URL starts with 'postgresql://'")
        print("  4. Check Railway service logs for connection errors")
        import traceback
        traceback.print_exc()
        return False


def check_local_development():
    """Check local development PostgreSQL setup"""
    
    print("🔍 Checking Local Development Setup")
    print("=" * 40)
    
    try:
        config = get_config()
        database_url = config.get('DATABASE_URL')
        
        if database_url:
            if database_url.startswith('postgresql'):
                print(f"✅ PostgreSQL DATABASE_URL configured")
                print(f"   URL: {database_url}")
                
                # Test connection
                db_manager = DatabaseManager()
                if db_manager.test_connection():
                    print("✅ Local PostgreSQL connection successful")
                    return True
                else:
                    print("❌ Local PostgreSQL connection failed")
                    return False
            else:
                print("❌ DATABASE_URL does not start with 'postgresql://'")
                return False
        else:
            print("❌ DATABASE_URL not set")
            print("\nLocal Development Setup Required:")
            print("  1. Set up PostgreSQL locally (Docker or native)")
            print("  2. Set DATABASE_URL environment variable")
            print("  3. Example: export DATABASE_URL=postgresql://user:pass@localhost:5432/trading_db")
            print("  4. See docs/RAILWAY_DATABASE_CENTRALIZATION_GUIDE.md for details")
            return False
            
    except Exception as e:
        print(f"❌ Error checking local development setup: {e}")
        return False


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Railway PostgreSQL Database Setup Helper")
    parser.add_argument("--verify", action="store_true", help="Verify database setup")
    parser.add_argument("--check-local", action="store_true", help="Check local development setup")
    
    args = parser.parse_args()
    
    if args.verify:
        success = verify_railway_setup()
        sys.exit(0 if success else 1)
    elif args.check_local:
        success = check_local_development()
        sys.exit(0 if success else 1)
    else:
        print_setup_instructions()


if __name__ == "__main__":
    main()