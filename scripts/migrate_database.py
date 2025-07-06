#!/usr/bin/env python3
"""
Database Migration Script

This script handles database schema migrations for the AI Trading Bot.
It ensures the database is up-to-date with the latest model definitions.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from sqlalchemy import create_engine, text
from database.models import Base
from database.manager import DatabaseManager
from config.paths import get_database_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """Handles database schema migrations"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or get_database_path()
        self.engine = create_engine(self.database_url)
        self.db_manager = DatabaseManager(database_url or self.database_url)
    
    def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database"""
        try:
            with self.engine.connect() as connection:
                # SQLite compatible query
                result = connection.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name"),
                    {"table_name": table_name}
                )
                return result.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False
    
    def create_all_tables(self):
        """Create all tables defined in the models"""
        try:
            logger.info("Creating database tables...")
            Base.metadata.create_all(self.engine)
            logger.info("✅ All tables created successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error creating tables: {e}")
            return False
    
    def migrate_to_latest(self) -> bool:
        """Run all necessary migrations to bring database to latest schema"""
        logger.info("🔄 Starting database migration...")
        
        # Check if AccountBalance table exists (added in persistent balance update)
        if not self.check_table_exists('account_balances'):
            logger.info("➕ Adding AccountBalance table...")
            try:
                # Create just the AccountBalance table
                from database.models import AccountBalance
                AccountBalance.__table__.create(self.engine, checkfirst=True)
                logger.info("✅ AccountBalance table created")
            except Exception as e:
                logger.error(f"❌ Error creating AccountBalance table: {e}")
                return False
        else:
            logger.info("✓ AccountBalance table already exists")
        
        # Ensure all other tables exist
        missing_tables = []
        expected_tables = [
            'trades', 'positions', 'account_history', 'performance_metrics',
            'trading_sessions', 'system_events', 'strategy_executions',
            'account_balances'
        ]
        
        for table in expected_tables:
            if not self.check_table_exists(table):
                missing_tables.append(table)
        
        if missing_tables:
            logger.info(f"➕ Creating missing tables: {missing_tables}")
            try:
                Base.metadata.create_all(self.engine, checkfirst=True)
                logger.info("✅ Missing tables created")
            except Exception as e:
                logger.error(f"❌ Error creating missing tables: {e}")
                return False
        else:
            logger.info("✓ All expected tables exist")
        
        # Add migration for balance recovery
        self._migrate_existing_sessions()
        
        logger.info("✅ Database migration completed successfully")
        return True
    
    def _migrate_existing_sessions(self):
        """Migrate existing sessions to have balance tracking"""
        try:
            # Find active sessions without balance records
            query = """
            SELECT ts.id, ts.initial_balance
            FROM trading_sessions ts
            LEFT JOIN account_balances ab ON ts.id = ab.session_id
            WHERE ts.is_active = 1 AND ab.id IS NULL
            """
            
            result = self.db_manager.execute_query(query)
            
            if result:
                logger.info(f"🔄 Migrating {len(result)} existing sessions to have balance tracking...")
                
                for session in result:
                    session_id = session['id']
                    initial_balance = session['initial_balance']
                    
                    # Calculate current balance from trades
                    trades_query = """
                    SELECT COALESCE(SUM(pnl), 0) as total_pnl
                    FROM trades
                    WHERE session_id = :session_id
                    """
                    
                    trades_result = self.db_manager.execute_query(trades_query, {"session_id": session_id})
                    total_pnl = trades_result[0]['total_pnl'] if trades_result else 0
                    current_balance = initial_balance + total_pnl
                    
                    # Create balance record
                    self.db_manager.update_balance(
                        current_balance,
                        'migrated_from_existing_session',
                        'migration_script',
                        session_id
                    )
                    
                    logger.info(f"✅ Migrated session #{session_id}: ${initial_balance:,.2f} → ${current_balance:,.2f}")
                
                logger.info("✅ Session migration completed")
            else:
                logger.info("✓ No sessions need migration")
                
        except Exception as e:
            logger.error(f"❌ Error migrating existing sessions: {e}")
    
    def validate_schema(self) -> bool:
        """Validate that the database schema matches the expected structure"""
        logger.info("🔍 Validating database schema...")
        
        try:
            # Test basic operations
            current_balance = self.db_manager.get_current_balance()
            logger.info(f"✓ Balance retrieval test: ${current_balance:,.2f}")
            
            active_positions = self.db_manager.get_active_positions()
            logger.info(f"✓ Position retrieval test: {len(active_positions)} positions")
            
            balance_history = self.db_manager.get_balance_history(limit=1)
            logger.info(f"✓ Balance history test: {len(balance_history)} records")
            
            logger.info("✅ Schema validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Schema validation failed: {e}")
            return False
    
    def reset_database(self) -> bool:
        """Reset the database (DANGEROUS - deletes all data)"""
        logger.warning("⚠️  RESETTING DATABASE - ALL DATA WILL BE LOST")
        
        try:
            # Drop all tables
            Base.metadata.drop_all(self.engine)
            logger.info("🗑️  All tables dropped")
            
            # Recreate all tables
            Base.metadata.create_all(self.engine)
            logger.info("✅ Database reset completed")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error resetting database: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Database Migration Tool')
    parser.add_argument('action', choices=['migrate', 'validate', 'reset', 'create'], 
                       help='Migration action to perform')
    parser.add_argument('--database-url', help='Database URL (optional)')
    parser.add_argument('--force', action='store_true', 
                       help='Force operation without confirmation (dangerous for reset)')
    
    args = parser.parse_args()
    
    migrator = DatabaseMigrator(args.database_url)
    
    if args.action == 'migrate':
        success = migrator.migrate_to_latest()
        if success:
            # Also validate after migration
            migrator.validate_schema()
        sys.exit(0 if success else 1)
    
    elif args.action == 'validate':
        success = migrator.validate_schema()
        sys.exit(0 if success else 1)
    
    elif args.action == 'create':
        success = migrator.create_all_tables()
        sys.exit(0 if success else 1)
    
    elif args.action == 'reset':
        if not args.force:
            print("⚠️  WARNING: This will delete ALL data in the database!")
            print("This includes all trades, positions, and balance history.")
            confirmation = input("Type 'DELETE ALL DATA' to confirm: ")
            if confirmation != "DELETE ALL DATA":
                print("❌ Reset cancelled")
                sys.exit(1)
        
        success = migrator.reset_database()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()