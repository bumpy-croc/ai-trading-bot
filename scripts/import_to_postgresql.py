#!/usr/bin/env python3
"""
Import JSON export data into a PostgreSQL database.

This utility can restore data that was previously exported (e.g. from an
older SQLite deployment) into the current PostgreSQL schema.  The tool
expects the JSON structure produced by the now-deprecated
``export_sqlite_data.py`` script.
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from config.paths import get_data_dir
from database.manager import DatabaseManager
from database.models import (
    Base, Trade, Position, AccountHistory, PerformanceMetrics, 
    TradingSession, SystemEvent, StrategyExecution,
    PositionSide, OrderStatus, TradeSource, EventType
)
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def import_postgresql_data(import_file=None, database_url=None):
    """Import data from JSON export into PostgreSQL"""
    
    if import_file is None:
        import_file = get_data_dir() / "migration_export.json"
    
    if not os.path.exists(import_file):
        print(f"❌ Import file not found at: {import_file}")
        return False
    
    print(f"📊 Importing data from: {import_file}")
    
    # Load export data
    try:
        with open(import_file, 'r') as f:
            export_data = json.load(f)
        print(f"✅ Loaded export data from {export_data['export_timestamp']}")
    except Exception as e:
        print(f"❌ Error loading export file: {e}")
        return False
    
    # Initialize database connection
    try:
        if database_url:
            db_manager = DatabaseManager(database_url)
        else:
            # Use environment DATABASE_URL
            db_manager = DatabaseManager()
        
        print(f"✅ Connected to PostgreSQL database")
        
        # Create tables if they don't exist
        with db_manager.get_session() as session:
            print("🔧 Creating database tables...")
            Base.metadata.create_all(db_manager.engine)
            
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return False
    
    # Import data table by table
    tables = export_data.get("tables", {})
    
    # Define import order (respecting foreign key constraints)
    import_order = [
        'trading_sessions',
        'trades',
        'positions', 
        'account_history',
        'performance_metrics',
        'system_events',
        'strategy_executions'
    ]
    
    total_imported = 0
    
    for table_name in import_order:
        if table_name not in tables:
            print(f"⚠️  Table {table_name} not found in export, skipping...")
            continue
            
        table_data = tables[table_name]
        if not table_data:
            print(f"⚠️  Table {table_name} is empty, skipping...")
            continue
            
        print(f"📥 Importing {len(table_data)} records into {table_name}...")
        
        try:
            imported_count = import_table_data(db_manager, table_name, table_data)
            total_imported += imported_count
            print(f"✅ Imported {imported_count} records into {table_name}")
            
        except Exception as e:
            print(f"❌ Error importing table {table_name}: {e}")
            continue
    
    print(f"\n✅ Import completed successfully!")
    print(f"📈 Total records imported: {total_imported}")
    
    return True


def import_table_data(db_manager, table_name, table_data):
    """Import data for a specific table"""
    
    # Map table names to SQLAlchemy models
    model_mapping = {
        'trading_sessions': TradingSession,
        'trades': Trade,
        'positions': Position,
        'account_history': AccountHistory,
        'performance_metrics': PerformanceMetrics,
        'system_events': SystemEvent,
        'strategy_executions': StrategyExecution
    }
    
    if table_name not in model_mapping:
        raise ValueError(f"Unknown table: {table_name}")
    
    model_class = model_mapping[table_name]
    imported_count = 0
    
    with db_manager.get_session() as session:
        for record in table_data:
            try:
                # Convert enum string values back to enum objects
                processed_record = convert_enum_values(table_name, record)
                
                # Create model instance
                instance = model_class(**processed_record)
                
                # Add to session
                session.add(instance)
                imported_count += 1
                
            except Exception as e:
                print(f"⚠️  Error processing record in {table_name}: {e}")
                continue
        
        # Commit all records for this table
        session.commit()
    
    return imported_count


def convert_enum_values(table_name, record):
    """Convert string enum values back to enum objects"""
    
    processed_record = record.copy()
    
    # Convert enum fields based on table
    if table_name == 'trades':
        if 'side' in processed_record and processed_record['side']:
            processed_record['side'] = PositionSide(processed_record['side'])
        if 'source' in processed_record and processed_record['source']:
            processed_record['source'] = TradeSource(processed_record['source'])
    
    elif table_name == 'positions':
        if 'side' in processed_record and processed_record['side']:
            processed_record['side'] = PositionSide(processed_record['side'])
        if 'status' in processed_record and processed_record['status']:
            processed_record['status'] = OrderStatus(processed_record['status'])
    
    elif table_name == 'trading_sessions':
        if 'mode' in processed_record and processed_record['mode']:
            processed_record['mode'] = TradeSource(processed_record['mode'])
    
    elif table_name == 'system_events':
        if 'event_type' in processed_record and processed_record['event_type']:
            processed_record['event_type'] = EventType(processed_record['event_type'])
    
    # Convert datetime strings to datetime objects
    for key, value in processed_record.items():
        if isinstance(value, str) and ('time' in key.lower() or 'timestamp' in key.lower()):
            try:
                if value:
                    processed_record[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except:
                pass  # Keep original value if parsing fails
    
    return processed_record


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Import JSON data into PostgreSQL")
    parser.add_argument("--import-file", help="Path to JSON export file")
    parser.add_argument("--database-url", help="PostgreSQL database URL")
    
    args = parser.parse_args()
    
    success = import_postgresql_data(args.import_file, args.database_url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()