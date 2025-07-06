#!/usr/bin/env python3
"""
Export SQLite data to JSON format for migration to PostgreSQL
"""

import sys
import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from config.paths import get_data_dir
from database.models import (
    Trade, Position, AccountHistory, PerformanceMetrics, 
    TradingSession, SystemEvent, StrategyExecution
)


def export_sqlite_data(db_path=None, output_file=None):
    """Export SQLite data to JSON format"""
    
    if db_path is None:
        db_path = get_data_dir() / "trading_bot.db"
    
    if output_file is None:
        output_file = get_data_dir() / "migration_export.json"
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found at: {db_path}")
        return False
    
    print(f"📊 Exporting SQLite data from: {db_path}")
    print(f"📁 Output file: {output_file}")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable dict-like access
    
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "source_database": str(db_path),
        "tables": {}
    }
    
    # Define tables to export
    tables = [
        'trading_sessions',
        'trades', 
        'positions',
        'account_history',
        'performance_metrics',
        'system_events',
        'strategy_executions'
    ]
    
    cursor = conn.cursor()
    
    for table in tables:
        try:
            # Check if table exists
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if not cursor.fetchone():
                print(f"⚠️  Table {table} not found, skipping...")
                continue
                
            # Get table data
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            
            # Convert rows to dictionaries
            table_data = []
            for row in rows:
                row_dict = dict(row)
                # Convert any datetime strings to ISO format
                for key, value in row_dict.items():
                    if isinstance(value, str) and ('time' in key.lower() or 'timestamp' in key.lower()):
                        try:
                            # Try to parse and reformat datetime
                            if value:
                                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                                row_dict[key] = dt.isoformat()
                        except:
                            pass  # Keep original value if parsing fails
                table_data.append(row_dict)
            
            export_data["tables"][table] = table_data
            print(f"✅ Exported {len(table_data)} records from {table}")
            
        except Exception as e:
            print(f"❌ Error exporting table {table}: {e}")
    
    conn.close()
    
    # Write to JSON file
    try:
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"\n✅ Export completed successfully!")
        print(f"📊 Total tables exported: {len(export_data['tables'])}")
        
        # Show summary
        total_records = sum(len(records) for records in export_data["tables"].values())
        print(f"📈 Total records exported: {total_records}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error writing export file: {e}")
        return False


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export SQLite data for PostgreSQL migration")
    parser.add_argument("--db-path", help="Path to SQLite database file")
    parser.add_argument("--output", help="Output JSON file path")
    
    args = parser.parse_args()
    
    success = export_sqlite_data(args.db_path, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()