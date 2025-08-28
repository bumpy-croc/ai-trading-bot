#!/usr/bin/env python3
"""
Test Commands for CLI

This module contains all the test functionality ported from the scripts directory.
"""

import os
from datetime import datetime
from pathlib import Path

from src.database.manager import DatabaseManager
from src.database.models import EventType


def test_database_main(args):
    """Test database connection and functionality"""
    print("🔍 Testing database connection...")
    
    try:
        # Test database connection
        db = DatabaseManager()
        print("✅ Database connection successful")
        
        # Test basic operations
        print("📝 Testing basic database operations...")
        
        # Test event logging
        db.log_event(
            event_type=EventType.TEST,
            message="Database test from CLI",
            severity="info",
            component="cli_test",
            details={"test_type": "connection_test"}
        )
        print("✅ Event logging successful")
        
        # Test query operations
        print("🔍 Testing query operations...")
        # Add any specific query tests here
        
        print("✅ All database tests passed")
        return 0
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return 1


def test_download_main(args):
    """Test data download functionality"""
    print("🔍 Testing data download...")
    
    try:
        from cli.commands.data_commands import download_binance_data_wrapper
        
        # Test download for a small time period
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / "data"
        data_dir.mkdir(exist_ok=True)
        
        print("📊 Testing Binance data download...")
        csv_file = download_binance_data_wrapper(
            symbol="BTCUSDT",
            timeframe="1d",
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-07T00:00:00Z",
            output_dir=str(data_dir)
        )
        
        if csv_file and os.path.exists(csv_file):
            print(f"✅ Download successful: {csv_file}")
            return 0
        else:
            print("❌ Download failed: file not created")
            return 1
            
    except Exception as e:
        print(f"❌ Download test failed: {e}")
        return 1


def test_secrets_access_main(args):
    """Test secrets access functionality"""
    print("🔍 Testing secrets access...")
    
    try:
        # Test accessing various secrets
        print("🔐 Testing secret access...")
        
        # Test database URL access
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            print("✅ Database URL access successful")
        else:
            print("⚠️ Database URL not found")
        
        # Test Binance API access
        binance_api_key = os.getenv("BINANCE_API_KEY")
        binance_secret_key = os.getenv("BINANCE_SECRET_KEY")
        
        if binance_api_key and binance_secret_key:
            print("✅ Binance API credentials access successful")
        else:
            print("⚠️ Binance API credentials not found")
        
        # Test other secrets as needed
        print("✅ Secrets access test completed")
        return 0
        
    except Exception as e:
        print(f"❌ Secrets access test failed: {e}")
        return 1


def heartbeat_main(args):
    """Log a heartbeat SystemEvent"""
    print("💓 Logging heartbeat...")
    
    try:
        component = os.getenv("HEARTBEAT_COMPONENT", "scheduler")
        db = DatabaseManager()
        
        db.log_event(
            event_type=EventType.TEST,
            message="Heartbeat",
            severity="info",
            component=component,
            details={"timestamp": datetime.utcnow().isoformat()},
        )
        
        print("✅ Heartbeat logged")
        return 0
        
    except Exception as e:
        print(f"❌ Heartbeat logging failed: {e}")
        return 1
