#!/usr/bin/env python3
"""
Test script to diagnose monitoring dashboard startup issues
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.insert(1, str(project_root / "src"))

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")

    try:
        from src.dashboards.monitoring.dashboard import MonitoringDashboard
        print("✅ MonitoringDashboard import successful")
    except Exception as e:
        print(f"❌ MonitoringDashboard import failed: {e}")
        return False

    try:
        from src.database.manager import DatabaseManager
        print("✅ DatabaseManager import successful")
    except Exception as e:
        print(f"❌ DatabaseManager import failed: {e}")
        return False

    try:
        from src.data_providers.binance_provider import BinanceProvider
        print("✅ BinanceProvider import successful")
    except Exception as e:
        print(f"❌ BinanceProvider import failed: {e}")
        return False

    return True

def test_database_connection():
    """Test database connection"""
    print("\nTesting database connection...")

    try:
        from src.database.manager import DatabaseManager
        from sqlalchemy import text

        db = DatabaseManager()
        with db.get_session() as session:
            session.execute(text("SELECT 1"))
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_dashboard_initialization():
    """Test dashboard initialization"""
    print("\nTesting dashboard initialization...")

    try:
        from src.dashboards.monitoring.dashboard import MonitoringDashboard
        dashboard = MonitoringDashboard()
        print("✅ Dashboard initialization successful")
        return True
    except Exception as e:
        print(f"❌ Dashboard initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_health_endpoint():
    """Test health endpoint"""
    print("\nTesting health endpoint...")

    try:
        from src.dashboards.monitoring.dashboard import MonitoringDashboard
        dashboard = MonitoringDashboard()

        # Test the health endpoint directly
        with dashboard.app.test_client() as client:
            response = client.get('/health')
            if response.status_code == 200:
                data = response.get_json()
                if data.get('status') == 'ok':
                    print("✅ Health endpoint test successful")
                    return True
            print(f"❌ Health endpoint test failed: {response.status_code} - {response.get_data(as_text=True)}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔍 Monitoring Dashboard Startup Diagnostic")
    print("=" * 50)

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_database_connection()
    all_passed &= test_dashboard_initialization()
    all_passed &= test_health_endpoint()

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Monitoring dashboard should start successfully.")
    else:
        print("❌ Some tests failed. Check the output above for issues.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
