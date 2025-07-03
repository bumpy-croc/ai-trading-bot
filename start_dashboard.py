#!/usr/bin/env python3
"""
Trading Bot Monitoring Dashboard Launcher

Simple script to start the monitoring dashboard with sensible defaults.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def main():
    """Launch the monitoring dashboard"""
    try:
        from monitoring.dashboard import MonitoringDashboard
        
        print("🚀 Starting Trading Bot Monitoring Dashboard...")
        print("📊 Dashboard will be available at: http://localhost:8080")
        print("⚙️  Configuration panel accessible via the gear icon")
        print("🔄 Updates every 1 hour")
        print("❌ Press Ctrl+C to stop")
        print("-" * 60)
        
        # Create and run dashboard
        dashboard = MonitoringDashboard(
            db_url=None,  # Will use default SQLite database
            update_interval=3600  # 1 hour updates
        )
        
        dashboard.run(
            host='0.0.0.0',
            port=8080,
            debug=False
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Please install requirements: pip install -r monitoring/requirements.txt")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()