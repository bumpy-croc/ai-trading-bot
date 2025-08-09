#!/usr/bin/env python3
"""
Trading Bot Monitoring Dashboard Launcher

Simple script to start the monitoring dashboard with sensible defaults.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Also add 'src' directory to ensure internal packages resolve
src_path = project_root / 'src'
sys.path.append(str(src_path))

def main():
    """Launch the monitoring dashboard"""
    try:
        from monitoring.dashboard import MonitoringDashboard
        
        print("🚀 Starting Trading Bot Monitoring Dashboard...")
        port = int(os.environ.get("PORT", 8000))
        print(f"📊 Dashboard will be available at: http://localhost:{port}")
        print("⚙️  Configuration panel accessible via the gear icon")
        print("🔄 Updates every 1 hour")
        print("❌ Press Ctrl+C to stop")
        print("-" * 60)
        
        # Create and run dashboard
        dashboard = MonitoringDashboard(
            db_url=None,  # Uses DATABASE_URL from environment (PostgreSQL)
            update_interval=3600  # 1 hour updates
        )
        
        # Respect Railway (and other PaaS) dynamic port assignment (see assignment above)
        dashboard.run(
            host="0.0.0.0",
            port=port,
            debug=False
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Please install requirements: pip install -r requirements-server.txt")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()