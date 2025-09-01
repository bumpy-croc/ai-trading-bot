#!/usr/bin/env python3
"""
Robust startup script for monitoring dashboard in Railway
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "src") not in sys.path:
    sys.path.insert(1, str(project_root / "src"))

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def wait_for_database(max_retries=30, retry_delay=2):
    """Wait for database to be ready"""
    from src.database.manager import DatabaseManager
    from sqlalchemy import text

    logger.info("Waiting for database connection...")

    for attempt in range(max_retries):
        try:
            db = DatabaseManager()
            with db.get_session() as session:
                session.execute(text("SELECT 1"))
            logger.info("✅ Database connection established")
            return True
        except Exception as e:
            logger.warning(f"Database connection attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    logger.error("❌ Failed to connect to database after all retries")
    return False

def start_dashboard():
    """Start the monitoring dashboard with proper error handling"""
    try:
        logger.info("Starting monitoring dashboard...")

        # Import and create dashboard
        from src.dashboards.monitoring.dashboard import MonitoringDashboard

        # Create dashboard instance (this will test database connection)
        dashboard = MonitoringDashboard()
        logger.info("✅ Monitoring dashboard initialized successfully")

        # Get port from environment (Railway sets PORT)
        port = int(os.getenv("PORT", "8000"))
        host = os.getenv("HOST", "0.0.0.0")

        logger.info(f"Starting server on {host}:{port}")

        # Start the dashboard
        dashboard.run(host=host, port=port, debug=False)

    except Exception as e:
        logger.error(f"❌ Failed to start monitoring dashboard: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """Main startup routine"""
    logger.info("🚀 Starting AI Trading Bot Monitoring Dashboard")
    logger.info("=" * 60)

    # Wait for database to be ready
    if not wait_for_database():
        logger.error("Cannot proceed without database connection")
        sys.exit(1)

    # Start the dashboard
    start_dashboard()

if __name__ == "__main__":
    main()
