#!/usr/bin/env python3
"""
Live trading runner with built-in health check server
"""
import sys
import os
import threading
import signal
import time
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import health check server
from health_check import run_health_server

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n[{datetime.now()}] Received signal {signum}, shutting down...")
    sys.exit(0)

def main():
    """Main entry point"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get command line arguments (strategy name)
    strategy = sys.argv[1] if len(sys.argv) > 1 else "adaptive"
    
    print(f"[{datetime.now()}] Starting AI Trading Bot with strategy: {strategy}")
    
    # Start health check server in background thread
    health_port = int(os.getenv('HEALTH_CHECK_PORT', '8000'))
    health_thread = threading.Thread(
        target=run_health_server, 
        args=(health_port,),
        daemon=True
    )
    health_thread.start()
    
    print(f"[{datetime.now()}] Health check server started on port {health_port}")
    
    # Give health server a moment to start
    time.sleep(2)
    
    # Import and run the main trading application
    try:
        print(f"[{datetime.now()}] Starting live trading with {strategy} strategy...")
        
        # Import here to avoid circular imports
        from run_live_trading import main as run_trading
        
        # Run the trading bot
        sys.argv = ['run_live_trading.py', strategy]  # Set argv for the trading script
        run_trading()
        
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] Keyboard interrupt received")
    except Exception as e:
        print(f"[{datetime.now()}] Error in main application: {e}")
        raise
    finally:
        print(f"[{datetime.now()}] AI Trading Bot shutting down")

if __name__ == '__main__':
    main()