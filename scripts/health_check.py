#!/usr/bin/env python3
"""
Health check endpoint for Railway deployments
"""
import sys
import os
import json
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.config_manager import get_config
    from data_providers import BinanceDataProvider
    from database.manager import DatabaseManager
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback for minimal health check
    pass


class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check requests"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health':
            self.handle_health_check()
        elif self.path == '/status':
            self.handle_status_check()
        else:
            self.send_error(404, "Not Found")
    
    def handle_health_check(self):
        """Basic health check - returns 200 if service is responding"""
        try:
            response = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "service": "ai-trading-bot"
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_error(500, f"Health check failed: {str(e)}")
    
    def handle_status_check(self):
        """Detailed status check - includes component checks"""
        try:
            status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "service": "ai-trading-bot",
                "components": {}
            }
            
            # Check configuration
            try:
                config = get_config()
                status["components"]["config"] = {
                    "status": "healthy",
                    "providers": [p.provider_name for p in config.providers if p.is_available()]
                }
            except Exception as e:
                status["components"]["config"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Check database connectivity
            try:
                db_manager = DatabaseManager()
                # Test database by creating a session and checking it works
                with db_manager.get_session() as session:
                    session.execute("SELECT 1")
                status["components"]["database"] = {"status": "healthy"}
            except Exception as e:
                status["components"]["database"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Check Binance API connectivity
            try:
                provider = BinanceDataProvider()
                # Get recent data to test API connectivity
                data = provider.get_live_data('BTCUSDT', '1h', limit=1)
                if not data.empty and 'close' in data.columns:
                    price = data['close'].iloc[-1]
                    status["components"]["binance_api"] = {
                        "status": "healthy",
                        "btc_price": float(price)
                    }
                else:
                    status["components"]["binance_api"] = {
                        "status": "unhealthy",
                        "error": "No price data returned"
                    }
            except Exception as e:
                status["components"]["binance_api"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Determine overall status
            unhealthy_components = [
                name for name, comp in status["components"].items() 
                if comp.get("status") != "healthy"
            ]
            
            if unhealthy_components:
                status["status"] = "degraded"
                status["unhealthy_components"] = unhealthy_components
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(status, indent=2).encode())
            
        except Exception as e:
            error_response = {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
            
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())
    
    def log_message(self, format, *args):
        """Override to reduce verbose logging"""
        pass


def run_health_server(port=8000):
    """Run the health check server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, HealthCheckHandler)
    print(f"Health check server running on port {port}")
    print(f"Endpoints: /health (basic), /status (detailed)")
    httpd.serve_forever()


if __name__ == '__main__':
    import threading
    import time
    
    # Run health server in background
    port = int(os.getenv('HEALTH_CHECK_PORT', '8000'))
    health_thread = threading.Thread(target=run_health_server, args=(port,))
    health_thread.daemon = True
    health_thread.start()
    
    print(f"Health check server started on port {port}")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Health check server stopping...")
        sys.exit(0)