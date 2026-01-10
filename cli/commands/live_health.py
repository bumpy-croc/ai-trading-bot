from __future__ import annotations

import argparse
import json
import os
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

from cli.core.forward import forward_to_module_main


class _HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/status":
            self._handle_status()
        else:
            self.send_error(404, "Not Found")

    def _handle_health(self):
        try:
            response = {
                "status": "healthy",
                "timestamp": datetime.now(UTC).isoformat(),
                "service": "ai-trading-bot",
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:  # pragma: no cover
            self.send_error(500, f"Health check failed: {str(e)}")

    def _handle_status(self):
        try:
            status = {
                "status": "healthy",
                "timestamp": datetime.now(UTC).isoformat(),
                "service": "ai-trading-bot",
                "components": {},
            }
            # Config providers
            try:
                from config.config_manager import get_config

                cfg = get_config()
                status["components"]["config"] = {
                    "status": "healthy",
                    "providers": [p.provider_name for p in cfg.providers if p.is_available()],
                }
            except Exception as e:
                status["components"]["config"] = {"status": "unhealthy", "error": str(e)}

            # Database
            try:
                from database.manager import DatabaseManager

                dbm = DatabaseManager()
                with dbm.get_session() as s:
                    s.execute("SELECT 1")
                status["components"]["database"] = {"status": "healthy"}
            except Exception as e:
                status["components"]["database"] = {"status": "unhealthy", "error": str(e)}

            # Binance API (best-effort)
            try:
                from data_providers.binance_provider import BinanceProvider

                prov = BinanceProvider()
                df = prov.get_live_data("BTCUSDT", "1h", limit=1)
                if df is not None and not df.empty and "close" in df.columns:
                    status["components"]["binance_api"] = {
                        "status": "healthy",
                        "btc_price": float(df["close"].iloc[-1]),
                    }
                else:
                    status["components"]["binance_api"] = {
                        "status": "unhealthy",
                        "error": "No price data returned",
                    }
            except Exception as e:
                status["components"]["binance_api"] = {"status": "unhealthy", "error": str(e)}

            unhealthy = [
                name
                for name, comp in status["components"].items()
                if comp.get("status") != "healthy"
            ]
            if unhealthy:
                status["status"] = "degraded"
                status["unhealthy_components"] = unhealthy

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status, indent=2).encode())
        except Exception as e:  # pragma: no cover
            error_response = {
                "status": "unhealthy",
                "timestamp": datetime.now(UTC).isoformat(),
                "error": str(e),
            }
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())

    def log_message(self, format, *args):  # noqa: A003
        pass


def _run_health_server(port: int) -> None:
    try:
        httpd = HTTPServer(("", port), _HealthCheckHandler)
        print(
            f"Health check server running on port {port}\nEndpoints: /health (basic), /status (detailed)"
        )
        httpd.serve_forever()
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"⚠️  Port {port} is already in use. Health server will not start.")
            print("   You can set a different port with: PORT=<port> make live-health")
        else:
            raise


def _handle(ns: argparse.Namespace) -> int:
    # start health server with same port logic as original script
    port = int(os.getenv("PORT", os.getenv("HEALTH_CHECK_PORT", "8000")))
    t = threading.Thread(target=_run_health_server, args=(port,), daemon=True)
    t.start()
    # forward to live runner with paper-trading default safety
    # Filter out --port and --help arguments that shouldn't be passed to the live trading script
    tail = ns.args or []
    filtered_args = []
    i = 0
    while i < len(tail):
        if (
            tail[i] in ["--port", "--help"]
            and i + 1 < len(tail)
            and not tail[i + 1].startswith("-")
        ):
            # Skip --port and its value
            i += 2
        elif tail[i] in ["--port", "--help"]:
            # Skip --port without value
            i += 1
        else:
            filtered_args.append(tail[i])
            i += 1
    return forward_to_module_main("src.engines.live.runner", filtered_args)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "live-health",
        help="Run live trading with embedded health server (uses PORT env var, defaults to 8000)",
    )
    p.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed through to runner")
    p.set_defaults(func=_handle)
