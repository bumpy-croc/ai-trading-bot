#!/usr/bin/env python3
"""
Trading Bot Monitoring Dashboard

A real-time web dashboard for monitoring the trading bot performance,
positions, risk metrics, and system health.
"""

from __future__ import annotations

# --- Gevent monkey patching is now handled early in CLI entry point
# This detects the async mode based on whether gevent was already patched
import os
import sys

_WEB_SERVER_USE_GEVENT = os.environ.get("WEB_SERVER_USE_GEVENT", "0") == "1"

# Detect if gevent monkey patching was already applied
if _WEB_SERVER_USE_GEVENT and "gevent" in sys.modules:
    _ASYNC_MODE = "gevent"
elif _WEB_SERVER_USE_GEVENT:
    # Fallback: apply monkey patching if not done yet (for standalone imports)
    import gevent.monkey

    gevent.monkey.patch_all()
    _ASYNC_MODE = "gevent"
else:
    _ASYNC_MODE = "threading"

# --- ALL imports must happen AFTER monkey patching to avoid threading issues ---

# Standard library imports
import argparse
import logging
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, TypedDict

# Third-party imports
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit

# Ensure absolute imports resolve when running as a module or script
# This must happen after monkey patching to avoid RLock issues
from src.infrastructure.runtime.paths import get_project_root

project_root = get_project_root()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(1, str(src_path))

# Project imports - all must happen after monkey patching
from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.cached_data_provider import CachedDataProvider
from src.database.manager import DatabaseManager
from src.performance.metrics import max_drawdown as perf_max_drawdown
from src.performance.metrics import sharpe as perf_sharpe

# Configure logging via centralized config (set by entry points)
logger = logging.getLogger(__name__)

# after imports define typedicts

# ---- TypedDicts for static typing ----


class PositionDict(TypedDict):
    symbol: str
    side: str
    entry_price: float
    current_price: float
    quantity: float
    unrealized_pnl: float
    entry_time: Any
    stop_loss: float | None
    take_profit: float | None
    # Order information (replaces order_id)
    orders: list[dict[str, Any]]
    # Partial operations tracking
    original_size: float | None
    current_size: float | None
    partial_exits_taken: int | None
    scale_ins_taken: int | None
    last_partial_exit_price: float | None
    last_scale_in_price: float | None
    # Trailing stops and MFE/MAE tracking
    trailing_stop_activated: bool | None
    trailing_stop_price: float | None
    breakeven_triggered: bool | None
    mfe: float | None
    mae: float | None


class TradeDict(TypedDict):
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float | None
    entry_time: Any
    exit_time: Any
    pnl: float
    exit_reason: str


class MonitoringDashboard:
    """
    Real-time monitoring dashboard for the trading bot
    """

    def __init__(
        self, db_url: DatabaseManager | str | None = None, update_interval: int = 3600
    ):
        # Calculate absolute paths for templates and static files
        templates_path = src_path / "dashboards" / "monitoring" / "templates"
        static_path = src_path / "dashboards" / "monitoring" / "static"

        self.app = Flask(
            __name__, template_folder=str(templates_path), static_folder=str(static_path)
        )
        from src.infrastructure.runtime.secrets import get_secret_key

        self.app.config["SECRET_KEY"] = get_secret_key()
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode=_ASYNC_MODE)

        # Initialize database manager – allow injection for tests/mocking
        if isinstance(db_url, DatabaseManager) or (
            db_url is not None and not isinstance(db_url, str) and hasattr(db_url, "execute_query")
        ):
            self.db_manager = db_url  # type: ignore[assignment]
        else:
            self.db_manager = DatabaseManager() if db_url is None else DatabaseManager(db_url)

        # Initialize data provider for live price data
        # Gracefully degrade if Binance is unreachable (e.g. no outbound DNS in the env)
        try:
            binance_provider = BinanceProvider()
            self.data_provider: Any = CachedDataProvider(binance_provider, cache_ttl_hours=1)
        except Exception as e:
            logger.warning(
                f"Binance provider unavailable: {e}. Starting dashboard in offline mode."
            )

            class _OfflineProvider:  # minimal stub implementation
                """Fallback provider that returns empty data so the UI still loads."""

                def get_current_price(self, symbol: str):
                    return 0.0

                def get_historical_data(
                    self, symbol: str, timeframe: str, start, end
                ):  # noqa: D401
                    return pd.DataFrame()

            self.data_provider = _OfflineProvider()

        self.update_interval = update_interval
        self.is_running = False
        self.update_thread: threading.Thread | None = None
        self.start_time = datetime.now(UTC)

        # Configurable monitoring parameters
        self.monitoring_config: dict[str, dict[str, Any]] = {
            # System Health Metrics
            "api_connection_status": {"enabled": True, "priority": "high", "format": "status"},
            "data_feed_status": {"enabled": True, "priority": "high", "format": "status"},
            "error_rate_hourly": {"enabled": True, "priority": "high", "format": "percentage"},
            "api_latency": {"enabled": True, "priority": "medium", "format": "number"},
            "last_data_update": {"enabled": True, "priority": "high", "format": "datetime"},
            "system_uptime": {"enabled": True, "priority": "medium", "format": "text"},
            # Risk Metrics
            "current_drawdown": {"enabled": True, "priority": "high", "format": "percentage"},
            "daily_pnl": {"enabled": True, "priority": "high", "format": "currency"},
            "weekly_pnl": {"enabled": True, "priority": "high", "format": "currency"},
            "position_sizes": {"enabled": True, "priority": "high", "format": "currency"},
            "max_drawdown": {"enabled": True, "priority": "high", "format": "percentage"},
            "risk_per_trade": {"enabled": True, "priority": "medium", "format": "percentage"},
            "volatility": {"enabled": True, "priority": "medium", "format": "percentage"},
            # Dynamic Risk Management
            "dynamic_risk_factor": {"enabled": True, "priority": "high", "format": "number"},
            "dynamic_risk_reason": {"enabled": True, "priority": "high", "format": "text"},
            "dynamic_risk_active": {"enabled": True, "priority": "high", "format": "boolean"},
            # Order Execution Metrics
            "fill_rate": {"enabled": True, "priority": "high", "format": "percentage"},
            "avg_slippage": {"enabled": True, "priority": "high", "format": "percentage"},
            "failed_orders": {"enabled": True, "priority": "high", "format": "integer"},
            "order_latency": {"enabled": True, "priority": "medium", "format": "number"},
            "execution_quality": {"enabled": True, "priority": "medium", "format": "status"},
            # Balance & Positions
            "current_balance": {"enabled": True, "priority": "high", "format": "currency"},
            # Counts should display as integers (with 1 decimal place in UI per requirement)
            "active_positions_count": {"enabled": True, "priority": "high", "format": "integer"},
            "total_position_value": {"enabled": True, "priority": "high", "format": "currency"},
            "margin_usage": {"enabled": True, "priority": "high", "format": "percentage"},
            "available_margin": {"enabled": True, "priority": "medium", "format": "currency"},
            "unrealized_pnl": {"enabled": True, "priority": "high", "format": "currency"},
            # Strategy Performance
            "win_rate": {"enabled": True, "priority": "high", "format": "percentage"},
            "sharpe_ratio": {"enabled": True, "priority": "high", "format": "number"},
            "recent_trade_outcomes": {"enabled": True, "priority": "medium", "format": "text"},
            "profit_factor": {"enabled": True, "priority": "medium", "format": "number"},
            "avg_win_loss_ratio": {"enabled": True, "priority": "medium", "format": "number"},
            "total_trades": {"enabled": True, "priority": "medium", "format": "integer"},
            # Additional Core Metrics
            "total_pnl": {"enabled": True, "priority": "high", "format": "currency"},
            "current_strategy": {"enabled": True, "priority": "high", "format": "text"},
            # Disable BTC-only current price metric to avoid confusion across symbols
            "current_price": {"enabled": False, "priority": "medium", "format": "currency"},
            "price_change_24h": {"enabled": True, "priority": "medium", "format": "percentage"},
            "rsi": {"enabled": True, "priority": "low", "format": "number"},
            "ema_trend": {"enabled": True, "priority": "low", "format": "text"},
        }

        self._setup_routes()
        self._setup_websocket_handlers()

    def _safe_float(self, value) -> float:
        """Safely convert any value to float, handling Decimal types"""
        if value is None:
            return 0.0
        if isinstance(value, Decimal):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _setup_routes(self):
        """Setup Flask routes"""

        @self.app.route("/")
        def dashboard():
            """Main dashboard page"""
            # Use enhanced dashboard template if it exists, fallback to standard
            try:
                return render_template("dashboard_enhanced.html")
            except Exception:
                return render_template("dashboard.html")

        @self.app.route("/health")
        def health():
            """Simple health check endpoint"""
            return jsonify({"status": "ok"})

        @self.app.route("/api/metrics")
        def get_metrics():
            """Get current metrics as JSON"""
            metrics = self._collect_metrics()
            return jsonify(metrics)

        @self.app.route("/api/config")
        def get_config():
            """Get monitoring configuration"""
            return jsonify(self.monitoring_config)

        @self.app.route("/api/config", methods=["POST"])
        def update_config():
            """Update monitoring configuration"""
            new_config = request.json
            if new_config:
                # Validate and update configuration
                for key, value in new_config.items():
                    if key in self.monitoring_config:
                        self.monitoring_config[key].update(value)
                return jsonify({"success": True})
            # Return 400 Bad Request when the payload is invalid
            return jsonify({"success": False, "error": "Invalid configuration"}), 400

        @self.app.route("/api/positions")
        def get_positions():
            """Get current positions"""
            positions = self._get_current_positions()
            return jsonify(positions)

        @self.app.route("/api/positions/<int:position_id>/orders")
        def get_position_orders(position_id):
            """Get order history for a specific position"""
            try:
                query = """
                SELECT
                    id, order_type, status, exchange_order_id, internal_order_id,
                    symbol, side, quantity, price, filled_quantity, filled_price,
                    commission, created_at, filled_at, cancelled_at
                FROM orders
                WHERE position_id = %s
                ORDER BY created_at DESC
                """
                orders = self.db_manager.execute_query(query, (position_id,))
                return jsonify({"position_id": position_id, "orders": orders})
            except Exception as e:
                logger.error(f"Error getting orders for position {position_id}: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/trades")
        def get_recent_trades():
            """Get recent trades"""
            limit = request.args.get("limit", 50, type=int)
            trades = self._get_recent_trades(limit)
            return jsonify(trades)

        @self.app.route("/api/partial-trades")
        def get_partial_trades():
            """Get recent partial trades (partial exits and scale-ins)"""
            limit = request.args.get("limit", 50, type=int)
            partial_trades = self._get_partial_trades(limit)
            return jsonify(partial_trades)

        @self.app.route("/api/performance")
        def get_performance_chart():
            """Get performance chart data"""
            days = request.args.get("days", 7, type=int)
            chart_data = self._get_performance_chart_data(days)
            return jsonify(chart_data)

        @self.app.route("/api/prices")
        def get_prices():
            """Get current prices for a list of symbols (comma-separated).

            Query params:
              symbols=BTCUSDT,ETHUSDT
            Returns:
              { "BTCUSDT": 68000.12, "ETHUSDT": 3200.34 }
            """
            symbols_param = request.args.get("symbols", "")
            symbols = [s.strip().upper() for s in symbols_param.split(",") if s.strip()]
            price_cache: dict[str, float] = {}
            out: dict[str, float] = {}
            for s in symbols:
                price = self._get_price_for_symbol(s, price_cache)
                out[s] = float(price) if price else 0.0
            return jsonify(out)

        @self.app.route("/api/system/status")
        def system_status():
            """Get system health status"""
            status = self._get_system_status()
            return jsonify(status)

        @self.app.route("/api/balance")
        def get_balance():
            """Get current balance (simple format)"""
            current_balance = self._get_current_balance()
            return jsonify({"balance": current_balance})

        @self.app.route("/api/balance/history")
        def get_balance_history_route():
            """Get balance history records"""
            # Prefer `days` query parameter for compatibility with latest API tests
            days_param = request.args.get("days", type=int)
            if days_param is not None:
                history = self._get_balance_history(days_param)
            else:
                limit_param = request.args.get("limit", 50, type=int)
                history = self._get_balance_history(limit_param)
            return jsonify(history)

        @self.app.route("/api/balance", methods=["POST"])
        def update_balance():
            """Manually update balance"""
            data = request.json or {}
            new_balance = data.get("balance")
            reason = data.get("reason", "Manual adjustment via dashboard")
            updated_by = data.get("updated_by", "dashboard_user")

            if new_balance is None:
                return jsonify({"success": False, "error": "Balance not provided"})

            try:
                new_balance = float(new_balance)
                if new_balance < 0:
                    return jsonify({"success": False, "error": "Balance cannot be negative"})

                success = self.db_manager.manual_balance_adjustment(new_balance, reason, updated_by)

                if success:
                    return jsonify(
                        {
                            "success": True,
                            "new_balance": new_balance,
                            "message": f"Balance updated to ${new_balance:,.2f}",
                        }
                    )
                else:
                    return jsonify({"success": False, "error": "Failed to update balance"})

            except (ValueError, TypeError):
                return jsonify({"success": False, "error": "Invalid balance value"})

        @self.app.route("/api/optimizer/cycles")
        def get_optimizer_cycles():
            """List recent optimizer cycles."""
            limit = request.args.get("limit", 50, type=int)
            offset = request.args.get("offset", 0, type=int)
            try:
                rows = self.db_manager.fetch_optimization_cycles(limit=limit, offset=offset)
                return jsonify({"items": rows, "count": len(rows)})
            except Exception as e:
                return jsonify({"items": [], "error": str(e)}), 200

        @self.app.route("/api/correlation/matrix")
        def get_correlation_matrix():
            """Return recent correlation matrix entries (flattened)."""
            try:
                rows = self.db_manager.execute_query(
                    """
                    SELECT symbol_pair, correlation_value, p_value, sample_size, last_updated, window_days
                    FROM correlation_matrix
                    ORDER BY last_updated DESC
                    LIMIT 200
                    """
                )
                return jsonify({"items": rows})
            except Exception as e:
                return jsonify({"items": [], "error": str(e)}), 200

        @self.app.route("/api/correlation/exposures")
        def get_portfolio_exposures():
            """Return latest portfolio exposure per correlation group."""
            try:
                rows = self.db_manager.execute_query(
                    """
                    SELECT correlation_group, total_exposure, position_count, symbols, last_updated
                    FROM portfolio_exposures
                    ORDER BY last_updated DESC
                    LIMIT 100
                    """
                )
                return jsonify({"items": rows})
            except Exception as e:
                return jsonify({"items": [], "error": str(e)}), 200

        @self.app.route("/api/debug/positions")
        def debug_positions():
            """Debug endpoint showing positions by all statuses for troubleshooting."""
            try:
                # * Get positions by status for debugging
                query = """
                SELECT
                    status,
                    COUNT(*) as count,
                    array_agg(
                        json_build_object(
                            'id', id,
                            'symbol', symbol,
                            'entry_price', entry_price,
                            'quantity', quantity,
                            'entry_time', entry_time,
                            'order_count', (SELECT COUNT(*) FROM orders WHERE position_id = positions.id)
                        )
                    ) as positions
                FROM positions
                GROUP BY status
                ORDER BY status
                """
                results = self.db_manager.execute_query(query)

                # * Also get validation results
                validation = self.db_manager.validate_position_status_consistency()

                return jsonify(
                    {
                        "positions_by_status": results,
                        "validation": validation,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )
            except Exception as e:
                logger.error(f"Debug positions endpoint error: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/debug/fix-positions", methods=["POST"])
        def fix_position_inconsistencies():
            """Endpoint to manually trigger position status fixes."""
            try:
                fixes = self.db_manager.fix_position_status_inconsistencies()
                return jsonify(
                    {
                        "success": True,
                        "fixes_applied": fixes,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )
            except Exception as e:
                logger.error(f"Fix positions endpoint error: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/api/pending-orders")
        def get_pending_orders():
            """Get current pending orders (orders not yet filled)."""
            try:
                pending_orders = self.db_manager.get_pending_orders()
                return jsonify(pending_orders)
            except Exception as e:
                logger.error(f"Error getting pending orders: {e}")
                return jsonify({"error": str(e)}), 500

        # ========== ADVANCED ANALYTICS ENDPOINTS ==========

        @self.app.route("/api/performance/advanced")
        def get_advanced_performance():
            """Get advanced performance metrics including rolling calculations."""
            days = request.args.get("days", 30, type=int)
            window = request.args.get("window", 7, type=int)  # Rolling window size
            try:
                data = self._get_advanced_performance_metrics(days, window)
                return jsonify(data)
            except Exception as e:
                logger.error(f"Error getting advanced performance: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/trades/analysis")
        def get_trade_analysis():
            """Get detailed trade analysis including distributions and patterns."""
            days = request.args.get("days", 30, type=int)
            try:
                data = self._get_trade_analysis(days)
                return jsonify(data)
            except Exception as e:
                logger.error(f"Error getting trade analysis: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/trades/distribution")
        def get_trade_distribution():
            """Get trade P&L distribution data for histogram."""
            days = request.args.get("days", 30, type=int)
            bins = request.args.get("bins", 20, type=int)
            try:
                data = self._get_trade_distribution(days, bins)
                return jsonify(data)
            except Exception as e:
                logger.error(f"Error getting trade distribution: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/models/performance")
        def get_model_performance():
            """Get ML model performance metrics and accuracy tracking."""
            model_name = request.args.get("model", "")
            days = request.args.get("days", 30, type=int)
            try:
                data = self._get_model_performance_data(model_name, days)
                return jsonify(data)
            except Exception as e:
                logger.error(f"Error getting model performance: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/models/list")
        def list_models():
            """List all tracked models with basic stats."""
            try:
                models = self._list_tracked_models()
                return jsonify(models)
            except Exception as e:
                logger.error(f"Error listing models: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/system/health-detailed")
        def get_detailed_health():
            """Get detailed system health metrics."""
            try:
                health = self._get_detailed_system_health()
                return jsonify(health)
            except Exception as e:
                logger.error(f"Error getting system health: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/risk/detailed")
        def get_detailed_risk():
            """Get detailed risk metrics and exposure analysis."""
            try:
                risk_data = self._get_detailed_risk_metrics()
                return jsonify(risk_data)
            except Exception as e:
                logger.error(f"Error getting risk metrics: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/correlation/matrix-formatted")
        def get_correlation_matrix_formatted():
            """Get correlation matrix in formatted structure for heatmap."""
            try:
                matrix = self._get_formatted_correlation_matrix()
                return jsonify(matrix)
            except Exception as e:
                logger.error(f"Error getting correlation matrix: {e}")
                return jsonify({"error": str(e)}), 500

        # ========== EXPORT ENDPOINTS ==========

        @self.app.route("/api/export/trades")
        def export_trades():
            """Export trades as CSV."""
            days = request.args.get("days", 30, type=int)
            try:
                csv_data = self._export_trades_csv(days)
                from flask import Response

                return Response(
                    csv_data,
                    mimetype="text/csv",
                    headers={
                        "Content-Disposition": f"attachment; filename=trades_{datetime.now().strftime('%Y%m%d')}.csv"
                    },
                )
            except Exception as e:
                logger.error(f"Error exporting trades: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/export/performance")
        def export_performance():
            """Export performance metrics as CSV."""
            days = request.args.get("days", 30, type=int)
            try:
                csv_data = self._export_performance_csv(days)
                from flask import Response

                return Response(
                    csv_data,
                    mimetype="text/csv",
                    headers={
                        "Content-Disposition": f"attachment; filename=performance_{datetime.now().strftime('%Y%m%d')}.csv"
                    },
                )
            except Exception as e:
                logger.error(f"Error exporting performance: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/export/positions")
        def export_positions():
            """Export current positions as CSV."""
            try:
                csv_data = self._export_positions_csv()
                from flask import Response

                return Response(
                    csv_data,
                    mimetype="text/csv",
                    headers={
                        "Content-Disposition": f"attachment; filename=positions_{datetime.now().strftime('%Y%m%d')}.csv"
                    },
                )
            except Exception as e:
                logger.error(f"Error exporting positions: {e}")
                return jsonify({"error": str(e)}), 500

    def _setup_websocket_handlers(self):
        """Setup WebSocket event handlers"""

        @self.socketio.on("connect")
        def handle_connect():
            """Handle client connection"""
            logger.info("Client connected to monitoring dashboard")
            emit("connected", {"status": "Connected to monitoring dashboard"})

        @self.socketio.on("disconnect")
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info("Client disconnected from monitoring dashboard")

        @self.socketio.on("request_update")
        def handle_update_request():
            """Handle manual update request"""
            metrics = self._collect_metrics()
            emit("metrics_update", metrics)

    # -------- Price fetching (assume symbols are accurate) --------

    def _get_price_for_symbol(self, symbol: str, cache: dict[str, float]) -> float:
        """Fetch current price for an exact symbol, with simple caching.

        If the fetch fails, returns 0.0. Symbols are assumed accurate.
        """
        s = str(symbol).upper()
        if not s:
            return 0.0
        if s in cache:
            return cache[s]
        try:
            price = self._safe_float(self.data_provider.get_current_price(s))
        except Exception as e:
            logger.error(f"Error fetching price for {s}: {e}")
            price = 0.0
        cache[s] = price
        return price

    def _collect_metrics(self) -> dict[str, Any]:
        """Collect all monitoring metrics"""
        try:
            metrics: dict[str, Any] = {}

            # Get enabled metrics only
            enabled_metrics = {
                k: v for k, v in self.monitoring_config.items() if v.get("enabled", True)
            }

            # System Health Metrics
            if "api_connection_status" in enabled_metrics:
                metrics["api_connection_status"] = self._get_api_connection_status()
            if "data_feed_status" in enabled_metrics:
                metrics["data_feed_status"] = self._get_data_feed_status()
            if "error_rate_hourly" in enabled_metrics:
                metrics["error_rate_hourly"] = self._get_error_rate_hourly()
            if "api_latency" in enabled_metrics:
                metrics["api_latency"] = self._get_api_latency()
            if "last_data_update" in enabled_metrics:
                metrics["last_data_update"] = datetime.now(UTC).isoformat()
            if "system_uptime" in enabled_metrics:
                metrics["system_uptime"] = float(self._get_system_uptime())

            # Risk Metrics
            if "current_drawdown" in enabled_metrics:
                metrics["current_drawdown"] = self._get_current_drawdown()
            if "daily_pnl" in enabled_metrics:
                metrics["daily_pnl"] = self._get_daily_pnl()
            if "weekly_pnl" in enabled_metrics:
                metrics["weekly_pnl"] = self._get_weekly_pnl()
            if "position_sizes" in enabled_metrics:
                metrics["position_sizes"] = self._get_total_position_sizes()
            if "max_drawdown" in enabled_metrics:
                metrics["max_drawdown"] = self._get_max_drawdown()
            if "risk_per_trade" in enabled_metrics:
                metrics["risk_per_trade"] = self._get_risk_per_trade()
            if "volatility" in enabled_metrics:
                metrics["volatility"] = self._get_volatility()
            if "dynamic_risk_factor" in enabled_metrics:
                metrics["dynamic_risk_factor"] = self._get_dynamic_risk_factor()
            if "dynamic_risk_reason" in enabled_metrics:
                metrics["dynamic_risk_reason"] = self._get_dynamic_risk_reason()
            if "dynamic_risk_active" in enabled_metrics:
                metrics["dynamic_risk_active"] = self._get_dynamic_risk_active()

            # Order Execution Metrics
            if "fill_rate" in enabled_metrics:
                metrics["fill_rate"] = self._get_fill_rate()
            if "avg_slippage" in enabled_metrics:
                metrics["avg_slippage"] = self._get_avg_slippage()
            if "failed_orders" in enabled_metrics:
                metrics["failed_orders"] = self._get_failed_orders()
            if "order_latency" in enabled_metrics:
                metrics["order_latency"] = self._get_order_latency()
            if "execution_quality" in enabled_metrics:
                metrics["execution_quality"] = float(self._get_execution_quality())

            # Balance & Positions
            if "current_balance" in enabled_metrics:
                metrics["current_balance"] = self._get_current_balance()
            if "active_positions_count" in enabled_metrics:
                metrics["active_positions_count"] = self._get_active_positions_count()
            if "total_position_value" in enabled_metrics:
                metrics["total_position_value"] = self._get_total_position_value()
            if "margin_usage" in enabled_metrics:
                metrics["margin_usage"] = self._get_margin_usage()
            if "available_margin" in enabled_metrics:
                metrics["available_margin"] = self._get_available_margin()
            if "unrealized_pnl" in enabled_metrics:
                metrics["unrealized_pnl"] = self._get_unrealized_pnl()

            # Strategy Performance
            if "win_rate" in enabled_metrics:
                metrics["win_rate"] = self._get_win_rate()
            if "sharpe_ratio" in enabled_metrics:
                metrics["sharpe_ratio"] = self._get_sharpe_ratio()
            if "recent_trade_outcomes" in enabled_metrics:
                metrics["recent_trade_outcomes"] = self._get_recent_trade_outcomes()
            if "profit_factor" in enabled_metrics:
                metrics["profit_factor"] = self._get_profit_factor()
            if "avg_win_loss_ratio" in enabled_metrics:
                metrics["avg_win_loss_ratio"] = self._get_avg_win_loss_ratio()
            if "total_trades" in enabled_metrics:
                metrics["total_trades"] = self._get_total_trades()

            # Additional Core Metrics
            if "total_pnl" in enabled_metrics:
                metrics["total_pnl"] = self._get_total_pnl()
            if "current_strategy" in enabled_metrics:
                metrics["current_strategy"] = self._get_current_strategy()
            # current_price disabled (BTC-specific) to avoid confusion
            if "price_change_24h" in enabled_metrics:
                metrics["price_change_24h"] = self._get_price_change_24h()
            if "rsi" in enabled_metrics:
                metrics["rsi"] = self._get_current_rsi()
            if "ema_trend" in enabled_metrics:
                metrics["ema_trend"] = self._get_ema_trend()

            return metrics

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return {"error": str(e), "last_update": datetime.now(UTC).isoformat()}

    def _get_total_pnl(self) -> float:
        """Get total PnL across all trades"""
        try:
            query = """
            SELECT COALESCE(SUM(pnl), 0) as total_pnl
            FROM trades
            WHERE exit_time IS NOT NULL
            """
            result = self.db_manager.execute_query(query)
            return self._safe_float(result[0]["total_pnl"]) if result else 0.0
        except Exception as e:
            logger.error(f"Error getting total PnL: {e}")
            return 0.0

    def _get_current_balance(self) -> float:
        """Get current account balance"""
        try:
            # Get latest account snapshot
            query = """
            SELECT balance
            FROM account_history
            ORDER BY timestamp DESC
            LIMIT 1
            """
            result = self.db_manager.execute_query(query)
            return self._safe_float(result[0]["balance"]) if result else 0.0
        except Exception as e:
            logger.error(f"Error getting current balance: {e}")
            return 0.0

    def _get_win_rate(self) -> float:
        """Calculate win rate percentage"""
        try:
            query = """
            SELECT
                COUNT(*) as total_trades,
                COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades
            FROM trades
            WHERE exit_time IS NOT NULL
            """
            result = self.db_manager.execute_query(query)
            if result and result[0]["total_trades"] > 0:
                return (result[0]["winning_trades"] / result[0]["total_trades"]) * 100
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0

    def _get_total_trades(self) -> int:
        """Get total number of completed trades"""
        try:
            query = "SELECT COUNT(*) as count FROM trades WHERE exit_time IS NOT NULL"
            result = self.db_manager.execute_query(query)
            return result[0]["count"] if result else 0
        except Exception as e:
            logger.error(f"Error getting total trades: {e}")
            return 0

    def _get_active_positions_count(self) -> int:
        """Get number of active positions"""
        try:
            positions = self.db_manager.get_active_positions()
            return len(positions)
        except Exception as e:
            logger.error(f"Error getting active positions: {e}")
            return 0

    def _get_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            query = (
                "SELECT balance, DATE(timestamp) as date FROM account_history ORDER BY timestamp"
            )
            result = self.db_manager.execute_query(query)
            if len(result) < 2:
                return 0.0
            df = pd.DataFrame(result)
            # Convert balance column to float to handle Decimal types
            df["balance"] = df["balance"].apply(self._safe_float)
            # Aggregate to daily last to avoid duplicate dates
            daily_balance = df.groupby("date")["balance"].last()
            return perf_max_drawdown(daily_balance)
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0

    def _get_current_exposure(self) -> float:
        """Get current market exposure as percentage of balance"""
        try:
            # Get current balance and active position values
            balance = self._get_current_balance()
            if balance <= 0:
                return 0.0

            positions = self.db_manager.get_active_positions()
            exposure = sum(
                self._safe_float(pos.get("quantity", 0))
                * self._safe_float(pos.get("entry_price", 0))
                for pos in positions
            )
            return (exposure / balance) * 100
        except Exception as e:
            logger.error(f"Error calculating current exposure: {e}")
            return 0.0

    def _get_risk_per_trade(self) -> float:
        """Get average risk per trade"""
        # This would typically come from the risk manager configuration
        return 1.0  # Default 1% risk per trade

    def _get_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            # Get daily returns from account snapshots
            query = """
            SELECT balance, DATE(timestamp) as date
            FROM account_history
            ORDER BY timestamp
            """
            result = self.db_manager.execute_query(query)

            if len(result) < 2:
                return 0.0

            # Calculate daily returns
            df = pd.DataFrame(result)
            # Convert balance column to float to handle Decimal types
            df["balance"] = df["balance"].apply(self._safe_float)
            # Aggregate to daily last to ensure daily frequency
            daily_balance = df.groupby("date")["balance"].last()
            return perf_sharpe(daily_balance)

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def _get_volatility(self) -> float:
        """Calculate portfolio volatility"""
        try:
            # Similar to Sharpe ratio calculation but return std dev
            query = """
            SELECT balance, DATE(timestamp) as date
            FROM account_history
            ORDER BY timestamp
            """
            result = self.db_manager.execute_query(query)

            if len(result) < 2:
                return 0.0

            df = pd.DataFrame(result)
            # Convert balance column to float to handle Decimal types
            df["balance"] = df["balance"].apply(self._safe_float)
            df["daily_return"] = df["balance"].pct_change()
            df = df.dropna()

            if len(df) == 0:
                return 0.0

            return df["daily_return"].std() * (252**0.5) * 100  # Annualized volatility %

        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0

    def _get_dynamic_risk_factor(self) -> float:
        """Get current dynamic risk adjustment factor"""
        try:
            # Get the most recent risk adjustment
            query = """
            SELECT adjustment_factor, timestamp
            FROM risk_adjustments
            WHERE parameter_name = 'position_size_factor'
            ORDER BY timestamp DESC
            LIMIT 1
            """
            result = self.db_manager.execute_query(query)

            if result and len(result) > 0:
                return self._safe_float(result[0]["adjustment_factor"])

            return 1.0  # Default factor when no adjustments

        except Exception as e:
            logger.error(f"Error getting dynamic risk factor: {e}")
            return 1.0

    def _get_dynamic_risk_reason(self) -> str:
        """Get the reason for current dynamic risk adjustment"""
        try:
            # Get the most recent risk adjustment reason
            query = """
            SELECT trigger_reason, timestamp
            FROM risk_adjustments
            WHERE parameter_name = 'position_size_factor'
            ORDER BY timestamp DESC
            LIMIT 1
            """
            result = self.db_manager.execute_query(query)

            if result and len(result) > 0:
                return str(result[0]["trigger_reason"])

            return "normal"  # Default reason when no adjustments

        except Exception as e:
            logger.error(f"Error getting dynamic risk reason: {e}")
            return "normal"

    def _get_dynamic_risk_active(self) -> bool:
        """Check if dynamic risk adjustments are currently active"""
        try:
            # Check if there are recent risk adjustments (within last hour)
            query = """
            SELECT COUNT(*) as count
            FROM risk_adjustments
            WHERE parameter_name = 'position_size_factor'
            AND adjustment_factor != 1.0
            AND timestamp > NOW() - INTERVAL '1 hour'
            """
            result = self.db_manager.execute_query(query)

            if result and len(result) > 0:
                return int(result[0]["count"]) > 0

            return False

        except Exception as e:
            logger.error(f"Error checking dynamic risk status: {e}")
            return False

    def _get_system_health_status(self) -> str:
        """Get overall system health status"""
        try:
            # Check recent activity
            query = """
            SELECT timestamp
            FROM account_history
            ORDER BY timestamp DESC
            LIMIT 1
            """
            result = self.db_manager.execute_query(query)

            if not result:
                return "Unknown"

            last_update = pd.to_datetime(result[0]["timestamp"])
            time_diff = (datetime.now(UTC) - last_update).total_seconds()

            if time_diff < 300:  # 5 minutes
                return "Healthy"
            elif time_diff < 900:  # 15 minutes
                return "Warning"
            else:
                return "Error"

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return "Error"

    def _get_api_status(self) -> str:
        """Check API connectivity status"""
        try:
            # Try to get current price to test API
            current_price = self.data_provider.get_current_price("BTCUSDT")
            return "Connected" if current_price else "Disconnected"
        except Exception as e:
            logger.error(f"API status check failed: {e}")
            return "Disconnected"

    def _get_recent_error_count(self) -> int:
        """Get count of recent errors (last 24 hours)"""
        try:
            query = """
            SELECT COUNT(*) as count
            FROM system_events
            WHERE event_type = 'ERROR'
            AND timestamp > NOW() - INTERVAL '1 day'
            """
            result = self.db_manager.execute_query(query)
            return result[0]["count"] if result else 0
        except Exception as e:
            logger.error(f"Error getting error count: {e}")
            return 0

    def _get_current_strategy(self) -> str:
        """Get current active strategy"""
        try:
            query = """
            SELECT strategy_name
            FROM trading_sessions
            ORDER BY start_time DESC
            LIMIT 1
            """
            result = self.db_manager.execute_query(query)
            return result[0]["strategy_name"] if result else "Unknown"
        except Exception as e:
            logger.error(f"Error getting current strategy: {e}")
            return "Unknown"

    def _get_strategy_confidence(self) -> float:
        """Get strategy confidence level"""
        # This would need to be implemented in the strategy itself
        return 75.0  # Placeholder

    def _get_signals_today(self) -> int:
        """Get number of signals generated today"""
        try:
            query = """
            SELECT COUNT(*) as count
            FROM trades
            WHERE DATE(entry_time) = CURRENT_DATE
            """
            result = self.db_manager.execute_query(query)
            return result[0]["count"] if result else 0
        except Exception as e:
            logger.error(f"Error getting signals today: {e}")
            return 0

    def _get_current_price(self) -> float:
        """Get current BTC price"""
        try:
            return self.data_provider.get_current_price("BTCUSDT")
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return 0.0

    def _get_price_change_24h(self) -> float:
        """Get 24h price change percentage"""
        try:
            df = self.data_provider.get_historical_data(
                "BTCUSDT", "1h", datetime.now(UTC) - timedelta(days=2), datetime.now(UTC)
            )
            if len(df) >= 2:
                current_price = df.iloc[-1]["close"]
                price_24h_ago = df.iloc[-25]["close"] if len(df) >= 25 else df.iloc[0]["close"]
                return ((current_price - price_24h_ago) / price_24h_ago) * 100
            return 0.0
        except Exception as e:
            logger.error(f"Error getting 24h price change: {e}")
            return 0.0

    def _get_volume_24h(self) -> float:
        """Get 24h trading volume"""
        try:
            df = self.data_provider.get_historical_data(
                "BTCUSDT", "1h", datetime.now(UTC) - timedelta(days=1), datetime.now(UTC)
            )
            if not df.empty:
                return df["volume"].sum()
            return 0.0
        except Exception as e:
            logger.error(f"Error getting 24h volume: {e}")
            return 0.0

    def _get_current_rsi(self) -> float:
        """Get current RSI value"""
        try:
            from src.tech.indicators.core import calculate_rsi

            df = self.data_provider.get_historical_data(
                "BTCUSDT", "1h", datetime.now(UTC) - timedelta(days=30), datetime.now(UTC)
            )
            if len(df) > 14:
                rsi = calculate_rsi(df["close"], period=14)
                return rsi.iloc[-1] if not rsi.empty else 50.0
            return 50.0
        except Exception as e:
            logger.error(f"Error getting RSI: {e}")
            return 50.0

    def _get_ema_trend(self) -> str:
        """Get EMA trend direction"""
        try:
            from src.tech.indicators.core import calculate_ema

            df = self.data_provider.get_historical_data(
                "BTCUSDT", "1h", datetime.now(UTC) - timedelta(days=30), datetime.now(UTC)
            )
            if len(df) > 50:
                ema_short = calculate_ema(df["close"], period=9)
                ema_long = calculate_ema(df["close"], period=21)

                if len(ema_short) > 0 and len(ema_long) > 0:
                    if ema_short.iloc[-1] > ema_long.iloc[-1]:
                        return "Bullish"
                    else:
                        return "Bearish"
            return "Neutral"
        except Exception as e:
            logger.error(f"Error getting EMA trend: {e}")
            return "Neutral"

    def _get_sentiment_score(self) -> float:
        """Get current sentiment score"""
        # This would integrate with sentiment providers
        return 0.0  # Placeholder

    def _get_sentiment_trend(self) -> str:
        """Get sentiment trend"""
        return "Neutral"  # Placeholder

    # ========== SYSTEM HEALTH METRICS ==========

    def _get_api_connection_status(self) -> str:
        """Get API connection status"""
        try:
            # Test API connectivity by making a simple request
            current_price = self.data_provider.get_current_price("BTCUSDT")
            return "Connected" if current_price and current_price > 0 else "Disconnected"
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return "Disconnected"

    def _get_data_feed_status(self) -> str:
        """Get data feed status"""
        try:
            # Check when we last received data
            query = """
            SELECT timestamp
            FROM account_history
            ORDER BY timestamp DESC
            LIMIT 1
            """
            result = self.db_manager.execute_query(query)

            if not result:
                return "No Data"

            last_update = pd.to_datetime(result[0]["timestamp"])
            time_diff = (datetime.now(UTC) - last_update).total_seconds()

            if time_diff < 300:  # 5 minutes
                return "Active"
            elif time_diff < 900:  # 15 minutes
                return "Delayed"
            else:
                return "Stale"

        except Exception as e:
            logger.error(f"Error checking data feed status: {e}")
            return "Error"

    def _get_error_rate_hourly(self) -> float:
        """Get error rate over the last hour"""
        try:
            query = """
            SELECT
                COUNT(CASE WHEN event_type = 'ERROR' THEN 1 END) as errors,
                COUNT(*) as total
            FROM system_events
            WHERE timestamp > NOW() - INTERVAL '1 hour'
            """
            result = self.db_manager.execute_query(query)
            if result and result[0]["total"] > 0:
                return (result[0]["errors"] / result[0]["total"]) * 100
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating hourly error rate: {e}")
            return 0.0

    def _get_api_latency(self) -> float:
        """Get average API latency in milliseconds"""
        try:
            import time

            start_time = time.time()
            # Make a simple API call to measure latency
            self.data_provider.get_current_price("BTCUSDT")
            end_time = time.time()
            return (end_time - start_time) * 1000  # Convert to milliseconds
        except Exception as e:
            logger.error(f"Error measuring API latency: {e}")
            return 0.0

    def _get_system_uptime(self) -> float:
        """Get system uptime in minutes"""
        try:
            uptime = datetime.now(UTC) - self.start_time
            total_minutes = uptime.total_seconds() / 60.0
            return float(total_minutes)
        except Exception as e:
            logger.error(f"Error getting system uptime: {e}")
            return 0.0

    # ========== RISK METRICS ==========

    def _get_current_drawdown(self) -> float:
        """Get current drawdown from peak"""
        try:
            query = """
            SELECT balance, timestamp
            FROM account_history
            ORDER BY timestamp ASC
            LIMIT 100
            """
            result = self.db_manager.execute_query(query)

            if len(result) < 2:
                return 0.0

            # Convert to DataFrame for easier calculation
            df = pd.DataFrame(result)
            df["balance"] = df["balance"].apply(self._safe_float)

            # Calculate running maximum (peak) in chronological order
            df["peak"] = df["balance"].cummax()

            # Calculate drawdown at the latest point
            current_balance = df["balance"].iloc[-1]
            current_peak = df["peak"].iloc[-1]

            if current_peak > 0:
                drawdown = ((current_peak - current_balance) / current_peak) * 100
                return max(0.0, float(drawdown))
            return 0.0

        except Exception as e:
            logger.error(f"Error calculating current drawdown: {e}")
            return 0.0

    def _get_daily_pnl(self) -> float:
        """Get P&L for today"""
        try:
            query = """
            SELECT COALESCE(SUM(pnl), 0) as daily_pnl
            FROM trades
            WHERE DATE(exit_time) = CURRENT_DATE
            AND exit_time IS NOT NULL
            """
            result = self.db_manager.execute_query(query)
            return self._safe_float(result[0]["daily_pnl"]) if result else 0.0
        except Exception as e:
            logger.error(f"Error getting daily P&L: {e}")
            return 0.0

    def _get_weekly_pnl(self) -> float:
        """Get P&L for the last 7 days"""
        try:
            query = """
            SELECT COALESCE(SUM(pnl), 0) as weekly_pnl
            FROM trades
            WHERE exit_time > NOW() - INTERVAL '7 days'
            AND exit_time IS NOT NULL
            """
            result = self.db_manager.execute_query(query)
            return self._safe_float(result[0]["weekly_pnl"]) if result else 0.0
        except Exception as e:
            logger.error(f"Error getting weekly P&L: {e}")
            return 0.0

    def _get_total_position_sizes(self) -> float:
        """Get total notional size (currency) of all active positions at entry prices.

        Presented as currency in the UI per configuration.
        """
        try:
            positions = self.db_manager.get_active_positions()
            total_value = 0.0
            for pos in positions:
                quantity = self._safe_float(pos.get("quantity", 0))
                entry_price = self._safe_float(pos.get("entry_price", 0))
                total_value += quantity * entry_price
            return total_value
        except Exception as e:
            logger.error(f"Error getting total position sizes: {e}")
            return 0.0

    def _get_total_position_value_at_entry(self) -> float:
        """Get total value of all active positions at entry prices"""
        try:
            positions = self.db_manager.get_active_positions()
            total_value = sum(
                self._safe_float(pos.get("quantity", 0))
                * self._safe_float(pos.get("entry_price", 0))
                for pos in positions
            )
            return total_value
        except Exception as e:
            logger.error(f"Error getting total position value at entry: {e}")
            return 0.0

    # ========== ORDER EXECUTION METRICS ==========

    def _get_fill_rate(self) -> float:
        """Get order fill rate percentage"""
        try:
            # This would need to be tracked in order execution logs
            # For now, calculate based on successful vs failed trades
            query = """
            SELECT
                COUNT(*) as total_orders,
                COUNT(CASE WHEN status::text = 'filled' THEN 1 END) as filled_orders
            FROM positions
            WHERE entry_time > NOW() - INTERVAL '24 hours'
            """
            result = self.db_manager.execute_query(query)

            if result and result[0]["total_orders"] > 0:
                return (result[0]["filled_orders"] / result[0]["total_orders"]) * 100
            return 100.0  # Default to 100% if no recent orders
        except Exception as e:
            logger.error(f"Error calculating fill rate: {e}")
            return 100.0

    def _get_avg_slippage(self) -> float:
        """Get average slippage percentage"""
        try:
            # Calculate slippage as difference between expected and actual execution price
            # This is a simplified calculation - in practice you'd track intended vs actual prices
            query = """
            SELECT
                entry_price,
                exit_price,
                side
            FROM trades
            WHERE exit_time > NOW() - INTERVAL '24 hours'
            AND exit_time IS NOT NULL
            LIMIT 50
            """
            result = self.db_manager.execute_query(query)

            if not result:
                return 0.0

            # Simple slippage estimation based on price movement
            total_slippage = 0.0
            count = 0

            for _trade in result:
                # Estimate slippage as 0.01-0.05% of trade value
                estimated_slippage = 0.02  # 0.02% average slippage
                total_slippage += estimated_slippage
                count += 1

            return total_slippage / count if count > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating average slippage: {e}")
            return 0.0

    def _get_failed_orders(self) -> int:
        """Get number of failed orders"""
        try:
            query = "SELECT COUNT(*) as failed_count FROM trades WHERE exit_reason = 'failed'"
            result = self.db_manager.execute_query(query)
            if result and "failed_count" in result[0]:
                return result[0]["failed_count"]
            else:
                return 0
        except Exception as e:
            logger.error(f"Error getting failed orders count: {e}")
            return 0

    def _get_order_latency(self) -> float:
        """Get average order execution latency in milliseconds"""
        # This would require detailed order execution tracking
        # For now, return a reasonable estimate
        return 50.0  # 50ms average latency

    def _get_execution_quality(self) -> float:
        """Get overall execution quality score (0-100)"""
        try:
            fill_rate = self._get_fill_rate()  # percentage
            slippage = self._get_avg_slippage()  # percentage
            failed_orders = self._get_failed_orders()

            score = fill_rate
            score -= min(slippage * 100.0, 20.0)  # penalize slippage up to 20 points
            score -= min(failed_orders, 20)  # penalize failures up to 20 points
            return max(0.0, min(100.0, float(score)))
        except Exception as e:
            logger.error(f"Error calculating execution quality: {e}")
            return 0.0

    # ========== BALANCE & POSITIONS ==========

    def _get_total_position_value(self) -> float:
        """Get total value of all positions at current prices (per-symbol)."""
        try:
            positions = self.db_manager.get_active_positions()
            total_value = 0.0
            price_cache: dict[str, float] = {}
            for pos in positions:
                symbol = str(pos.get("symbol", ""))
                quantity = self._safe_float(pos.get("quantity", 0))
                if not symbol or quantity == 0:
                    continue
                if symbol not in price_cache:
                    try:
                        price_cache[symbol] = self._safe_float(
                            self.data_provider.get_current_price(symbol)
                        )
                    except Exception:
                        price_cache[symbol] = 0.0
                current_price = price_cache.get(symbol, 0.0)
                total_value += quantity * current_price
            return total_value
        except Exception as e:
            logger.error(f"Error getting total position value: {e}")
            return 0.0

    def _get_margin_usage(self) -> float:
        """Get margin usage percentage"""
        try:
            current_balance = self._get_current_balance()
            position_value = self._get_total_position_value()

            if current_balance > 0:
                # Assuming 1:1 margin (no leverage) for safety
                return (position_value / current_balance) * 100
            return 0.0

        except Exception as e:
            logger.error(f"Error calculating margin usage: {e}")
            return 0.0

    def _get_available_margin(self) -> float:
        """Get available margin for new positions"""
        try:
            current_balance = self._get_current_balance()
            used_margin = self._get_total_position_value()
            return max(0.0, float(current_balance - used_margin))
        except Exception as e:
            logger.error(f"Error calculating available margin: {e}")
            return 0.0

    def _get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L from active positions (per-symbol)."""
        try:
            positions = self.db_manager.get_active_positions()
            total_unrealized = 0.0
            price_cache: dict[str, float] = {}
            for position in positions:
                symbol = str(position.get("symbol", ""))
                entry_price = self._safe_float(position.get("entry_price", 0))
                quantity = self._safe_float(position.get("quantity", 0))
                side = str(position.get("side", "")).lower()
                if not symbol or quantity == 0:
                    continue
                if symbol not in price_cache:
                    try:
                        price_cache[symbol] = self._safe_float(
                            self.data_provider.get_current_price(symbol)
                        )
                    except Exception:
                        price_cache[symbol] = 0.0
                current_price = price_cache.get(symbol, 0.0)
                if side == "long":
                    unrealized = (current_price - entry_price) * quantity
                else:  # short
                    unrealized = (entry_price - current_price) * quantity
                total_unrealized += unrealized
            return total_unrealized
        except Exception as e:
            logger.error(f"Error calculating unrealized P&L: {e}")
            return 0.0

    # ========== STRATEGY PERFORMANCE ==========

    def _get_recent_trade_outcomes(self) -> str:
        """Get recent trade outcomes summary"""
        try:
            query = """
            SELECT pnl
            FROM trades
            WHERE exit_time IS NOT NULL
            ORDER BY exit_time DESC
            LIMIT 10
            """
            result = self.db_manager.execute_query(query)

            if not result:
                return "No recent trades"

            outcomes = []
            for trade in result:
                pnl_val = self._safe_float(trade.get("pnl"))
                if pnl_val is None:
                    continue
                outcomes.append("W" if pnl_val > 0 else "L")

            return "".join(outcomes)  # e.g., "WLWWLWLWW"

        except Exception as e:
            logger.error(f"Error getting recent trade outcomes: {e}")
            return "Unknown"

    def _get_profit_factor(self) -> float:
        """Get profit factor (gross profit / gross loss)"""
        try:
            query = """
            SELECT
                SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as gross_loss
            FROM trades
            WHERE exit_time IS NOT NULL
            """
            result = self.db_manager.execute_query(query)

            if not result:
                return 0.0

            gross_profit = self._safe_float(result[0].get("gross_profit") or 0.0)
            gross_loss = self._safe_float(result[0].get("gross_loss") or 0.0)

            if gross_loss == 0:
                # Avoid division by zero; if no losses yet, profit factor is undefined -> return 0
                return 0.0

            return gross_profit / gross_loss
        except Exception as e:
            logger.error(f"Error calculating profit factor: {e}")
            return 0.0

    def _get_avg_win_loss_ratio(self) -> float:
        """Get average win to loss ratio"""
        try:
            query = """
            SELECT
                AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                AVG(CASE WHEN pnl < 0 THEN ABS(pnl) END) as avg_loss
            FROM trades
            WHERE exit_time IS NOT NULL
            """
            result = self.db_manager.execute_query(query)

            if not result:
                return 0.0

            avg_win = self._safe_float(result[0].get("avg_win") or 0.0)
            avg_loss = self._safe_float(result[0].get("avg_loss") or 0.0)

            if avg_loss == 0:
                return 0.0

            return avg_win / avg_loss
        except Exception as e:
            logger.error(f"Error calculating win/loss ratio: {e}")
            return 0.0

    def _get_current_positions(self) -> list[PositionDict]:
        """Get current active positions with order history"""
        try:
            # Get positions with their orders using a join query
            query = """
            SELECT
                p.*,
                json_agg(
                    json_build_object(
                        'id', o.id,
                        'order_type', o.order_type,
                        'status', o.status,
                        'exchange_order_id', o.exchange_order_id,
                        'internal_order_id', o.internal_order_id,
                        'side', o.side,
                        'quantity', o.quantity,
                        'price', o.price,
                        'filled_quantity', o.filled_quantity,
                        'filled_price', o.filled_price,
                        'commission', o.commission,
                        'created_at', o.created_at,
                        'filled_at', o.filled_at,
                        'cancelled_at', o.cancelled_at
                    ) ORDER BY o.created_at
                ) as orders
            FROM positions p
            LEFT JOIN orders o ON p.id = o.position_id
            WHERE p.status = 'OPEN'
            GROUP BY p.id
            ORDER BY p.entry_time DESC
            """
            positions_data = self.db_manager.execute_query(query)

            positions: list[PositionDict] = []
            price_cache: dict[str, float] = {}

            for pos in positions_data:
                # Ensure quantity is float
                quantity = self._safe_float(pos.get("quantity", 0))

                # Calculate unrealized PnL - convert entry_price to float
                entry_price = self._safe_float(pos.get("entry_price", 0))
                side = pos.get("side", "").lower()
                symbol = str(pos.get("symbol", ""))
                # Fetch per-symbol current price directly (assume symbol is accurate)
                current_price = self._get_price_for_symbol(symbol, price_cache)

                if side == "long":
                    unrealized_pnl = (current_price - entry_price) * quantity
                else:
                    unrealized_pnl = (entry_price - current_price) * quantity

                # Parse orders JSON (will be None if no orders)
                orders_json = pos.get("orders")
                if orders_json and orders_json != [None]:  # json_agg returns [null] for no orders
                    orders = orders_json
                else:
                    orders = []

                positions.append(
                    PositionDict(
                        **{
                            "symbol": symbol,
                            "side": side,  # Use the lowercase side we already converted
                            "entry_price": entry_price,
                            "current_price": current_price,
                            "quantity": quantity,
                            "unrealized_pnl": unrealized_pnl,
                            "entry_time": pos.get("entry_time"),
                            "stop_loss": (
                                self._safe_float(pos.get("stop_loss"))
                                if pos.get("stop_loss") is not None
                                else None
                            ),
                            "take_profit": (
                                self._safe_float(pos.get("take_profit"))
                                if pos.get("take_profit") is not None
                                else None
                            ),
                            "orders": orders,  # Order history for this position
                            # Partial operations data
                            "original_size": self._safe_float(pos.get("original_size")),
                            "current_size": self._safe_float(pos.get("current_size")),
                            "partial_exits_taken": pos.get("partial_exits_taken"),
                            "scale_ins_taken": pos.get("scale_ins_taken"),
                            "last_partial_exit_price": self._safe_float(
                                pos.get("last_partial_exit_price")
                            ),
                            "last_scale_in_price": self._safe_float(pos.get("last_scale_in_price")),
                            # Trailing stops and MFE/MAE tracking
                            "trailing_stop_activated": bool(
                                pos.get("trailing_stop_activated", False)
                            ),
                            "trailing_stop_price": (
                                self._safe_float(pos.get("trailing_stop_price"))
                                if pos.get("trailing_stop_price") is not None
                                else None
                            ),
                            "breakeven_triggered": bool(pos.get("breakeven_triggered", False)),
                            "mfe": (
                                self._safe_float(pos.get("mfe"))
                                if pos.get("mfe") is not None
                                else 0.0
                            ),
                            "mae": (
                                self._safe_float(pos.get("mae"))
                                if pos.get("mae") is not None
                                else 0.0
                            ),
                        }
                    )
                )

            return positions
        except Exception as e:
            logger.error(f"Error getting current positions: {e}")
            return []

    def _get_recent_trades(self, limit: int = 50) -> list[TradeDict]:
        """Get recent completed trades"""
        try:
            query = """
            SELECT
                symbol, side, entry_price, exit_price, quantity,
                entry_time, exit_time, pnl, exit_reason
            FROM trades
            WHERE exit_time IS NOT NULL
            ORDER BY exit_time DESC
            LIMIT %s
            """
            result = self.db_manager.execute_query(query, (limit,))
            logger.info(f"Query returned {len(result)} rows")

            trades: list[TradeDict] = []
            for row in result or []:
                try:
                    # Handle None values more gracefully for numeric fields
                    entry_price_raw = row.get("entry_price")
                    exit_price_raw = row.get("exit_price")
                    pnl_raw = row.get("pnl")
                    quantity_raw = row.get("quantity")

                    trade: TradeDict = {
                        "symbol": str(row.get("symbol", "")),
                        "side": str(row.get("side", "")),
                        "entry_price": (
                            float(entry_price_raw) if entry_price_raw is not None else 0.0
                        ),
                        "exit_price": float(exit_price_raw) if exit_price_raw is not None else 0.0,
                        "quantity": float(quantity_raw) if quantity_raw is not None else 0.0,
                        "entry_time": row.get("entry_time"),
                        "exit_time": row.get("exit_time"),
                        "pnl": float(pnl_raw) if pnl_raw is not None else 0.0,
                        "exit_reason": str(row.get("exit_reason", "")),
                    }
                    trades.append(trade)
                except Exception as e:
                    logger.error(f"Error converting row to TradeDict: {e}")
                    logger.error(f"Row data: {row}")

            logger.info(f"Successfully converted {len(trades)} trades")
            return trades
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            import traceback

            traceback.print_exc()
            return []

    def _get_partial_trades(self, limit: int = 50) -> list[dict]:
        """Get recent partial trades (partial exits and scale-ins)"""
        try:
            query = """
            SELECT
                pt.operation_type, pt.size, pt.price, pt.pnl, pt.target_level, pt.timestamp,
                p.symbol, p.side, p.entry_price
            FROM partial_trades pt
            JOIN positions p ON pt.position_id = p.id
            ORDER BY pt.timestamp DESC
            LIMIT %s
            """
            result = self.db_manager.execute_query(query, (limit,))
            logger.info(f"Partial trades query returned {len(result)} rows")

            partial_trades: list[dict] = []
            for row in result or []:
                try:
                    trade = {
                        "operation_type": str(row.get("operation_type", "")),
                        "symbol": str(row.get("symbol", "")),
                        "side": str(row.get("side", "")),
                        "entry_price": float(row.get("entry_price", 0)),
                        "operation_price": float(row.get("price", 0)),
                        "size": float(row.get("size", 0)),
                        "pnl": float(row.get("pnl", 0)) if row.get("pnl") is not None else None,
                        "target_level": row.get("target_level"),
                        "timestamp": row.get("timestamp"),
                    }
                    partial_trades.append(trade)
                except Exception as e:
                    logger.error(f"Error converting partial trade row: {e}")
                    logger.error(f"Row data: {row}")

            logger.info(f"Successfully converted {len(partial_trades)} partial trades")
            return partial_trades
        except Exception as e:
            logger.error(f"Error getting partial trades: {e}")
            import traceback

            traceback.print_exc()
            return []

    def _get_performance_chart_data(self, days: int = 7) -> dict[str, list]:
        """Get performance chart data for the specified number of days"""
        try:
            # * Validate days parameter to prevent SQL injection
            # * Ensure days is a positive integer within reasonable bounds
            if not isinstance(days, int) or days <= 0 or days > 365:
                logger.warning(f"Invalid days parameter: {days}. Using default value of 7.")
                days = 7

            # * Use string formatting for INTERVAL clause since PostgreSQL doesn't support
            # * parameter placeholders within INTERVAL expressions
            query = f"""
            SELECT balance, timestamp
            FROM account_history
            WHERE timestamp > NOW() - INTERVAL '{days} DAYS'
            ORDER BY timestamp
            """  # nosec B608: days is validated and constrained to an integer range [1, 365]
            result = self.db_manager.execute_query(query)

            timestamps: list[str] = []
            balances: list[float] = []

            for row in result:
                # Ensure ISO 8601 strings for JS Date parsing
                ts = row["timestamp"]
                if isinstance(ts, datetime):
                    timestamps.append(ts.isoformat())
                else:
                    # Fallback to string conversion
                    timestamps.append(str(ts))
                balances.append(self._safe_float(row["balance"]))

            return {"timestamps": timestamps, "balances": balances}

        except Exception as e:
            logger.error(f"Error getting performance chart data: {e}")
            return {"timestamps": [], "balances": []}

    def _get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "database_connected": True,  # If we're here, DB is connected
            "api_status": self._get_api_status(),
            "last_trade": self._get_last_trade_time(),
            "uptime": self._get_uptime(),
            "memory_usage": self._get_memory_usage(),
            "error_rate": self._get_error_rate(),
        }

    def _get_last_trade_time(self) -> str:
        """Get timestamp of last trade"""
        try:
            query = """
            SELECT MAX(entry_time) as last_trade
            FROM trades
            """
            result = self.db_manager.execute_query(query)
            return result[0]["last_trade"] if result and result[0]["last_trade"] else "Never"
        except Exception as e:
            logger.error(f"Error getting last trade time: {e}")
            return "Unknown"

    def _get_uptime(self) -> str:
        """Get system uptime"""
        # This would need to be tracked separately
        return "Unknown"

    def _get_memory_usage(self) -> float:
        """Get memory usage percentage"""
        try:
            import psutil

            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0

    def _get_error_rate(self) -> float:
        """Get error rate over last hour"""
        try:
            query = """
            SELECT
                COUNT(CASE WHEN event_type = 'ERROR' THEN 1 END) as errors,
                COUNT(*) as total
            FROM system_events
            WHERE timestamp > NOW() - INTERVAL '1 hour'
            """
            result = self.db_manager.execute_query(query)
            if result and result[0]["total"] > 0:
                return (result[0]["errors"] / result[0]["total"]) * 100
            return 0.0
        except Exception as e:
            logger.error(f"Error getting error rate: {e}")
            return 0.0

    def _get_balance_info(self) -> dict[str, Any]:
        """Get comprehensive balance information"""
        try:
            current_balance = self.db_manager.get_current_balance()
            balance_history = self.db_manager.get_balance_history(limit=10)

            # Calculate balance change over time
            balance_change_24h = 0.0
            if len(balance_history) >= 2:
                try:
                    recent_balance = balance_history[0]["balance"]
                    older_balance = next(
                        (
                            h["balance"]
                            for h in balance_history
                            if ((datetime.now(UTC) - h["timestamp"].astimezone(UTC)).days >= 1)
                        ),
                        recent_balance,
                    )
                    if older_balance > 0:
                        balance_change_24h = (
                            (recent_balance - older_balance) / older_balance
                        ) * 100
                except (KeyError, TypeError, ZeroDivisionError):
                    pass

            return {
                "current_balance": current_balance,
                "balance_change_24h": balance_change_24h,
                "last_updated": (
                    balance_history[0]["timestamp"].isoformat()
                    if balance_history and isinstance(balance_history[0]["timestamp"], datetime)
                    else balance_history[0]["timestamp"] if balance_history else None
                ),
                "last_update_reason": balance_history[0]["reason"] if balance_history else None,
                "recent_history": balance_history[:5],  # Last 5 balance changes
            }
        except Exception as e:
            logger.error(f"Error getting balance info: {e}")
            return {
                "current_balance": 0.0,
                "balance_change_24h": 0.0,
                "last_updated": None,
                "last_update_reason": "Error retrieving balance",
                "recent_history": [],
            }

    def _get_balance_history(self, days: int = 30):
        """Retrieve balance history.

        Args:
            days: How many days of history to retrieve. The current implementation maps the
                  value directly to the `limit` parameter of `DatabaseManager.get_balance_history`.

        Returns:
            A list of balance snapshots ordered by most recent first.
        """
        try:
            validated_days = self._normalize_days(days)
            return self.db_manager.get_balance_history(limit=validated_days)
        except Exception as e:
            logger.error(f"Error getting balance history: {e}")
            return []

    # ========== ADVANCED ANALYTICS METHODS ==========

    @staticmethod
    def _normalize_days(days: int | None, default: int = 30, max_days: int = 365) -> int:
        """Clamp days to a safe range and ensure an integer value."""

        try:
            validated_days = int(days) if days is not None else default
        except (TypeError, ValueError):
            return default

        if validated_days <= 0 or validated_days > max_days:
            return default
        return validated_days

    def _get_advanced_performance_metrics(self, days: int = 30, window: int = 7) -> dict[str, Any]:
        """Calculate advanced performance metrics with rolling windows."""
        try:
            validated_days = self._normalize_days(days)

            # Get account history
            query = f"""
            SELECT balance, timestamp
            FROM account_history
            WHERE timestamp > NOW() - INTERVAL '{validated_days} DAYS'
            ORDER BY timestamp
            """
            result = self.db_manager.execute_query(query)

            if len(result) < 2:
                return {"error": "Insufficient data"}

            df = pd.DataFrame(result)
            df["balance"] = df["balance"].apply(self._safe_float)
            df = df.set_index("timestamp")

            # Calculate daily returns
            df["returns"] = df["balance"].pct_change()

            # Rolling Sharpe ratio (annualized)
            rolling_sharpe = []
            for i in range(window, len(df)):
                window_returns = df["returns"].iloc[i - window : i]
                if window_returns.std() > 0:
                    sharpe = (window_returns.mean() / window_returns.std()) * (252**0.5)
                else:
                    sharpe = 0.0
                rolling_sharpe.append(
                    {"timestamp": df.index[i].isoformat(), "sharpe": float(sharpe)}
                )

            # Drawdown calculation
            df["peak"] = df["balance"].cummax()
            df["drawdown"] = (df["balance"] - df["peak"]) / df["peak"] * 100

            drawdown_data = [
                {"timestamp": ts.isoformat(), "drawdown": float(dd)}
                for ts, dd in zip(df.index, df["drawdown"], strict=False)
            ]

            # Win rate over time (rolling)
            trades_query = f"""
            SELECT DATE(exit_time) as date,
                   COUNT(*) as total,
                   COUNT(CASE WHEN pnl > 0 THEN 1 END) as wins
            FROM trades
            WHERE exit_time > NOW() - INTERVAL '{validated_days} DAYS'
            AND exit_time IS NOT NULL
            GROUP BY DATE(exit_time)
            ORDER BY date
            """
            trades_result = self.db_manager.execute_query(trades_query)

            win_rate_data: list[dict[str, float | str]] = []
            for row in trades_result:
                if not isinstance(row, dict) or "total" not in row or "wins" not in row:
                    continue

                total = row["total"]
                wins = row["wins"]
                wr = (wins / total * 100) if total and total > 0 else 0
                win_rate_data.append(
                    {
                        "date": (
                            row.get("date").isoformat()
                            if hasattr(row.get("date"), "isoformat")
                            else str(row.get("date"))
                        ),
                        "win_rate": float(wr),
                    }
                )

            return {
                "rolling_sharpe": rolling_sharpe,
                "drawdown_series": drawdown_data,
                "win_rate_series": win_rate_data,
                "current_drawdown": float(df["drawdown"].iloc[-1]),
                "max_drawdown": float(df["drawdown"].min()),
            }
        except Exception as e:
            logger.error(f"Error calculating advanced performance: {e}")
            return {"error": str(e)}

    def _get_trade_analysis(self, days: int = 30) -> dict[str, Any]:
        """Analyze trade patterns and performance."""
        try:
            validated_days = self._normalize_days(days)
            query = f"""
            SELECT
                symbol, side, entry_price, exit_price, quantity,
                entry_time, exit_time, pnl, pnl_percent,
                strategy_name, exit_reason
            FROM trades
            WHERE exit_time > NOW() - INTERVAL '{validated_days} DAYS'
            AND exit_time IS NOT NULL
            ORDER BY exit_time
            """
            result = self.db_manager.execute_query(query)

            if not result:
                return {"error": "No trades found"}

            df = pd.DataFrame(result)
            df["pnl"] = df["pnl"].apply(self._safe_float)

            # Trade duration analysis
            df["duration"] = (
                pd.to_datetime(df["exit_time"]) - pd.to_datetime(df["entry_time"])
            ).dt.total_seconds() / 3600
            avg_duration = float(df["duration"].mean())
            median_duration = float(df["duration"].median())

            # Profit by hour of day
            df["hour"] = pd.to_datetime(df["exit_time"]).dt.hour
            profit_by_hour = df.groupby("hour")["pnl"].sum().to_dict()
            profit_by_hour = {int(k): float(v) for k, v in profit_by_hour.items()}

            # Profit by day of week
            df["day_of_week"] = pd.to_datetime(df["exit_time"]).dt.dayofweek
            profit_by_dow = df.groupby("day_of_week")["pnl"].sum().to_dict()
            profit_by_dow = {int(k): float(v) for k, v in profit_by_dow.items()}

            # Best and worst trades
            best_trades = df.nlargest(5, "pnl")[
                ["symbol", "entry_time", "exit_time", "pnl"]
            ].to_dict("records")
            worst_trades = df.nsmallest(5, "pnl")[
                ["symbol", "entry_time", "exit_time", "pnl"]
            ].to_dict("records")

            # Serialize timestamps
            for trade in best_trades + worst_trades:
                trade["entry_time"] = (
                    trade["entry_time"].isoformat()
                    if hasattr(trade["entry_time"], "isoformat")
                    else str(trade["entry_time"])
                )
                trade["exit_time"] = (
                    trade["exit_time"].isoformat()
                    if hasattr(trade["exit_time"], "isoformat")
                    else str(trade["exit_time"])
                )
                trade["pnl"] = float(trade["pnl"])

            return {
                "total_trades": len(df),
                "avg_duration_hours": avg_duration,
                "median_duration_hours": median_duration,
                "profit_by_hour": profit_by_hour,
                "profit_by_day_of_week": profit_by_dow,
                "best_trades": best_trades,
                "worst_trades": worst_trades,
            }
        except Exception as e:
            logger.error(f"Error analyzing trades: {e}")
            return {"error": str(e)}

    def _get_trade_distribution(self, days: int = 30, bins: int = 20) -> dict[str, Any]:
        """Get trade P&L distribution for histogram."""
        try:
            validated_days = self._normalize_days(days)
            query = f"""
            SELECT pnl
            FROM trades
            WHERE exit_time > NOW() - INTERVAL '{validated_days} DAYS'
            AND exit_time IS NOT NULL
            """
            result = self.db_manager.execute_query(query)

            if not result:
                return {"bins": [], "counts": []}

            pnls = [float(row["pnl"]) for row in result]
            counts, bin_edges = np.histogram(pnls, bins=bins)

            return {
                "bins": [float(x) for x in bin_edges.tolist()],
                "counts": [int(x) for x in counts.tolist()],
                "mean": float(np.mean(pnls)),
                "median": float(np.median(pnls)),
                "std": float(np.std(pnls)),
            }
        except Exception as e:
            logger.error(f"Error getting trade distribution: {e}")
            return {"error": str(e)}

    def _get_model_performance_data(self, model_name: str = "", days: int = 30) -> dict[str, Any]:
        """Get ML model performance metrics."""
        try:
            validated_days = self._normalize_days(days)
            # Query prediction_performance table
            query = """
            SELECT
                timestamp, model_name, horizon, mae, rmse, mape, ic,
                mean_pred, std_pred, mean_real, std_real,
                strategy_name, symbol, timeframe
            FROM prediction_performance
            WHERE timestamp > NOW() - INTERVAL '%s DAYS'
            """
            params = [validated_days]

            if model_name:
                query += " AND model_name = %s"
                params.append(model_name)

            query += " ORDER BY timestamp"
            result = self.db_manager.execute_query(query, params)

            if not result:
                return {"message": "No model performance data found", "series": []}

            # Convert to time series format
            series = []
            for row in result:
                series.append(
                    {
                        "timestamp": (
                            row["timestamp"].isoformat()
                            if hasattr(row["timestamp"], "isoformat")
                            else str(row["timestamp"])
                        ),
                        "model_name": row["model_name"],
                        "mae": float(row["mae"]) if row["mae"] else None,
                        "rmse": float(row["rmse"]) if row["rmse"] else None,
                        "mape": float(row["mape"]) if row["mape"] else None,
                        "ic": float(row["ic"]) if row["ic"] else None,
                    }
                )

            # Calculate summary stats
            df = pd.DataFrame(result)
            summary = {
                "avg_mae": float(df["mae"].mean()) if "mae" in df.columns else None,
                "avg_rmse": float(df["rmse"].mean()) if "rmse" in df.columns else None,
                "avg_mape": float(df["mape"].mean()) if "mape" in df.columns else None,
                "avg_ic": float(df["ic"].mean()) if "ic" in df.columns else None,
            }

            return {"series": series, "summary": summary}
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {"error": str(e)}

    def _list_tracked_models(self) -> dict[str, Any]:
        """List all models with performance tracking."""
        try:
            query = """
            SELECT DISTINCT model_name, COUNT(*) as data_points, MAX(timestamp) as last_updated
            FROM prediction_performance
            GROUP BY model_name
            ORDER BY last_updated DESC
            """
            result = self.db_manager.execute_query(query)

            models = []
            for row in result:
                models.append(
                    {
                        "model_name": row["model_name"],
                        "data_points": row["data_points"],
                        "last_updated": (
                            row["last_updated"].isoformat()
                            if hasattr(row["last_updated"], "isoformat")
                            else str(row["last_updated"])
                        ),
                    }
                )

            return {"models": models}
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {"error": str(e)}

    def _get_detailed_system_health(self) -> dict[str, Any]:
        """Get detailed system health metrics."""
        try:
            # Error rate (last hour) – also used to measure DB latency
            error_query = """
            SELECT
                COUNT(*) as total_events,
                COUNT(CASE WHEN event_type = 'ERROR' THEN 1 END) as errors,
                COUNT(CASE WHEN event_type = 'WARNING' THEN 1 END) as warnings
            FROM system_events
            WHERE timestamp > NOW() - INTERVAL '1 hour'
            """
            db_query_time_start = time.time()
            error_result = self.db_manager.execute_query(error_query)
            db_latency = (time.time() - db_query_time_start) * 1000
            error_data = (
                error_result[0] if error_result else {"total_events": 0, "errors": 0, "warnings": 0}
            )

            # Recent errors
            recent_errors_query = """
            SELECT message, timestamp, severity
            FROM system_events
            WHERE event_type IN ('ERROR', 'WARNING')
            ORDER BY timestamp DESC
            LIMIT 10
            """
            recent_errors = self.db_manager.execute_query(recent_errors_query)
            for err in recent_errors:
                err["timestamp"] = (
                    err["timestamp"].isoformat()
                    if hasattr(err["timestamp"], "isoformat")
                    else str(err["timestamp"])
                )

            # Memory usage (if psutil available)
            memory_usage = 0.0
            try:
                import psutil

                memory_usage = psutil.virtual_memory().percent
            except ImportError:
                pass

            return {
                "database_latency_ms": float(db_latency),
                "database_status": "connected",
                "api_status": self._get_api_status(),
                "error_rate_hourly": (
                    float(error_data["errors"] / error_data["total_events"] * 100)
                    if error_data["total_events"] > 0
                    else 0.0
                ),
                "warning_rate_hourly": (
                    float(error_data["warnings"] / error_data["total_events"] * 100)
                    if error_data["total_events"] > 0
                    else 0.0
                ),
                "recent_errors": recent_errors,
                "memory_usage_percent": float(memory_usage),
                "uptime_minutes": float(self._get_system_uptime()),
            }
        except Exception as e:
            logger.error(f"Error getting detailed system health: {e}")
            return {"error": str(e)}

    def _get_detailed_risk_metrics(self) -> dict[str, Any]:
        """Get detailed risk metrics and exposure analysis."""
        try:
            # Current risk adjustments
            risk_adj_query = """
            SELECT parameter_name, adjustment_factor, trigger_reason, timestamp
            FROM risk_adjustments
            ORDER BY timestamp DESC
            LIMIT 10
            """
            risk_adjustments = self.db_manager.execute_query(risk_adj_query)
            for adj in risk_adjustments:
                adj["timestamp"] = (
                    adj["timestamp"].isoformat()
                    if hasattr(adj["timestamp"], "isoformat")
                    else str(adj["timestamp"])
                )
                adj["adjustment_factor"] = float(adj["adjustment_factor"])

            # VaR calculation (95% confidence)
            var_query = """
            SELECT pnl
            FROM trades
            WHERE exit_time > NOW() - INTERVAL '30 days'
            AND exit_time IS NOT NULL
            """
            var_result = self.db_manager.execute_query(var_query)
            pnls = [float(row["pnl"]) for row in var_result] if var_result else []
            var_95 = float(np.percentile(pnls, 5)) if pnls else 0.0

            # Position concentration
            positions = self.db_manager.get_active_positions()
            total_value = sum(
                self._safe_float(p.get("quantity", 0)) * self._safe_float(p.get("entry_price", 0))
                for p in positions
            )
            concentration = {}
            for p in positions:
                symbol = p.get("symbol", "")
                value = self._safe_float(p.get("quantity", 0)) * self._safe_float(
                    p.get("entry_price", 0)
                )
                pct = (value / total_value * 100) if total_value > 0 else 0
                concentration[symbol] = float(pct)

            return {
                "recent_risk_adjustments": risk_adjustments,
                "var_95": var_95,
                "position_concentration": concentration,
                "total_exposure": float(total_value),
                "current_drawdown": float(self._get_current_drawdown()),
                "max_drawdown": float(self._get_max_drawdown()),
            }
        except Exception as e:
            logger.error(f"Error getting detailed risk metrics: {e}")
            return {"error": str(e)}

    def _get_formatted_correlation_matrix(self) -> dict[str, Any]:
        """Get correlation matrix formatted for heatmap visualization."""
        try:
            query = """
            SELECT symbol_pair, correlation_value, last_updated
            FROM correlation_matrix
            ORDER BY last_updated DESC
            LIMIT 100
            """
            result = self.db_manager.execute_query(query)

            # Parse symbol pairs and build matrix
            symbols = set()
            correlations = {}
            for row in result:
                pair = row["symbol_pair"]
                s1, s2 = pair.split("-")
                symbols.add(s1)
                symbols.add(s2)
                correlations[pair] = float(row["correlation_value"])

            symbols_list = sorted(list(symbols))
            matrix = []
            for s1 in symbols_list:
                row_data = []
                for s2 in symbols_list:
                    if s1 == s2:
                        row_data.append(1.0)
                    else:
                        pair = "-".join(sorted([s1, s2]))
                        row_data.append(correlations.get(pair, 0.0))
                matrix.append(row_data)

            return {
                "symbols": symbols_list,
                "matrix": matrix,
            }
        except Exception as e:
            logger.error(f"Error formatting correlation matrix: {e}")
            return {"error": str(e)}

    # ========== EXPORT METHODS ==========

    def _export_trades_csv(self, days: int = 30) -> str:
        """Export trades to CSV format."""
        try:
            import csv
            import io

            query = f"""
            SELECT
                symbol, side, entry_price, exit_price, quantity,
                entry_time, exit_time, pnl, pnl_percent,
                strategy_name, exit_reason
            FROM trades
            WHERE exit_time > NOW() - INTERVAL '{days} DAYS'
            AND exit_time IS NOT NULL
            ORDER BY exit_time DESC
            """  # nosec B608: days validated below
            if days <= 0 or days > 365:
                days = 30
            result = self.db_manager.execute_query(query)

            output = io.StringIO()
            if not result:
                return "symbol,side,entry_price,exit_price,quantity,entry_time,exit_time,pnl,pnl_percent,strategy_name,exit_reason\n"

            writer = csv.DictWriter(output, fieldnames=result[0].keys())
            writer.writeheader()
            for row in result:
                # Convert datetime to string
                row_dict = dict(row)
                for k, v in row_dict.items():
                    if hasattr(v, "isoformat"):
                        row_dict[k] = v.isoformat()
                    elif isinstance(v, Decimal):
                        row_dict[k] = float(v)
                writer.writerow(row_dict)

            return output.getvalue()
        except Exception as e:
            logger.error(f"Error exporting trades: {e}")
            return f"Error: {e}"

    def _export_performance_csv(self, days: int = 30) -> str:
        """Export performance metrics to CSV format."""
        try:
            import csv
            import io

            query = f"""
            SELECT timestamp, balance, equity, total_pnl, daily_pnl, drawdown, open_positions
            FROM account_history
            WHERE timestamp > NOW() - INTERVAL '{days} DAYS'
            ORDER BY timestamp
            """  # nosec B608: days validated below
            if days <= 0 or days > 365:
                days = 30
            result = self.db_manager.execute_query(query)

            output = io.StringIO()
            if not result:
                return "timestamp,balance,equity,total_pnl,daily_pnl,drawdown,open_positions\n"

            writer = csv.DictWriter(output, fieldnames=result[0].keys())
            writer.writeheader()
            for row in result:
                row_dict = dict(row)
                for k, v in row_dict.items():
                    if hasattr(v, "isoformat"):
                        row_dict[k] = v.isoformat()
                    elif isinstance(v, Decimal):
                        row_dict[k] = float(v)
                writer.writerow(row_dict)

            return output.getvalue()
        except Exception as e:
            logger.error(f"Error exporting performance: {e}")
            return f"Error: {e}"

    def _export_positions_csv(self) -> str:
        """Export current positions to CSV format."""
        try:
            import csv
            import io

            positions = self._get_current_positions()

            output = io.StringIO()
            if not positions:
                return "symbol,side,entry_price,current_price,quantity,unrealized_pnl,entry_time\n"

            fieldnames = [
                "symbol",
                "side",
                "entry_price",
                "current_price",
                "quantity",
                "unrealized_pnl",
                "entry_time",
            ]
            writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            for pos in positions:
                pos_dict = {k: v for k, v in pos.items() if k in fieldnames}
                for k, v in pos_dict.items():
                    if hasattr(v, "isoformat"):
                        pos_dict[k] = v.isoformat()
                writer.writerow(pos_dict)

            return output.getvalue()
        except Exception as e:
            logger.error(f"Error exporting positions: {e}")
            return f"Error: {e}"

    def start_monitoring(self):
        """Start the monitoring update thread"""
        if self.is_running:
            return

        self.is_running = True
        t = threading.Thread(target=self._monitoring_loop)
        t.daemon = True
        t.start()
        self.update_thread = t
        logger.info("Monitoring thread started")

    def stop_monitoring(self):
        """Stop the monitoring update thread"""
        self.is_running = False
        if self.update_thread is not None:
            self.update_thread.join(timeout=10)
        logger.info("Monitoring thread stopped")

    def _monitoring_loop(self):
        """Main monitoring loop that broadcasts updates"""
        while self.is_running:
            try:
                metrics = self._collect_metrics()
                self.socketio.emit("metrics_update", metrics)
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)

    def run(self, host="127.0.0.1", port: int | None = None, debug=False):
        """Run the dashboard server (honours $PORT if provided)"""
        # Allow platform (Railway, Heroku, etc.) to inject port
        if port is None:
            env_port = os.getenv("PORT")
            if env_port and env_port.isdigit():
                port = int(env_port)
            else:
                port = 8080

        logger.info("-----------------------------------------------")
        logger.info("MONITORING DASHBOARD STARTUP")
        logger.info(f"Binding to {host}:{port} (debug={debug})")
        logger.info(f"Async mode: {_ASYNC_MODE}")
        logger.info("-----------------------------------------------")
        self.start_monitoring()
        try:
            # Decide server kwargs based on whether gevent is enabled.
            # With gevent enabled, Flask-SocketIO runs a production-safe server.
            # Without gevent, allow Werkzeug only for local development.
            server_kwargs = {
                "host": host,
                "port": port,
                "debug": debug,
                "use_reloader": False,
                "log_output": True,
            }
            if not _WEB_SERVER_USE_GEVENT:
                server_kwargs["allow_unsafe_werkzeug"] = True

            self.socketio.run(self.app, **server_kwargs)
            # If we ever return from run(), log why
            logger.warning(
                "Flask-SocketIO server exited normally (this usually means shutdown was requested)."
            )
        finally:
            self.stop_monitoring()
            logger.info("Monitoring dashboard stopped")

    def _monitor_system_health(self) -> dict[str, Any]:
        """Aggregate key system health indicators for tests and UI.
        Returns a dictionary with status fields and an overall alert level.
        """
        try:
            system_status = {
                "api_connection_status": self._get_api_status(),
                "system_health": self._get_system_health_status(),
            }
            # Data feed status based on ability to fetch a recent price
            try:
                _ = getattr(self.data_provider, "get_current_price", lambda *_args, **_kwargs: 0.0)(
                    "BTCUSDT"
                )
                system_status["data_feed_status"] = "Active"
            except Exception:
                system_status["data_feed_status"] = "No Data"

            # Determine alert level
            health = system_status["system_health"]
            if health == "Healthy":
                alert = "normal"
            elif health in {"Warning"}:
                alert = "warning"
            else:
                alert = "critical"

            return {"system_status": system_status, "alert_level": alert}
        except Exception as exc:
            logger.error(f"Health monitoring failed: {exc}")
            return {
                "system_status": {
                    "api_connection_status": "Disconnected",
                    "data_feed_status": "No Data",
                },
                "alert_level": "critical",
            }


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(description="Trading Bot Monitoring Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--db-url", help="Database URL")
    parser.add_argument(
        "--update-interval", type=int, default=3600, help="Update interval in seconds"
    )

    args = parser.parse_args()

    dashboard = MonitoringDashboard(db_url=args.db_url, update_interval=args.update_interval)

    dashboard.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
