"""
Tests for the monitoring dashboard system.

The monitoring dashboard is critical for real-time system oversight. Tests cover:
- Dashboard initialization and configuration
- API endpoints and data retrieval
- Real-time metrics calculation
- WebSocket functionality
- System health monitoring
- Performance data aggregation
"""

from unittest.mock import Mock, patch

import pytest

# Import conditionally to handle missing dependencies
try:
    from src.dashboards.monitoring import MonitoringDashboard

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    MonitoringDashboard = Mock

try:
    from src.database.manager import DatabaseManager

    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    DatabaseManager = Mock


@pytest.mark.skipif(not MONITORING_AVAILABLE, reason="Monitoring components not available")
class TestMonitoringDashboard:
    """Test suite for the monitoring dashboard"""

    def test_dashboard_initialization(self):
        """Test dashboard initialization with default settings"""
        with patch("src.dashboards.monitoring.dashboard.DatabaseManager") as mock_db, patch(
            "src.data_providers.binance_provider.BinanceProvider"
        ) as mock_binance, patch(
            "src.data_providers.cached_data_provider.CachedDataProvider"
        ) as mock_cache:
            # Mock successful initialization
            mock_binance.return_value = Mock()
            mock_cache.return_value = Mock()
            mock_db.return_value = Mock()

            dashboard = MonitoringDashboard()

            assert dashboard.app is not None
            assert dashboard.socketio is not None
            assert dashboard.db_manager is not None
            assert dashboard.data_provider is not None
            assert not dashboard.is_running
            assert dashboard.update_interval == 3600

    def test_dashboard_initialization_with_offline_mode(self):
        """Test dashboard initialization when Binance is unavailable"""
        with patch("src.dashboards.monitoring.dashboard.DatabaseManager") as mock_db, patch(
            "src.dashboards.monitoring.dashboard.BinanceProvider"
        ) as mock_binance:
            # Mock Binance failure
            mock_binance.side_effect = Exception("Connection failed")
            mock_db.return_value = Mock()

            dashboard = MonitoringDashboard()

            # Should fall back to offline provider
            assert dashboard.data_provider is not None
            assert hasattr(dashboard.data_provider, "get_current_price")

    def test_dashboard_configuration(self):
        """Test dashboard configuration management"""
        with patch("src.dashboards.monitoring.dashboard.DatabaseManager"), patch(
            "src.data_providers.binance_provider.BinanceProvider"
        ), patch("src.data_providers.cached_data_provider.CachedDataProvider"):
            dashboard = MonitoringDashboard()

            # Test default configuration
            config = dashboard.monitoring_config

            # Check required metrics are present
            required_metrics = [
                "current_balance",
                "total_pnl",
                "win_rate",
                "max_drawdown",
                "current_drawdown",
                "api_connection_status",
            ]

            for metric in required_metrics:
                assert metric in config
                assert "enabled" in config[metric]
                assert "priority" in config[metric]
                assert "format" in config[metric]

    @pytest.mark.monitoring
    def test_metrics_collection(self):
        """Test metrics collection functionality"""
        with patch("src.dashboards.monitoring.dashboard.DatabaseManager"), patch(
            "src.data_providers.binance_provider.BinanceProvider"
        ), patch("src.data_providers.cached_data_provider.CachedDataProvider"):
            dashboard = MonitoringDashboard()

            # Mock database responses
            dashboard.db_manager.execute_query = Mock(return_value=[(1000.0,)])

            # Test metrics
            metrics = dashboard._collect_metrics()
            assert isinstance(metrics, dict)
            assert "current_balance" in metrics
            assert "total_pnl" in metrics

    def test_api_endpoints(self):
        """Test API endpoints for dashboard"""
        with patch("src.dashboards.monitoring.dashboard.DatabaseManager"), patch(
            "src.data_providers.binance_provider.BinanceProvider"
        ), patch("src.data_providers.cached_data_provider.CachedDataProvider"):
            dashboard = MonitoringDashboard()
            client = dashboard.app.test_client()

            # Test index route
            response = client.get("/")
            assert response.status_code == 200

            # Test balance endpoint
            response = client.get("/api/balance")
            assert response.status_code == 200
            data = response.get_json()
            assert "balance" in data

    def test_system_health_monitoring(self):
        """Test system health monitoring functionality"""
        with patch("src.dashboards.monitoring.dashboard.DatabaseManager"), patch(
            "src.data_providers.binance_provider.BinanceProvider"
        ), patch("src.data_providers.cached_data_provider.CachedDataProvider"):
            dashboard = MonitoringDashboard()

            # Mock data for health checks
            dashboard._get_current_balance = Mock(return_value=1000.0)
            dashboard._get_recent_trades = Mock(return_value=[{"pnl": 50.0}])
            dashboard._get_current_positions = Mock(
                return_value=[{"symbol": "BTCUSDT", "pnl": 20.0}]
            )
            dashboard._get_system_status = Mock(return_value={"api_connection_status": "Active"})

            health_status = dashboard._monitor_system_health()
            assert isinstance(health_status, dict)
            assert "system_status" in health_status
            assert "alert_level" in health_status
            # Updated acceptable statuses to include "No Data" fallback
            feed_status = health_status.get("system_status", {}).get("data_feed_status")
            assert feed_status in ["Active", "Inactive", "Error", "No Data"]


@pytest.mark.skipif(not MONITORING_AVAILABLE, reason="Monitoring components not available")
class TestMonitoringIntegration:
    def test_dashboard_with_real_data_provider(self):
        """Integration test with real data provider (mocked)"""
        with patch("src.dashboards.monitoring.dashboard.DatabaseManager"):
            dashboard = MonitoringDashboard()
            assert dashboard.app is not None
            assert dashboard.socketio is not None

    def test_dashboard_with_real_database(self):
        """Integration test with real database (mocked)"""
        with patch("src.dashboards.monitoring.dashboard.DatabaseManager"), patch(
            "src.data_providers.binance_provider.BinanceProvider"
        ), patch("src.data_providers.cached_data_provider.CachedDataProvider"):
            dashboard = MonitoringDashboard()
            assert dashboard.db_manager is not None
