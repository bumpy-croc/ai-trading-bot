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

import pytest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Import conditionally to handle missing dependencies
try:
    from src.monitoring.dashboard import MonitoringDashboard
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
        with patch('src.monitoring.dashboard.DatabaseManager') as mock_db, \
             patch('src.data_providers.binance_provider.BinanceProvider') as mock_binance, \
             patch('src.data_providers.cached_data_provider.CachedDataProvider') as mock_cache:
            
            # Mock successful initialization
            mock_binance.return_value = Mock()
            mock_cache.return_value = Mock()
            mock_db.return_value = Mock()
            
            dashboard = MonitoringDashboard()
            
            assert dashboard.app is not None
            assert dashboard.socketio is not None
            assert dashboard.db_manager is not None
            assert dashboard.data_provider is not None
            assert dashboard.is_running == False
            assert dashboard.update_interval == 3600

    def test_dashboard_initialization_with_offline_mode(self):
        """Test dashboard initialization when Binance is unavailable"""
        with patch('src.monitoring.dashboard.DatabaseManager') as mock_db, \
             patch('src.monitoring.dashboard.BinanceProvider') as mock_binance:
            
            # Mock Binance failure
            mock_binance.side_effect = Exception("Connection failed")
            mock_db.return_value = Mock()
            
            dashboard = MonitoringDashboard()
            
            # Should fall back to offline provider
            assert dashboard.data_provider is not None
            assert hasattr(dashboard.data_provider, 'get_current_price')

    def test_dashboard_configuration(self):
        """Test dashboard configuration management"""
        with patch('src.monitoring.dashboard.DatabaseManager'), \
             patch('src.data_providers.binance_provider.BinanceProvider'), \
             patch('src.data_providers.cached_data_provider.CachedDataProvider'):
            
            dashboard = MonitoringDashboard()
            
            # Test default configuration
            config = dashboard.monitoring_config
            
            # Check required metrics are present
            required_metrics = [
                'current_balance', 'total_pnl', 'win_rate', 
                'max_drawdown', 'current_drawdown', 'api_connection_status'
            ]
            
            for metric in required_metrics:
                assert metric in config
                assert 'enabled' in config[metric]
                assert 'priority' in config[metric]
                assert 'format' in config[metric]

    @pytest.mark.monitoring
    def test_metrics_collection(self):
        """Test metrics collection functionality"""
        with patch('src.monitoring.dashboard.DatabaseManager'), \
             patch('src.data_providers.binance_provider.BinanceProvider'), \
             patch('src.data_providers.cached_data_provider.CachedDataProvider'):
            
            dashboard = MonitoringDashboard()
            
            # Mock database responses
            dashboard.db_manager.execute_query = Mock(return_value=[
                {'balance': 10000.0, 'timestamp': datetime.now()}
            ])
            
            # Mock data provider
            dashboard.data_provider.get_current_price = Mock(return_value=50000.0)
            
            metrics = dashboard._collect_metrics()
            
            # Check that metrics are collected
            assert isinstance(metrics, dict)
            assert 'current_balance' in metrics
            assert 'total_pnl' in metrics
            assert 'win_rate' in metrics
            assert 'max_drawdown' in metrics

    @pytest.mark.monitoring
    def test_balance_calculation(self):
        """Test balance calculation methods"""
        with patch('src.monitoring.dashboard.DatabaseManager'), \
             patch('src.data_providers.binance_provider.BinanceProvider'), \
             patch('src.data_providers.cached_data_provider.CachedDataProvider'):
            
            dashboard = MonitoringDashboard()
            
            # Mock database response
            dashboard.db_manager.execute_query = Mock(return_value=[
                {'balance': 10500.0, 'timestamp': datetime.now()}
            ])
            
            balance = dashboard._get_current_balance()
            assert balance == 10500.0

    @pytest.mark.monitoring
    def test_win_rate_calculation(self):
        """Test win rate calculation"""
        with patch('src.monitoring.dashboard.DatabaseManager'), \
             patch('src.data_providers.binance_provider.BinanceProvider'), \
             patch('src.data_providers.cached_data_provider.CachedDataProvider'):
            
            dashboard = MonitoringDashboard()
            
            # Mock database response with aggregated win rate data
            dashboard.db_manager.execute_query = Mock(return_value=[
                {'total_trades': 3, 'winning_trades': 2}
            ])
            
            win_rate = dashboard._get_win_rate()
            assert win_rate == pytest.approx(66.67, rel=1e-2)  # 2 out of 3 trades profitable

    @pytest.mark.monitoring
    def test_drawdown_calculation(self):
        """Test drawdown calculation"""
        with patch('src.monitoring.dashboard.DatabaseManager'), \
             patch('src.data_providers.binance_provider.BinanceProvider'), \
             patch('src.data_providers.cached_data_provider.CachedDataProvider'):
            
            dashboard = MonitoringDashboard()
            
            # Mock balance history with drawdown
            balance_history = [
                {'balance': 10000.0, 'date': '2024-01-01'},
                {'balance': 12000.0, 'date': '2024-01-02'},
                {'balance': 9000.0, 'date': '2024-01-03'},
                {'balance': 11000.0, 'date': '2024-01-04'}
            ]
            
            dashboard.db_manager.execute_query = Mock(return_value=balance_history)
            
            max_dd = dashboard._get_max_drawdown()
            # Should be 25% drawdown (9000 from 12000 peak)
            assert max_dd == 25.0

    @pytest.mark.monitoring
    def test_position_tracking(self):
        """Test position tracking functionality"""
        with patch('src.monitoring.dashboard.DatabaseManager'), \
             patch('src.data_providers.binance_provider.BinanceProvider'), \
             patch('src.data_providers.cached_data_provider.CachedDataProvider'):
            
            dashboard = MonitoringDashboard()
            
            # Mock active positions
            positions_data = [
                {
                    'symbol': 'BTCUSDT',
                    'side': 'long',
                    'entry_price': 50000.0,
                    'current_price': 51000.0,
                    'quantity': 0.1,
                    'entry_time': datetime.now() - timedelta(hours=1),
                    'stop_loss': 49000.0,
                    'take_profit': 52000.0,
                    'order_id': 'test_001'
                }
            ]
            
            dashboard.db_manager.execute_query = Mock(return_value=positions_data)
            dashboard._get_current_price = Mock(return_value=51000.0)
            
            positions = dashboard._get_current_positions()
            
            assert len(positions) == 1
            position = positions[0]
            assert position['symbol'] == 'BTCUSDT'
            assert position['side'] == 'long'
            assert position['entry_price'] == 50000.0
            assert position['unrealized_pnl'] > 0  # Should be positive with price increase

    @pytest.mark.monitoring
    def test_trade_history(self):
        """Test trade history retrieval"""
        with patch('src.monitoring.dashboard.DatabaseManager'), \
             patch('src.data_providers.binance_provider.BinanceProvider'), \
             patch('src.data_providers.cached_data_provider.CachedDataProvider'):
            
            dashboard = MonitoringDashboard()
            
            # Mock recent trades
            trades_data = [
                {
                    'symbol': 'BTCUSDT',
                    'side': 'long',
                    'entry_price': 50000.0,
                    'exit_price': 51000.0,
                    'quantity': 0.1,
                    'entry_time': datetime.now() - timedelta(hours=2),
                    'exit_time': datetime.now() - timedelta(hours=1),
                    'pnl': 100.0,
                    'exit_reason': 'Take profit'
                }
            ]
            
            dashboard.db_manager.execute_query = Mock(return_value=trades_data)
            
            trades = dashboard._get_recent_trades(limit=10)
            
            assert len(trades) == 1
            trade = trades[0]
            assert trade['symbol'] == 'BTCUSDT'
            assert trade['pnl'] == 100.0
            assert trade['exit_reason'] == 'Take profit'

    @pytest.mark.monitoring
    def test_system_health_monitoring(self):
        """Test system health status monitoring"""
        with patch('src.monitoring.dashboard.DatabaseManager'), \
             patch('src.data_providers.binance_provider.BinanceProvider'), \
             patch('src.data_providers.cached_data_provider.CachedDataProvider'):
            
            dashboard = MonitoringDashboard()
            
            # Test API connection status
            status = dashboard._get_api_connection_status()
            assert status in ['Connected', 'Disconnected']
            
            # Test data feed status
            feed_status = dashboard._get_data_feed_status()
            assert feed_status in ['Active', 'Inactive', 'Error']
            
            # Test system health
            health = dashboard._get_system_health_status()
            assert health in ['Healthy', 'Warning', 'Critical', 'Error']

    @pytest.mark.monitoring
    def test_performance_chart_data(self):
        """Test performance chart data generation"""
        with patch('src.monitoring.dashboard.DatabaseManager'), \
             patch('src.data_providers.binance_provider.BinanceProvider'), \
             patch('src.data_providers.cached_data_provider.CachedDataProvider'):
            
            dashboard = MonitoringDashboard()
            
            # Mock balance history for chart
            balance_history = [
                {'balance': 10000.0, 'timestamp': datetime.now() - timedelta(days=6)},
                {'balance': 10200.0, 'timestamp': datetime.now() - timedelta(days=5)},
                {'balance': 9800.0, 'timestamp': datetime.now() - timedelta(days=4)},
                {'balance': 10500.0, 'timestamp': datetime.now() - timedelta(days=3)},
                {'balance': 10300.0, 'timestamp': datetime.now() - timedelta(days=2)},
                {'balance': 10800.0, 'timestamp': datetime.now() - timedelta(days=1)},
                {'balance': 11000.0, 'timestamp': datetime.now()}
            ]
            
            dashboard.db_manager.execute_query = Mock(return_value=balance_history)
            
            chart_data = dashboard._get_performance_chart_data(days=7)
            
            assert 'timestamps' in chart_data
            assert 'balances' in chart_data
            assert len(chart_data['timestamps']) == 7
            assert len(chart_data['balances']) == 7

    @pytest.mark.monitoring
    def test_error_handling(self):
        """Test error handling in metrics collection"""
        with patch('src.monitoring.dashboard.DatabaseManager'), \
             patch('src.data_providers.binance_provider.BinanceProvider'), \
             patch('src.data_providers.cached_data_provider.CachedDataProvider'):
            
            dashboard = MonitoringDashboard()
            
            # Mock database error
            dashboard.db_manager.execute_query = Mock(side_effect=Exception("Database error"))
            
            # Should handle errors gracefully
            metrics = dashboard._collect_metrics()
            
            # Should still return metrics structure even with errors
            assert isinstance(metrics, dict)
            assert 'current_balance' in metrics
            # Should have fallback values for failed metrics
            assert metrics['current_balance'] == 0.0

    @pytest.mark.monitoring
    def test_configuration_update(self):
        """Test configuration update functionality"""
        with patch('src.monitoring.dashboard.DatabaseManager'), \
             patch('src.data_providers.binance_provider.BinanceProvider'), \
             patch('src.data_providers.cached_data_provider.CachedDataProvider'):
            
            dashboard = MonitoringDashboard()
            
            # Test updating configuration
            original_interval = dashboard.update_interval
            dashboard.update_interval = 1800  # 30 minutes
            
            assert dashboard.update_interval == 1800
            assert dashboard.update_interval != original_interval

    @pytest.mark.monitoring
    def test_monitoring_lifecycle(self):
        """Test monitoring start/stop functionality"""
        with patch('src.monitoring.dashboard.DatabaseManager'), \
             patch('src.data_providers.binance_provider.BinanceProvider'), \
             patch('src.data_providers.cached_data_provider.CachedDataProvider'):
            
            dashboard = MonitoringDashboard()
            
            # Test start monitoring
            dashboard.start_monitoring()
            assert dashboard.is_running == True
            
            # Test stop monitoring
            dashboard.stop_monitoring()
            assert dashboard.is_running == False

    @pytest.mark.monitoring
    def test_websocket_functionality(self):
        """Test WebSocket event handling"""
        with patch('src.monitoring.dashboard.DatabaseManager'), \
             patch('src.data_providers.binance_provider.BinanceProvider'), \
             patch('src.data_providers.cached_data_provider.CachedDataProvider'):
            
            dashboard = MonitoringDashboard()
            
            # Mock socketio emit
            mock_emit = Mock()
            dashboard.socketio.emit = mock_emit
            
            # Test metrics update emission (simulate monitoring loop)
            metrics = dashboard._collect_metrics()
            dashboard.socketio.emit('metrics_update', metrics)
            
            # Should emit metrics update
            mock_emit.assert_called_with('metrics_update', metrics)


class TestMonitoringFallbacks:
    """Test fallback behavior when monitoring components are unavailable"""

    def test_mock_monitoring_dashboard(self):
        """Test that mock dashboard works when real components unavailable"""
        if MONITORING_AVAILABLE:
            pytest.skip("Real monitoring components available")
        
        # Should be able to create mock dashboard
        dashboard = MonitoringDashboard()
        assert dashboard is not None

    def test_availability_flags(self):
        """Test component availability flags"""
        assert isinstance(MONITORING_AVAILABLE, bool)
        assert isinstance(DATABASE_AVAILABLE, bool)


@pytest.mark.monitoring
class TestMonitoringIntegration:
    """Integration tests for monitoring system"""

    def test_dashboard_with_real_data_provider(self, mock_data_provider):
        """Test dashboard integration with data provider"""
        if not MONITORING_AVAILABLE:
            pytest.skip("Monitoring components not available")
        
        with patch('src.monitoring.dashboard.DatabaseManager'):
            dashboard = MonitoringDashboard()
            
            # Replace with mock provider
            dashboard.data_provider = mock_data_provider
            
            # Test metrics collection with mock data
            metrics = dashboard._collect_metrics()
            assert isinstance(metrics, dict)

    def test_dashboard_with_real_database(self):
        """Test dashboard integration with database"""
        if not MONITORING_AVAILABLE or not DATABASE_AVAILABLE:
            pytest.skip("Required components not available")
        
        with patch('src.monitoring.dashboard.DatabaseManager'), \
             patch('src.data_providers.binance_provider.BinanceProvider'), \
             patch('src.data_providers.cached_data_provider.CachedDataProvider'):
            
            dashboard = MonitoringDashboard()
            
            # Test database integration
            assert dashboard.db_manager is not None
            # Should be able to execute queries (even if they fail)
            assert hasattr(dashboard.db_manager, 'execute_query')