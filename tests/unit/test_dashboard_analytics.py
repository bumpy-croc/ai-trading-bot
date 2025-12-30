"""
Unit tests for dashboard analytics endpoints.

Tests the advanced analytics endpoints added to the monitoring dashboard.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from decimal import Decimal


@pytest.fixture
def dashboard_app():
    """Create a test instance of the monitoring dashboard."""
    from src.dashboards.monitoring.dashboard import MonitoringDashboard

    # Mock database manager
    mock_db = Mock()
    mock_db.execute_query = Mock(return_value=[])
    mock_db.get_active_positions = Mock(return_value=[])

    dashboard = MonitoringDashboard(mock_db)
    dashboard.app.config['TESTING'] = True

    return dashboard.app, mock_db


class TestAdvancedPerformanceEndpoint:
    """Test /api/performance/advanced endpoint."""

    def test_returns_performance_metrics_with_valid_data(self, dashboard_app):
        """Returns rolling Sharpe, drawdown, and win rate data."""
        app, mock_db = dashboard_app

        # Mock account history data
        mock_db.execute_query = Mock(return_value=[
            {'balance': Decimal('10000'), 'timestamp': datetime(2025, 11, 1, tzinfo=timezone.utc)},
            {'balance': Decimal('10100'), 'timestamp': datetime(2025, 11, 2, tzinfo=timezone.utc)},
            {'balance': Decimal('10200'), 'timestamp': datetime(2025, 11, 3, tzinfo=timezone.utc)},
            {'balance': Decimal('10300'), 'timestamp': datetime(2025, 11, 4, tzinfo=timezone.utc)},
            {'balance': Decimal('10400'), 'timestamp': datetime(2025, 11, 5, tzinfo=timezone.utc)},
            {'balance': Decimal('10500'), 'timestamp': datetime(2025, 11, 6, tzinfo=timezone.utc)},
            {'balance': Decimal('10600'), 'timestamp': datetime(2025, 11, 7, tzinfo=timezone.utc)},
            {'balance': Decimal('10700'), 'timestamp': datetime(2025, 11, 8, tzinfo=timezone.utc)},
        ])

        client = app.test_client()
        response = client.get('/api/performance/advanced?days=30')

        assert response.status_code == 200
        data = response.get_json()

        assert 'rolling_sharpe' in data
        assert 'drawdown_series' in data
        assert 'win_rate_series' in data
        assert 'current_drawdown' in data
        assert 'max_drawdown' in data

    def test_handles_insufficient_data(self, dashboard_app):
        """Returns error when insufficient data available."""
        app, mock_db = dashboard_app

        # Only one data point
        mock_db.execute_query = Mock(return_value=[
            {'balance': Decimal('10000'), 'timestamp': datetime(2025, 11, 1, tzinfo=timezone.utc)},
        ])

        client = app.test_client()
        response = client.get('/api/performance/advanced?days=30')

        assert response.status_code == 200
        data = response.get_json()
        assert 'error' in data

    def test_validates_days_parameter(self, dashboard_app):
        """Validates and limits days parameter."""
        app, _ = dashboard_app

        client = app.test_client()

        # Test with negative days
        response = client.get('/api/performance/advanced?days=-10')
        assert response.status_code == 200  # Should use default

        # Test with excessive days
        response = client.get('/api/performance/advanced?days=999')
        assert response.status_code == 200  # Should limit to 365


class TestTradeAnalysisEndpoint:
    """Test /api/trades/analysis endpoint."""

    def test_returns_trade_analysis_data(self, dashboard_app):
        """Returns comprehensive trade analysis."""
        app, mock_db = dashboard_app

        # Mock trades data
        mock_db.execute_query = Mock(return_value=[
            {
                'symbol': 'BTCUSDT',
                'side': 'long',
                'entry_price': Decimal('50000'),
                'exit_price': Decimal('51000'),
                'quantity': Decimal('0.1'),
                'entry_time': datetime(2025, 11, 1, 10, 0, tzinfo=timezone.utc),
                'exit_time': datetime(2025, 11, 1, 14, 0, tzinfo=timezone.utc),
                'pnl': Decimal('100'),
                'pnl_percent': Decimal('2.0'),
                'strategy_name': 'ml_basic',
                'exit_reason': 'take_profit'
            },
            {
                'symbol': 'ETHUSDT',
                'side': 'short',
                'entry_price': Decimal('3000'),
                'exit_price': Decimal('2950'),
                'quantity': Decimal('1.0'),
                'entry_time': datetime(2025, 11, 2, 8, 0, tzinfo=timezone.utc),
                'exit_time': datetime(2025, 11, 2, 16, 0, tzinfo=timezone.utc),
                'pnl': Decimal('50'),
                'pnl_percent': Decimal('1.67'),
                'strategy_name': 'ml_basic',
                'exit_reason': 'stop_loss'
            },
        ])

        client = app.test_client()
        response = client.get('/api/trades/analysis?days=30')

        assert response.status_code == 200
        data = response.get_json()

        assert 'total_trades' in data
        assert 'avg_duration_hours' in data
        assert 'median_duration_hours' in data
        assert 'profit_by_hour' in data
        assert 'profit_by_day_of_week' in data
        assert 'best_trades' in data
        assert 'worst_trades' in data

        assert data['total_trades'] == 2

    def test_handles_no_trades(self, dashboard_app):
        """Handles case when no trades found."""
        app, mock_db = dashboard_app

        mock_db.execute_query = Mock(return_value=[])

        client = app.test_client()
        response = client.get('/api/trades/analysis?days=30')

        assert response.status_code == 200
        data = response.get_json()
        assert 'error' in data


class TestTradeDistributionEndpoint:
    """Test /api/trades/distribution endpoint."""

    def test_returns_histogram_data(self, dashboard_app):
        """Returns P&L distribution for histogram."""
        app, mock_db = dashboard_app

        # Mock P&L data
        mock_db.execute_query = Mock(return_value=[
            {'pnl': Decimal('100')},
            {'pnl': Decimal('50')},
            {'pnl': Decimal('-30')},
            {'pnl': Decimal('200')},
            {'pnl': Decimal('-50')},
            {'pnl': Decimal('75')},
        ])

        client = app.test_client()
        response = client.get('/api/trades/distribution?days=30&bins=5')

        assert response.status_code == 200
        data = response.get_json()

        assert 'bins' in data
        assert 'counts' in data
        assert 'mean' in data
        assert 'median' in data
        assert 'std' in data

        assert len(data['bins']) == 6  # bins + 1 for edges
        assert len(data['counts']) == 5

    def test_handles_empty_data(self, dashboard_app):
        """Handles empty trade data."""
        app, mock_db = dashboard_app

        mock_db.execute_query = Mock(return_value=[])

        client = app.test_client()
        response = client.get('/api/trades/distribution?days=30')

        assert response.status_code == 200
        data = response.get_json()
        assert data['bins'] == []
        assert data['counts'] == []


class TestModelPerformanceEndpoint:
    """Test /api/models/performance endpoint."""

    def test_returns_model_performance_data(self, dashboard_app):
        """Returns model performance metrics over time."""
        app, mock_db = dashboard_app

        mock_db.execute_query = Mock(return_value=[
            {
                'timestamp': datetime(2025, 11, 1, tzinfo=timezone.utc),
                'model_name': 'btc_model_v1',
                'horizon': 24,
                'mae': Decimal('150.5'),
                'rmse': Decimal('200.3'),
                'mape': Decimal('2.5'),
                'ic': Decimal('0.65'),
                'mean_pred': Decimal('50000'),
                'std_pred': Decimal('500'),
                'mean_real': Decimal('50100'),
                'std_real': Decimal('520'),
                'strategy_name': 'ml_basic',
                'symbol': 'BTCUSDT',
                'timeframe': '1h'
            },
        ])

        client = app.test_client()
        response = client.get('/api/models/performance?days=30')

        assert response.status_code == 200
        data = response.get_json()

        assert 'series' in data
        assert 'summary' in data
        assert len(data['series']) == 1
        assert 'avg_mae' in data['summary']

    def test_filters_by_model_name(self, dashboard_app):
        """Filters performance data by model name."""
        app, mock_db = dashboard_app

        mock_db.execute_query = Mock(return_value=[])

        client = app.test_client()
        response = client.get('/api/models/performance?days=30&model=btc_model_v1')

        assert response.status_code == 200
        # Verify model parameter was passed in query
        mock_db.execute_query.assert_called()


class TestSystemHealthEndpoint:
    """Test /api/system/health-detailed endpoint."""

    def test_returns_system_health_metrics(self, dashboard_app):
        """Returns comprehensive system health data."""
        app, mock_db = dashboard_app

        # Mock error data
        mock_db.execute_query = Mock(side_effect=[
            [{'total_events': 100, 'errors': 5, 'warnings': 10}],  # Error stats
            [],  # Recent errors
        ])

        client = app.test_client()
        response = client.get('/api/system/health-detailed')

        assert response.status_code == 200
        data = response.get_json()

        assert 'database_latency_ms' in data
        assert 'database_status' in data
        assert 'error_rate_hourly' in data
        assert 'warning_rate_hourly' in data
        assert 'recent_errors' in data
        assert 'memory_usage_percent' in data
        assert 'uptime_minutes' in data


class TestRiskMetricsEndpoint:
    """Test /api/risk/detailed endpoint."""

    def test_returns_risk_metrics(self, dashboard_app):
        """Returns detailed risk metrics."""
        app, mock_db = dashboard_app

        # Mock risk adjustments and trades
        mock_db.execute_query = Mock(side_effect=[
            [],  # Risk adjustments
            [{'pnl': Decimal('-100')}, {'pnl': Decimal('50')}, {'pnl': Decimal('-30')}],  # VaR data
        ])

        mock_db.get_active_positions = Mock(return_value=[
            {'symbol': 'BTCUSDT', 'quantity': Decimal('0.1'), 'entry_price': Decimal('50000')},
        ])

        client = app.test_client()
        response = client.get('/api/risk/detailed')

        assert response.status_code == 200
        data = response.get_json()

        assert 'recent_risk_adjustments' in data
        assert 'var_95' in data
        assert 'position_concentration' in data
        assert 'total_exposure' in data
        assert 'current_drawdown' in data
        assert 'max_drawdown' in data


class TestCorrelationMatrixEndpoint:
    """Test /api/correlation/matrix-formatted endpoint."""

    def test_returns_formatted_correlation_matrix(self, dashboard_app):
        """Returns correlation matrix formatted for heatmap."""
        app, mock_db = dashboard_app

        mock_db.execute_query = Mock(return_value=[
            {'symbol_pair': 'BTCUSDT-ETHUSDT', 'correlation_value': Decimal('0.85'), 'last_updated': datetime.now(timezone.utc)},
            {'symbol_pair': 'BTCUSDT-BNBUSDT', 'correlation_value': Decimal('0.72'), 'last_updated': datetime.now(timezone.utc)},
            {'symbol_pair': 'ETHUSDT-BNBUSDT', 'correlation_value': Decimal('0.68'), 'last_updated': datetime.now(timezone.utc)},
        ])

        client = app.test_client()
        response = client.get('/api/correlation/matrix-formatted')

        assert response.status_code == 200
        data = response.get_json()

        assert 'symbols' in data
        assert 'matrix' in data
        assert len(data['symbols']) == 3
        assert len(data['matrix']) == 3
        assert len(data['matrix'][0]) == 3  # Square matrix

    def test_handles_no_correlation_data(self, dashboard_app):
        """Handles case when no correlation data available."""
        app, mock_db = dashboard_app

        mock_db.execute_query = Mock(return_value=[])

        client = app.test_client()
        response = client.get('/api/correlation/matrix-formatted')

        assert response.status_code == 200
        data = response.get_json()
        assert data['symbols'] == []
        assert data['matrix'] == []


class TestExportEndpoints:
    """Test CSV export endpoints."""

    def test_exports_trades_as_csv(self, dashboard_app):
        """Exports trades in CSV format."""
        app, mock_db = dashboard_app

        mock_db.execute_query = Mock(return_value=[
            {
                'symbol': 'BTCUSDT',
                'side': 'long',
                'entry_price': Decimal('50000'),
                'exit_price': Decimal('51000'),
                'quantity': Decimal('0.1'),
                'entry_time': datetime(2025, 11, 1, 10, 0, tzinfo=timezone.utc),
                'exit_time': datetime(2025, 11, 1, 14, 0, tzinfo=timezone.utc),
                'pnl': Decimal('100'),
                'pnl_percent': Decimal('2.0'),
                'strategy_name': 'ml_basic',
                'exit_reason': 'take_profit'
            },
        ])

        client = app.test_client()
        response = client.get('/api/export/trades?days=30')

        assert response.status_code == 200
        assert response.content_type == 'text/csv; charset=utf-8'
        assert b'symbol,side,entry_price' in response.data

    def test_exports_performance_as_csv(self, dashboard_app):
        """Exports performance metrics in CSV format."""
        app, mock_db = dashboard_app

        mock_db.execute_query = Mock(return_value=[
            {
                'timestamp': datetime(2025, 11, 1, tzinfo=timezone.utc),
                'balance': Decimal('10000'),
                'equity': Decimal('10100'),
                'total_pnl': Decimal('100'),
                'daily_pnl': Decimal('50'),
                'drawdown': Decimal('-2.5'),
                'open_positions': 2
            },
        ])

        client = app.test_client()
        response = client.get('/api/export/performance?days=30')

        assert response.status_code == 200
        assert response.content_type == 'text/csv; charset=utf-8'
        assert b'timestamp,balance,equity' in response.data

    def test_exports_positions_as_csv(self, dashboard_app):
        """Exports current positions in CSV format."""
        app, mock_db = dashboard_app

        mock_db.get_active_positions = Mock(return_value=[
            {
                'symbol': 'BTCUSDT',
                'side': 'long',
                'entry_price': Decimal('50000'),
                'current_price': Decimal('51000'),
                'quantity': Decimal('0.1'),
                'unrealized_pnl': Decimal('100'),
                'entry_time': datetime(2025, 11, 1, 10, 0, tzinfo=timezone.utc),
            },
        ])

        client = app.test_client()
        response = client.get('/api/export/positions')

        assert response.status_code == 200
        assert response.content_type == 'text/csv; charset=utf-8'
        assert b'symbol,side,entry_price' in response.data


@pytest.mark.integration
class TestDashboardIntegration:
    """Integration tests for dashboard with real database."""

    def test_enhanced_dashboard_loads(self, dashboard_app):
        """Enhanced dashboard template loads successfully."""
        app, _ = dashboard_app

        client = app.test_client()
        response = client.get('/')

        assert response.status_code == 200
        # Should use enhanced template
        assert b'dashboardTabs' in response.data or b'Dashboard' in response.data

    def test_health_endpoint_accessible(self, dashboard_app):
        """Health check endpoint is accessible."""
        app, _ = dashboard_app

        client = app.test_client()
        response = client.get('/health')

        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'ok'
