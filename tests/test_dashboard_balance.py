import pytest
from unittest.mock import patch

# Patch the DatabaseManager before importing the dashboard to avoid real DB connections

import builtins
from types import SimpleNamespace


class _MockDBManager:
    def execute_query(self, *args, **kwargs):
        return []


# Apply patch via import machinery before the dashboard module creates an instance
import importlib

dashboard_module = importlib.import_module("monitoring.dashboard")
setattr(dashboard_module, "DatabaseManager", _MockDBManager)

# Patch external providers used during dashboard initialization
setattr(dashboard_module, "BinanceDataProvider", lambda *args, **kwargs: SimpleNamespace())
setattr(
    dashboard_module,
    "CachedDataProvider",
    lambda provider, cache_ttl_hours=0: provider,
)

# Ensure src/ directory is on path (handled by sitecustomize), then import
from monitoring.dashboard import MonitoringDashboard


@pytest.fixture
def dashboard():
    """Create an instance of the MonitoringDashboard for testing."""
    # Use an in-memory DB (or None) – internal queries will be monkey-patched
    return MonitoringDashboard(db_url=None)


def test_get_balance_endpoint(dashboard, monkeypatch):
    """/api/balance should return current balance and HTTP 200."""
    expected_balance = 1234.56
    monkeypatch.setattr(dashboard, "_get_current_balance", lambda: expected_balance)

    client = dashboard.app.test_client()
    resp = client.get("/api/balance")

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["balance"] == expected_balance


def test_get_balance_history_endpoint(dashboard, monkeypatch):
    """/api/balance/history should return balance history list and HTTP 200."""
    mock_history = [
        {"timestamp": "2025-01-01T00:00:00", "balance": 1000},
        {"timestamp": "2025-01-02T00:00:00", "balance": 1100},
    ]
    monkeypatch.setattr(dashboard, "_get_balance_history", lambda days: mock_history)

    client = dashboard.app.test_client()
    resp = client.get("/api/balance/history?days=2")

    assert resp.status_code == 200
    data = resp.get_json()
    assert data == mock_history 