"""Integration tests for the V2 monitoring dashboard.

Covers the bundled ``/api/dashboard/state`` endpoint and the
``_get_bot_meta`` helper that backs the dashboard topbar / strategy view.

These tests deliberately patch out the database and data-provider layers so
they can run without a live Postgres instance — they exercise the contract
of the new code paths added in the V2 redesign (#dashboard-v2).
"""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest

try:
    from src.dashboards.monitoring import MonitoringDashboard

    MONITORING_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard for CI without deps
    MONITORING_AVAILABLE = False
    MonitoringDashboard = Mock  # type: ignore


pytestmark = pytest.mark.integration


@pytest.fixture
def dashboard():
    """Construct a MonitoringDashboard with stubbed deps for HTTP tests."""
    if not MONITORING_AVAILABLE:
        pytest.skip("Monitoring components not available")
    with (
        patch("src.dashboards.monitoring.dashboard.DatabaseManager") as mock_db_cls,
        patch("src.data_providers.binance_provider.BinanceProvider"),
        patch("src.data_providers.cached_data_provider.CachedDataProvider"),
    ):
        mock_db = Mock()
        mock_db_cls.return_value = mock_db
        d = MonitoringDashboard()
        # Default mocks return empty so the endpoint exercises empty paths
        mock_db.execute_query.return_value = []
        mock_db.get_active_positions.return_value = []
        # Tame the periodic monitoring thread for tests
        d.is_running = False
        return d


def _client(d):
    return d.app.test_client()


# ─────────────────────────── /api/dashboard/state ──────────────


def test_dashboard_state_endpoint_returns_bundled_keys(dashboard):
    """All four top-level keys + server_time must be present."""
    # Make _collect_metrics deterministic (avoid Binance probe)
    with patch.object(
        dashboard,
        "_collect_metrics",
        return_value={"current_balance": 1100.0, "api_connection_status": "Connected"},
    ):
        resp = _client(dashboard).get("/api/dashboard/state")
    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert set(body.keys()) >= {"bot", "metrics", "positions", "trades", "server_time"}
    assert isinstance(body["positions"], list)
    assert isinstance(body["trades"], list)


def test_dashboard_state_clamps_trades_limit_above_500(dashboard):
    """trades_limit must be clamped to the validator's max (500)."""
    captured = {}

    real_get_recent = dashboard._get_recent_trades

    def spy(limit):
        captured["limit"] = limit
        return real_get_recent(limit)

    with (
        patch.object(
            dashboard, "_collect_metrics", return_value={"api_connection_status": "Disconnected"}
        ),
        patch.object(dashboard, "_get_recent_trades", side_effect=spy),
    ):
        resp = _client(dashboard).get("/api/dashboard/state?trades_limit=99999")
    assert resp.status_code == 200
    assert captured["limit"] <= 500, "trades_limit should be clamped"


def test_dashboard_state_internal_error_does_not_leak_exception_text(dashboard):
    """500 path must return a generic 'internal_error' code — no SQL leakage."""
    with patch.object(
        dashboard,
        "_collect_metrics",
        side_effect=RuntimeError("postgres detail: column foo missing in table bar"),
    ):
        resp = _client(dashboard).get("/api/dashboard/state")
    assert resp.status_code == 500
    body = json.loads(resp.data)
    # The message must not contain the leaky details
    assert "postgres" not in json.dumps(body)
    assert "column foo" not in json.dumps(body)
    assert body == {"error": "internal_error"}


def test_dashboard_state_passes_metrics_hint_to_bot_meta(dashboard):
    """`_get_bot_meta` must reuse api_connection_status from metrics, not re-probe."""
    metrics = {"api_connection_status": "Connected"}
    with (
        patch.object(dashboard, "_collect_metrics", return_value=metrics),
        patch.object(dashboard, "_get_api_status") as mock_probe,
    ):
        # Ensure the probe is NOT called when metrics_hint is present
        resp = _client(dashboard).get("/api/dashboard/state")
    assert resp.status_code == 200
    mock_probe.assert_not_called()


# ─────────────────────────── _get_bot_meta ──────────────


def test_bot_meta_empty_db_returns_safe_defaults(dashboard):
    """No session rows → default placeholders, initial_balance is None."""
    dashboard.db_manager.execute_query.return_value = []
    with patch.object(dashboard, "_get_api_status", return_value="Disconnected"):
        meta = dashboard._get_bot_meta()
    assert meta["name"] == "Unknown"
    assert meta["mode"] == "paper"
    assert meta["initial_balance"] is None
    assert meta["max_open_positions"] is None
    assert meta["status"] == "running"
    assert meta["connected"] is False


def test_bot_meta_prefers_running_session_over_stale_paper(dashboard):
    """A live session (end_time IS NULL) wins over a more-recent stopped row.

    The implementation issues two queries: first ``WHERE end_time IS NULL``,
    then a fallback. We verify the running query is consulted first.
    """
    running_row = {
        "strategy_name": "ml_basic",
        "symbol": "ETHUSDT",
        "timeframe": "1h",
        "mode": "live",
        "initial_balance": 5000.0,
        "strategy_config": None,
        "start_time": None,
        "end_time": None,
    }
    # First call (running query) returns the running row, second call shouldn't fire
    dashboard.db_manager.execute_query.side_effect = [[running_row]]
    with patch.object(dashboard, "_get_api_status", return_value="Connected"):
        meta = dashboard._get_bot_meta()
    assert meta["mode"] == "live"
    assert meta["initial_balance"] == 5000.0
    assert meta["symbols"] == ["ETHUSDT"]
    assert meta["status"] == "running"


def test_bot_meta_falls_back_to_recent_session_when_no_running(dashboard):
    """When no running session exists, the most-recent overall row is used."""
    stopped_row = {
        "strategy_name": "ml_adaptive",
        "symbol": "BTCUSDT",
        "timeframe": "4h",
        "mode": "paper",
        "initial_balance": 1000.0,
        "strategy_config": None,
        "start_time": None,
        "end_time": "2026-04-26T00:00:00Z",
    }
    # First call (running) returns nothing → fallback (recent) returns stopped
    dashboard.db_manager.execute_query.side_effect = [[], [stopped_row]]
    with patch.object(dashboard, "_get_api_status", return_value="Connected"):
        meta = dashboard._get_bot_meta()
    assert meta["status"] == "stopped"
    assert meta["mode"] == "paper"


def test_bot_meta_normalises_comma_separated_symbol_field(dashboard):
    """`symbol` may store one symbol or a comma-separated list."""
    row = {
        "strategy_name": "ml_basic",
        "symbol": "btcusdt,ethusdt , solusdt",
        "timeframe": "1h",
        "mode": "paper",
        "initial_balance": 1000.0,
        "strategy_config": None,
        "start_time": None,
        "end_time": None,
    }
    dashboard.db_manager.execute_query.side_effect = [[row]]
    with patch.object(dashboard, "_get_api_status", return_value="Connected"):
        meta = dashboard._get_bot_meta()
    assert meta["symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_bot_meta_extracts_max_open_positions_from_strategy_config(dashboard):
    """`max_open_positions` is parsed out of `strategy_config` JSON."""
    row = {
        "strategy_name": "ml_basic",
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "mode": "paper",
        "initial_balance": 1000.0,
        "strategy_config": json.dumps({"risk_manager": {"max_open_positions": 5}}),
        "start_time": None,
        "end_time": None,
    }
    dashboard.db_manager.execute_query.side_effect = [[row]]
    with patch.object(dashboard, "_get_api_status", return_value="Connected"):
        meta = dashboard._get_bot_meta()
    assert meta["max_open_positions"] == 5


def test_bot_meta_handles_db_failure_gracefully(dashboard):
    """A DB error should NOT bubble up — meta returns safe defaults."""
    dashboard.db_manager.execute_query.side_effect = RuntimeError("db down")
    with (
        patch.object(dashboard, "_get_active_symbol", return_value="BTCUSDT"),
        patch.object(dashboard, "_get_api_status", return_value="Disconnected"),
    ):
        meta = dashboard._get_bot_meta()
    assert meta["symbols"] == ["BTCUSDT"]
    assert meta["initial_balance"] is None
    assert meta["max_open_positions"] is None


def test_bot_meta_uses_metrics_hint_for_connected(dashboard):
    """`metrics_hint` short-circuits the API probe."""
    dashboard.db_manager.execute_query.side_effect = [[], []]
    with patch.object(dashboard, "_get_api_status") as probe:
        meta = dashboard._get_bot_meta(metrics_hint={"api_connection_status": "Connected"})
    probe.assert_not_called()
    assert meta["connected"] is True


def test_bot_meta_dedupes_repeated_symbols(dashboard):
    """A misconfigured `BTCUSDT,btcusdt` should emit one upper-cased symbol."""
    row = {
        "strategy_name": "ml_basic",
        "symbol": "BTCUSDT,btcusdt,ETHUSDT",
        "timeframe": "1h",
        "mode": "paper",
        "initial_balance": 1000.0,
        "strategy_config": None,
        "start_time": None,
        "end_time": None,
    }
    dashboard.db_manager.execute_query.side_effect = [[row]]
    with patch.object(dashboard, "_get_api_status", return_value="Connected"):
        meta = dashboard._get_bot_meta()
    assert meta["symbols"] == ["BTCUSDT", "ETHUSDT"]


def test_bot_meta_extracts_risk_per_trade_from_strategy_config(dashboard):
    """`risk_per_trade` should be parsed from strategy_config.risk."""
    row = {
        "strategy_name": "ml_basic",
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "mode": "paper",
        "initial_balance": 1000.0,
        "strategy_config": json.dumps({"risk": {"risk_per_trade": 0.5}}),
        "start_time": None,
        "end_time": None,
    }
    dashboard.db_manager.execute_query.side_effect = [[row]]
    with patch.object(dashboard, "_get_api_status", return_value="Connected"):
        meta = dashboard._get_bot_meta()
    assert meta["risk_per_trade"] == 0.5


def test_bot_meta_risk_per_trade_is_none_when_not_configured(dashboard):
    """When the config doesn't carry risk_per_trade, meta returns None.

    The UI then renders `—` instead of fabricating a 1.0% baseline.
    """
    row = {
        "strategy_name": "ml_basic",
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "mode": "paper",
        "initial_balance": 1000.0,
        "strategy_config": json.dumps({"risk": {}}),  # no risk_per_trade key
        "start_time": None,
        "end_time": None,
    }
    dashboard.db_manager.execute_query.side_effect = [[row]]
    with patch.object(dashboard, "_get_api_status", return_value="Connected"):
        meta = dashboard._get_bot_meta()
    assert meta["risk_per_trade"] is None
    assert meta["max_open_positions"] is None


def test_recent_trades_includes_db_id(dashboard):
    """`_get_recent_trades` must return the DB primary key.

    The V2 dashboard builds stable client-side trade IDs from this column;
    if it's missing, the inspector silently picks the wrong trade after a
    refetch.
    """
    captured = {}

    def fake_execute(query, _params=None):
        captured["query"] = query
        return [
            {
                "id": 42,
                "symbol": "BTCUSDT",
                "side": "long",
                "entry_price": 60000.0,
                "exit_price": 61000.0,
                "quantity": 0.01,
                "entry_time": None,
                "exit_time": None,
                "pnl": 10.0,
                "exit_reason": "take_profit",
            }
        ]

    dashboard.db_manager.execute_query = fake_execute
    trades = dashboard._get_recent_trades(50)
    assert "id," in captured["query"], "SELECT must include id"
    assert len(trades) == 1
    assert trades[0]["id"] == 42


def test_bot_meta_extracts_both_knobs_in_one_pass(dashboard):
    """Both max_open_positions and risk_per_trade can be parsed from same row."""
    row = {
        "strategy_name": "ml_basic",
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "mode": "paper",
        "initial_balance": 1000.0,
        "strategy_config": json.dumps(
            {"risk_manager": {"max_open_positions": 2, "risk_per_trade": 1.5}}
        ),
        "start_time": None,
        "end_time": None,
    }
    dashboard.db_manager.execute_query.side_effect = [[row]]
    with patch.object(dashboard, "_get_api_status", return_value="Connected"):
        meta = dashboard._get_bot_meta()
    assert meta["max_open_positions"] == 2
    assert meta["risk_per_trade"] == 1.5
