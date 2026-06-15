"""Contract tests for WebSocketHealthMonitor wiring (#486).

Behavioral coverage of the WS health subsystem lives in
test_trading_engine_ws_health.py and test_trading_engine_websocket.py
(exercised via the engine wrappers). These tests pin the extraction's own
contract: the engine builds the monitor bound to itself, the thin wrappers
delegate to it, and the three test-mocked sibling calls
(``_handle_kline_disconnect`` / ``_handle_user_stream_disconnect`` /
``_start_ws_health_monitor``) route back through the engine wrappers so
``patch.object(engine, ...)`` still intercepts them.
"""

from unittest.mock import MagicMock, create_autospec, patch

import pytest

from src.data_providers.data_provider import DataProvider
from src.engines.live.trading_engine import LiveTradingEngine
from src.engines.live.ws_health import WebSocketHealthMonitor
from src.strategies.ml_basic import create_ml_basic_strategy

pytestmark = pytest.mark.fast


def make_engine() -> LiveTradingEngine:
    with patch("src.engines.live.trading_engine.DatabaseManager"):
        return LiveTradingEngine(
            strategy=create_ml_basic_strategy(),
            data_provider=create_autospec(DataProvider, instance=True),
            initial_balance=1000.0,
        )


class TestWiring:
    def test_engine_builds_monitor_bound_to_itself(self):
        engine = make_engine()
        assert isinstance(engine.ws_health_monitor, WebSocketHealthMonitor)
        assert engine.ws_health_monitor._state is engine

    def test_wrappers_delegate_to_monitor(self):
        engine = make_engine()
        engine.ws_health_monitor = create_autospec(WebSocketHealthMonitor, instance=True)

        engine._check_kline_health()
        engine.ws_health_monitor.check_kline_health.assert_called_once_with()

        engine._check_user_stream_health()
        engine.ws_health_monitor.check_user_stream_health.assert_called_once_with()

        engine._drain_pending_fill_exits()
        engine.ws_health_monitor.drain_pending_fill_exits.assert_called_once_with()

    def test_static_predicate_wrappers_delegate(self):
        # The static probe predicates delegate to the monitor's static methods.
        from src.config.constants import DEFAULT_WS_USER_DEGRADED_PROBE_FIRST_BOUNDARY

        assert LiveTradingEngine._user_probe_boundary_reached(
            DEFAULT_WS_USER_DEGRADED_PROBE_FIRST_BOUNDARY
        ) == WebSocketHealthMonitor.user_probe_boundary_reached(
            DEFAULT_WS_USER_DEGRADED_PROBE_FIRST_BOUNDARY
        )


class TestMockedCrossCallRoutesThroughEngine:
    """The monitor routes its test-mocked sibling calls back through the engine
    wrappers, so a mock on the ENGINE method still intercepts them. (The full
    set is exercised by test_trading_engine_ws_health.py's 30 disconnect-mock
    tests; this is a minimal kline guard for the routing contract itself.)"""

    def test_kline_disconnect_mock_on_engine_intercepts(self):
        engine = make_engine()
        provider = MagicMock()
        provider.ws_healthy = False
        engine._ws_kline_provider = provider
        engine._ws_kline_active = True
        with patch.object(engine, "_handle_kline_disconnect") as mock_disc:
            engine._check_kline_health()
        assert mock_disc.called
