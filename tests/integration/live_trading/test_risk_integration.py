from unittest.mock import Mock

import pytest

from src.strategies.ml_basic import create_ml_basic_strategy

pytestmark = pytest.mark.integration

try:
    from src.live.trading_engine import LiveTradingEngine, PositionSide

    LIVE_TRADING_AVAILABLE = True
except ImportError:
    LIVE_TRADING_AVAILABLE = False

    class LiveTradingEngine:
        def __init__(self, strategy=None, data_provider=None, **kwargs):
            self.strategy = strategy
            self.data_provider = data_provider
            self.positions = {}

            class DummyRisk:
                def get_max_concurrent_positions(self):
                    return 1

            self.risk_manager = DummyRisk()

        def _open_position(self, **kwargs):
            if len(self.positions) < self.risk_manager.get_max_concurrent_positions():
                self.positions[str(len(self.positions))] = Mock(size=kwargs.get("size", 0))

    PositionSide = Mock(LONG="LONG")


class TestRiskIntegration:
    @pytest.mark.live_trading
    @pytest.mark.risk_management
    def test_risk_manager_integration(self, mock_data_provider, risk_parameters):
        if not LIVE_TRADING_AVAILABLE:
            pytest.skip("Live trading components not available")
        strategy = create_ml_basic_strategy(fast_mode=True)
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            risk_parameters=risk_parameters,
            max_position_size=0.05,
        )
        engine._open_position(symbol="BTCUSDT", side=PositionSide.LONG, size=0.5, price=50000)
        position = list(engine.positions.values())[0]
        assert position.size <= 0.05

    @pytest.mark.live_trading
    @pytest.mark.risk_management
    def test_drawdown_monitoring(self, mock_strategy, mock_data_provider):
        if not LIVE_TRADING_AVAILABLE:
            pytest.skip("Live trading components not available")
        engine = LiveTradingEngine(
            strategy=mock_strategy, data_provider=mock_data_provider, initial_balance=10000
        )
        engine.current_balance = 7000
        engine.peak_balance = 10000
        if hasattr(engine, "_update_performance_metrics") and hasattr(
            engine, "get_performance_summary"
        ):
            engine._update_performance_metrics()
            performance = engine.get_performance_summary()
            assert performance["max_drawdown_pct"] == 30.0

    @pytest.mark.live_trading
    def test_maximum_positions_limit(self, mock_strategy, mock_data_provider):
        if not LIVE_TRADING_AVAILABLE:
            pytest.skip("Live trading components not available")
        engine = LiveTradingEngine(strategy=mock_strategy, data_provider=mock_data_provider)
        max_positions = engine.risk_manager.get_max_concurrent_positions()
        for i in range(max_positions + 2):
            engine._open_position(
                symbol=f"COIN{i}USDT", side=PositionSide.LONG, size=0.01, price=1000
            )
        assert len(engine.positions) <= max_positions
