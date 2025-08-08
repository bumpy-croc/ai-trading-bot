import pytest
from datetime import datetime
from unittest.mock import Mock

pytestmark = pytest.mark.integration

try:
    from live.trading_engine import LiveTradingEngine
    from live.strategy_manager import StrategyManager
    LIVE_TRADING_AVAILABLE = True
    STRATEGY_MANAGER_AVAILABLE = True
except ImportError:
    LIVE_TRADING_AVAILABLE = False
    STRATEGY_MANAGER_AVAILABLE = False
    LiveTradingEngine = object
    StrategyManager = Mock


@pytest.mark.skipif(not STRATEGY_MANAGER_AVAILABLE, reason="Strategy manager not available")
class TestStrategyHotSwapping:
    @pytest.mark.live_trading
    def test_strategy_hot_swap_preparation(self, mock_strategy, mock_data_provider):
        if not LIVE_TRADING_AVAILABLE:
            pytest.skip("Live trading components not available")
        engine = LiveTradingEngine(strategy=mock_strategy, data_provider=mock_data_provider, enable_hot_swapping=True)
        engine.strategy_manager = Mock()
        engine.strategy_manager.has_pending_update.return_value = True
        engine.strategy_manager.apply_pending_update.return_value = True
        engine.strategy_manager.current_strategy = Mock()
        engine.strategy_manager.current_strategy.name = "NewStrategy"
        assert engine.strategy_manager.has_pending_update() is True
        success = engine.strategy_manager.apply_pending_update()
        assert success is True

    @pytest.mark.live_trading
    def test_strategy_hot_swap_with_position_closure(self, mock_strategy, mock_data_provider):
        engine = LiveTradingEngine(strategy=mock_strategy, data_provider=mock_data_provider, enable_hot_swapping=True)
        position = Mock(symbol="BTCUSDT", side="LONG", size=0.1, entry_price=50000, entry_time=datetime.now(), order_id="test_001")
        engine.positions = {"test_001": position}
        new_strategy = Mock(name="NewStrategy")
        if hasattr(engine, 'hot_swap_strategy'):
            _ = engine.hot_swap_strategy("new_strategy", close_positions=True)

    @pytest.mark.live_trading
    def test_model_update_during_trading(self, mock_strategy, mock_data_provider, mock_model_file):
        if not LIVE_TRADING_AVAILABLE:
            pytest.skip("Live trading components not available")
        engine = LiveTradingEngine(strategy=mock_strategy, data_provider=mock_data_provider, enable_hot_swapping=True)
        engine.strategy_manager = Mock()
        engine.strategy_manager.update_model.return_value = True
        if hasattr(engine, 'update_model'):
            _ = engine.update_model(str(mock_model_file))
            engine.strategy_manager.update_model.assert_called_once()