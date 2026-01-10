import pytest

from src.engines.live.strategy_manager import StrategyManager

pytestmark = pytest.mark.integration


class TestStrategyManagerIntegration:
    @pytest.mark.live_trading
    def test_strategy_manager_with_live_engine(self, temp_directory, mock_data_provider):
        from src.engines.live.trading_engine import LiveTradingEngine

        manager = StrategyManager(staging_dir=str(temp_directory))
        initial_strategy = manager.load_strategy("ml_basic")
        engine = LiveTradingEngine(
            strategy=initial_strategy, data_provider=mock_data_provider, enable_hot_swapping=True
        )
        engine.strategy_manager = manager
        manager.hot_swap_strategy("ml_basic", new_config={"sequence_length": 120})
        assert manager.has_pending_update() is True
        success = manager.apply_pending_update()
        assert success is True
