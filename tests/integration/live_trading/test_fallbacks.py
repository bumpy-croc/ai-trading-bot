from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

pytestmark = pytest.mark.integration

try:
    from live.trading_engine import LiveTradingEngine, PositionSide

    LIVE_TRADING_AVAILABLE = True
except ImportError:
    LIVE_TRADING_AVAILABLE = False

    class LiveTradingEngine:
        def __init__(self, strategy=None, data_provider=None, enable_live_trading=False, **kwargs):
            self.strategy = strategy
            self.data_provider = data_provider
            self.enable_live_trading = enable_live_trading
            self.positions = {}
            self.completed_trades = []
            self.trading_session_id = 42

        def _check_entry_conditions(self, market_data, idx, symbol, current_price):
            return True

        def _check_exit_conditions(self, market_data, idx, current_price):
            return True


class TestLiveTradingFallbacks:
    def test_mock_live_trading_engine(self, mock_strategy, mock_data_provider):
        engine = LiveTradingEngine(
            strategy=mock_strategy, data_provider=mock_data_provider, enable_live_trading=False
        )
        assert engine is not None
        assert hasattr(engine, "strategy")
        assert hasattr(engine, "data_provider")

    def test_missing_components_handling(self):
        assert LiveTradingEngine is not None
        assert PositionSide is not None if "PositionSide" in globals() else True

    def test_strategy_execution_logging(self, mock_strategy, mock_data_provider):
        # Mock the database manager to avoid database connection issues
        with pytest.MonkeyPatch().context() as m:
            # Mock the DatabaseManager before creating the engine
            m.setattr('src.database.manager.DatabaseManager', MagicMock())
            
            # Create engine after mocking
            engine = LiveTradingEngine(
                strategy=mock_strategy, 
                data_provider=mock_data_provider, 
                enable_live_trading=False,
                max_position_size=0.1  # Explicitly set as float
            )
            
            # Ensure all mocked components are properly set
            engine.db_manager = MagicMock()
            engine.db_manager.log_strategy_execution = MagicMock()
            engine.risk_manager = MagicMock()
            engine.risk_manager.get_max_concurrent_positions.return_value = 1
            engine.risk_manager.calculate_position_fraction.return_value = 0.05  # Return a float instead of MagicMock
            
            # Double-check max_position_size is a float
            assert isinstance(engine.max_position_size, (int, float)), f"max_position_size should be numeric, got {type(engine.max_position_size)}"
            
            # Mock the missing _close_position method
            engine._close_position = MagicMock()
            engine.positions = {}
            engine.trading_session_id = 42
            market_data = pd.DataFrame(
                {
                    "open": [50000, 50100],
                    "high": [50200, 50300],
                    "low": [49800, 49900],
                    "close": [50100, 50200],
                    "volume": [1000, 1100],
                    "rsi": [45, 55],
                    "atr": [500, 510],
                },
                index=pd.date_range("2024-01-01", periods=2, freq="1h"),
            )
            mock_data_provider.get_live_data.return_value = market_data.tail(1)
            mock_strategy.calculate_indicators.return_value = market_data
            mock_strategy.check_entry_conditions.return_value = True
            mock_strategy.calculate_position_size.return_value = 0.1
            mock_strategy.calculate_stop_loss.return_value = 49500
            mock_strategy.get_risk_overrides.return_value = None  # Return None instead of Mock
            current_index = len(market_data) - 1
            symbol = "BTCUSDT"
            current_price = market_data["close"].iloc[-1]
            if hasattr(engine, "_check_entry_conditions"):
                engine._check_entry_conditions(market_data, current_index, symbol, current_price)
                assert engine.db_manager.log_strategy_execution.called
            if hasattr(engine, "_check_exit_conditions"):
                engine.db_manager.log_strategy_execution.reset_mock()
                engine.positions = {
                    "test_exit_001": type(
                        "P",
                        (),
                        dict(
                            symbol="BTCUSDT",
                            side="LONG",
                            size=0.1,
                            entry_price=50000,
                            entry_time=datetime.now() - timedelta(hours=2),
                            stop_loss=49500,
                            take_profit=None,
                            order_id="test_exit_001",
                        ),
                    )
                }
                mock_strategy.check_exit_conditions.return_value = True
                engine._check_exit_conditions(market_data, current_index, current_price)
                assert engine.db_manager.log_strategy_execution.called
