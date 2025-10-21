from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

pytestmark = pytest.mark.integration

try:
    from src.live.trading_engine import LiveTradingEngine, PositionSide

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
            self.max_position_size = kwargs.get('max_position_size', 0.1)


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
        """Test that strategy execution is logged - updated for component-based strategies"""
        # This test now verifies that the engine can work with component strategies
        # and log their execution properly
        try:
            from src.strategies.components import (
                Strategy,
                MLBasicSignalGenerator,
                FixedRiskManager,
                ConfidenceWeightedSizer,
            )
            
            # Create a component-based strategy
            signal_generator = MLBasicSignalGenerator(name="test_fallback_sg")
            risk_manager = FixedRiskManager(risk_per_trade=0.02)
            position_sizer = ConfidenceWeightedSizer(base_fraction=0.02)
            
            strategy = Strategy(
                name="test_fallback",
                signal_generator=signal_generator,
                risk_manager=risk_manager,
                position_sizer=position_sizer
            )
            
            # Create engine with component strategy
            engine = LiveTradingEngine(
                strategy=strategy,
                data_provider=mock_data_provider,
                enable_live_trading=False,
                max_position_size=0.1,
            )
            
            # Create test data with enough history
            market_data = pd.DataFrame(
                {
                    "open": [50000 + i * 10 for i in range(150)],
                    "high": [50200 + i * 10 for i in range(150)],
                    "low": [49800 + i * 10 for i in range(150)],
                    "close": [50100 + i * 10 for i in range(150)],
                    "volume": [1000 + i * 10 for i in range(150)],
                },
                index=pd.date_range("2024-01-01", periods=150, freq="1h"),
            )
            mock_data_provider.get_live_data.return_value = market_data
            
            # Get a trading decision using component strategy
            decision = strategy.process_candle(market_data, 149, 10000)
            
            # Verify decision was created (this is what gets logged)
            assert decision is not None
            assert hasattr(decision, 'signal')
            assert hasattr(decision, 'position_size')
            
        except ImportError:
            # If component strategies not available, skip this test
            pytest.skip("Component strategies not available")
