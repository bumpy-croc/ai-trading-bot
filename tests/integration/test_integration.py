"""
Integration tests for the trading bot system.

These tests validate end-to-end workflows and component interactions.
They are slower but critical for ensuring the system works as a whole.
"""

from datetime import datetime
from unittest.mock import Mock

import pandas as pd
import pytest

from src.engines.backtest.engine import Backtester
from src.engines.live.strategy_manager import StrategyManager
from src.engines.live.trading_engine import LiveTradingEngine
from src.risk.risk_manager import RiskManager, RiskParameters
from src.strategies.components import SignalDirection
from src.strategies.ml_basic import create_ml_basic_strategy

pytestmark = pytest.mark.integration


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Test complete trading workflows"""

    def test_complete_backtesting_workflow(self, mock_data_provider, sample_ohlcv_data):
        """Test complete backtesting from start to finish"""
        # Setup
        strategy = create_ml_basic_strategy(fast_mode=True)
        risk_params = RiskParameters()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            risk_parameters=risk_params,
            initial_balance=10000,
        )

        # Mock data provider to return sample data
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data

        # Run backtest
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 7)

        results = backtester.run("BTCUSDT", "1h", start_date, end_date)

        # Verify complete results structure
        assert isinstance(results, dict)
        required_metrics = [
            "total_trades",
            "win_rate",
            "total_return",
            "max_drawdown",
            "sharpe_ratio",
            "final_balance",
        ]
        for metric in required_metrics:
            assert metric in results

        # Verify realistic results
        assert results["final_balance"] > 0
        assert 0 <= results["win_rate"] <= 100
        assert results["max_drawdown"] >= 0

    def test_strategy_to_live_trading_workflow(self, mock_data_provider, temp_directory):
        """Test strategy development to live trading deployment"""
        # 1. Strategy Development Phase
        strategy = create_ml_basic_strategy(fast_mode=True)

        # Verify strategy has required component-based interface
        assert hasattr(strategy, "process_candle")
        assert hasattr(strategy, "should_exit_position")
        assert hasattr(strategy, "get_stop_loss_price")

        # 2. Backtesting Phase
        backtester = Backtester(
            strategy=strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        # Mock successful backtest
        mock_data_provider.get_historical_data.return_value = pd.DataFrame(
            {
                "open": [50000, 50100, 50200],
                "high": [50200, 50300, 50400],
                "low": [49800, 49900, 50000],
                "close": [50100, 50200, 50300],
                "volume": [1000, 1100, 1200],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="1h"),
        )

        backtest_results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))
        assert backtest_results["total_return"] is not None

        # 3. Live Trading Deployment
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            enable_live_trading=False,  # Paper trading for test
            initial_balance=10000,
        )

        # Verify engine is properly configured
        assert engine.strategy == strategy
        assert not engine.enable_live_trading
        assert engine.current_balance == 10000

    def test_data_flow_integration(self, mock_data_provider):
        """Test data flow from provider through strategy to trading decisions"""
        # Setup components
        strategy = create_ml_basic_strategy(fast_mode=True)
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            enable_live_trading=False,
        )

        # Mock data provider responses
        market_data = pd.DataFrame(
            {
                "open": [50000, 50100, 50200, 50300],
                "high": [50200, 50300, 50400, 50500],
                "low": [49800, 49900, 50000, 50100],
                "close": [50100, 50200, 50300, 50400],
                "volume": [1000, 1100, 1200, 1300],
            },
            index=pd.date_range("2024-01-01", periods=4, freq="1h"),
        )

        mock_data_provider.get_live_data.return_value = market_data.tail(1)

        # Test data processing pipeline
        # 1. Data retrieval
        latest_data = engine._get_latest_data("BTCUSDT", "1h")
        assert latest_data is not None

        # 2. Strategy decision pipeline via component interface
        decision = strategy.process_candle(market_data, len(market_data) - 1, balance=10000.0)
        assert decision.signal.direction in {
            SignalDirection.BUY,
            SignalDirection.SELL,
            SignalDirection.HOLD,
        }
        assert decision.position_size >= 0.0
        assert isinstance(decision.metadata, dict)

    @pytest.mark.live_trading
    def test_hot_swapping_integration(self, mock_data_provider, temp_directory):
        """Test hot-swapping strategies during live trading"""
        # Setup strategy manager
        manager = StrategyManager(staging_dir=str(temp_directory))
        initial_strategy = manager.load_strategy("ml_basic", version="v1")

        # Setup trading engine with hot-swapping enabled
        engine = LiveTradingEngine(
            strategy=initial_strategy,
            data_provider=mock_data_provider,
            enable_hot_swapping=True,
            enable_live_trading=False,
        )

        # Mock strategy manager in engine
        engine.strategy_manager = manager

        # Test hot-swap workflow
        # 1. Prepare new strategy
        swap_success = manager.hot_swap_strategy("ml_basic", new_config={"sequence_length": 120})
        assert swap_success

        # 2. Engine detects pending update
        has_update = manager.has_pending_update()
        assert has_update

        # 3. Apply update
        apply_success = manager.apply_pending_update()
        assert apply_success

        # 4. Update engine's strategy reference (simulating trading loop behavior)
        engine.strategy = manager.current_strategy

        # 5. Verify strategy was updated
        assert engine.strategy != initial_strategy

    def test_risk_management_integration(self, mock_data_provider):
        """Test risk management integration across all components"""
        # Setup with specific risk parameters
        risk_params = RiskParameters(
            base_risk_per_trade=0.01,  # 1% per trade
            max_position_size=0.10,  # 10% max position
            max_daily_risk=0.05,  # 5% daily risk
        )

        strategy = create_ml_basic_strategy()
        risk_manager = RiskManager(risk_params)

        LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            risk_parameters=risk_params,
            initial_balance=10000,
            enable_live_trading=False,
        )

        # Test risk integration
        # 1. Risk manager calculates appropriate position sizes
        test_position_size = risk_manager.calculate_position_size(
            price=50000,
            atr=2500,
            balance=10000,  # Large ATR for smaller position
        )
        assert test_position_size > 0

        # 2. Position size respects maximum position limit
        max_position_value = 10000 * risk_params.max_position_size
        actual_position_value = test_position_size * 50000
        assert actual_position_value <= max_position_value

        # 3. Risk manager validates drawdown properly
        excessive_drawdown = risk_manager.check_drawdown(current_balance=7000, peak_balance=10000)
        assert excessive_drawdown  # 30% drawdown should trigger

        acceptable_drawdown = risk_manager.check_drawdown(current_balance=9000, peak_balance=10000)
        assert not acceptable_drawdown  # 10% drawdown should be acceptable


@pytest.mark.integration
class TestComponentInteractions:
    """Test interactions between major components"""

    def test_strategy_data_provider_interaction(self, mock_data_provider):
        """Test strategy working with different data providers"""
        strategy = create_ml_basic_strategy()

        # Test with mock provider
        market_data = pd.DataFrame(
            {
                "open": [50000, 50100],
                "high": [50200, 50300],
                "low": [49800, 49900],
                "close": [50100, 50200],
                "volume": [1000, 1100],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="1h"),
        )

        mock_data_provider.get_historical_data.return_value = market_data

        fetched_data = mock_data_provider.get_historical_data("BTCUSDT", "1h")
        assert fetched_data.equals(market_data)

        decision = strategy.process_candle(fetched_data, len(fetched_data) - 1, balance=10000.0)
        assert decision.signal.direction in {
            SignalDirection.BUY,
            SignalDirection.SELL,
            SignalDirection.HOLD,
        }
        assert decision.position_size >= 0.0

    def test_backtester_strategy_integration(self, mock_data_provider, sample_ohlcv_data):
        """Test backtester working with different strategies"""
        strategies = [create_ml_basic_strategy()]
        for strategy in strategies:
            backtester = Backtester(
                strategy=strategy, data_provider=mock_data_provider, initial_balance=10000
            )

            mock_data_provider.get_historical_data.return_value = sample_ohlcv_data

            results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

            # Each strategy should produce valid results
            assert isinstance(results, dict)
            assert "total_return" in results
            assert "final_balance" in results

    def test_live_engine_component_integration(self, mock_data_provider):
        """Test live engine integrating all components"""
        # Setup all components
        strategy = create_ml_basic_strategy()
        risk_params = RiskParameters()

        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            risk_parameters=risk_params,
            enable_live_trading=False,
            initial_balance=10000,
        )

        # Mock data provider
        market_data = pd.DataFrame(
            {"open": [50000], "high": [50200], "low": [49800], "close": [50100], "volume": [1000]},
            index=[datetime.now()],
        )

        mock_data_provider.get_live_data.return_value = market_data

        # Test integrated workflow
        # 1. Data retrieval
        data = engine._get_latest_data("BTCUSDT", "1h")
        assert data is not None

        # 2. Position opening (if strategy allows)
        from src.engines.live.trading_engine import PositionSide

        initial_position_count = len(engine.live_position_tracker._positions)
        engine._execute_entry(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            price=50000,
            stop_loss=None,
            take_profit=None,
            signal_strength=0.0,
            signal_confidence=0.0,
        )
        assert len(engine.live_position_tracker._positions) == initial_position_count + 1

        # 3. Performance tracking
        engine._update_performance_metrics()
        performance = engine.get_performance_summary()
        assert isinstance(performance, dict)


@pytest.mark.integration
class TestRealTimeScenarios:
    """Test scenarios that simulate real-time trading conditions"""

    def test_market_volatility_scenario(self, mock_data_provider):
        """Test system behavior during high volatility"""
        strategy = create_ml_basic_strategy()
        engine = LiveTradingEngine(
            strategy=strategy, data_provider=mock_data_provider, enable_live_trading=False
        )

        # Simulate volatile market data
        volatile_data = []
        base_price = 50000
        for i in range(10):
            # Random 5% moves
            price_change = 0.05 if i % 2 == 0 else -0.05
            new_price = base_price * (1 + price_change)

            volatile_data.append(
                {
                    "open": base_price,
                    "high": max(base_price, new_price) * 1.01,
                    "low": min(base_price, new_price) * 0.99,
                    "close": new_price,
                    "volume": 1000,
                }
            )
            base_price = new_price

        df = pd.DataFrame(volatile_data, index=pd.date_range("2024-01-01", periods=10, freq="1h"))
        mock_data_provider.get_live_data.return_value = df.tail(1)

        # System should handle volatility without crashing
        try:
            latest_data = engine._get_latest_data("BTCUSDT", "1h")
            assert latest_data is not None
        except Exception as e:
            pytest.fail(f"System failed during volatility: {e}")

    def test_network_interruption_scenario(self, mock_data_provider):
        """Test system behavior during network issues"""
        strategy = create_ml_basic_strategy()
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            enable_live_trading=False,
            max_consecutive_errors=3,
        )

        # Simulate network failures
        from requests.exceptions import ConnectionError

        mock_data_provider.get_live_data.side_effect = [
            ConnectionError("Network error"),
            ConnectionError("Network error"),
            pd.DataFrame(
                {  # Recovery
                    "open": [50000],
                    "high": [50100],
                    "low": [49900],
                    "close": [50050],
                    "volume": [1000],
                },
                index=[datetime.now()],
            ),
        ]

        # Test error handling
        # First two calls should fail
        result1 = engine._get_latest_data("BTCUSDT", "1h")
        assert result1 is None

        result2 = engine._get_latest_data("BTCUSDT", "1h")
        assert result2 is None

        # Third call should succeed
        result3 = engine._get_latest_data("BTCUSDT", "1h")
        assert result3 is not None

    def test_memory_usage_during_extended_operation(self, mock_data_provider):
        """Test memory usage during extended operation"""
        strategy = create_ml_basic_strategy()
        engine = LiveTradingEngine(
            strategy=strategy, data_provider=mock_data_provider, enable_live_trading=False
        )

        # Mock continuous data
        mock_data_provider.get_live_data.return_value = pd.DataFrame(
            {"open": [50000], "high": [50100], "low": [49900], "close": [50050], "volume": [1000]},
            index=[datetime.now()],
        )

        # Simulate extended operation
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Simulate many data requests
        for _i in range(100):
            engine._get_latest_data("BTCUSDT", "1h")

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 50MB for this test)
        assert memory_increase < 50 * 1024 * 1024


@pytest.mark.integration
class TestProductionReadiness:
    """Test production readiness scenarios"""

    def test_system_startup_sequence(self, mock_data_provider, temp_directory):
        """Test complete system startup sequence"""
        # 1. Initialize strategy manager
        manager = StrategyManager(staging_dir=str(temp_directory))

        # 2. Load strategy
        strategy = manager.load_strategy("ml_basic")
        assert strategy is not None

        # 3. Initialize trading engine
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            enable_live_trading=False,
            enable_hot_swapping=True,
        )

        # 4. Connect strategy manager
        engine.strategy_manager = manager

        # 5. Verify system is ready
        assert engine.strategy is not None
        assert engine.data_provider is not None
        assert engine.strategy_manager is not None

        # System should be ready to start
        assert not engine.is_running

    def test_graceful_shutdown_sequence(self, mock_data_provider):
        """Test graceful shutdown of all components"""
        strategy = create_ml_basic_strategy()
        engine = LiveTradingEngine(
            strategy=strategy, data_provider=mock_data_provider, enable_live_trading=False
        )

        # Mock data for closing positions - make sure the mock is configured properly
        mock_data_provider.get_live_data.return_value = pd.DataFrame(
            {"close": [51000]}, index=[datetime.now()]
        )

        # Add some mock positions
        from src.engines.live.trading_engine import Position, PositionSide

        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50000,
            entry_time=datetime.now(),
            order_id="test_001",
        )
        engine.positions["test_001"] = position

        # Set engine as "running" initially so stop() can work properly
        engine.is_running = True

        # Test shutdown
        engine.stop()

        # Verify clean shutdown
        assert len(engine.positions) == 0
        assert not engine.is_running

    def test_configuration_validation(self):
        """Test that configurations are properly validated"""
        # Test invalid configurations
        with pytest.raises((ValueError, AssertionError)):
            # Negative initial balance
            LiveTradingEngine(strategy=Mock(), data_provider=Mock(), initial_balance=-1000)

        with pytest.raises((ValueError, AssertionError, TypeError)):
            # Invalid risk parameters
            RiskParameters(base_risk_per_trade=-0.01)  # Negative risk

    def test_logging_integration(self, mock_data_provider, caplog):
        """Test that logging works correctly across components"""
        import logging

        strategy = create_ml_basic_strategy(fast_mode=True)
        engine = LiveTradingEngine(
            strategy=strategy, data_provider=mock_data_provider, enable_live_trading=False
        )

        # Generate some log messages
        with caplog.at_level(logging.INFO):
            engine._get_latest_data("BTCUSDT", "1h")

        # Verify logging is working
        # Note: Actual log checking depends on implementation details
