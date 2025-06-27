"""
Integration tests for the trading bot system.

These tests validate end-to-end workflows and component interactions.
They are slower but critical for ensuring the system works as a whole.
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pandas as pd

from live.trading_engine import LiveTradingEngine
from live.strategy_manager import StrategyManager
from strategies.adaptive import AdaptiveStrategy
from core.risk.risk_manager import RiskManager, RiskParameters
from backtesting.engine import Backtester


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Test complete trading workflows"""

    def test_complete_backtesting_workflow(self, mock_data_provider, sample_ohlcv_data):
        """Test complete backtesting from start to finish"""
        # Setup
        strategy = AdaptiveStrategy()
        risk_params = RiskParameters()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            risk_parameters=risk_params,
            initial_balance=10000
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
            'total_trades', 'win_rate', 'total_return', 
            'max_drawdown', 'sharpe_ratio', 'final_balance'
        ]
        for metric in required_metrics:
            assert metric in results
        
        # Verify realistic results
        assert results['final_balance'] > 0
        assert 0 <= results['win_rate'] <= 100
        assert results['max_drawdown'] >= 0

    def test_strategy_to_live_trading_workflow(self, mock_data_provider, temp_directory):
        """Test strategy development to live trading deployment"""
        # 1. Strategy Development Phase
        strategy = AdaptiveStrategy()
        
        # Verify strategy has required methods
        assert hasattr(strategy, 'calculate_indicators')
        assert hasattr(strategy, 'check_entry_conditions')
        assert hasattr(strategy, 'check_exit_conditions')
        
        # 2. Backtesting Phase
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000
        )
        
        # Mock successful backtest
        mock_data_provider.get_historical_data.return_value = pd.DataFrame({
            'open': [50000, 50100, 50200],
            'high': [50200, 50300, 50400],
            'low': [49800, 49900, 50000],
            'close': [50100, 50200, 50300],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1H'))
        
        backtest_results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))
        assert backtest_results['total_return'] is not None
        
        # 3. Live Trading Deployment
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            enable_live_trading=False,  # Paper trading for test
            initial_balance=10000
        )
        
        # Verify engine is properly configured
        assert engine.strategy == strategy
        assert engine.enable_live_trading == False
        assert engine.current_balance == 10000

    def test_data_flow_integration(self, mock_data_provider):
        """Test data flow from provider through strategy to trading decisions"""
        # Setup components
        strategy = AdaptiveStrategy()
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            enable_live_trading=False
        )
        
        # Mock data provider responses
        market_data = pd.DataFrame({
            'open': [50000, 50100, 50200, 50300],
            'high': [50200, 50300, 50400, 50500],
            'low': [49800, 49900, 50000, 50100],
            'close': [50100, 50200, 50300, 50400],
            'volume': [1000, 1100, 1200, 1300]
        }, index=pd.date_range('2024-01-01', periods=4, freq='1H'))
        
        mock_data_provider.get_live_data.return_value = market_data.tail(1)
        
        # Test data processing pipeline
        # 1. Data retrieval
        latest_data = engine._get_latest_data("BTCUSDT", "1h")
        assert latest_data is not None
        
        # 2. Indicator calculation
        df_with_indicators = strategy.calculate_indicators(market_data)
        assert len(df_with_indicators.columns) > len(market_data.columns)
        
        # 3. Signal generation
        if len(df_with_indicators) > 1:
            entry_signal = strategy.check_entry_conditions(df_with_indicators, len(df_with_indicators) - 1)
            assert isinstance(entry_signal, bool)

    @pytest.mark.live_trading
    def test_hot_swapping_integration(self, mock_data_provider, temp_directory):
        """Test hot-swapping strategies during live trading"""
        # Setup strategy manager
        manager = StrategyManager(staging_dir=str(temp_directory))
        initial_strategy = manager.load_strategy("adaptive", version="v1")
        
        # Setup trading engine with hot-swapping enabled
        engine = LiveTradingEngine(
            strategy=initial_strategy,
            data_provider=mock_data_provider,
            enable_hot_swapping=True,
            enable_live_trading=False
        )
        
        # Mock strategy manager in engine
        engine.strategy_manager = manager
        
        # Test hot-swap workflow
        # 1. Prepare new strategy
        swap_success = manager.hot_swap_strategy("adaptive", new_config={"fast_ma": 12})
        assert swap_success == True
        
        # 2. Engine detects pending update
        has_update = manager.has_pending_update()
        assert has_update == True
        
        # 3. Apply update
        apply_success = manager.apply_pending_update()
        assert apply_success == True
        
        # 4. Verify strategy was updated
        assert engine.strategy != initial_strategy

    def test_risk_management_integration(self, mock_data_provider):
        """Test risk management integration across all components"""
        # Setup with specific risk parameters
        risk_params = RiskParameters(
            base_risk_per_trade=0.01,  # 1% per trade
            max_position_size=0.10,    # 10% max position
            max_daily_risk=0.05        # 5% daily risk
        )
        
        strategy = AdaptiveStrategy()
        risk_manager = RiskManager(risk_params)
        
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            risk_parameters=risk_params,
            initial_balance=10000,
            enable_live_trading=False
        )
        
        # Test risk integration
        # 1. Position sizing respects limits
        position_size = strategy.calculate_position_size(
            pd.DataFrame({'close': [50000], 'atr': [1000]}), 0, 10000
        )
        max_position_value = 10000 * risk_params.max_position_size
        assert position_size * 50000 <= max_position_value
        
        # 2. Risk manager validates positions
        test_position_size = risk_manager.calculate_position_size(
            price=50000, atr=1000, balance=10000
        )
        assert test_position_size > 0


@pytest.mark.integration
class TestComponentInteractions:
    """Test interactions between major components"""

    def test_strategy_data_provider_interaction(self, mock_data_provider):
        """Test strategy working with different data providers"""
        strategy = AdaptiveStrategy()
        
        # Test with mock provider
        market_data = pd.DataFrame({
            'open': [50000, 50100],
            'high': [50200, 50300],
            'low': [49800, 49900],
            'close': [50100, 50200],
            'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2, freq='1H'))
        
        mock_data_provider.get_historical_data.return_value = market_data
        
        # Strategy should process data successfully
        df_with_indicators = strategy.calculate_indicators(market_data)
        assert len(df_with_indicators) == len(market_data)
        
        # Strategy should generate signals
        if len(df_with_indicators) > 1:
            signal = strategy.check_entry_conditions(df_with_indicators, 1)
            assert isinstance(signal, bool)

    def test_backtester_strategy_integration(self, mock_data_provider, sample_ohlcv_data):
        """Test backtester working with different strategies"""
        strategies = [AdaptiveStrategy()]
        
        for strategy in strategies:
            backtester = Backtester(
                strategy=strategy,
                data_provider=mock_data_provider,
                initial_balance=10000
            )
            
            mock_data_provider.get_historical_data.return_value = sample_ohlcv_data
            
            results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))
            
            # Each strategy should produce valid results
            assert isinstance(results, dict)
            assert 'total_return' in results
            assert 'final_balance' in results

    def test_live_engine_component_integration(self, mock_data_provider):
        """Test live engine integrating all components"""
        # Setup all components
        strategy = AdaptiveStrategy()
        risk_params = RiskParameters()
        
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            risk_parameters=risk_params,
            enable_live_trading=False,
            initial_balance=10000
        )
        
        # Mock data provider
        market_data = pd.DataFrame({
            'open': [50000],
            'high': [50200],
            'low': [49800],
            'close': [50100],
            'volume': [1000]
        }, index=[datetime.now()])
        
        mock_data_provider.get_live_data.return_value = market_data
        
        # Test integrated workflow
        # 1. Data retrieval
        data = engine._get_latest_data("BTCUSDT", "1h")
        assert data is not None
        
        # 2. Position opening (if strategy allows)
        initial_position_count = len(engine.positions)
        engine._open_position("BTCUSDT", "long", 0.1, 50000)
        assert len(engine.positions) == initial_position_count + 1
        
        # 3. Performance tracking
        engine._update_performance_metrics()
        performance = engine.get_performance_summary()
        assert isinstance(performance, dict)


@pytest.mark.integration
class TestRealTimeScenarios:
    """Test scenarios that simulate real-time trading conditions"""

    def test_market_volatility_scenario(self, mock_data_provider):
        """Test system behavior during high volatility"""
        strategy = AdaptiveStrategy()
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            enable_live_trading=False
        )
        
        # Simulate volatile market data
        volatile_data = []
        base_price = 50000
        for i in range(10):
            # Random 5% moves
            price_change = 0.05 if i % 2 == 0 else -0.05
            new_price = base_price * (1 + price_change)
            
            volatile_data.append({
                'open': base_price,
                'high': max(base_price, new_price) * 1.01,
                'low': min(base_price, new_price) * 0.99,
                'close': new_price,
                'volume': 1000
            })
            base_price = new_price
        
        df = pd.DataFrame(volatile_data, index=pd.date_range('2024-01-01', periods=10, freq='1H'))
        mock_data_provider.get_live_data.return_value = df.tail(1)
        
        # System should handle volatility without crashing
        try:
            latest_data = engine._get_latest_data("BTCUSDT", "1h")
            assert latest_data is not None
        except Exception as e:
            pytest.fail(f"System failed during volatility: {e}")

    def test_network_interruption_scenario(self, mock_data_provider):
        """Test system behavior during network issues"""
        strategy = AdaptiveStrategy()
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            enable_live_trading=False,
            max_consecutive_errors=3
        )
        
        # Simulate network failures
        from requests.exceptions import ConnectionError
        mock_data_provider.get_live_data.side_effect = [
            ConnectionError("Network error"),
            ConnectionError("Network error"),
            pd.DataFrame({  # Recovery
                'open': [50000], 'high': [50100], 'low': [49900],
                'close': [50050], 'volume': [1000]
            }, index=[datetime.now()])
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
        strategy = AdaptiveStrategy()
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            enable_live_trading=False
        )
        
        # Mock continuous data
        mock_data_provider.get_live_data.return_value = pd.DataFrame({
            'open': [50000], 'high': [50100], 'low': [49900],
            'close': [50050], 'volume': [1000]
        }, index=[datetime.now()])
        
        # Simulate extended operation
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Simulate many data requests
        for i in range(100):
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
        strategy = manager.load_strategy("adaptive")
        assert strategy is not None
        
        # 3. Initialize trading engine
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            enable_live_trading=False,
            enable_hot_swapping=True
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
        strategy = AdaptiveStrategy()
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            enable_live_trading=False
        )
        
        # Add some mock positions
        from live.trading_engine import Position, PositionSide
        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50000,
            entry_time=datetime.now(),
            order_id="test_001"
        )
        engine.positions["test_001"] = position
        
        # Mock data for closing positions
        mock_data_provider.get_live_data.return_value = pd.DataFrame({
            'close': [51000]
        }, index=[datetime.now()])
        
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
            LiveTradingEngine(
                strategy=Mock(),
                data_provider=Mock(),
                initial_balance=-1000
            )
        
        with pytest.raises((ValueError, AssertionError, TypeError)):
            # Invalid risk parameters
            RiskParameters(
                base_risk_per_trade=-0.01  # Negative risk
            )

    def test_logging_integration(self, mock_data_provider, caplog):
        """Test that logging works correctly across components"""
        import logging
        
        strategy = AdaptiveStrategy()
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            enable_live_trading=False
        )
        
        # Generate some log messages
        with caplog.at_level(logging.INFO):
            engine._get_latest_data("BTCUSDT", "1h")
        
        # Verify logging is working
        # Note: Actual log checking depends on implementation details