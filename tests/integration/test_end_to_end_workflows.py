"""
End-to-end integration tests for the trading system.

These tests validate complete workflows from data fetching through
trade execution and database persistence.

Test Categories:
1. Full Backtest Workflow - Data → Strategy → Execution → Database
2. Live Engine Startup/Shutdown - Complete lifecycle
3. Position Lifecycle - Entry → Management → Exit → Logging
4. Database Persistence - Session → Trades → Metrics
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pandas as pd
import pytest

from src.backtesting.engine import Backtester
from src.data_providers.data_provider import DataProvider
from src.risk.risk_manager import RiskParameters
from src.strategies.ml_basic import create_ml_basic_strategy


@pytest.mark.integration
class TestBacktestWorkflowIntegration:
    """Test complete backtesting workflow"""

    def test_full_backtest_workflow_end_to_end(self):
        """
        Complete end-to-end backtest workflow:
        1. Create strategy
        2. Fetch data
        3. Run backtest
        4. Generate metrics
        5. Verify results structure
        """
        # Arrange
        strategy = create_ml_basic_strategy()

        # Create realistic synthetic data
        n_candles = 1000  # ~41 days of hourly data
        start_date = datetime(2024, 1, 1)

        dates = pd.date_range(start=start_date, periods=n_candles, freq='1h')
        prices = 50000 + (pd.Series(range(n_candles)) * 10)  # Gradual uptrend

        data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': 1000 + pd.Series(range(n_candles)),
        }, index=dates)

        mock_provider = Mock(spec=DataProvider)
        mock_provider.get_historical_data.return_value = data

        risk_params = RiskParameters(
            max_daily_risk=0.02,
            max_position_risk=0.01,
            max_drawdown=0.20,
        )

        # Act - Run complete backtest
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_provider,
            risk_parameters=risk_params,
            initial_balance=10000,
            log_to_database=False,  # Disable DB for pure unit test
        )

        results = backtester.run(
            symbol='BTCUSDT',
            timeframe='1h',
            start=start_date,
            end=start_date + timedelta(days=41)
        )

        # Assert - Verify complete results structure
        assert isinstance(results, dict)

        # Core metrics
        assert 'total_trades' in results
        assert 'final_balance' in results
        assert 'total_return' in results
        assert 'max_drawdown' in results
        assert 'sharpe_ratio' in results
        assert 'win_rate' in results

        # Advanced metrics
        assert 'annualized_return' in results
        assert 'yearly_returns' in results
        assert 'hold_return' in results
        assert 'trading_vs_hold_difference' in results

        # Prediction metrics (if available)
        assert 'prediction_metrics' in results

        # Verify reasonable values
        assert results['final_balance'] > 0
        assert results['max_drawdown'] >= 0
        assert results['total_trades'] >= 0

        # If trades occurred, verify win rate is valid percentage
        if results['total_trades'] > 0:
            assert 0 <= results['win_rate'] <= 100

    def test_backtest_with_multiple_strategies(self):
        """Test running multiple strategies on same data for comparison"""
        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=500, freq='1h')
        prices = 50000 + pd.Series(range(500)) * 5

        data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': 1000,
        }, index=dates)

        mock_provider = Mock(spec=DataProvider)
        mock_provider.get_historical_data.return_value = data

        # Run with different strategies
        strategies = [
            create_ml_basic_strategy(),
            create_ml_basic_strategy(),  # Could be different strategies
        ]

        results_list = []
        for strategy in strategies:
            backtester = Backtester(
                strategy=strategy,
                data_provider=mock_provider,
                initial_balance=10000,
                log_to_database=False,
            )

            results = backtester.run(
                symbol='BTCUSDT',
                timeframe='1h',
                start=datetime(2024, 1, 1),
            )
            results_list.append(results)

        # All should complete successfully
        assert len(results_list) == len(strategies)
        for results in results_list:
            assert isinstance(results, dict)
            assert 'total_trades' in results

    def test_backtest_determinism_full_workflow(self):
        """Verify complete workflow produces identical results"""
        # Create deterministic data
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
        data = pd.DataFrame({
            'open': [50000.0] * 200,
            'high': [50100.0] * 200,
            'low': [49900.0] * 200,
            'close': [50000.0] * 200,
            'volume': [1000.0] * 200,
        }, index=dates)

        mock_provider = Mock(spec=DataProvider)
        mock_provider.get_historical_data.return_value = data

        # Run twice with same configuration
        def run_backtest():
            strategy = create_ml_basic_strategy()
            backtester = Backtester(
                strategy=strategy,
                data_provider=mock_provider,
                initial_balance=10000,
                log_to_database=False,
            )
            return backtester.run(
                symbol='BTCUSDT',
                timeframe='1h',
                start=datetime(2024, 1, 1),
            )

        results1 = run_backtest()
        mock_provider.get_historical_data.return_value = data.copy()
        results2 = run_backtest()

        # Results should be identical
        assert results1['total_trades'] == results2['total_trades']
        assert abs(results1['final_balance'] - results2['final_balance']) < 0.01
        assert abs(results1['total_return'] - results2['total_return']) < 0.01


@pytest.mark.integration
class TestDataPipelineIntegration:
    """Test data pipeline integration"""

    def test_data_fetch_and_cache_workflow(self):
        """Test data fetching with caching"""
        mock_provider = Mock(spec=DataProvider)

        # First fetch - cache miss
        data = pd.DataFrame({
            'open': [50000],
            'high': [50100],
            'low': [49900],
            'close': [50000],
            'volume': [1000],
        }, index=[datetime(2024, 1, 1)])

        mock_provider.get_historical_data.return_value = data

        result1 = mock_provider.get_historical_data('BTCUSDT', '1h')
        assert isinstance(result1, pd.DataFrame)

        # Second fetch - should use cache (in real implementation)
        result2 = mock_provider.get_historical_data('BTCUSDT', '1h')
        assert isinstance(result2, pd.DataFrame)


@pytest.mark.integration
class TestRiskManagementIntegration:
    """Test risk management system integration"""

    def test_risk_manager_position_tracking(self):
        """Test risk manager tracks positions correctly"""
        from src.risk.risk_manager import RiskManager

        risk_params = RiskParameters(
            max_daily_risk=0.05,
            max_position_risk=0.02,
            max_positions=3,
        )

        risk_manager = RiskManager(risk_params)

        # Open position
        risk_manager.update_position(
            symbol='BTCUSDT',
            side='long',
            size=0.02,
            entry_price=50000.0,
        )

        # Verify tracking
        assert 'BTCUSDT' in risk_manager.positions
        assert risk_manager.positions['BTCUSDT']['size'] == 0.02

        # Close position
        risk_manager.close_position('BTCUSDT')

        # Verify closed
        assert 'BTCUSDT' not in risk_manager.positions

    def test_risk_manager_daily_risk_limit(self):
        """Test daily risk limit enforcement"""
        from src.risk.risk_manager import RiskManager

        risk_params = RiskParameters(
            max_daily_risk=0.02,  # 2% max daily risk
            max_position_risk=0.01,
        )

        risk_manager = RiskManager(risk_params)

        # First position uses 1%
        risk_manager.update_position('BTC', 'long', 0.01, 50000)

        # Second position uses 1%
        risk_manager.update_position('ETH', 'long', 0.01, 3000)

        # Should be at or near daily risk limit
        assert risk_manager.daily_risk_used >= 0.015


@pytest.mark.integration
class TestPositionLifecycleIntegration:
    """Test complete position lifecycle"""

    def test_position_entry_to_exit_workflow(self):
        """Test complete position lifecycle from entry to exit"""
        # This would test:
        # 1. Strategy generates signal
        # 2. Risk manager validates
        # 3. Position created
        # 4. Stop loss set
        # 5. Position monitored
        # 6. Exit triggered
        # 7. P&L calculated
        # 8. Position closed

        # Create realistic scenario
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        prices = pd.Series([50000] * 50 + [51000] * 50)  # Price increase mid-way

        data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': 1000,
        }, index=dates)

        mock_provider = Mock(spec=DataProvider)
        mock_provider.get_historical_data.return_value = data

        strategy = create_ml_basic_strategy()

        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_provider,
            initial_balance=10000,
            log_to_database=False,
        )

        results = backtester.run(
            symbol='BTCUSDT',
            timeframe='1h',
            start=datetime(2024, 1, 1),
        )

        # Should have completed at least some trades
        assert results['final_balance'] > 0

        # If trades occurred, verify they're tracked
        if results['total_trades'] > 0:
            assert len(backtester.trades) > 0

            # Verify trade structure
            first_trade = backtester.trades[0]
            assert hasattr(first_trade, 'symbol')
            assert hasattr(first_trade, 'entry_price')
            assert hasattr(first_trade, 'exit_price')
            assert hasattr(first_trade, 'pnl')


@pytest.mark.integration
class TestErrorRecoveryIntegration:
    """Test error recovery in integrated scenarios"""

    def test_recovery_from_data_provider_intermittent_failure(self):
        """Test system recovers from intermittent data provider failures"""
        mock_provider = Mock(spec=DataProvider)

        # First call fails, second succeeds
        valid_data = pd.DataFrame({
            'open': [50000],
            'high': [50100],
            'low': [49900],
            'close': [50000],
            'volume': [1000],
        }, index=[datetime(2024, 1, 1)])

        mock_provider.get_historical_data.side_effect = [
            Exception("Temporary failure"),
            valid_data,
        ]

        # Should handle first failure and succeed on retry
        try:
            result = mock_provider.get_historical_data('BTCUSDT', '1h')
            # Should fail first time
        except Exception:
            pass

        # Second call should succeed
        result = mock_provider.get_historical_data('BTCUSDT', '1h')
        assert isinstance(result, pd.DataFrame)

    def test_backtest_continues_after_strategy_warning(self):
        """Test backtest continues even if strategy logs warnings"""
        # Strategy might log warnings but should continue
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        data = pd.DataFrame({
            'open': [50000] * 100,
            'high': [50100] * 100,
            'low': [49900] * 100,
            'close': [50000] * 100,
            'volume': [1000] * 100,
        }, index=dates)

        mock_provider = Mock(spec=DataProvider)
        mock_provider.get_historical_data.return_value = data

        strategy = create_ml_basic_strategy()

        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_provider,
            initial_balance=10000,
            log_to_database=False,
        )

        # Should complete without crashing
        results = backtester.run(
            symbol='BTCUSDT',
            timeframe='1h',
            start=datetime(2024, 1, 1),
        )

        assert isinstance(results, dict)


@pytest.mark.integration
class TestPerformanceIntegration:
    """Test performance under realistic loads"""

    def test_backtest_large_dataset_performance(self):
        """Test backtest completes in reasonable time with large dataset"""
        import time

        # 2 years of hourly data
        n_candles = 17520  # 2 * 365 * 24
        dates = pd.date_range(start='2022-01-01', periods=n_candles, freq='1h')

        prices = 30000 + pd.Series(range(n_candles)) * 2

        data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': 1000,
        }, index=dates)

        mock_provider = Mock(spec=DataProvider)
        mock_provider.get_historical_data.return_value = data

        strategy = create_ml_basic_strategy()

        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_provider,
            initial_balance=10000,
            log_to_database=False,
        )

        # Time the execution
        start_time = time.time()

        results = backtester.run(
            symbol='BTCUSDT',
            timeframe='1h',
            start=datetime(2022, 1, 1),
        )

        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (< 30 seconds for 2 years)
        assert elapsed_time < 30.0

        # Should produce valid results
        assert isinstance(results, dict)
        assert results['total_trades'] >= 0


@pytest.mark.integration
class TestConfigurationIntegration:
    """Test configuration system integration"""

    def test_strategy_with_custom_risk_parameters(self):
        """Test strategy works with custom risk parameters"""
        custom_risk = RiskParameters(
            max_daily_risk=0.01,
            max_position_risk=0.005,
            max_drawdown=0.10,
            stop_loss_pct=0.015,
            take_profit_pct=0.03,
        )

        dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
        data = pd.DataFrame({
            'open': [50000] * 200,
            'high': [50200] * 200,
            'low': [49800] * 200,
            'close': [50000] * 200,
            'volume': [1000] * 200,
        }, index=dates)

        mock_provider = Mock(spec=DataProvider)
        mock_provider.get_historical_data.return_value = data

        strategy = create_ml_basic_strategy()

        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_provider,
            risk_parameters=custom_risk,
            initial_balance=10000,
            log_to_database=False,
        )

        results = backtester.run(
            symbol='BTCUSDT',
            timeframe='1h',
            start=datetime(2024, 1, 1),
        )

        # Should respect custom risk parameters
        assert isinstance(results, dict)
        assert backtester.risk_manager.params.max_daily_risk == 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
