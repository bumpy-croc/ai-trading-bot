"""Integration tests for dynamic risk management in backtesting"""

from datetime import datetime

import pandas as pd
import pytest

from src.backtesting.engine import Backtester
from src.position_management.dynamic_risk import DynamicRiskConfig
from src.strategies.components.strategy import Strategy
from src.strategies.components.signal_generator import SignalGenerator, Signal, SignalDirection
from src.strategies.components.risk_manager import RiskManager
from src.strategies.components.position_sizer import PositionSizer

pytestmark = pytest.mark.integration


class MockSignalGenerator(SignalGenerator):
    """Mock signal generator that never signals"""

    def __init__(self):
        super().__init__(name="mock_signal")

    def generate_signal(self, df: pd.DataFrame, index: int, regime=None) -> Signal:
        return Signal(
            direction=SignalDirection.HOLD,
            confidence=0.0,
            strength=0.0,
            metadata={"timestamp": df.index[index] if len(df) > index else datetime.now()},
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        return 0.0


class MockRiskManager(RiskManager):
    """Mock risk manager with fixed risk"""

    def __init__(self):
        super().__init__(name="mock_risk")

    def calculate_position_size(self, signal: Signal, balance: float, regime=None) -> float:
        return 0.05 * balance

    def should_exit(self, position, current_data, regime=None) -> bool:
        return False

    def get_stop_loss(self, entry_price: float, signal: Signal, regime=None) -> float:
        return entry_price * 0.95


class MockPositionSizer(PositionSizer):
    """Mock position sizer with fixed fraction"""

    def __init__(self):
        super().__init__(name="mock_sizer")

    def calculate_size(
        self, signal: Signal, balance: float, risk_amount: float, regime=None
    ) -> float:
        return 0.05


def create_mock_strategy() -> Strategy:
    """Create a mock component-based strategy for testing"""
    return Strategy(
        name="MockStrategy",
        signal_generator=MockSignalGenerator(),
        risk_manager=MockRiskManager(),
        position_sizer=MockPositionSizer(),
    )


class MockDataProvider:
    """Mock data provider for testing"""

    def get_historical_data(self, symbol, timeframe, start, end):
        # Return empty DataFrame for quick test
        return pd.DataFrame()


class TestBacktestingDynamicRiskIntegration:
    """Integration tests for dynamic risk management in backtesting"""

    def test_backtester_dynamic_risk_enabled_by_default(self):
        """Dynamic risk should follow the live engine defaults."""
        backtester = Backtester(
            strategy=create_mock_strategy(), data_provider=MockDataProvider(), log_to_database=False
        )

        assert backtester.enable_dynamic_risk is True
        assert backtester.dynamic_risk_manager is not None

    def test_backtester_can_disable_dynamic_risk(self):
        """Strategies can still opt out of dynamic risk sizing."""
        backtester = Backtester(
            strategy=create_mock_strategy(),
            data_provider=MockDataProvider(),
            enable_dynamic_risk=False,
            log_to_database=False,
        )

        assert backtester.enable_dynamic_risk is False
        assert backtester.dynamic_risk_manager is None

    def test_backtester_with_dynamic_risk_enabled(self):
        """Test backtester creation with explicit dynamic risk configuration"""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05, 0.10, 0.15],
            risk_reduction_factors=[0.8, 0.6, 0.4],
        )

        backtester = Backtester(
            strategy=create_mock_strategy(),
            data_provider=MockDataProvider(),
            enable_dynamic_risk=True,
            dynamic_risk_config=config,
            log_to_database=False,
        )

        assert backtester.enable_dynamic_risk is True
        assert backtester.dynamic_risk_manager is not None
        assert backtester.dynamic_risk_manager.config.enabled is True

    def test_dynamic_risk_size_adjustment_in_backtest(self):
        """Test dynamic risk size adjustment functionality"""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05, 0.10, 0.15],
            risk_reduction_factors=[0.8, 0.6, 0.4],
        )

        backtester = Backtester(
            strategy=create_mock_strategy(),
            data_provider=MockDataProvider(),
            enable_dynamic_risk=True,
            dynamic_risk_config=config,
            initial_balance=10000,
            log_to_database=False,
        )

        # Test no drawdown scenario
        backtester.balance = 10000
        backtester.peak_balance = 10000

        adjusted_size = backtester._get_dynamic_risk_adjusted_size(0.05, datetime.now())

        # Should be close to original size (no significant adjustment)
        assert 0.04 <= adjusted_size <= 0.06

        # Test 15% drawdown scenario
        backtester.balance = 8500  # 15% drawdown
        backtester.peak_balance = 10000

        adjusted_size = backtester._get_dynamic_risk_adjusted_size(0.05, datetime.now())

        # Should be significantly reduced (0.05 * 0.4 = 0.02)
        assert adjusted_size == pytest.approx(0.02, rel=0.1)

    def test_peak_balance_tracking(self):
        """Test peak balance tracking functionality"""
        backtester = Backtester(
            strategy=create_mock_strategy(),
            data_provider=MockDataProvider(),
            initial_balance=10000,
            log_to_database=False,
        )

        # Initial state
        assert backtester.balance == 10000
        assert backtester.peak_balance == 10000

        # Increase balance
        backtester.balance = 12000
        backtester._update_peak_balance()
        assert backtester.peak_balance == 12000

        # Decrease balance (should not affect peak)
        backtester.balance = 9000
        backtester._update_peak_balance()
        assert backtester.peak_balance == 12000  # Peak should remain

    def test_dynamic_risk_with_custom_config(self):
        """Test dynamic risk with custom configuration"""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.03, 0.07],  # Custom thresholds
            risk_reduction_factors=[0.9, 0.5],  # Custom factors
            volatility_adjustment_enabled=False,
        )

        backtester = Backtester(
            strategy=create_mock_strategy(),
            data_provider=MockDataProvider(),
            enable_dynamic_risk=True,
            dynamic_risk_config=config,
            initial_balance=10000,
            log_to_database=False,
        )

        # Test 5% drawdown (between first and second threshold)
        backtester.balance = 9500
        backtester.peak_balance = 10000

        adjusted_size = backtester._get_dynamic_risk_adjusted_size(0.04, datetime.now())

        # Should apply first reduction factor (0.04 * 0.9 = 0.036)
        assert adjusted_size == pytest.approx(0.036, rel=0.1)

    def test_dynamic_risk_graceful_failure(self):
        """Test that dynamic risk fails gracefully"""
        # Create backtester with broken dynamic risk manager
        backtester = Backtester(
            strategy=create_mock_strategy(),
            data_provider=MockDataProvider(),
            enable_dynamic_risk=True,
            log_to_database=False,
        )

        # Break the dynamic risk manager
        backtester.dynamic_risk_manager = None

        # Should return original size without error
        original_size = 0.05
        adjusted_size = backtester._get_dynamic_risk_adjusted_size(original_size, datetime.now())

        assert adjusted_size == original_size
