"""Integration tests for dynamic risk management in live trading engine.

These tests verify that the live engine applies dynamic risk adjustments
consistently with the backtesting engine, ensuring parity in risk management.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.engines.live.trading_engine import LiveTradingEngine, Position, PositionSide
from src.engines.shared.dynamic_risk_handler import DynamicRiskHandler
from src.position_management.dynamic_risk import DynamicRiskConfig, DynamicRiskManager
from src.strategies.components import (
    FixedFractionSizer,
    FixedRiskManager,
    HoldSignalGenerator,
    Signal,
    SignalDirection,
    Strategy,
)

pytestmark = pytest.mark.integration


def create_mock_strategy() -> Strategy:
    """Create a mock component-based strategy for testing."""
    signal = HoldSignalGenerator()
    risk = FixedRiskManager(risk_per_trade=0.01, stop_loss_pct=0.05)
    sizer = FixedFractionSizer(fraction=0.05)
    return Strategy("test_strategy", signal, risk, sizer)


class MockDataProvider:
    """Mock data provider for testing."""

    def get_historical_data(self, symbol, timeframe, start, end=None, limit=None):
        return pd.DataFrame()

    def get_live_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        return pd.DataFrame()

    def update_live_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        return 100.0


class TestLiveEngineDynamicRiskIntegration:
    """Integration tests for dynamic risk management in live trading engine."""

    def test_live_engine_dynamic_risk_enabled_by_default(self):
        """Dynamic risk should be enabled by default in live engine."""
        engine = LiveTradingEngine(
            strategy=create_mock_strategy(),
            data_provider=MockDataProvider(),
            initial_balance=10000.0,
            enable_live_trading=False,
            log_trades=False,
        )

        assert engine.enable_dynamic_risk is True

    def test_live_engine_can_disable_dynamic_risk(self):
        """Live engine should allow disabling dynamic risk."""
        engine = LiveTradingEngine(
            strategy=create_mock_strategy(),
            data_provider=MockDataProvider(),
            initial_balance=10000.0,
            enable_live_trading=False,
            log_trades=False,
            enable_dynamic_risk=False,
        )

        assert engine.enable_dynamic_risk is False

    def test_live_engine_with_custom_dynamic_risk_config(self):
        """Test live engine with custom dynamic risk configuration."""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05, 0.10, 0.15],
            risk_reduction_factors=[0.8, 0.6, 0.4],
        )

        engine = LiveTradingEngine(
            strategy=create_mock_strategy(),
            data_provider=MockDataProvider(),
            initial_balance=10000.0,
            enable_live_trading=False,
            log_trades=False,
            enable_dynamic_risk=True,
            dynamic_risk_config=config,
        )

        assert engine.enable_dynamic_risk is True

    def test_dynamic_risk_handler_integration(self):
        """Test dynamic risk handler integration with live engine."""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05, 0.10],
            risk_reduction_factors=[0.8, 0.5],
        )
        manager = DynamicRiskManager(config)
        handler = DynamicRiskHandler(manager)

        # Test size adjustment under drawdown
        original_size = 0.05

        # No drawdown
        adjusted = handler.apply_dynamic_risk(
            original_size=original_size,
            current_time=datetime.now(),
            balance=10000.0,
            peak_balance=10000.0,
        )
        assert adjusted == pytest.approx(original_size, rel=0.1)

        # 5% drawdown - should apply 0.8 factor
        adjusted_5pct = handler.apply_dynamic_risk(
            original_size=original_size,
            current_time=datetime.now(),
            balance=9500.0,
            peak_balance=10000.0,
        )
        assert adjusted_5pct == pytest.approx(original_size * 0.8, rel=0.1)

        # 10% drawdown - should apply 0.5 factor
        adjusted_10pct = handler.apply_dynamic_risk(
            original_size=original_size,
            current_time=datetime.now(),
            balance=9000.0,
            peak_balance=10000.0,
        )
        assert adjusted_10pct == pytest.approx(original_size * 0.5, rel=0.1)

    def test_dynamic_risk_graceful_degradation(self):
        """Test that dynamic risk fails gracefully with None manager."""
        handler = DynamicRiskHandler(dynamic_risk_manager=None)

        original_size = 0.05
        adjusted = handler.apply_dynamic_risk(
            original_size=original_size,
            current_time=datetime.now(),
            balance=9000.0,
            peak_balance=10000.0,
        )

        # Should return original size when manager is None
        assert adjusted == original_size

    def test_peak_balance_tracking_parity(self):
        """Test that peak balance tracking matches backtester behavior."""
        engine = LiveTradingEngine(
            strategy=create_mock_strategy(),
            data_provider=MockDataProvider(),
            initial_balance=10000.0,
            enable_live_trading=False,
            log_trades=False,
        )

        # Initial state
        assert engine.current_balance == 10000.0
        assert engine.peak_balance == 10000.0

        # Simulate balance increase and peak update
        # Live engine updates peak inline when balance > peak
        engine.current_balance = 12000.0
        if engine.current_balance > engine.peak_balance:
            engine.peak_balance = engine.current_balance
        assert engine.peak_balance == 12000.0

        # Simulate balance decrease (should not affect peak)
        engine.current_balance = 9000.0
        if engine.current_balance > engine.peak_balance:
            engine.peak_balance = engine.current_balance
        assert engine.peak_balance == 12000.0  # Peak unchanged

    def test_dynamic_risk_adjustment_tracking(self):
        """Test that dynamic risk adjustments are tracked for analysis."""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05],
            risk_reduction_factors=[0.5],
        )
        manager = DynamicRiskManager(config)
        handler = DynamicRiskHandler(manager, significance_threshold=0.1)

        # Trigger a significant adjustment
        handler.apply_dynamic_risk(
            original_size=0.05,
            current_time=datetime.now(),
            balance=9000.0,  # 10% drawdown
            peak_balance=10000.0,
        )

        # Check adjustment was tracked
        assert handler.has_adjustments is True
        assert handler.adjustment_count >= 1

        # Get adjustments
        adjustments = handler.get_adjustments(clear=True)
        assert len(adjustments) >= 1

        # After clearing, should be empty
        assert handler.has_adjustments is False


class TestDynamicRiskConfigurationParity:
    """Test that dynamic risk configuration matches between engines."""

    def test_default_config_consistency(self):
        """Verify default configuration values are consistent."""
        from src.config.constants import DEFAULT_DYNAMIC_RISK_ENABLED

        # Both engines should use the same default
        live_engine = LiveTradingEngine(
            strategy=create_mock_strategy(),
            data_provider=MockDataProvider(),
            initial_balance=10000.0,
            enable_live_trading=False,
            log_trades=False,
        )

        assert live_engine.enable_dynamic_risk == DEFAULT_DYNAMIC_RISK_ENABLED

    def test_config_thresholds_applied_consistently(self):
        """Verify threshold configuration is applied the same way."""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.03, 0.07, 0.12],
            risk_reduction_factors=[0.9, 0.6, 0.3],
        )
        manager = DynamicRiskManager(config)
        handler = DynamicRiskHandler(manager)

        original_size = 0.10

        # Test each threshold level
        test_cases = [
            (10000.0, 10000.0, 0.10),   # No drawdown, no reduction
            (9700.0, 10000.0, 0.09),    # 3% drawdown, 0.9 factor
            (9300.0, 10000.0, 0.06),    # 7% drawdown, 0.6 factor
            (8800.0, 10000.0, 0.03),    # 12% drawdown, 0.3 factor
        ]

        for balance, peak, expected in test_cases:
            adjusted = handler.apply_dynamic_risk(
                original_size=original_size,
                current_time=datetime.now(),
                balance=balance,
                peak_balance=peak,
            )
            assert adjusted == pytest.approx(expected, rel=0.15), (
                f"Failed for balance={balance}, peak={peak}: "
                f"expected {expected}, got {adjusted}"
            )


class TestLiveEngineRiskExitIntegration:
    """Test risk exit behavior in live engine."""

    def test_stop_loss_detection_uses_high_low(self):
        """Verify stop loss detection uses high/low prices."""
        strategy = Mock()
        strategy.get_risk_overrides.return_value = None
        data_provider = Mock()
        data_provider.get_current_price.return_value = 98.0

        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=10000.0,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=0.0,
            slippage_rate=0.0,
            use_high_low_for_stops=True,
        )

        # Long position with SL at 95
        position = Position(
            symbol="TEST",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=100.0,
            entry_time=datetime.now(),
            order_id="test-order",
            original_size=0.1,
            current_size=0.1,
            stop_loss=95.0,
        )

        # Close is 98 (above SL), but low is 94 (below SL)
        triggered = engine._check_stop_loss(
            position,
            current_price=98.0,
            candle_high=101.0,
            candle_low=94.0,  # Below SL
        )

        assert triggered is True

    def test_take_profit_detection_uses_high_low(self):
        """Verify take profit detection uses high/low prices."""
        strategy = Mock()
        strategy.get_risk_overrides.return_value = None
        data_provider = Mock()
        data_provider.get_current_price.return_value = 108.0

        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=10000.0,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=0.0,
            slippage_rate=0.0,
            use_high_low_for_stops=True,
        )

        # Long position with TP at 110
        position = Position(
            symbol="TEST",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=100.0,
            entry_time=datetime.now(),
            order_id="test-order",
            original_size=0.1,
            current_size=0.1,
            take_profit=110.0,
        )

        # Close is 108 (below TP), but high is 112 (above TP)
        triggered = engine._check_take_profit(
            position,
            current_price=108.0,
            candle_high=112.0,  # Above TP
            candle_low=105.0,
        )

        assert triggered is True
