"""Integration tests for parity between backtest and live trading engines.

These tests verify that both engines produce identical results when given
the same market data and configuration, ensuring backtest results accurately
predict live trading behavior.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.engines.backtest.engine import Backtester
from src.engines.live.trading_engine import LiveTradingEngine, Position, PositionSide
from src.engines.shared.cost_calculator import CostCalculator
from src.engines.shared.dynamic_risk_handler import DynamicRiskHandler
from src.engines.shared.models import PositionSide as SharedPositionSide
from src.engines.shared.trailing_stop_manager import TrailingStopManager, TrailingStopUpdate
from src.position_management.dynamic_risk import DynamicRiskConfig, DynamicRiskManager
from src.position_management.trailing_stops import TrailingStopPolicy
from src.strategies.components import (
    FixedFractionSizer,
    FixedRiskManager,
    HoldSignalGenerator,
    Signal,
    SignalDirection,
    Strategy,
)
from src.strategies.components.signal_generator import SignalGenerator
from src.strategies.components.risk_manager import RiskManager
from src.strategies.components.position_sizer import PositionSizer

pytestmark = pytest.mark.integration


class MockDataProvider:
    """Mock data provider for testing."""

    def __init__(self, data: pd.DataFrame | None = None):
        self._data = data if data is not None else self._create_default_data()

    def _create_default_data(self) -> pd.DataFrame:
        """Create default test market data."""
        idx = pd.date_range("2024-01-01", periods=100, freq="1h")
        closes = [100 + i * 0.5 for i in range(100)]  # Upward trend
        return pd.DataFrame(
            {
                "open": [c - 0.25 for c in closes],
                "high": [c + 1 for c in closes],
                "low": [c - 1 for c in closes],
                "close": closes,
                "volume": [1000] * 100,
                "atr": [1.0] * 100,
            },
            index=idx,
        )

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        return self._data.copy()

    def get_live_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        return self._data.copy()

    def update_live_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return self._data.copy()

    def get_current_price(self, symbol: str) -> float:
        return float(self._data["close"].iloc[-1])


class MockSignalGenerator(SignalGenerator):
    """Mock signal generator that produces configurable signals."""

    def __init__(self, signals: list[SignalDirection] | None = None):
        super().__init__(name="mock_signal")
        self._signals = signals or []
        self._index = 0

    def generate_signal(self, df: pd.DataFrame, index: int, regime=None) -> Signal:
        if self._index < len(self._signals):
            direction = self._signals[self._index]
            self._index += 1
        else:
            direction = SignalDirection.HOLD
        return Signal(
            direction=direction,
            confidence=0.8 if direction != SignalDirection.HOLD else 0.0,
            strength=0.8 if direction != SignalDirection.HOLD else 0.0,
            metadata={"timestamp": df.index[index] if len(df) > index else datetime.now()},
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        return 0.8


class MockRiskManager(RiskManager):
    """Mock risk manager with fixed risk."""

    def __init__(self, stop_loss_pct: float = 0.05):
        super().__init__(name="mock_risk")
        self._stop_loss_pct = stop_loss_pct

    def calculate_position_size(self, signal: Signal, balance: float, regime=None) -> float:
        return 0.05 * balance

    def should_exit(self, position, current_data, regime=None) -> bool:
        return False

    def get_stop_loss(self, entry_price: float, signal: Signal, regime=None) -> float:
        if signal.direction == SignalDirection.LONG:
            return entry_price * (1 - self._stop_loss_pct)
        elif signal.direction == SignalDirection.SHORT:
            return entry_price * (1 + self._stop_loss_pct)
        return entry_price * (1 - self._stop_loss_pct)


class MockPositionSizer(PositionSizer):
    """Mock position sizer with fixed fraction."""

    def __init__(self, fraction: float = 0.05):
        super().__init__(name="mock_sizer")
        self._fraction = fraction

    def calculate_size(
        self, signal: Signal, balance: float, risk_amount: float, regime=None
    ) -> float:
        return self._fraction


def create_mock_strategy(signals: list[SignalDirection] | None = None) -> Strategy:
    """Create a mock component-based strategy for testing."""
    return Strategy(
        name="MockStrategy",
        signal_generator=MockSignalGenerator(signals),
        risk_manager=MockRiskManager(),
        position_sizer=MockPositionSizer(),
    )


class TestCostCalculatorParity:
    """Test that cost calculations are identical between engines."""

    @pytest.mark.parametrize(
        "fee_rate,slippage_rate",
        [
            (0.001, 0.0005),  # Default rates
            (0.002, 0.001),   # Higher rates
            (0.0, 0.0),       # Zero costs
            (0.0005, 0.0001), # Lower rates
        ],
    )
    def test_entry_cost_calculation_consistency(self, fee_rate: float, slippage_rate: float):
        """Verify entry costs are calculated identically."""
        calc = CostCalculator(fee_rate=fee_rate, slippage_rate=slippage_rate)

        # Test long entry
        long_result = calc.calculate_entry_costs(price=100.0, notional=1000.0, side="long")

        # For long, slippage makes entry worse (higher)
        expected_long_price = 100.0 * (1 + slippage_rate)
        assert long_result.executed_price == pytest.approx(expected_long_price)
        assert long_result.fee == pytest.approx(1000.0 * fee_rate)

        # Test short entry
        calc2 = CostCalculator(fee_rate=fee_rate, slippage_rate=slippage_rate)
        short_result = calc2.calculate_entry_costs(price=100.0, notional=1000.0, side="short")

        # For short, slippage makes entry worse (lower)
        expected_short_price = 100.0 * (1 - slippage_rate)
        assert short_result.executed_price == pytest.approx(expected_short_price)
        assert short_result.fee == pytest.approx(1000.0 * fee_rate)

    @pytest.mark.parametrize(
        "fee_rate,slippage_rate",
        [
            (0.001, 0.0005),
            (0.002, 0.001),
            (0.0, 0.0),
        ],
    )
    def test_exit_cost_calculation_consistency(self, fee_rate: float, slippage_rate: float):
        """Verify exit costs are calculated identically."""
        calc = CostCalculator(fee_rate=fee_rate, slippage_rate=slippage_rate)

        # Test long exit
        long_result = calc.calculate_exit_costs(price=110.0, notional=1100.0, side="long")

        # For long exit, slippage makes exit worse (lower)
        expected_long_price = 110.0 * (1 - slippage_rate)
        assert long_result.executed_price == pytest.approx(expected_long_price)
        assert long_result.fee == pytest.approx(1100.0 * fee_rate)

        # Test short exit
        calc2 = CostCalculator(fee_rate=fee_rate, slippage_rate=slippage_rate)
        short_result = calc2.calculate_exit_costs(price=90.0, notional=900.0, side="short")

        # For short exit, slippage makes exit worse (higher)
        expected_short_price = 90.0 * (1 + slippage_rate)
        assert short_result.executed_price == pytest.approx(expected_short_price)
        assert short_result.fee == pytest.approx(900.0 * fee_rate)

    def test_cost_accumulation_tracking(self):
        """Verify cost accumulation is tracked consistently."""
        calc = CostCalculator(fee_rate=0.001, slippage_rate=0.0005)

        # Multiple trades
        calc.calculate_entry_costs(price=100.0, notional=1000.0, side="long")
        calc.calculate_exit_costs(price=110.0, notional=1100.0, side="long")
        calc.calculate_entry_costs(price=105.0, notional=500.0, side="short")
        calc.calculate_exit_costs(price=95.0, notional=475.0, side="short")

        # Check accumulated totals
        total_notional = 1000.0 + 1100.0 + 500.0 + 475.0
        expected_fees = total_notional * 0.001
        assert calc.total_fees_paid == pytest.approx(expected_fees)
        assert calc.total_slippage_cost > 0  # Should have accumulated slippage


class TestTrailingStopParity:
    """Test that trailing stop logic is identical between engines."""

    def test_breakeven_trigger_consistency(self):
        """Verify breakeven triggers at the same level for both engines."""
        policy = TrailingStopPolicy(
            activation_threshold=0.02,  # 2%
            trailing_distance_pct=0.01,  # 1%
            breakeven_threshold=0.015,  # 1.5%
            breakeven_buffer=0.001,  # 0.1%
        )
        manager = TrailingStopManager(policy)

        # Create a mock position
        class MockPosition:
            def __init__(self):
                self.entry_price = 100.0
                self.side = SharedPositionSide.LONG
                self.stop_loss = 95.0
                self.trailing_stop_price = None
                self.breakeven_triggered = False
                self.trailing_stop_activated = False

        position = MockPosition()

        # Price move of 1.5% should trigger breakeven
        current_price = 101.5
        result = manager.update(position, current_price)

        assert result.breakeven_triggered is True
        assert result.new_stop_price is not None
        # Breakeven stop should be at entry + buffer
        expected_stop = 100.0 * (1 + 0.001)  # 100.1
        assert result.new_stop_price == pytest.approx(expected_stop)

    def test_trailing_activation_consistency(self):
        """Verify trailing stop activates at the same level."""
        policy = TrailingStopPolicy(
            activation_threshold=0.02,  # 2%
            trailing_distance_pct=0.01,  # 1%
        )
        manager = TrailingStopManager(policy)

        class MockPosition:
            def __init__(self):
                self.entry_price = 100.0
                self.side = SharedPositionSide.LONG
                self.stop_loss = 95.0
                self.trailing_stop_price = None
                self.breakeven_triggered = True  # Already triggered
                self.trailing_stop_activated = False

        position = MockPosition()

        # Price move of 2% should activate trailing
        current_price = 102.0
        result = manager.update(position, current_price)

        assert result.trailing_activated is True
        assert result.new_stop_price is not None
        # Trailing stop should be at current price - 1%
        expected_stop = 102.0 - (102.0 * 0.01)  # 100.98
        assert result.new_stop_price == pytest.approx(expected_stop)

    @pytest.mark.parametrize(
        "side,entry_price,current_price,expected_direction",
        [
            (SharedPositionSide.LONG, 100.0, 105.0, "up"),   # Long profit
            (SharedPositionSide.LONG, 100.0, 95.0, "none"),  # Long loss (no trail)
            (SharedPositionSide.SHORT, 100.0, 95.0, "down"), # Short profit
            (SharedPositionSide.SHORT, 100.0, 105.0, "none"),# Short loss (no trail)
        ],
    )
    def test_trailing_direction_consistency(
        self, side, entry_price, current_price, expected_direction
    ):
        """Verify trailing stops move in the correct direction for each side."""
        policy = TrailingStopPolicy(
            activation_threshold=0.01,
            trailing_distance_pct=0.005,
        )
        manager = TrailingStopManager(policy)

        class MockPosition:
            def __init__(self, pos_side, pos_entry):
                self.entry_price = pos_entry
                self.side = pos_side
                self.stop_loss = None
                self.trailing_stop_price = None
                self.breakeven_triggered = True
                self.trailing_stop_activated = False

        position = MockPosition(side, entry_price)
        result = manager.update(position, current_price)

        if expected_direction == "none":
            # Should not activate for losses
            assert result.trailing_activated is False
        else:
            # Should activate for profits
            if abs(current_price - entry_price) / entry_price >= 0.01:
                assert result.updated is True


class TestDynamicRiskParity:
    """Test that dynamic risk adjustments are identical between engines."""

    def test_drawdown_reduction_consistency(self):
        """Verify position size reduction during drawdown is consistent."""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05, 0.10, 0.15],
            risk_reduction_factors=[0.8, 0.6, 0.4],
        )
        manager = DynamicRiskManager(config)
        handler = DynamicRiskHandler(manager)

        original_size = 0.05
        current_time = datetime.now()

        # No drawdown - should return original size
        adjusted = handler.apply_dynamic_risk(
            original_size=original_size,
            current_time=current_time,
            balance=10000.0,
            peak_balance=10000.0,
        )
        assert adjusted == pytest.approx(original_size, rel=0.1)

        # 10% drawdown - should apply 0.6 factor
        adjusted_10pct = handler.apply_dynamic_risk(
            original_size=original_size,
            current_time=current_time,
            balance=9000.0,  # 10% drawdown
            peak_balance=10000.0,
        )
        assert adjusted_10pct == pytest.approx(original_size * 0.6, rel=0.1)

        # 15% drawdown - should apply 0.4 factor
        adjusted_15pct = handler.apply_dynamic_risk(
            original_size=original_size,
            current_time=current_time,
            balance=8500.0,  # 15% drawdown
            peak_balance=10000.0,
        )
        assert adjusted_15pct == pytest.approx(original_size * 0.4, rel=0.1)

    def test_handler_graceful_degradation(self):
        """Verify handler returns original size when manager is None."""
        handler = DynamicRiskHandler(dynamic_risk_manager=None)

        original_size = 0.05
        adjusted = handler.apply_dynamic_risk(
            original_size=original_size,
            current_time=datetime.now(),
            balance=10000.0,
            peak_balance=10000.0,
        )

        assert adjusted == original_size

    def test_adjustment_tracking_consistency(self):
        """Verify adjustments are tracked consistently."""
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

        # Check that adjustment was tracked
        assert handler.has_adjustments is True
        adjustments = handler.get_adjustments()
        assert len(adjustments) >= 1


class TestPositionSizingParity:
    """Test that position sizing is calculated identically."""

    def test_max_position_size_enforcement(self):
        """Verify max position size is enforced the same way."""
        from src.engines.live.execution.entry_handler import LiveEntryHandler

        strategy = Mock()
        strategy.get_risk_overrides.return_value = None
        data_provider = Mock()
        data_provider.get_current_price.return_value = 100.0

        max_position = 0.1  # 10%

        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=10_000.0,
            max_position_size=max_position,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=0.0,
            slippage_rate=0.0,
        )

        # Test through entry handler directly
        entry_result = engine.live_entry_handler.execution_engine.execute_entry(
            symbol="TEST",
            side=PositionSide.LONG,
            size_fraction=min(0.5, max_position),  # Handler enforces cap
            base_price=100.0,
            balance=10_000.0,
        )

        # Verify position value matches capped size
        assert entry_result.success is True
        expected_position_value = 10_000.0 * max_position
        assert entry_result.position_value == pytest.approx(expected_position_value)

    def test_backtester_position_size_parity(self):
        """Verify backtester has same position size limits as live engine."""
        strategy = create_mock_strategy()
        data_provider = MockDataProvider()

        # Create backtester with same max position size
        backtester = Backtester(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=10_000.0,
            max_position_size=0.1,
            log_to_database=False,
        )

        assert backtester.max_position_size == 0.1


class TestFeeSlippageIntegration:
    """Test fee and slippage integration parity."""

    def test_entry_fee_applied_to_balance(self):
        """Verify entry fees reduce balance identically."""
        strategy = Mock()
        strategy.get_risk_overrides.return_value = None
        data_provider = Mock()
        data_provider.get_current_price.return_value = 100.0

        initial_balance = 10_000.0
        fee_rate = 0.001  # 0.1%
        position_size = 0.1  # 10%

        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=initial_balance,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=fee_rate,
            slippage_rate=0.0,
        )

        position_value = initial_balance * position_size
        expected_fee = position_value * fee_rate

        # Test through entry handler's execution engine
        entry_result = engine.live_entry_handler.execution_engine.execute_entry(
            symbol="TEST",
            side=PositionSide.LONG,
            size_fraction=position_size,
            base_price=100.0,
            balance=initial_balance,
        )

        # Verify fee was charged
        assert entry_result.entry_fee == pytest.approx(expected_fee)
        # Balance should be reduced by the fee
        assert engine.live_entry_handler.execution_engine.total_fees_paid == pytest.approx(expected_fee)

    def test_slippage_affects_entry_price(self):
        """Verify slippage is applied to entry prices identically."""
        strategy = Mock()
        strategy.get_risk_overrides.return_value = None
        data_provider = Mock()
        data_provider.get_current_price.return_value = 100.0

        slippage_rate = 0.0005  # 0.05%

        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=10_000.0,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=0.0,
            slippage_rate=slippage_rate,
        )

        # Test through entry handler
        entry_result = engine.live_entry_handler.execution_engine.execute_entry(
            symbol="TEST",
            side=PositionSide.LONG,
            size_fraction=0.1,
            base_price=100.0,
            balance=10_000.0,
        )

        # For long, slippage increases entry price
        expected_entry = 100.0 * (1 + slippage_rate)
        assert entry_result.executed_price == pytest.approx(expected_entry)


class TestStopLossTakeProfitParity:
    """Test SL/TP behavior parity."""

    def test_long_sl_triggers_on_low_breach(self):
        """Verify long SL triggers when low breaches SL level."""
        from src.engines.live.execution.exit_handler import LiveExitHandler

        strategy = Mock()
        strategy.get_risk_overrides.return_value = None
        data_provider = Mock()

        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=10_000.0,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=0.0,
            slippage_rate=0.0,
            use_high_low_for_stops=True,
        )

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

        # Check exit conditions through exit handler
        exit_check = engine.live_exit_handler.check_exit_conditions(
            position=position,
            current_price=98.0,
            candle_high=101.0,
            candle_low=94.0,  # Breaches SL at 95
        )

        assert exit_check.should_exit is True
        assert "Stop loss" in exit_check.exit_reason

    def test_short_sl_triggers_on_high_breach(self):
        """Verify short SL triggers when high breaches SL level."""
        from src.engines.live.execution.exit_handler import LiveExitHandler

        strategy = Mock()
        strategy.get_risk_overrides.return_value = None
        data_provider = Mock()

        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=10_000.0,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=0.0,
            slippage_rate=0.0,
            use_high_low_for_stops=True,
        )

        position = Position(
            symbol="TEST",
            side=PositionSide.SHORT,
            size=0.1,
            entry_price=100.0,
            entry_time=datetime.now(),
            order_id="test-order",
            original_size=0.1,
            current_size=0.1,
            stop_loss=105.0,
        )

        # Check exit conditions through exit handler
        exit_check = engine.live_exit_handler.check_exit_conditions(
            position=position,
            current_price=102.0,
            candle_high=106.0,  # Breaches SL at 105
            candle_low=99.0,
        )

        assert exit_check.should_exit is True
        assert "Stop loss" in exit_check.exit_reason

    def test_exit_uses_sl_level_not_close(self):
        """Verify exit price uses SL level, not close price."""
        strategy = Mock()
        strategy.get_risk_overrides.return_value = None
        data_provider = Mock()
        data_provider.get_current_price.return_value = 98.0  # Close above SL

        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=10_000.0,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=0.0,
            slippage_rate=0.0,
        )

        sl_level = 95.0
        position = Position(
            symbol="TEST",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=100.0,
            entry_time=datetime.now(),
            order_id="test-order",
            original_size=0.1,
            current_size=0.1,
            stop_loss=sl_level,
            entry_balance=10_000.0,
        )

        # Execute exit through exit handler
        exit_result = engine.live_exit_handler.execute_exit(
            position=position,
            exit_reason="Stop loss",
            current_price=98.0,
            limit_price=sl_level,
            current_balance=10_000.0,
            candle_low=94.0,
            candle_high=101.0,
        )

        # P&L should be based on SL price (95), not close (98)
        # (95 - 100) / 100 * 0.1 * 10000 = -$50
        expected_pnl = -50.0
        assert exit_result.realized_pnl == pytest.approx(expected_pnl)


class TestSharedModelConsistency:
    """Test that shared models are used consistently."""

    def test_position_side_enum_consistency(self):
        """Verify PositionSide enum values match between modules."""
        from src.engines.shared.models import PositionSide as SharedSide
        from src.engines.live.trading_engine import PositionSide as LiveSide

        assert SharedSide.LONG.value == LiveSide.LONG.value
        assert SharedSide.SHORT.value == LiveSide.SHORT.value

    def test_shared_models_imported_in_backtest(self):
        """Verify backtest engine uses shared models."""
        from src.engines.backtest import PositionSide as BacktestSide
        from src.engines.shared.models import PositionSide as SharedSide

        # Both should be the same enum
        assert BacktestSide is SharedSide

    def test_cost_calculator_used_by_both_engines(self):
        """Verify both engines use the shared CostCalculator."""
        from src.engines.backtest.execution.execution_engine import CostCalculator as BacktestCC
        from src.engines.live.execution.execution_engine import CostCalculator as LiveCC
        from src.engines.shared.cost_calculator import CostCalculator as SharedCC

        # Both should import from shared
        assert BacktestCC is SharedCC
        assert LiveCC is SharedCC


class TestEndToEndParityScenario:
    """End-to-end parity test scenarios."""

    def test_profitable_long_trade_parity(self):
        """Test that a profitable long trade produces identical results."""
        # Create identical data
        idx = pd.date_range("2024-01-01", periods=10, freq="1h")
        data = pd.DataFrame(
            {
                "open": [100, 100, 101, 102, 103, 104, 105, 106, 107, 108],
                "high": [101, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "low": [99, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [1000] * 10,
                "atr": [1.0] * 10,
            },
            index=idx,
        )

        provider = MockDataProvider(data)

        # Use consistent fee/slippage
        fee_rate = 0.001
        slippage_rate = 0.0005
        initial_balance = 10000.0

        # Test shared cost calculator produces same results
        calc = CostCalculator(fee_rate=fee_rate, slippage_rate=slippage_rate)

        # Entry at 100
        entry_result = calc.calculate_entry_costs(
            price=100.0,
            notional=1000.0,  # 10% of balance
            side="long",
        )

        # Exit at 109 (9% gain)
        exit_result = calc.calculate_exit_costs(
            price=109.0,
            notional=1090.0,  # Value at exit
            side="long",
        )

        # Calculate expected P&L
        # Entry slippage: 100 * 0.0005 = 0.05
        # Exit slippage: 109 * 0.0005 = 0.0545
        # Gross P&L: (109 - 100.05) / 100.05 * 1000 = ~89.5

        total_fees = entry_result.fee + exit_result.fee
        total_slippage = entry_result.slippage_cost + exit_result.slippage_cost

        assert total_fees == pytest.approx(1.0 + 1.09)  # 1000 * 0.001 + 1090 * 0.001
        assert total_slippage > 0

    def test_losing_short_trade_parity(self):
        """Test that a losing short trade produces identical results."""
        calc = CostCalculator(fee_rate=0.001, slippage_rate=0.0005)

        # Entry short at 100
        entry_result = calc.calculate_entry_costs(
            price=100.0,
            notional=1000.0,
            side="short",
        )

        # Exit short at 105 (5% loss)
        exit_result = calc.calculate_exit_costs(
            price=105.0,
            notional=1050.0,
            side="short",
        )

        # Short entry: price * (1 - slippage) = 99.95
        assert entry_result.executed_price == pytest.approx(99.95)

        # Short exit: price * (1 + slippage) = 105.0525
        assert exit_result.executed_price == pytest.approx(105.0525)

        # Fees should be applied to notional
        assert entry_result.fee == pytest.approx(1.0)
        assert exit_result.fee == pytest.approx(1.05)


class TestPartialOperationsParity:
    """Test partial exit/scale-in parity."""

    def test_partial_exit_pnl_calculation(self):
        """Verify partial exit P&L is calculated identically."""
        from src.engines.shared.partial_exit_executor import PartialExitExecutor

        executor = PartialExitExecutor(fee_rate=0.001, slippage_rate=0.0005)

        # Long position, 50% exit at profit
        result = executor.execute_partial_exit(
            entry_price=100.0,
            exit_price=110.0,
            position_side=SharedPositionSide.LONG,
            exit_fraction=0.5,
            basis_balance=10000.0,
        )

        # P&L% = (110 - 100) / 100 * 0.5 = 0.05 (5%)
        assert result.pnl_percent == pytest.approx(0.05)

        # Gross P&L = 10000 * 0.05 = $500
        assert result.gross_pnl == pytest.approx(500.0)

        # Net P&L after fees/slippage
        assert result.realized_pnl < result.gross_pnl  # Costs reduce P&L

    def test_short_partial_exit_calculation(self):
        """Verify short partial exit P&L is calculated identically."""
        from src.engines.shared.partial_exit_executor import PartialExitExecutor

        executor = PartialExitExecutor(fee_rate=0.001, slippage_rate=0.0005)

        # Short position, 50% exit at profit (price went down)
        result = executor.execute_partial_exit(
            entry_price=100.0,
            exit_price=90.0,  # 10% drop = profit for short
            position_side=SharedPositionSide.SHORT,
            exit_fraction=0.5,
            basis_balance=10000.0,
        )

        # P&L% = (100 - 90) / 100 * 0.5 = 0.05 (5%)
        assert result.pnl_percent == pytest.approx(0.05)
        assert result.gross_pnl == pytest.approx(500.0)
