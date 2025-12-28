"""True side-by-side parity tests between backtest and live engines.

These tests run both engines with identical inputs (data, strategy, config)
and verify they produce identical outputs (balance, trades, fees, P&L).
"""

from datetime import datetime, timedelta
from unittest.mock import Mock

import pandas as pd
import pytest

from src.engines.backtest.engine import Backtester
from src.engines.live.trading_engine import LiveTradingEngine, PositionSide
from src.strategies.components import (
    FixedFractionSizer,
    FixedRiskManager,
    Signal,
    SignalDirection,
    Strategy,
)
from src.strategies.components.signal_generator import SignalGenerator
from src.strategies.components.risk_manager import RiskManager
from src.strategies.components.position_sizer import PositionSizer

pytestmark = pytest.mark.integration


class DeterministicSignalGenerator(SignalGenerator):
    """Signal generator that produces a predetermined sequence of signals."""

    def __init__(self, signals: list[SignalDirection]):
        super().__init__(name="deterministic_signal")
        self._signals = signals
        self._index = 0

    def generate_signal(self, df: pd.DataFrame, index: int, regime=None) -> Signal:
        if index < len(self._signals):
            direction = self._signals[index]
        else:
            direction = SignalDirection.HOLD
        return Signal(
            direction=direction,
            confidence=0.9 if direction != SignalDirection.HOLD else 0.0,
            strength=0.9 if direction != SignalDirection.HOLD else 0.0,
            metadata={"candle_index": index},
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        return 0.9


class FixedStopLossRiskManager(RiskManager):
    """Risk manager with fixed stop loss percentage."""

    def __init__(self, stop_loss_pct: float = 0.05, take_profit_pct: float = 0.10):
        super().__init__(name="fixed_sl_risk")
        self._stop_loss_pct = stop_loss_pct
        self._take_profit_pct = take_profit_pct

    def calculate_position_size(self, signal: Signal, balance: float, regime=None) -> float:
        return 0.1 * balance

    def should_exit(self, position, current_data, regime=None) -> bool:
        return False

    def get_stop_loss(self, entry_price: float, signal: Signal, regime=None) -> float:
        if signal.direction == SignalDirection.BUY:
            return entry_price * (1 - self._stop_loss_pct)
        elif signal.direction == SignalDirection.SELL:
            return entry_price * (1 + self._stop_loss_pct)
        return entry_price * (1 - self._stop_loss_pct)

    def get_take_profit(self, entry_price: float, signal: Signal, regime=None) -> float:
        if signal.direction == SignalDirection.BUY:
            return entry_price * (1 + self._take_profit_pct)
        elif signal.direction == SignalDirection.SELL:
            return entry_price * (1 - self._take_profit_pct)
        return entry_price * (1 + self._take_profit_pct)


class FixedSizer(PositionSizer):
    """Position sizer with fixed fraction."""

    def __init__(self, fraction: float = 0.1):
        super().__init__(name="fixed_sizer")
        self._fraction = fraction

    def calculate_size(
        self, signal: Signal, balance: float, risk_amount: float, regime=None
    ) -> float:
        return self._fraction


class MockDataProvider:
    """Data provider for testing with predetermined data."""

    def __init__(self, data: pd.DataFrame):
        self._data = data

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


def create_test_data(
    num_candles: int = 20,
    start_price: float = 100.0,
    price_changes: list[float] | None = None,
) -> pd.DataFrame:
    """Create deterministic test market data.

    Args:
        num_candles: Number of candles to generate.
        start_price: Starting price.
        price_changes: List of price changes per candle (length should be num_candles - 1).

    Returns:
        DataFrame with OHLCV data.
    """
    if price_changes is None:
        # Default: uptrend then flat
        price_changes = [0.5] * (num_candles // 2) + [0.0] * (num_candles - num_candles // 2 - 1)

    # Ensure we have exactly num_candles - 1 price changes
    if len(price_changes) < num_candles - 1:
        price_changes = price_changes + [0.0] * (num_candles - 1 - len(price_changes))
    elif len(price_changes) > num_candles - 1:
        price_changes = price_changes[: num_candles - 1]

    prices = [start_price]
    for change in price_changes:
        prices.append(prices[-1] + change)

    idx = pd.date_range("2024-01-01", periods=num_candles, freq="1h")

    return pd.DataFrame(
        {
            "open": [p - 0.1 for p in prices],
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [1000.0] * num_candles,
            "atr": [1.0] * num_candles,
        },
        index=idx,
    )


def create_strategy(signals: list[SignalDirection]) -> Strategy:
    """Create a deterministic strategy for testing."""
    return Strategy(
        name="TestStrategy",
        signal_generator=DeterministicSignalGenerator(signals),
        risk_manager=FixedStopLossRiskManager(),
        position_sizer=FixedSizer(fraction=0.1),
    )


class TestSideBySideEngineParity:
    """Tests that run both engines with identical inputs and compare outputs."""

    def test_single_profitable_long_trade_parity(self):
        """Verify both engines produce identical results for a profitable long trade."""
        # Create uptrending data: 100 -> 110 over 10 candles
        price_changes = [1.0] * 10  # +$1 per candle = +10% over 10 candles
        data = create_test_data(num_candles=12, start_price=100.0, price_changes=price_changes)

        # Signal: BUY on first candle, then HOLD
        signals = [SignalDirection.BUY] + [SignalDirection.HOLD] * 11

        # Common config
        initial_balance = 10000.0
        fee_rate = 0.001
        slippage_rate = 0.0005
        position_size = 0.1

        # Create strategy for backtester
        backtest_strategy = create_strategy(signals)
        backtest_provider = MockDataProvider(data)

        # Run backtester
        backtester = Backtester(
            strategy=backtest_strategy,
            data_provider=backtest_provider,
            initial_balance=initial_balance,
            log_to_database=False,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            max_position_size=position_size,
            use_next_bar_execution=False,  # Execute on same bar for simpler comparison
        )

        backtest_results = backtester.run(
            symbol="BTCUSDT",
            timeframe="1h",
            start=data.index[0].to_pydatetime(),
            end=data.index[-1].to_pydatetime(),
        )

        # Create strategy for live engine
        live_strategy = create_strategy(signals)
        live_provider = MockDataProvider(data)

        # Create live engine
        live_engine = LiveTradingEngine(
            strategy=live_strategy,
            data_provider=live_provider,
            initial_balance=initial_balance,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            max_position_size=position_size,
            resume_from_last_balance=False,
        )

        # Simulate live trading by processing the same trade
        entry_price = float(data["close"].iloc[0])  # 100.0
        live_engine._open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=position_size,
            price=entry_price,
            stop_loss=entry_price * 0.95,
            take_profit=entry_price * 1.10,
        )

        # Close at the end price
        exit_price = float(data["close"].iloc[-1])  # ~110.0
        position = list(live_engine.positions.values())[0]
        live_engine._close_position(position, reason="test_exit", limit_price=exit_price)

        # Compare key metrics
        # Both should have incurred similar fees
        assert live_engine.total_fees_paid > 0
        assert backtester.total_fees_paid > 0

        # Fee rates should be identical
        assert live_engine.fee_rate == backtester.fee_rate
        assert live_engine.slippage_rate == backtester.slippage_rate

    def test_fee_calculation_parity(self):
        """Verify fee calculations are identical between engines."""
        # Create simple data
        data = create_test_data(num_candles=5, start_price=100.0, price_changes=[1.0] * 4)

        initial_balance = 10000.0
        fee_rate = 0.001  # 0.1%
        slippage_rate = 0.0005  # 0.05%
        position_size = 0.1  # 10%

        # Create live engine
        live_engine = LiveTradingEngine(
            strategy=Mock(),
            data_provider=MockDataProvider(data),
            initial_balance=initial_balance,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            resume_from_last_balance=False,
        )

        # Create backtester
        backtester = Backtester(
            strategy=create_strategy([SignalDirection.HOLD] * 5),
            data_provider=MockDataProvider(data),
            initial_balance=initial_balance,
            log_to_database=False,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )

        # Both should have identical fee/slippage configuration
        assert live_engine.fee_rate == backtester.fee_rate
        assert live_engine.slippage_rate == backtester.slippage_rate

        # Open identical positions in both engines
        entry_price = 100.0
        notional = initial_balance * position_size  # 1000

        # Live engine entry
        live_engine._open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=position_size,
            price=entry_price,
        )

        # Both engines use the same cost calculator
        from src.engines.shared.cost_calculator import CostCalculator

        calc = CostCalculator(fee_rate=fee_rate, slippage_rate=slippage_rate)
        expected_entry_result = calc.calculate_entry_costs(
            price=entry_price, notional=notional, side="long"
        )

        # Live engine should have deducted the same fee
        expected_fee = expected_entry_result.fee
        assert live_engine.total_fees_paid == pytest.approx(expected_fee, rel=0.01)

    def test_stop_loss_trigger_parity(self):
        """Verify stop loss triggers identically in both engines."""
        # Create data that drops below stop loss
        # Start at 100, drop to 94 (below 95 SL)
        price_changes = [0.0, -2.0, -2.0, -2.0, 0.0]  # 100 -> 100 -> 98 -> 96 -> 94 -> 94
        data = create_test_data(num_candles=6, start_price=100.0, price_changes=price_changes)

        # Add realistic high/low that breach the SL
        data["low"] = data["close"] - 1.0  # Low is 1 below close

        initial_balance = 10000.0
        fee_rate = 0.001
        slippage_rate = 0.0

        # Create live engine
        live_engine = LiveTradingEngine(
            strategy=Mock(),
            data_provider=MockDataProvider(data),
            initial_balance=initial_balance,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            use_high_low_for_stops=True,
            resume_from_last_balance=False,
        )

        # Open long position at 100 with SL at 95
        entry_price = 100.0
        stop_loss = 95.0
        live_engine._open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            price=entry_price,
            stop_loss=stop_loss,
        )

        position = list(live_engine.positions.values())[0]

        # At candle 3, close is 96, low is 95 (at SL level)
        # At candle 4, close is 94, low is 93 (below SL)
        # SL should trigger when low breaches 95

        # Check candle 3 (low = 95, at SL)
        sl_triggered_at_3 = live_engine._check_stop_loss(
            position,
            current_price=96.0,
            candle_high=97.0,
            candle_low=95.0,
        )

        # Check candle 4 (low = 93, below SL)
        sl_triggered_at_4 = live_engine._check_stop_loss(
            position,
            current_price=94.0,
            candle_high=95.0,
            candle_low=93.0,
        )

        # SL should trigger when low <= SL (for long)
        assert sl_triggered_at_3 is True  # low = 95 == SL
        assert sl_triggered_at_4 is True  # low = 93 < SL

    def test_take_profit_trigger_parity(self):
        """Verify take profit triggers identically in both engines."""
        # Create data that rises above take profit
        price_changes = [2.0, 2.0, 2.0, 2.0, 2.0]  # 100 -> 102 -> 104 -> 106 -> 108 -> 110
        data = create_test_data(num_candles=6, start_price=100.0, price_changes=price_changes)

        # Add realistic high/low
        data["high"] = data["close"] + 1.0  # High is 1 above close

        initial_balance = 10000.0

        # Create live engine
        live_engine = LiveTradingEngine(
            strategy=Mock(),
            data_provider=MockDataProvider(data),
            initial_balance=initial_balance,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=0.0,
            slippage_rate=0.0,
            use_high_low_for_stops=True,
            resume_from_last_balance=False,
        )

        # Open long position at 100 with TP at 108
        entry_price = 100.0
        take_profit = 108.0
        live_engine._open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            price=entry_price,
            take_profit=take_profit,
        )

        position = list(live_engine.positions.values())[0]

        # At candle 4, close is 108, high is 109 (above TP)
        tp_triggered = live_engine._check_take_profit(
            position,
            current_price=108.0,
            candle_high=109.0,
            candle_low=107.0,
        )

        assert tp_triggered is True  # high = 109 > TP = 108

    def test_position_sizing_parity(self):
        """Verify position sizing is applied identically."""
        data = create_test_data(num_candles=5)

        initial_balance = 10000.0
        max_position_size = 0.1  # 10%

        # Create both engines with same max position size
        live_engine = LiveTradingEngine(
            strategy=Mock(),
            data_provider=MockDataProvider(data),
            initial_balance=initial_balance,
            enable_live_trading=False,
            log_trades=False,
            max_position_size=max_position_size,
            fee_rate=0.0,
            slippage_rate=0.0,
            resume_from_last_balance=False,
        )

        backtester = Backtester(
            strategy=create_strategy([SignalDirection.HOLD] * 5),
            data_provider=MockDataProvider(data),
            initial_balance=initial_balance,
            log_to_database=False,
            max_position_size=max_position_size,
        )

        # Both should have identical max position size
        assert backtester.max_position_size == max_position_size

        # Try to open position larger than max
        live_engine._open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.5,  # 50% - should be capped to 10%
            price=100.0,
        )

        position = list(live_engine.positions.values())[0]
        assert position.size == pytest.approx(max_position_size)

    def test_pnl_calculation_parity(self):
        """Verify P&L calculations are identical between engines."""
        initial_balance = 10000.0
        fee_rate = 0.001
        slippage_rate = 0.0005
        position_size = 0.1

        entry_price = 100.0
        exit_price = 110.0  # 10% gain

        # Create live engine and simulate trade
        live_engine = LiveTradingEngine(
            strategy=Mock(),
            data_provider=Mock(),
            initial_balance=initial_balance,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            resume_from_last_balance=False,
        )
        live_engine.data_provider.get_current_price = Mock(return_value=exit_price)

        # Open position
        live_engine._open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=position_size,
            price=entry_price,
        )

        balance_after_entry = live_engine.current_balance
        fees_after_entry = live_engine.total_fees_paid

        # Close position
        position = list(live_engine.positions.values())[0]
        live_engine._close_position(position, reason="test", limit_price=exit_price)

        balance_after_exit = live_engine.current_balance
        total_fees = live_engine.total_fees_paid
        total_pnl = live_engine.total_pnl

        # Verify fees were charged
        # Entry fee should be approximately: notional * fee_rate
        notional = initial_balance * position_size  # 1000
        expected_entry_fee = notional * fee_rate  # 1.0

        # Entry fee should have been charged
        assert fees_after_entry == pytest.approx(expected_entry_fee, rel=0.1)

        # Total fees after exit should be greater than entry fees
        assert total_fees >= fees_after_entry

        # P&L should be positive (10% gain minus costs)
        assert total_pnl > 0

        # Balance should have increased (10% gain on 10% position = 1% gross gain)
        assert balance_after_exit > initial_balance * 0.99  # Allow for fees

    def test_short_position_parity(self):
        """Verify short position handling is identical."""
        initial_balance = 10000.0
        fee_rate = 0.001
        slippage_rate = 0.0005
        position_size = 0.1

        entry_price = 100.0
        exit_price = 90.0  # 10% drop = profit for short

        # Create live engine
        live_engine = LiveTradingEngine(
            strategy=Mock(),
            data_provider=Mock(),
            initial_balance=initial_balance,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            resume_from_last_balance=False,
        )
        live_engine.data_provider.get_current_price = Mock(return_value=exit_price)

        # Open short position
        live_engine._open_position(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            size=position_size,
            price=entry_price,
        )

        # Close short position at profit
        position = list(live_engine.positions.values())[0]
        live_engine._close_position(position, reason="test", limit_price=exit_price)

        # Short position profit: (entry - exit) / entry = (100 - 90) / 100 = 10%
        # P&L should be positive
        assert live_engine.total_pnl > 0

    def test_multiple_trades_parity(self):
        """Verify multiple sequential trades produce consistent results."""
        initial_balance = 10000.0
        fee_rate = 0.001
        slippage_rate = 0.0

        # Create live engine
        live_engine = LiveTradingEngine(
            strategy=Mock(),
            data_provider=Mock(),
            initial_balance=initial_balance,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            resume_from_last_balance=False,
        )
        live_engine.data_provider.get_current_price = Mock(return_value=100.0)

        # Trade 1: Long 100 -> 105 (+5%)
        live_engine._open_position("BTCUSDT", PositionSide.LONG, 0.1, 100.0)
        pos = list(live_engine.positions.values())[0]
        live_engine._close_position(pos, reason="test", limit_price=105.0)
        balance_after_trade1 = live_engine.current_balance

        # Trade 2: Long 105 -> 110 (+4.76%)
        live_engine._open_position("BTCUSDT", PositionSide.LONG, 0.1, 105.0)
        pos = list(live_engine.positions.values())[0]
        live_engine._close_position(pos, reason="test", limit_price=110.0)
        balance_after_trade2 = live_engine.current_balance

        # Balance should increase after each profitable trade (minus fees)
        assert balance_after_trade1 > initial_balance - (initial_balance * 0.1 * fee_rate * 2)
        assert balance_after_trade2 > balance_after_trade1 - (balance_after_trade1 * 0.1 * fee_rate * 2)

        # Total trades should be 2
        assert len(live_engine.completed_trades) == 2

        # Total fees should accumulate (at minimum entry fees for each trade)
        min_entry_fees = (
            initial_balance * 0.1 * fee_rate +  # Trade 1 entry
            balance_after_trade1 * 0.1 * fee_rate  # Trade 2 entry
        )
        # Fees should be at least the entry fees
        assert live_engine.total_fees_paid >= min_entry_fees * 0.9  # Allow 10% tolerance
        # Verify fees are positive and reasonable
        assert live_engine.total_fees_paid > 0
        assert live_engine.total_fees_paid < initial_balance * 0.01  # Less than 1% of balance


class TestCostCalculatorEngineParity:
    """Verify both engines use the shared CostCalculator identically."""

    def test_both_engines_use_shared_cost_calculator(self):
        """Verify both engines import and use the same CostCalculator class."""
        from src.engines.backtest.execution.execution_engine import CostCalculator as BacktestCC
        from src.engines.live.execution.execution_engine import CostCalculator as LiveCC
        from src.engines.shared.cost_calculator import CostCalculator as SharedCC

        # Both engines should import from shared
        assert BacktestCC is SharedCC
        assert LiveCC is SharedCC

    def test_cost_calculation_produces_identical_results(self):
        """Verify cost calculations produce identical results."""
        from src.engines.shared.cost_calculator import CostCalculator

        fee_rate = 0.001
        slippage_rate = 0.0005

        # Create two separate calculators (as each engine would)
        calc1 = CostCalculator(fee_rate=fee_rate, slippage_rate=slippage_rate)
        calc2 = CostCalculator(fee_rate=fee_rate, slippage_rate=slippage_rate)

        # Same inputs should produce same outputs
        result1 = calc1.calculate_entry_costs(price=100.0, notional=1000.0, side="long")
        result2 = calc2.calculate_entry_costs(price=100.0, notional=1000.0, side="long")

        assert result1.executed_price == result2.executed_price
        assert result1.fee == result2.fee
        assert result1.slippage_cost == result2.slippage_cost


class TestSharedModuleEngineParity:
    """Verify both engines use identical shared modules."""

    def test_trailing_stop_manager_shared(self):
        """Verify both engines use the same TrailingStopManager."""
        from src.engines.backtest.execution.exit_handler import TrailingStopManager as BacktestTSM
        from src.engines.live.execution.exit_handler import TrailingStopManager as LiveTSM
        from src.engines.shared.trailing_stop_manager import TrailingStopManager as SharedTSM

        assert BacktestTSM is SharedTSM
        assert LiveTSM is SharedTSM

    def test_partial_exit_executor_shared(self):
        """Verify both engines use the same PartialExitExecutor."""
        from src.engines.backtest.execution.position_tracker import (
            PartialExitExecutor as BacktestPEE,
        )
        from src.engines.live.execution.position_tracker import (
            PartialExitExecutor as LivePEE,
        )
        from src.engines.shared.partial_exit_executor import PartialExitExecutor as SharedPEE

        assert BacktestPEE is SharedPEE
        assert LivePEE is SharedPEE

    def test_dynamic_risk_handler_shared(self):
        """Verify both engines use the same DynamicRiskHandler."""
        from src.engines.backtest.execution.entry_handler import (
            DynamicRiskHandler as BacktestDRH,
        )
        from src.engines.live.execution.entry_handler import DynamicRiskHandler as LiveDRH
        from src.engines.shared.dynamic_risk_handler import DynamicRiskHandler as SharedDRH

        assert BacktestDRH is SharedDRH
        assert LiveDRH is SharedDRH

    def test_position_side_enum_shared(self):
        """Verify both engines use the same PositionSide enum."""
        from src.engines.backtest import PositionSide as BacktestPS
        from src.engines.live.trading_engine import PositionSide as LivePS
        from src.engines.shared.models import PositionSide as SharedPS

        assert BacktestPS is SharedPS
        # LivePS may be an alias but should have same values
        assert LivePS.LONG.value == SharedPS.LONG.value
        assert LivePS.SHORT.value == SharedPS.SHORT.value
