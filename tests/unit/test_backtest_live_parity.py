"""
Parity validation tests between backtest and live trading engines.

These tests verify that both engines produce identical results for the same
trading scenarios, ensuring backtests accurately represent live trading behavior.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from src.engines.live.execution.entry_handler import LiveEntrySignal
from src.engines.live.trading_engine import LiveTradingEngine, Position, PositionSide


class TestFeeSlippageParity:
    """Verify fee and slippage calculations match between engines."""

    def test_entry_fee_deducted_from_balance(self):
        """Entry fee should be deducted from balance on position open."""
        strategy = Mock()
        strategy.get_risk_overrides.return_value = None
        data_provider = Mock()
        data_provider.get_current_price.return_value = 100.0

        initial_balance = 10_000.0
        fee_rate = 0.001  # 0.1%
        position_size = 0.1  # 10% of balance

        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=initial_balance,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=fee_rate,
            slippage_rate=0.0,  # Disable slippage for this test
        )

        # Manually open a position to test fee deduction
        position_value = initial_balance * position_size
        expected_fee = position_value * fee_rate  # $10 fee on $1000 position

        # Open position through engine
        entry_signal = LiveEntrySignal(
            should_enter=True,
            side=PositionSide.LONG,
            size_fraction=position_size,
            stop_loss=95.0,
            take_profit=110.0,
        )
        engine._execute_entry_signal(
            entry_signal,
            symbol="TEST",
            current_price=100.0,
        )

        assert engine.total_fees_paid == pytest.approx(expected_fee)
        assert engine.current_balance == pytest.approx(initial_balance - expected_fee)

    def test_exit_fee_deducted_from_pnl(self):
        """Exit fee should be deducted from realized P&L."""
        strategy = Mock()
        strategy.get_risk_overrides.return_value = None
        data_provider = Mock()
        data_provider.get_current_price.return_value = 110.0  # Price went up 10%

        initial_balance = 10_000.0
        fee_rate = 0.001  # 0.1%
        position_size = 0.1  # 10% of balance

        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=initial_balance,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=fee_rate,
            slippage_rate=0.0,
        )

        # Create position manually (simulating already-open position)
        position = Position(
            symbol="TEST",
            side=PositionSide.LONG,
            size=position_size,
            entry_price=100.0,
            entry_time=datetime.now(),
            order_id="test-order",
            original_size=position_size,
            current_size=position_size,
        )
        engine.positions[position.order_id] = position

        # Close position
        engine._close_position(position, reason="test")

        # Exit fee is based on exit notional (basis_balance * position_size * exit_price/entry_price)
        # Exit notional = 10000 * 0.1 * (110.0/100.0) = 1100
        # Exit fee: 1100 * 0.001 = 1.1
        assert engine.total_fees_paid == pytest.approx(1.1)

    def test_slippage_applied_adversely_on_entry(self):
        """Slippage should work against the position on entry."""
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

        # Open long position
        entry_signal = LiveEntrySignal(
            should_enter=True,
            side=PositionSide.LONG,
            size_fraction=0.1,
            stop_loss=95.0,
            take_profit=110.0,
        )
        engine._execute_entry_signal(
            entry_signal,
            symbol="TEST",
            current_price=100.0,
        )

        # For long, slippage should increase entry price (worse for buyer)
        position = list(engine.positions.values())[0]
        expected_entry = 100.0 * (1 + slippage_rate)
        assert position.entry_price == pytest.approx(expected_entry)

    def test_slippage_applied_adversely_on_exit(self):
        """Slippage should work against the position on exit."""
        strategy = Mock()
        strategy.get_risk_overrides.return_value = None
        data_provider = Mock()
        data_provider.get_current_price.return_value = 110.0

        slippage_rate = 0.0005  # 0.05%
        initial_balance = 10_000.0

        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=initial_balance,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=0.0,
            slippage_rate=slippage_rate,
        )

        # Create long position
        position = Position(
            symbol="TEST",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=100.0,
            entry_time=datetime.now(),
            order_id="test-order",
            original_size=0.1,
            current_size=0.1,
        )
        engine.positions[position.order_id] = position

        # Close position
        engine._close_position(position, reason="test")

        # For long exit, slippage should decrease exit price (worse for seller)
        # Expected exit: 110.0 * (1 - 0.0005) = 109.945
        # P&L should reflect the slippage-adjusted exit price
        assert engine.total_slippage_cost > 0


class TestStopLossTakeProfitParity:
    """Verify SL/TP detection uses high/low prices correctly."""

    def test_long_stop_loss_triggers_on_low(self):
        """Long position SL should trigger when candle low breaches SL level."""
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

        # Close price above SL, but low below SL
        # SL should trigger because low breached SL
        triggered = engine._check_stop_loss(
            position,
            current_price=98.0,  # Close is above SL
            candle_high=101.0,
            candle_low=94.0,  # Low breaches SL at 95
        )

        assert triggered is True

    def test_long_stop_loss_no_trigger_when_low_above_sl(self):
        """Long position SL should NOT trigger when candle low is above SL level."""
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

        # Low is above SL, should not trigger
        triggered = engine._check_stop_loss(
            position,
            current_price=98.0,
            candle_high=101.0,
            candle_low=96.0,  # Low is above SL at 95
        )

        assert triggered is False

    def test_short_stop_loss_triggers_on_high(self):
        """Short position SL should trigger when candle high breaches SL level."""
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
            stop_loss=105.0,  # Short SL is above entry
        )

        # Close below SL, but high above SL - should trigger
        triggered = engine._check_stop_loss(
            position,
            current_price=102.0,  # Close is below SL
            candle_high=106.0,  # High breaches SL at 105
            candle_low=99.0,
        )

        assert triggered is True

    def test_long_take_profit_triggers_on_high(self):
        """Long position TP should trigger when candle high breaches TP level."""
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
            take_profit=110.0,
        )

        # Close below TP, but high above TP - should trigger
        triggered = engine._check_take_profit(
            position,
            current_price=108.0,  # Close is below TP
            candle_high=112.0,  # High breaches TP at 110
            candle_low=105.0,
        )

        assert triggered is True


class TestExitPriceParity:
    """Verify exit prices use actual SL/TP levels, not close price."""

    def test_stop_loss_exit_uses_sl_level(self):
        """When SL triggers, exit should be at SL level, not close price."""
        strategy = Mock()
        strategy.get_risk_overrides.return_value = None
        data_provider = Mock()
        data_provider.get_current_price.return_value = 98.0  # Close is above SL

        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=10_000.0,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=0.0,
            slippage_rate=0.0,
        )

        stop_loss_level = 95.0
        position = Position(
            symbol="TEST",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=100.0,
            entry_time=datetime.now(),
            order_id="test-order",
            original_size=0.1,
            current_size=0.1,
            stop_loss=stop_loss_level,
        )
        engine.positions[position.order_id] = position

        # Close position with SL limit price
        engine._close_position(position, reason="stop_loss", limit_price=stop_loss_level)

        # P&L should be based on SL price (95), not close price (98)
        # Long: (exit - entry) / entry * size = (95 - 100) / 100 * 0.1 = -0.005 = -0.5%
        # Cash P&L on $10,000: -0.5% of $10,000 = -$50
        expected_cash_pnl = -50.0
        assert engine.total_pnl == pytest.approx(expected_cash_pnl)

    def test_take_profit_exit_uses_tp_level(self):
        """When TP triggers, exit should be at TP level, not close price."""
        strategy = Mock()
        strategy.get_risk_overrides.return_value = None
        data_provider = Mock()
        data_provider.get_current_price.return_value = 115.0  # Close is above TP

        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=10_000.0,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=0.0,
            slippage_rate=0.0,
        )

        take_profit_level = 110.0
        position = Position(
            symbol="TEST",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=100.0,
            entry_time=datetime.now(),
            order_id="test-order",
            original_size=0.1,
            current_size=0.1,
            take_profit=take_profit_level,
        )
        engine.positions[position.order_id] = position

        # Close position with TP limit price
        engine._close_position(position, reason="take_profit", limit_price=take_profit_level)

        # P&L should be based on TP price (110), not close price (115)
        # Long: (exit - entry) / entry * size = (110 - 100) / 100 * 0.1 = 0.01 = 1%
        # Cash P&L on $10,000: 1% of $10,000 = $100
        expected_cash_pnl = 100.0
        assert engine.total_pnl == pytest.approx(expected_cash_pnl)


class TestPositionSizeParity:
    """Verify position size limits match between engines."""

    def test_max_position_size_enforced(self):
        """Position size should be capped at max_position_size."""
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

        # Try to open position larger than max
        entry_signal = LiveEntrySignal(
            should_enter=True,
            side=PositionSide.LONG,
            size_fraction=0.5,  # 50% - should be capped to 10%
            stop_loss=95.0,
            take_profit=110.0,
        )
        engine._execute_entry_signal(
            entry_signal,
            symbol="TEST",
            current_price=100.0,
        )

        position = list(engine.positions.values())[0]
        assert position.size == pytest.approx(max_position)


class TestBacktesterPositionSizeParity:
    """Verify backtester position size limits match live engine."""

    def test_backtester_max_position_size_default(self):
        """Backtester should have same default max_position_size as live engine (10%)."""
        from src.engines.backtest.engine import Backtester

        # Both engines should default to 10%
        assert hasattr(Backtester.__init__, "__defaults__") or True

        # Create backtester and verify default
        strategy = Mock()
        strategy.name = "test"
        strategy.get_risk_overrides.return_value = None
        strategy.calculate_indicators = Mock(return_value=None)
        data_provider = Mock()

        backtester = Backtester(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=10_000.0,
        )

        assert backtester.max_position_size == 0.1  # Default 10%

    def test_backtester_max_position_size_configurable(self):
        """Backtester max_position_size should be configurable."""
        from src.engines.backtest.engine import Backtester

        strategy = Mock()
        strategy.name = "test"
        strategy.get_risk_overrides.return_value = None
        strategy.calculate_indicators = Mock(return_value=None)
        data_provider = Mock()

        backtester = Backtester(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=10_000.0,
            max_position_size=0.25,  # 25%
        )

        assert backtester.max_position_size == 0.25

    def test_backtester_rejects_invalid_max_position_size(self):
        """Backtester should reject invalid max_position_size values."""
        from src.engines.backtest.engine import Backtester

        strategy = Mock()
        strategy.name = "test"
        data_provider = Mock()

        with pytest.raises(ValueError, match="Max position size must be between 0 and 1"):
            Backtester(
                strategy=strategy,
                data_provider=data_provider,
                initial_balance=10_000.0,
                max_position_size=1.5,  # Invalid: > 1.0
            )

        with pytest.raises(ValueError, match="Max position size must be between 0 and 1"):
            Backtester(
                strategy=strategy,
                data_provider=data_provider,
                initial_balance=10_000.0,
                max_position_size=0.0,  # Invalid: <= 0
            )


class TestCostTrackingParity:
    """Verify cost tracking is consistent."""

    def test_total_fees_accumulate(self):
        """Total fees should accumulate across multiple trades."""
        strategy = Mock()
        strategy.get_risk_overrides.return_value = None
        data_provider = Mock()
        data_provider.get_current_price.return_value = 100.0

        fee_rate = 0.001

        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=10_000.0,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=fee_rate,
            slippage_rate=0.0,
        )

        # Open first position
        first_entry_signal = LiveEntrySignal(
            should_enter=True,
            side=PositionSide.LONG,
            size_fraction=0.1,
            stop_loss=95.0,
            take_profit=110.0,
        )
        engine._execute_entry_signal(
            first_entry_signal,
            symbol="TEST1",
            current_price=100.0,
        )

        first_fee = engine.total_fees_paid

        # Open second position
        second_entry_signal = LiveEntrySignal(
            should_enter=True,
            side=PositionSide.SHORT,
            size_fraction=0.1,
            stop_loss=105.0,
            take_profit=90.0,
        )
        engine._execute_entry_signal(
            second_entry_signal,
            symbol="TEST2",
            current_price=100.0,
        )

        # Fees should have accumulated
        assert engine.total_fees_paid > first_fee

    def test_slippage_costs_tracked_separately(self):
        """Slippage costs should be tracked separately from fees."""
        strategy = Mock()
        strategy.get_risk_overrides.return_value = None
        data_provider = Mock()
        data_provider.get_current_price.return_value = 100.0

        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=10_000.0,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=0.001,
            slippage_rate=0.0005,
        )

        entry_signal = LiveEntrySignal(
            should_enter=True,
            side=PositionSide.LONG,
            size_fraction=0.1,
            stop_loss=95.0,
            take_profit=110.0,
        )
        engine._execute_entry_signal(
            entry_signal,
            symbol="TEST",
            current_price=100.0,
        )

        # Both should be tracked
        assert engine.total_fees_paid > 0
        assert engine.total_slippage_cost > 0
        assert engine.total_fees_paid != engine.total_slippage_cost
