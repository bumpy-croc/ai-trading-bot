from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from src.engines.live.trading_engine import LiveTradingEngine, Position, PositionSide
from src.performance.metrics import Side, cash_pnl, pnl_percent
from src.strategies.components import (
    FixedFractionSizer,
    FixedRiskManager,
    HoldSignalGenerator,
    Strategy,
    StrategyRuntime,
)


@pytest.mark.parametrize(
    "side, fraction, entry_price, exit_price",
    [
        (PositionSide.LONG, 0.25, 100.0, 110.0),
        (PositionSide.SHORT, 0.4, 50.0, 45.0),
    ],
)
def test_close_position_cash_matches_backtester(side, fraction, entry_price, exit_price):
    """Ensure live engine realizes the same cash PnL as the backtester when closing."""

    strategy = Mock()
    strategy.get_risk_overrides.return_value = None
    data_provider = Mock()
    data_provider.get_current_price.return_value = exit_price

    initial_balance = 1_000.0
    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=data_provider,
        initial_balance=initial_balance,
        enable_live_trading=False,
        log_trades=False,
        # Disable fees/slippage to test pure P&L calculation
        fee_rate=0.0,
        slippage_rate=0.0,
    )

    position = Position(
        symbol="TEST",
        side=side,
        size=fraction,
        entry_price=entry_price,
        entry_time=datetime.now(UTC),
        order_id="order-1",
        original_size=fraction,
        current_size=fraction,
    )
    engine.live_position_tracker.track_recovered_position(position, db_id=None)

    expected_pct = pnl_percent(entry_price, exit_price, Side(side.value), fraction)
    expected_cash = cash_pnl(expected_pct, initial_balance)

    engine._execute_exit(
        position=position,
        reason="unit-test",
        limit_price=None,
        current_price=exit_price,
        candle_high=None,
        candle_low=None,
        candle=None,
    )

    assert engine.current_balance == pytest.approx(initial_balance + expected_cash)
    assert engine.performance_tracker.get_metrics().total_pnl == pytest.approx(expected_cash)
    assert position.order_id not in engine.live_position_tracker._positions


def _build_component_strategy() -> Strategy:
    signal = HoldSignalGenerator()
    risk = FixedRiskManager(risk_per_trade=0.01, stop_loss_pct=0.05)
    sizer = FixedFractionSizer(fraction=0.05)
    return Strategy("runtime_component", signal, risk, sizer)


def test_live_engine_accepts_strategy_runtime():
    component_strategy = _build_component_strategy()
    runtime = StrategyRuntime(component_strategy)

    data_provider = Mock()
    data_provider.get_current_price.return_value = 100.0

    engine = LiveTradingEngine(
        strategy=runtime,
        data_provider=data_provider,
        initial_balance=1_000.0,
        enable_live_trading=False,
        log_trades=False,
        enable_hot_swapping=False,
        # Disable fees/slippage for this structural test
        fee_rate=0.0,
        slippage_rate=0.0,
    )

    assert engine.strategy is component_strategy
    assert engine._runtime is runtime
    assert engine._component_strategy is component_strategy
