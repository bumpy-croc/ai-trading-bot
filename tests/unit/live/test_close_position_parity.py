from datetime import datetime
from unittest.mock import Mock

import pytest

from src.live.trading_engine import LiveTradingEngine, Position, PositionSide
from src.performance.metrics import Side, cash_pnl, pnl_percent


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
    )

    position = Position(
        symbol="TEST",
        side=side,
        size=fraction,
        entry_price=entry_price,
        entry_time=datetime.now(),
        order_id="order-1",
        original_size=fraction,
        current_size=fraction,
    )
    engine.positions[position.order_id] = position

    expected_pct = pnl_percent(entry_price, exit_price, Side(side.value), fraction)
    expected_cash = cash_pnl(expected_pct, initial_balance)

    engine._close_position(position, reason="unit-test")

    assert engine.current_balance == pytest.approx(initial_balance + expected_cash)
    assert engine.total_pnl == pytest.approx(expected_cash)
    assert position.order_id not in engine.positions
