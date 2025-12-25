import math

from src.engines.live.pnl import BalanceTracker


def test_balance_tracker_basic_progression():
    bt = BalanceTracker.start(1000.0)
    assert bt.current_balance == 1000.0
    assert math.isclose(bt.total_return_pct, 0.0)
    assert math.isclose(bt.current_drawdown_pct, 0.0)

    bt.apply_pnl(100.0)
    assert bt.current_balance == 1100.0
    assert bt.peak_balance == 1100.0
    assert math.isclose(bt.total_return_pct, 10.0)
    assert math.isclose(bt.current_drawdown_pct, 0.0)

    # Drawdown
    bt.apply_pnl(-200.0)
    assert bt.current_balance == 900.0
    assert bt.peak_balance == 1100.0
    assert math.isclose(bt.current_drawdown_pct, (1100.0 - 900.0) / 1100.0 * 100.0)
    # Max DD captured as fraction
    assert math.isclose(bt.max_drawdown, (1100.0 - 900.0) / 1100.0)


def test_balance_tracker_never_negative_divide():
    bt = BalanceTracker.start(0.0)
    assert math.isclose(bt.total_return_pct, 0.0)
    assert math.isclose(bt.current_drawdown_pct, 0.0)

    bt.apply_pnl(0.0)
    assert math.isclose(bt.total_return_pct, 0.0)
    assert math.isclose(bt.current_drawdown_pct, 0.0)
