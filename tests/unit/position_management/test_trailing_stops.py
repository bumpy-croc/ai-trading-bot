import pytest

from src.position_management.trailing_stops import TrailingStopPolicy

pytestmark = pytest.mark.unit


def test_no_activation_before_threshold():
    policy = TrailingStopPolicy(
        activation_threshold=0.015, trailing_distance_pct=0.005, breakeven_threshold=0.02, breakeven_buffer=0.001
    )
    # +1.0% sized PnL (position size 1.0 for clarity) < 1.5% => not activated
    new_sl, activated, be = policy.update_trailing_stop(
        side="long",
        entry_price=100.0,
        current_price=101.0,
        existing_stop=None,
        position_fraction=1.0,
    )
    assert activated is False
    assert be is False
    assert new_sl is None


def test_activation_and_trailing_percentage_long():
    policy = TrailingStopPolicy(activation_threshold=0.01, trailing_distance_pct=0.005)
    # +2% pnl -> activate; distance = 0.5% of price = 102 * 0.005 = 0.51 => SL = 101.49
    new_sl, activated, be = policy.update_trailing_stop(
        side="long",
        entry_price=100.0,
        current_price=102.0,
        existing_stop=None,
        position_fraction=1.0,
    )
    assert activated is True
    assert be is False
    assert pytest.approx(new_sl, rel=1e-6) == 102.0 - (102.0 * 0.005)

    # Move to 103 -> SL should tighten only upwards
    new_sl2, activated2, be2 = policy.update_trailing_stop(
        side="long",
        entry_price=100.0,
        current_price=103.0,
        existing_stop=new_sl,
        position_fraction=1.0,
    )
    assert activated2 is True and be2 is False
    assert new_sl2 >= new_sl


def test_activation_and_trailing_percentage_short():
    policy = TrailingStopPolicy(activation_threshold=0.01, trailing_distance_pct=0.01)
    # Short side, +2% pnl -> price falls to 98; distance = 1% * 98 = 0.98 -> SL = 98 + 0.98 = 98.98
    new_sl, activated, be = policy.update_trailing_stop(
        side="short",
        entry_price=100.0,
        current_price=98.0,
        existing_stop=None,
        position_fraction=1.0,
    )
    assert activated is True
    assert be is False
    assert pytest.approx(new_sl, rel=1e-6) == 98.0 + (98.0 * 0.01)

    # Price moves further to 97; SL should tighten downward (lower)
    new_sl2, _, _ = policy.update_trailing_stop(
        side="short",
        entry_price=100.0,
        current_price=97.0,
        existing_stop=new_sl,
        position_fraction=1.0,
    )
    assert new_sl2 <= new_sl


def test_trailing_atr_precedence_over_pct():
    policy = TrailingStopPolicy(activation_threshold=0.0, trailing_distance_pct=0.02, atr_multiplier=2.0)
    # ATR-based distance used when atr provided
    new_sl, activated, be = policy.update_trailing_stop(
        side="long",
        entry_price=100.0,
        current_price=110.0,
        existing_stop=None,
        position_fraction=1.0,
        atr=1.5,
    )
    assert activated is True
    assert be is False
    assert pytest.approx(new_sl, rel=1e-6) == 110.0 - (1.5 * 2.0)


def test_breakeven_trigger_and_buffer_long():
    policy = TrailingStopPolicy(
        activation_threshold=0.005, trailing_distance_pct=0.005, breakeven_threshold=0.02, breakeven_buffer=0.001
    )
    # First activate but below breakeven
    sl1, act1, be1 = policy.update_trailing_stop(
        side="long",
        entry_price=100.0,
        current_price=101.0,  # +1% pnl
        existing_stop=None,
        position_fraction=1.0,
    )
    assert act1 is True and be1 is False and sl1 is not None

    # Now reach breakeven threshold at +2%
    sl2, act2, be2 = policy.update_trailing_stop(
        side="long",
        entry_price=100.0,
        current_price=102.0,
        existing_stop=sl1,
        position_fraction=1.0,
    )
    assert act2 is True and be2 is True
    expected_be = 100.0 * (1 + 0.001)
    assert sl2 >= expected_be


def test_never_loosen_stop():
    policy = TrailingStopPolicy(activation_threshold=0.0, trailing_distance_pct=0.01)
    # Long: initial SL
    sl1, act1, be1 = policy.update_trailing_stop(
        side="long",
        entry_price=100.0,
        current_price=105.0,
        existing_stop=None,
        position_fraction=1.0,
    )
    # Price pulls back; stop should not move down
    sl2, _, _ = policy.update_trailing_stop(
        side="long",
        entry_price=100.0,
        current_price=104.0,
        existing_stop=sl1,
        position_fraction=1.0,
    )
    assert sl2 == sl1