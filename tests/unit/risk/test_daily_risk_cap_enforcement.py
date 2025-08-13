import pytest
import pandas as pd

from risk.risk_manager import RiskManager, RiskParameters

pytestmark = pytest.mark.unit


def test_daily_risk_cap_enforced_across_positions():
    params = RiskParameters(
        base_risk_per_trade=0.03,
        max_risk_per_trade=0.05,
        max_position_size=0.5,
        max_daily_risk=0.06,  # 6%
    )
    rm = RiskManager(parameters=params)

    # Minimal df with required columns
    df = pd.DataFrame({
        'close': [100, 101, 102],
        'atr': [1.0, 1.0, 1.0],
        'prediction_confidence': [1.0, 1.0, 1.0],
    })

    overrides = {
        'position_sizer': 'fixed_fraction',
        'base_fraction': 0.03,  # 3%
        'max_fraction': 0.5,
    }

    # First allocation should be ~3%
    f1 = rm.calculate_position_fraction(df=df, index=1, balance=10000, price=101.0, indicators={}, strategy_overrides=overrides)
    assert 0.0299 <= f1 <= 0.03
    rm.update_position(symbol='BTCUSDT_1', side='long', size=f1, entry_price=101.0)
    assert rm.daily_risk_used == pytest.approx(f1)

    # Second allocation should be limited by remaining daily risk (~3%)
    f2 = rm.calculate_position_fraction(df=df, index=2, balance=10000, price=102.0, indicators={}, strategy_overrides=overrides)
    assert 0.0299 <= f2 <= 0.03
    rm.update_position(symbol='BTCUSDT_2', side='long', size=f2, entry_price=102.0)
    assert rm.daily_risk_used == pytest.approx(f1 + f2)

    # Third allocation should be 0 due to max_daily_risk reached
    f3 = rm.calculate_position_fraction(df=df, index=2, balance=10000, price=102.0, indicators={}, strategy_overrides=overrides)
    assert f3 == 0.0