from datetime import time

import pytest

from src.data_providers.binance_provider import BinanceProvider
from src.live.trading_engine import LiveTradingEngine
from src.position_management.time_exits import MarketSessionDef, TimeExitPolicy
from src.strategies.ml_basic import MlBasic


@pytest.mark.skip(reason="Live integration smoke - requires exchange provider; kept for CI opt-in")
def test_live_engine_accepts_time_exit_policy_unique_name():
    strategy = MlBasic()
    dp = BinanceProvider()
    session = MarketSessionDef(
        name="US_EQUITIES",
        timezone="UTC",
        open_time=time(14, 30),
        close_time=time(21, 0),
        days_of_week=[1, 2, 3, 4, 5],
        is_24h=False,
    )
    policy = TimeExitPolicy(end_of_day_flat=True, market_session=session)

    engine = LiveTradingEngine(strategy=strategy, data_provider=dp, enable_live_trading=False, time_exit_policy=policy)
    assert engine.time_exit_policy is policy
