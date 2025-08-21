from datetime import datetime, time, timedelta

import pytest

from src.position_management.time_exits import (
    TimeExitPolicy,
    MarketSessionDef,
    TimeRestrictions,
)


def test_max_holding_hours_triggers_exit():
    policy = TimeExitPolicy(max_holding_hours=24)
    entry = datetime(2024, 1, 1, 0, 0)
    now = entry + timedelta(hours=25)
    should_exit, reason = policy.check_time_exit_conditions(entry, now)
    assert should_exit
    assert reason == "Max holding period"


def test_weekend_flat_triggers_on_weekend():
    policy = TimeExitPolicy(weekend_flat=True)
    entry = datetime(2024, 1, 5, 12, 0)  # Friday
    now = datetime(2024, 1, 6, 10, 0)  # Saturday
    should_exit, reason = policy.check_time_exit_conditions(entry, now)
    assert should_exit
    assert reason == "Weekend flat"


def test_trading_hours_only_exits_outside_session():
    session = MarketSessionDef(
        name="US_EQUITIES",
        timezone="UTC",
        open_time=time(14, 30),  # 9:30 ET
        close_time=time(21, 0),  # 16:00 ET
        days_of_week=[1, 2, 3, 4, 5],
        is_24h=False,
    )
    policy = TimeExitPolicy(
        time_restrictions=TimeRestrictions(trading_hours_only=True),
        market_session=session,
    )

    entry = datetime(2024, 1, 2, 15, 0)  # Tuesday 15:00 UTC (inside session)
    now_outside = datetime(2024, 1, 2, 22, 0)  # After close
    should_exit, reason = policy.check_time_exit_conditions(entry, now_outside)
    assert should_exit
    assert reason in ("Outside trading hours", "No overnight")


def test_end_of_day_flat_triggers_at_close():
    session = MarketSessionDef(
        name="US_EQUITIES",
        timezone="UTC",
        open_time=time(14, 30),
        close_time=time(21, 0),
        days_of_week=[1, 2, 3, 4, 5],
        is_24h=False,
    )
    policy = TimeExitPolicy(end_of_day_flat=True, market_session=session)
    entry = datetime(2024, 1, 2, 15, 0)
    # At close - 1 second tolerance
    now = datetime(2024, 1, 2, 21, 0)
    should_exit, reason = policy.check_time_exit_conditions(entry, now)
    assert should_exit
    assert reason == "End of day flat"


def test_get_next_exit_time_returns_soonest():
    session = MarketSessionDef(
        name="US_EQUITIES",
        timezone="UTC",
        open_time=time(14, 30),
        close_time=time(21, 0),
        days_of_week=[1, 2, 3, 4, 5],
        is_24h=False,
    )
    policy = TimeExitPolicy(max_holding_hours=48, end_of_day_flat=True, market_session=session)
    entry = datetime(2024, 1, 2, 15, 0)
    now = datetime(2024, 1, 2, 16, 0)
    nxt = policy.get_next_exit_time(entry, now)
    assert nxt is not None
    # Should be same day's close or max holding, whichever comes first
    assert nxt <= entry + timedelta(hours=48)
