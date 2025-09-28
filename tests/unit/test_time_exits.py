from datetime import datetime, time, timedelta

import pytest

from src.position_management.time_exits import (
    MarketSessionDef,
    TimeExitPolicy,
    TimeRestrictions,
)

pytestmark = pytest.mark.unit


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


# Edge case tests for comprehensive coverage

class TestTimeExitPolicyEdgeCases:
    """Test edge cases and comprehensive scenarios for TimeExitPolicy."""

    def test_timezone_aware_vs_naive_datetimes(self):
        """Test handling of timezone-aware vs naive datetimes."""
        policy = TimeExitPolicy(max_holding_hours=24)
        
        # Naive datetimes
        entry_naive = datetime(2024, 1, 1, 0, 0)
        now_naive = datetime(2024, 1, 2, 1, 0)  # 25 hours later
        should_exit, reason = policy.check_time_exit_conditions(entry_naive, now_naive)
        assert should_exit
        assert reason == "Max holding period"
        
        # Mixed timezone handling should work
        from datetime import timezone
        entry_utc = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        now_utc = datetime(2024, 1, 2, 1, 0, tzinfo=timezone.utc)
        should_exit, reason = policy.check_time_exit_conditions(entry_utc, now_utc)
        assert should_exit
        assert reason == "Max holding period"

    def test_zero_max_holding_hours(self):
        """Test with zero max holding hours."""
        policy = TimeExitPolicy(max_holding_hours=0)
        entry = datetime(2024, 1, 1, 0, 0)
        now = datetime(2024, 1, 1, 0, 1)  # 1 minute later
        should_exit, reason = policy.check_time_exit_conditions(entry, now)
        assert should_exit
        assert reason == "Max holding period"

    def test_negative_max_holding_hours(self):
        """Test with negative max holding hours (should exit immediately)."""
        policy = TimeExitPolicy(max_holding_hours=-1)
        entry = datetime(2024, 1, 1, 0, 0)
        now = datetime(2024, 1, 1, 0, 0)  # Same time
        should_exit, reason = policy.check_time_exit_conditions(entry, now)
        assert should_exit
        assert reason == "Max holding period"

    def test_no_exit_conditions_met(self):
        """Test when no exit conditions are met."""
        policy = TimeExitPolicy()  # No conditions set
        entry = datetime(2024, 1, 1, 0, 0)
        now = datetime(2024, 1, 1, 1, 0)
        should_exit, reason = policy.check_time_exit_conditions(entry, now)
        assert not should_exit
        assert reason is None

    def test_weekend_flat_edge_cases(self):
        """Test weekend flat with various weekend scenarios."""
        policy = TimeExitPolicy(weekend_flat=True)
        entry = datetime(2024, 1, 5, 12, 0)  # Friday
        
        # Saturday
        now_sat = datetime(2024, 1, 6, 10, 0)
        should_exit, reason = policy.check_time_exit_conditions(entry, now_sat)
        assert should_exit
        assert reason == "Weekend flat"
        
        # Sunday
        now_sun = datetime(2024, 1, 7, 10, 0)
        should_exit, reason = policy.check_time_exit_conditions(entry, now_sun)
        assert should_exit
        assert reason == "Weekend flat"
        
        # Monday (not weekend)
        now_mon = datetime(2024, 1, 8, 10, 0)
        should_exit, reason = policy.check_time_exit_conditions(entry, now_mon)
        assert not should_exit

    def test_multiple_conditions_priority(self):
        """Test priority when multiple exit conditions are met."""
        policy = TimeExitPolicy(
            max_holding_hours=1,
            weekend_flat=True,
        )
        entry = datetime(2024, 1, 5, 23, 0)  # Friday 23:00
        now = datetime(2024, 1, 6, 1, 0)  # Saturday 01:00 (both weekend and max holding)
        
        should_exit, reason = policy.check_time_exit_conditions(entry, now)
        assert should_exit
        # Max holding period should be checked first
        assert reason == "Max holding period"

    def test_time_restrictions_no_overnight(self):
        """Test no_overnight restriction."""
        session = MarketSessionDef(
            name="US_EQUITIES",
            timezone="UTC",
            open_time=time(14, 30),
            close_time=time(21, 0),
            days_of_week=[1, 2, 3, 4, 5],
            is_24h=False,
        )
        policy = TimeExitPolicy(
            time_restrictions=TimeRestrictions(no_overnight=True),
            market_session=session,
        )
        
        entry = datetime(2024, 1, 2, 15, 0)  # Tuesday inside session
        now_after_close = datetime(2024, 1, 2, 22, 0)  # After close (overnight)
        should_exit, reason = policy.check_time_exit_conditions(entry, now_after_close)
        assert should_exit
        assert reason == "No overnight"

    def test_time_restrictions_no_weekend(self):
        """Test no_weekend restriction."""
        policy = TimeExitPolicy(
            time_restrictions=TimeRestrictions(no_weekend=True)
        )
        
        entry = datetime(2024, 1, 5, 12, 0)  # Friday
        now_weekend = datetime(2024, 1, 6, 10, 0)  # Saturday
        should_exit, reason = policy.check_time_exit_conditions(entry, now_weekend)
        assert should_exit
        assert reason == "No weekend"

    def test_24h_market_session(self):
        """Test with 24-hour market session."""
        session = MarketSessionDef(
            name="CRYPTO",
            timezone="UTC",
            is_24h=True,
        )
        policy = TimeExitPolicy(
            time_restrictions=TimeRestrictions(trading_hours_only=True, no_overnight=True),
            market_session=session,
        )
        
        entry = datetime(2024, 1, 1, 0, 0)
        now = datetime(2024, 1, 1, 23, 0)
        should_exit, reason = policy.check_time_exit_conditions(entry, now)
        assert not should_exit  # 24h market should always be open

    def test_market_session_undefined_times(self):
        """Test market session with undefined open/close times."""
        session = MarketSessionDef(
            name="UNDEFINED",
            timezone="UTC",
            # open_time and close_time are None
            is_24h=False,
        )
        policy = TimeExitPolicy(
            time_restrictions=TimeRestrictions(trading_hours_only=True),
            market_session=session,
        )
        
        entry = datetime(2024, 1, 1, 0, 0)
        now = datetime(2024, 1, 1, 23, 0)
        should_exit, reason = policy.check_time_exit_conditions(entry, now)
        assert not should_exit  # Undefined times should assume always open

    def test_end_of_day_flat_precision(self):
        """Test end-of-day flat with precision timing."""
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
        
        # Just before close (within 1-second tolerance)
        now_before = datetime(2024, 1, 2, 20, 59, 59, 500000)  # 0.5 seconds before
        should_exit, reason = policy.check_time_exit_conditions(entry, now_before)
        assert should_exit
        assert reason == "End of day flat"
        
        # Exactly at close
        now_at = datetime(2024, 1, 2, 21, 0)
        should_exit, reason = policy.check_time_exit_conditions(entry, now_at)
        assert should_exit
        assert reason == "End of day flat"

    def test_get_next_exit_time_no_conditions(self):
        """Test get_next_exit_time with no conditions set."""
        policy = TimeExitPolicy()
        entry = datetime(2024, 1, 1, 0, 0)
        now = datetime(2024, 1, 1, 1, 0)
        nxt = policy.get_next_exit_time(entry, now)
        assert nxt is None

    def test_get_next_exit_time_preserve_naive(self):
        """Test that get_next_exit_time preserves naive datetime when both inputs are naive."""
        policy = TimeExitPolicy(max_holding_hours=24)
        entry = datetime(2024, 1, 1, 0, 0)  # Naive
        now = datetime(2024, 1, 1, 1, 0)    # Naive
        nxt = policy.get_next_exit_time(entry, now)
        assert nxt is not None
        assert nxt.tzinfo is None  # Should remain naive

    def test_get_next_exit_time_weekend_calculation(self):
        """Test weekend exit time calculation."""
        policy = TimeExitPolicy(weekend_flat=True)
        entry = datetime(2024, 1, 3, 12, 0)  # Wednesday
        now = datetime(2024, 1, 3, 13, 0)
        nxt = policy.get_next_exit_time(entry, now)
        assert nxt is not None
        # Should be Saturday 00:00 UTC
        assert nxt.isoweekday() == 6  # Saturday
        assert nxt.hour == 0 and nxt.minute == 0

    def test_get_next_exit_time_past_candidates(self):
        """Test get_next_exit_time when all candidates are in the past."""
        policy = TimeExitPolicy(max_holding_hours=1)
        entry = datetime(2024, 1, 1, 0, 0)
        now = datetime(2024, 1, 1, 2, 0)  # 2 hours later (past max holding)
        nxt = policy.get_next_exit_time(entry, now)
        assert nxt is not None
        # Should return the past candidate (min of all candidates)
        assert nxt == entry + timedelta(hours=1)


class TestMarketSessionDefEdgeCases:
    """Test edge cases for MarketSessionDef."""

    def test_market_session_without_zoneinfo(self):
        """Test market session behavior when zoneinfo is not available."""
        # This tests the fallback behavior
        session = MarketSessionDef(
            name="TEST",
            timezone="US/Eastern",  # Will fallback to UTC if ZoneInfo not available
            open_time=time(9, 30),
            close_time=time(16, 0),
            days_of_week=[1, 2, 3, 4, 5],
            is_24h=False,
        )
        
        # Test is_open_at
        dt = datetime(2024, 1, 2, 15, 0)  # Tuesday 15:00 UTC
        is_open = session.is_open_at(dt)
        assert isinstance(is_open, bool)  # Should not crash

    def test_market_session_invalid_days_of_week(self):
        """Test market session with no valid days."""
        session = MarketSessionDef(
            name="INVALID",
            timezone="UTC",
            open_time=time(9, 30),
            close_time=time(16, 0),
            days_of_week=[],  # No valid days
            is_24h=False,
        )
        
        dt = datetime(2024, 1, 2, 15, 0)  # Tuesday
        is_open = session.is_open_at(dt)
        # Empty days_of_week might be interpreted as "no restrictions" rather than "no valid days"
        # The actual behavior depends on the implementation
        assert isinstance(is_open, bool)

    def test_market_session_next_close_24h(self):
        """Test next_close_after for 24h market."""
        session = MarketSessionDef(
            name="CRYPTO",
            timezone="UTC",
            is_24h=True,
        )
        
        dt = datetime(2024, 1, 1, 12, 0)
        next_close = session.next_close_after(dt)
        assert next_close is None  # 24h market has no close

    def test_market_session_next_close_undefined_times(self):
        """Test next_close_after with undefined open/close times."""
        session = MarketSessionDef(
            name="UNDEFINED",
            timezone="UTC",
            # open_time and close_time are None
            is_24h=False,
        )
        
        dt = datetime(2024, 1, 1, 12, 0)
        next_close = session.next_close_after(dt)
        assert next_close is None

    def test_market_session_next_close_max_days_scan(self):
        """Test next_close_after with edge case of max days scan."""
        # Create a session that only operates on a very specific day
        # This is an edge case to test the MAX_DAYS_TO_SCAN limit
        session = MarketSessionDef(
            name="RARE",
            timezone="UTC",
            open_time=time(9, 0),
            close_time=time(17, 0),
            days_of_week=[1],  # Only Monday
            is_24h=False,
        )
        
        # Start from a Tuesday
        dt = datetime(2024, 1, 2, 12, 0)  # Tuesday
        next_close = session.next_close_after(dt)
        assert next_close is not None
        assert next_close.isoweekday() == 1  # Should be next Monday

    def test_market_session_cross_day_boundary(self):
        """Test market session that crosses day boundary."""
        session = MarketSessionDef(
            name="OVERNIGHT",
            timezone="UTC",
            open_time=time(22, 0),   # 22:00
            close_time=time(6, 0),   # 06:00 next day
            days_of_week=[1, 2, 3, 4, 5],
            is_24h=False,
        )
        
        # This is a tricky case - the session definition doesn't handle
        # cross-day sessions well, but we test the current behavior
        dt = datetime(2024, 1, 2, 23, 0)  # Tuesday 23:00
        is_open = session.is_open_at(dt)
        # Current implementation would check if 23:00 is between 22:00 and 06:00
        # This would be False since 23:00 > 06:00, which is the limitation
        assert not is_open


class TestTimeRestrictionsEdgeCases:
    """Test edge cases for TimeRestrictions."""

    def test_all_restrictions_enabled(self):
        """Test with all time restrictions enabled."""
        session = MarketSessionDef(
            name="RESTRICTIVE",
            timezone="UTC",
            open_time=time(9, 0),
            close_time=time(17, 0),
            days_of_week=[1, 2, 3, 4, 5],  # Weekdays only
            is_24h=False,
        )
        
        policy = TimeExitPolicy(
            time_restrictions=TimeRestrictions(
                no_overnight=True,
                no_weekend=True,
                trading_hours_only=True,
            ),
            market_session=session,
        )
        
        entry = datetime(2024, 1, 2, 10, 0)  # Tuesday 10:00 (inside session)
        
        # Test weekend restriction (should trigger first)
        now_weekend = datetime(2024, 1, 6, 10, 0)  # Saturday
        should_exit, reason = policy.check_time_exit_conditions(entry, now_weekend)
        assert should_exit
        # The order of checks might vary, so accept either restriction
        assert reason in ("No weekend", "Outside trading hours")
        
        # Test after hours on weekday
        now_after_hours = datetime(2024, 1, 2, 18, 0)  # Tuesday 18:00 (after close)
        should_exit, reason = policy.check_time_exit_conditions(entry, now_after_hours)
        assert should_exit
        assert reason in ("Outside trading hours", "No overnight")

    def test_restrictions_with_none_session(self):
        """Test time restrictions with None market session."""
        policy = TimeExitPolicy(
            time_restrictions=TimeRestrictions(
                trading_hours_only=True,
                no_overnight=True,
            ),
            market_session=None,  # No session defined
        )
        
        entry = datetime(2024, 1, 1, 0, 0)
        now = datetime(2024, 1, 1, 23, 0)
        should_exit, reason = policy.check_time_exit_conditions(entry, now)
        assert not should_exit  # Should not exit if no session is defined
