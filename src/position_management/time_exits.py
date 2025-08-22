"""
Time-based exit engine and utilities.

Implements configurable time-based exit strategies including maximum holding
periods, end-of-day flat, weekend flat, and time-of-day restrictions with
market-session and timezone awareness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Optional

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    # Fallback for environments without zoneinfo; will behave as naive UTC
    ZoneInfo = None  # type: ignore


@dataclass
class MarketSessionDef:
    """In-memory market session definition used by the policy.

    This mirrors the database `MarketSession` model but avoids DB dependency
    for core time computations so it can be used in backtests and live loops.
    """

    name: str
    timezone: str = "UTC"
    open_time: Optional[time] = None
    close_time: Optional[time] = None
    days_of_week: Optional[Sequence[int]] = None  # 1=Mon .. 7=Sun
    is_24h: bool = False

    def is_open_at(self, dt_utc: datetime) -> bool:
        if self.is_24h:
            return True

        if not self.open_time or not self.close_time:
            return True  # If undefined, assume always open

        tz = ZoneInfo(self.timezone) if ZoneInfo else None
        local_dt = dt_utc.astimezone(tz) if tz else dt_utc
        dow = local_dt.isoweekday()
        if self.days_of_week and dow not in self.days_of_week:
            return False
        local_t = local_dt.time()
        return self.open_time <= local_t <= self.close_time

    def next_close_after(self, dt_utc: datetime) -> Optional[datetime]:
        # Return None when we cannot determine a close time
        if self.is_24h:
            return None
        if not self.open_time or not self.close_time:
            return None

        # Bound the search to a reasonable number of days to avoid infinite loops
        MAX_DAYS_TO_SCAN = 14

        tz = ZoneInfo(self.timezone) if ZoneInfo else None
        local_dt = dt_utc.astimezone(tz) if tz else dt_utc

        # Start from today; if already past close, start from next day
        start_date = local_dt.date()
        close_today = datetime(
            year=start_date.year,
            month=start_date.month,
            day=start_date.day,
            hour=self.close_time.hour,
            minute=self.close_time.minute,
            second=self.close_time.second,
            microsecond=0,
            tzinfo=local_dt.tzinfo,
        )

        candidate_close = close_today
        if local_dt > candidate_close:
            candidate_close = candidate_close + timedelta(days=1)

        # If days_of_week is provided, move forward to the next valid weekday
        if self.days_of_week:
            for _ in range(MAX_DAYS_TO_SCAN):
                if candidate_close.isoweekday() in self.days_of_week:
                    break
                candidate_close = candidate_close + timedelta(days=1)
            else:
                raise ValueError("No valid session day found within MAX_DAYS_TO_SCAN")

        # Align the time to the session close on the chosen day (in local tz)
        candidate_close = candidate_close.replace(
            hour=self.close_time.hour,
            minute=self.close_time.minute,
            second=self.close_time.second,
            microsecond=0,
        )

        # Convert back to UTC if we used a timezone
        return candidate_close.astimezone(ZoneInfo("UTC")) if tz else candidate_close


@dataclass
class TimeRestrictions:
    no_overnight: bool = False
    no_weekend: bool = False
    trading_hours_only: bool = False


@dataclass
class TimeExitPolicy:
    max_holding_hours: Optional[int] = None
    end_of_day_flat: bool = False
    weekend_flat: bool = False
    market_timezone: str = "UTC"
    time_restrictions: TimeRestrictions = field(default_factory=TimeRestrictions)
    market_session: Optional[MarketSessionDef] = None

    def _as_utc(self, dt: datetime) -> datetime:
        # Normalize any naive datetime to UTC; assume already-UTC if tz-aware
        if dt.tzinfo is None:
            return dt.replace(tzinfo=ZoneInfo("UTC")) if ZoneInfo else dt
        return dt.astimezone(ZoneInfo("UTC")) if ZoneInfo else dt

    def check_time_exit_conditions(self, entry_time: datetime, now_time: datetime) -> tuple[bool, str | None]:
        """Return (should_exit, reason). Times can be naive (assumed UTC) or tz-aware.
        """
        now_utc = self._as_utc(now_time)
        entry_utc = self._as_utc(entry_time)

        # 1) Maximum holding period
        if self.max_holding_hours is not None:
            max_until = entry_utc + timedelta(hours=self.max_holding_hours)
            if now_utc >= max_until:
                return True, "Max holding period"

        # 2) Weekend flat: close before market enters weekend or if weekend
        if self.weekend_flat:
            dow = now_utc.isoweekday()
            if dow in (6, 7):
                return True, "Weekend flat"

        # 3) End-of-day flat: close at daily market close
        if self.end_of_day_flat and self.market_session:
            next_close = self.market_session.next_close_after(now_utc)
            if next_close:
                # If we are at or beyond close, exit
                # Use 1-minute tolerance to avoid precision issues
                if now_utc >= next_close - timedelta(seconds=1):
                    return True, "End of day flat"

        # 4) Time-of-day restrictions
        if self.time_restrictions.trading_hours_only and self.market_session:
            if not self.market_session.is_open_at(now_utc):
                return True, "Outside trading hours"

        if self.time_restrictions.no_overnight and self.market_session and not self.market_session.is_24h:
            # If session is currently closed (overnight), exit
            if not self.market_session.is_open_at(now_utc):
                return True, "No overnight"

        if self.time_restrictions.no_weekend:
            if now_utc.isoweekday() in (6, 7):
                return True, "No weekend"

        return False, None

    def get_next_exit_time(self, entry_time: datetime, now_time: datetime) -> Optional[datetime]:
        """Return the next scheduled exit time (UTC) based on policy, if any.
        """
        # Preserve naivety if both inputs are naive
        preserve_naive = entry_time.tzinfo is None and now_time.tzinfo is None
        now_utc = self._as_utc(now_time)
        entry_utc = self._as_utc(entry_time)

        candidates: list[datetime] = []

        if self.max_holding_hours is not None:
            candidates.append(entry_utc + timedelta(hours=self.max_holding_hours))

        if self.weekend_flat or self.time_restrictions.no_weekend:
            dow = now_utc.isoweekday()
            # Next Saturday 00:00 UTC
            days_until_sat = (6 - dow) % 7
            weekend_start = (now_utc + timedelta(days=days_until_sat)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            candidates.append(weekend_start)

        if self.end_of_day_flat and self.market_session:
            close = self.market_session.next_close_after(now_utc)
            if close:
                candidates.append(close)

        if not candidates:
            return None

        # Return soonest in the future
        future = [c for c in candidates if c > now_utc]
        nxt = min(future) if future else min(candidates)
        if preserve_naive:
            try:
                return nxt.replace(tzinfo=None)
            except Exception:
                return nxt
        return nxt

