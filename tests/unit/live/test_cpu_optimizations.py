"""
Unit tests for CPU optimization features in the live trading engine.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

from src.config.constants import (
    DEFAULT_CHECK_INTERVAL,
    DEFAULT_DATA_FRESHNESS_THRESHOLD,
    DEFAULT_MAX_CHECK_INTERVAL,
    DEFAULT_MIN_CHECK_INTERVAL,
    DEFAULT_SLEEP_POLL_INTERVAL,
)


class MockTradingEngine:
    """Mock trading engine for testing CPU optimizations without full dependencies"""

    def __init__(self):
        self.base_check_interval = DEFAULT_CHECK_INTERVAL
        self.min_check_interval = DEFAULT_MIN_CHECK_INTERVAL
        self.max_check_interval = DEFAULT_MAX_CHECK_INTERVAL
        self.data_freshness_threshold = DEFAULT_DATA_FRESHNESS_THRESHOLD
        self.positions = {}

    def _calculate_adaptive_interval(self, current_price=None):
        """Calculate adaptive check interval based on recent trading activity and market conditions"""
        interval = self.base_check_interval

        # Factor in recent trading activity
        recent_trades = len(
            [
                p
                for p in self.positions.values()
                if hasattr(p, "entry_time") and p.entry_time > datetime.now() - timedelta(hours=1)
            ]
        )
        if recent_trades > 0:
            interval = max(self.min_check_interval, interval // 2)
        elif len(self.positions) == 0:
            interval = min(self.max_check_interval, interval * 2)

        # Consider time of day (basic market hours awareness)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Off-hours (UTC)
            interval = min(self.max_check_interval, interval * 1.5)

        return int(interval)

    def _is_data_fresh(self, df):
        """Check if the data is fresh enough to warrant processing"""
        if df is None or df.empty:
            return False

        latest_timestamp = df.index[-1] if hasattr(df.index[-1], "timestamp") else datetime.now()
        if isinstance(latest_timestamp, str):
            try:
                latest_timestamp = pd.to_datetime(latest_timestamp)
            except (ValueError, TypeError):
                return True  # Assume fresh if we can't parse timestamp

        age_seconds = (datetime.now() - latest_timestamp).total_seconds()
        return age_seconds <= self.data_freshness_threshold


class MockPosition:
    """Mock position for testing"""

    def __init__(self, entry_time=None):
        self.entry_time = entry_time or datetime.now()


class TestCPUOptimizations(unittest.TestCase):
    """Test CPU optimization features"""

    def setUp(self):
        self.engine = MockTradingEngine()

    def test_adaptive_interval_no_positions(self):
        """Test that interval increases when there are no active positions"""
        # No positions should result in increased interval
        interval = self.engine._calculate_adaptive_interval()
        self.assertGreater(interval, self.engine.base_check_interval)
        self.assertLessEqual(interval, self.engine.max_check_interval)

    def test_adaptive_interval_with_recent_activity(self):
        """Test that interval decreases with recent trading activity"""
        # Add a recent position
        recent_position = MockPosition(datetime.now() - timedelta(minutes=30))
        self.engine.positions["TEST"] = recent_position

        interval = self.engine._calculate_adaptive_interval()
        self.assertLess(interval, self.engine.base_check_interval)
        self.assertGreaterEqual(interval, self.engine.min_check_interval)

    def test_adaptive_interval_off_hours(self):
        """Test that interval increases during off-market hours"""
        with patch("datetime.datetime") as mock_datetime:
            # Mock current hour to be off-hours (3 AM UTC)
            mock_datetime.now.return_value = datetime(2024, 1, 1, 3, 0, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            interval = self.engine._calculate_adaptive_interval()
            # Should be increased due to off-hours
            self.assertGreater(interval, self.engine.base_check_interval)

    def test_adaptive_interval_bounds(self):
        """Test that adaptive interval respects min/max bounds"""
        # Test maximum bound
        interval = self.engine._calculate_adaptive_interval()
        self.assertLessEqual(interval, self.engine.max_check_interval)

        # Test minimum bound with multiple recent positions
        for i in range(5):
            recent_position = MockPosition(datetime.now() - timedelta(minutes=10))
            self.engine.positions[f"TEST_{i}"] = recent_position

        interval = self.engine._calculate_adaptive_interval()
        self.assertGreaterEqual(interval, self.engine.min_check_interval)

    def test_data_freshness_check_fresh_data(self):
        """Test data freshness check with recent data"""
        # Create DataFrame with recent timestamp
        current_time = datetime.now()
        df = pd.DataFrame({"close": [100]}, index=[current_time - timedelta(seconds=30)])

        is_fresh = self.engine._is_data_fresh(df)
        self.assertTrue(is_fresh)

    def test_data_freshness_check_stale_data(self):
        """Test data freshness check with stale data"""
        # Create DataFrame with old timestamp
        old_time = datetime.now() - timedelta(seconds=self.engine.data_freshness_threshold + 60)
        df = pd.DataFrame({"close": [100]}, index=[old_time])

        is_fresh = self.engine._is_data_fresh(df)
        self.assertFalse(is_fresh)

    def test_data_freshness_check_empty_data(self):
        """Test data freshness check with empty data"""
        df = pd.DataFrame()
        is_fresh = self.engine._is_data_fresh(df)
        self.assertFalse(is_fresh)

        is_fresh = self.engine._is_data_fresh(None)
        self.assertFalse(is_fresh)

    def test_constants_are_reasonable(self):
        """Test that the CPU optimization constants have reasonable values"""
        self.assertGreater(DEFAULT_CHECK_INTERVAL, 0)
        self.assertGreater(DEFAULT_MIN_CHECK_INTERVAL, 0)
        self.assertGreater(DEFAULT_MAX_CHECK_INTERVAL, DEFAULT_CHECK_INTERVAL)
        self.assertLess(DEFAULT_MIN_CHECK_INTERVAL, DEFAULT_CHECK_INTERVAL)
        self.assertGreater(DEFAULT_SLEEP_POLL_INTERVAL, 0)
        self.assertLess(DEFAULT_SLEEP_POLL_INTERVAL, 1)  # Should be subsecond
        self.assertGreater(DEFAULT_DATA_FRESHNESS_THRESHOLD, 0)


if __name__ == "__main__":
    unittest.main()
