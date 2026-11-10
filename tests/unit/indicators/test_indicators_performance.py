import time

import numpy as np
import pandas as pd
import pytest

from src.tech.indicators.core import calculate_ema, calculate_rsi

# Not `fast`: wall-clock timing is inherently sensitive to machine load, and this
# repo routinely runs several concurrent pytest workers across worktrees (see GH #1174).
# `slow` keeps this out of the pre-push `fast` gate while still running in CI/full suites.
pytestmark = [pytest.mark.unit, pytest.mark.slow, pytest.mark.performance, pytest.mark.mock_only]


class TestIndicatorPerformance:
    @pytest.mark.timeout(30)
    def test_indicators_performance(self):
        """EMA/RSI on 10k rows should complete quickly - guards against gross regressions.

        Threshold is deliberately generous (not a tight benchmark) so it tolerates
        contention from concurrent test runs while still catching an accidental
        algorithmic blowup (e.g. O(n^2) creeping into a rolling calculation).
        """
        large_data = pd.Series(np.random.randn(10000))
        start_time = time.time()
        _ = calculate_ema(large_data, period=20)
        _ = calculate_rsi(large_data, period=14)
        assert time.time() - start_time < 15.0
