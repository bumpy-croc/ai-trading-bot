from __future__ import annotations

import pytest

from src.optimizer.strategy_drift import (
    DriftConfig,
    DriftReport,
    DriftSeverity,
    StrategyDriftDetector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _baseline_kwargs(
    sharpe_mean: float = 1.0,
    sharpe_std: float = 0.3,
    win_rate_mean: float = 55.0,
    win_rate_std: float = 5.0,
    drawdown_mean: float = 10.0,
    drawdown_std: float = 3.0,
) -> dict:
    """Return baseline keyword arguments for detect()."""
    return {
        "baseline_sharpe_mean": sharpe_mean,
        "baseline_sharpe_std": sharpe_std,
        "baseline_win_rate_mean": win_rate_mean,
        "baseline_win_rate_std": win_rate_std,
        "baseline_drawdown_mean": drawdown_mean,
        "baseline_drawdown_std": drawdown_std,
    }


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

class TestDriftSeverityNone:
    def test_performance_within_range(self):
        detector = StrategyDriftDetector()
        report = detector.detect(
            **_baseline_kwargs(),
            live_sharpe=0.9,
            live_win_rate=53.0,
            live_max_drawdown=11.0,
        )
        assert report.severity == DriftSeverity.NONE
        assert "No action" in report.recommendation


class TestDriftSeverityMild:
    def test_slightly_degraded_sharpe(self):
        detector = StrategyDriftDetector()
        # Sharpe z = (0.5 - 1.0) / 0.3 ≈ -1.67 → beyond mild (1.5) but below severe (2.0)
        report = detector.detect(
            **_baseline_kwargs(),
            live_sharpe=0.5,
            live_win_rate=54.0,
            live_max_drawdown=10.0,
        )
        assert report.severity == DriftSeverity.MILD
        assert "Monitor" in report.recommendation


class TestDriftSeveritySevere:
    def test_significant_sharpe_drop(self):
        detector = StrategyDriftDetector()
        # Sharpe z = (0.3 - 1.0) / 0.3 ≈ -2.33 → beyond severe (2.0) but below critical (2.5)
        report = detector.detect(
            **_baseline_kwargs(),
            live_sharpe=0.3,
            live_win_rate=54.0,
            live_max_drawdown=10.0,
        )
        assert report.severity == DriftSeverity.SEVERE
        assert "reducing position sizes" in report.recommendation


class TestDriftSeverityCritical:
    def test_extreme_performance_degradation(self):
        detector = StrategyDriftDetector()
        # Sharpe z = (-0.5 - 1.0) / 0.3 = -5.0 → well beyond critical
        report = detector.detect(
            **_baseline_kwargs(),
            live_sharpe=-0.5,
            live_win_rate=35.0,
            live_max_drawdown=25.0,
        )
        assert report.severity == DriftSeverity.CRITICAL
        assert "pausing" in report.recommendation.lower()

    def test_drawdown_spike_triggers_critical(self):
        detector = StrategyDriftDetector()
        # Drawdown z = -(20.0 - 10.0)/3.0 = -3.33 → critical
        report = detector.detect(
            **_baseline_kwargs(),
            live_sharpe=0.9,
            live_win_rate=54.0,
            live_max_drawdown=20.0,
        )
        assert report.severity == DriftSeverity.CRITICAL


# ---------------------------------------------------------------------------
# Z-score edge cases
# ---------------------------------------------------------------------------

class TestZScore:
    def test_zero_std_no_deviation(self):
        z = StrategyDriftDetector._z_score(1.0, 1.0, 0.0)
        assert z == 0.0

    def test_zero_std_with_deviation(self):
        z = StrategyDriftDetector._z_score(0.5, 1.0, 0.0)
        assert z < 0  # worse than baseline

    def test_normal_z_score(self):
        z = StrategyDriftDetector._z_score(0.5, 1.0, 0.25)
        assert z == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# Custom thresholds
# ---------------------------------------------------------------------------

class TestCustomConfig:
    def test_tighter_thresholds(self):
        """Tighter thresholds should flag drift sooner."""
        cfg = DriftConfig(mild_z=0.5, severe_z=1.0, critical_z=1.5)
        detector = StrategyDriftDetector(cfg)
        report = detector.detect(
            **_baseline_kwargs(),
            live_sharpe=0.7,
            live_win_rate=54.0,
            live_max_drawdown=10.0,
        )
        # Sharpe z = (0.7 - 1.0) / 0.3 = -1.0, abs(1.0) >= severe(1.0) but < critical(1.5)
        assert report.severity == DriftSeverity.SEVERE

    def test_looser_thresholds(self):
        """Looser thresholds should tolerate more deviation."""
        cfg = DriftConfig(mild_z=3.0, severe_z=4.0, critical_z=5.0)
        detector = StrategyDriftDetector(cfg)
        report = detector.detect(
            **_baseline_kwargs(),
            live_sharpe=0.3,
            live_win_rate=45.0,
            live_max_drawdown=15.0,
        )
        assert report.severity == DriftSeverity.NONE


# ---------------------------------------------------------------------------
# Report details
# ---------------------------------------------------------------------------

class TestReportDetails:
    def test_details_contain_live_and_baseline(self):
        detector = StrategyDriftDetector()
        report = detector.detect(
            **_baseline_kwargs(),
            live_sharpe=0.8,
            live_win_rate=50.0,
            live_max_drawdown=12.0,
        )
        assert report.details["live_sharpe"] == 0.8
        assert report.details["live_win_rate"] == 50.0
        assert report.details["live_max_drawdown"] == 12.0
        assert report.details["baseline_sharpe_mean"] == 1.0

    def test_z_scores_populated(self):
        detector = StrategyDriftDetector()
        report = detector.detect(
            **_baseline_kwargs(),
            live_sharpe=0.8,
            live_win_rate=50.0,
            live_max_drawdown=12.0,
        )
        assert isinstance(report.sharpe_z, float)
        assert isinstance(report.win_rate_z, float)
        assert isinstance(report.drawdown_z, float)


# ---------------------------------------------------------------------------
# Drift severity enum
# ---------------------------------------------------------------------------

class TestDriftSeverityEnum:
    def test_values(self):
        assert DriftSeverity.NONE.value == "NONE"
        assert DriftSeverity.MILD.value == "MILD"
        assert DriftSeverity.SEVERE.value == "SEVERE"
        assert DriftSeverity.CRITICAL.value == "CRITICAL"

    def test_ordering_by_severity(self):
        """Verify the enum members exist in expected order."""
        members = list(DriftSeverity)
        assert members == [
            DriftSeverity.NONE,
            DriftSeverity.MILD,
            DriftSeverity.SEVERE,
            DriftSeverity.CRITICAL,
        ]
