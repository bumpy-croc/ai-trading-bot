from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.optimizer.schemas import ExperimentConfig, ExperimentResult
from src.optimizer.walk_forward import (
    FoldResult,
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardResult,
    compute_windows,
    robustness_label,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    sharpe: float = 1.0,
    total_return: float = 10.0,
    max_drawdown: float = 5.0,
    win_rate: float = 55.0,
    total_trades: int = 50,
    **overrides,
) -> ExperimentResult:
    """Factory for ExperimentResult with sensible defaults."""
    cfg = ExperimentConfig(
        strategy_name="ml_basic",
        symbol="BTCUSDT",
        timeframe="1h",
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 6, 1, tzinfo=UTC),
        initial_balance=10_000.0,
        provider="mock",
    )
    base = {
        "config": cfg,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "total_return": total_return,
        "annualized_return": total_return * 2,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "final_balance": 10_000 + total_return * 100,
    }
    base.update(overrides)
    return ExperimentResult(**base)


# ---------------------------------------------------------------------------
# Window splitting
# ---------------------------------------------------------------------------

class TestComputeWindows:
    def test_single_fold(self):
        end = datetime(2025, 7, 1, tzinfo=UTC)
        windows = compute_windows(end, train_days=180, test_days=30, step_days=30, num_folds=1)

        assert len(windows) == 1
        train_start, train_end, test_start, test_end = windows[0]
        assert test_end == end
        assert test_start == end - timedelta(days=30)
        assert train_end == test_start
        assert train_start == train_end - timedelta(days=180)

    def test_multiple_folds_ordered_chronologically(self):
        end = datetime(2025, 12, 31, tzinfo=UTC)
        windows = compute_windows(end, train_days=90, test_days=30, step_days=30, num_folds=4)

        assert len(windows) == 4
        # Verify chronological order
        for i in range(1, len(windows)):
            assert windows[i][0] > windows[i - 1][0], "Windows should be chronological"

    def test_no_overlap_between_train_and_test(self):
        end = datetime(2025, 12, 31, tzinfo=UTC)
        windows = compute_windows(end, train_days=60, test_days=30, step_days=30, num_folds=3)

        for train_start, train_end, test_start, test_end in windows:
            assert train_end == test_start, "Train end should equal test start (no gap/overlap)"
            assert train_start < train_end
            assert test_start < test_end

    def test_step_days_controls_spacing(self):
        end = datetime(2025, 12, 31, tzinfo=UTC)
        windows = compute_windows(end, train_days=60, test_days=30, step_days=15, num_folds=3)

        # Last fold test_end is 'end', second-to-last is 'end - 15 days'
        assert windows[-1][3] == end
        assert windows[-2][3] == end - timedelta(days=15)
        assert windows[-3][3] == end - timedelta(days=30)


# ---------------------------------------------------------------------------
# Robustness label
# ---------------------------------------------------------------------------

class TestRobustnessLabel:
    def test_robust(self):
        assert robustness_label(0.8, 0.5, 0.7) == "ROBUST"

    def test_acceptable(self):
        assert robustness_label(0.6, 0.5, 0.7) == "ACCEPTABLE"

    def test_weak(self):
        assert robustness_label(0.3, 0.5, 0.7) == "WEAK"

    def test_poor_negative(self):
        assert robustness_label(-0.1, 0.5, 0.7) == "POOR"

    def test_poor_zero(self):
        assert robustness_label(0.0, 0.5, 0.7) == "POOR"

    def test_boundary_good(self):
        assert robustness_label(0.5, 0.5, 0.7) == "ACCEPTABLE"

    def test_boundary_strong(self):
        assert robustness_label(0.7, 0.5, 0.7) == "ROBUST"


# ---------------------------------------------------------------------------
# Safe robustness ratio
# ---------------------------------------------------------------------------

class TestSafeRobustness:
    def test_normal_positive(self):
        ratio = WalkForwardAnalyzer._safe_robustness(0.8, 1.0)
        assert ratio == pytest.approx(0.8)

    def test_is_sharpe_zero_returns_zero(self):
        # Zero IS Sharpe provides no valid baseline
        assert WalkForwardAnalyzer._safe_robustness(0.5, 0.0) == 0.0
        assert WalkForwardAnalyzer._safe_robustness(-0.5, 0.0) == 0.0

    def test_is_sharpe_negative_returns_zero(self):
        # Negative IS Sharpe means no valid baseline to compare against
        assert WalkForwardAnalyzer._safe_robustness(0.8, -1.0) == 0.0
        assert WalkForwardAnalyzer._safe_robustness(-0.5, -0.3) == 0.0

    def test_clamped_high(self):
        ratio = WalkForwardAnalyzer._safe_robustness(5.0, 1.0)
        assert ratio == 2.0

    def test_clamped_low(self):
        ratio = WalkForwardAnalyzer._safe_robustness(-5.0, 1.0)
        assert ratio == -1.0


# ---------------------------------------------------------------------------
# Overfitting detection
# ---------------------------------------------------------------------------

class TestOverfittingDetection:
    def _make_analyzer(self) -> WalkForwardAnalyzer:
        cfg = WalkForwardConfig(provider="mock", random_seed=42)
        return WalkForwardAnalyzer(cfg)

    def test_overfitting_detected_when_is_much_higher(self):
        analyzer = self._make_analyzer()
        folds = [
            FoldResult(
                fold_index=0,
                train_start=datetime(2025, 1, 1, tzinfo=UTC),
                train_end=datetime(2025, 4, 1, tzinfo=UTC),
                test_start=datetime(2025, 4, 1, tzinfo=UTC),
                test_end=datetime(2025, 5, 1, tzinfo=UTC),
                is_sharpe=2.0,
                oos_sharpe=0.3,
                is_return=20.0,
                oos_return=2.0,
                is_max_drawdown=5.0,
                oos_max_drawdown=10.0,
                is_win_rate=65.0,
                oos_win_rate=45.0,
                is_total_trades=100,
                oos_total_trades=20,
                robustness_ratio=0.15,
            )
        ]
        result = analyzer._aggregate(folds)
        assert result.overfitting_detected is True

    def test_no_overfitting_when_oos_close_to_is(self):
        analyzer = self._make_analyzer()
        folds = [
            FoldResult(
                fold_index=0,
                train_start=datetime(2025, 1, 1, tzinfo=UTC),
                train_end=datetime(2025, 4, 1, tzinfo=UTC),
                test_start=datetime(2025, 4, 1, tzinfo=UTC),
                test_end=datetime(2025, 5, 1, tzinfo=UTC),
                is_sharpe=1.0,
                oos_sharpe=0.9,
                is_return=10.0,
                oos_return=8.0,
                is_max_drawdown=5.0,
                oos_max_drawdown=6.0,
                is_win_rate=55.0,
                oos_win_rate=52.0,
                is_total_trades=100,
                oos_total_trades=20,
                robustness_ratio=0.9,
            )
        ]
        result = analyzer._aggregate(folds)
        assert result.overfitting_detected is False


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class TestAggregation:
    def _make_analyzer(self) -> WalkForwardAnalyzer:
        cfg = WalkForwardConfig(provider="mock", random_seed=42)
        return WalkForwardAnalyzer(cfg)

    def test_empty_folds_returns_defaults(self):
        analyzer = self._make_analyzer()
        result = analyzer._aggregate([])
        assert result.num_folds == 0
        assert result.mean_robustness_ratio == 0.0
        assert result.robustness_label == "POOR"

    def test_mean_and_median_computed_correctly(self):
        analyzer = self._make_analyzer()
        base = {
            "train_start": datetime(2025, 1, 1, tzinfo=UTC),
            "train_end": datetime(2025, 4, 1, tzinfo=UTC),
            "test_start": datetime(2025, 4, 1, tzinfo=UTC),
            "test_end": datetime(2025, 5, 1, tzinfo=UTC),
            "is_return": 10.0,
            "oos_return": 8.0,
            "is_max_drawdown": 5.0,
            "oos_max_drawdown": 6.0,
            "is_win_rate": 55.0,
            "oos_win_rate": 52.0,
            "is_total_trades": 100,
            "oos_total_trades": 20,
        }
        folds = [
            FoldResult(fold_index=0, is_sharpe=1.0, oos_sharpe=0.8, robustness_ratio=0.8, **base),
            FoldResult(fold_index=1, is_sharpe=1.2, oos_sharpe=0.6, robustness_ratio=0.5, **base),
            FoldResult(fold_index=2, is_sharpe=0.8, oos_sharpe=0.7, robustness_ratio=0.875, **base),
        ]
        result = analyzer._aggregate(folds)

        assert result.num_folds == 3
        assert result.mean_is_sharpe == pytest.approx(1.0, abs=0.01)
        assert result.mean_oos_sharpe == pytest.approx(0.7, abs=0.01)
        expected_mean = (0.8 + 0.5 + 0.875) / 3
        assert result.mean_robustness_ratio == pytest.approx(expected_mean, abs=0.01)
        # Median of [0.5, 0.8, 0.875] = 0.8
        assert result.median_robustness_ratio == pytest.approx(0.8, abs=0.01)


# ---------------------------------------------------------------------------
# Fold resolution
# ---------------------------------------------------------------------------

class TestFoldResolution:
    def test_explicit_num_folds(self):
        cfg = WalkForwardConfig(num_folds=10, train_days=90, test_days=30)
        analyzer = WalkForwardAnalyzer(cfg)
        num, step = analyzer._resolve_folds()
        assert num == 10
        assert step == 30  # defaults to test_days

    def test_explicit_step_days(self):
        cfg = WalkForwardConfig(num_folds=5, step_days=15, test_days=30)
        analyzer = WalkForwardAnalyzer(cfg)
        num, step = analyzer._resolve_folds()
        assert num == 5
        assert step == 15

    def test_derived_from_total_days(self):
        # total=360, train=180, test=30 → available=150, folds=150/30+1=6
        cfg = WalkForwardConfig(total_days=360, train_days=180, test_days=30)
        analyzer = WalkForwardAnalyzer(cfg)
        num, step = analyzer._resolve_folds()
        assert num == 6
        assert step == 30

    def test_total_days_too_small_raises(self):
        cfg = WalkForwardConfig(total_days=100, train_days=180, test_days=30)
        analyzer = WalkForwardAnalyzer(cfg)
        with pytest.raises(ValueError, match="too small"):
            analyzer._resolve_folds()

    def test_num_folds_and_total_days_raises(self):
        cfg = WalkForwardConfig(num_folds=5, total_days=360, train_days=180, test_days=30)
        analyzer = WalkForwardAnalyzer(cfg)
        with pytest.raises(ValueError, match="mutually exclusive"):
            analyzer._resolve_folds()

    def test_default_folds(self):
        cfg = WalkForwardConfig()
        analyzer = WalkForwardAnalyzer(cfg)
        num, step = analyzer._resolve_folds()
        assert num == 6


# ---------------------------------------------------------------------------
# Integration with mock runner
# ---------------------------------------------------------------------------

class TestWalkForwardRun:
    def test_run_with_mock_runner(self):
        """Verify end-to-end run with a mocked ExperimentRunner."""
        cfg = WalkForwardConfig(
            strategy_name="ml_basic",
            symbol="BTCUSDT",
            timeframe="1h",
            train_days=30,
            test_days=10,
            num_folds=3,
            provider="mock",
            random_seed=42,
        )

        mock_runner = MagicMock()
        # Alternate IS/OOS results for 3 folds (6 calls total)
        mock_runner.run.side_effect = [
            _make_result(sharpe=1.2),   # fold 0 IS
            _make_result(sharpe=0.8),   # fold 0 OOS
            _make_result(sharpe=1.0),   # fold 1 IS
            _make_result(sharpe=0.7),   # fold 1 OOS
            _make_result(sharpe=1.5),   # fold 2 IS
            _make_result(sharpe=1.0),   # fold 2 OOS
        ]

        analyzer = WalkForwardAnalyzer(cfg, runner=mock_runner)
        result = analyzer.run(end=datetime(2025, 12, 31, tzinfo=UTC))

        assert result.num_folds == 3
        assert mock_runner.run.call_count == 6

        # Check fold-level data
        assert result.folds[0].is_sharpe == 1.2
        assert result.folds[0].oos_sharpe == 0.8

        # Robustness should be acceptable (0.8/1.2 ≈ 0.67, 0.7/1.0 = 0.7, 1.0/1.5 ≈ 0.67)
        assert result.mean_robustness_ratio > 0.5
        assert result.robustness_label == "ACCEPTABLE"

    def test_run_detects_overfitting(self):
        """High IS Sharpe with low OOS Sharpe triggers overfitting flag."""
        cfg = WalkForwardConfig(
            train_days=30,
            test_days=10,
            num_folds=2,
            provider="mock",
            random_seed=42,
        )

        mock_runner = MagicMock()
        mock_runner.run.side_effect = [
            _make_result(sharpe=3.0),   # fold 0 IS  (great)
            _make_result(sharpe=0.2),   # fold 0 OOS (terrible)
            _make_result(sharpe=2.5),   # fold 1 IS  (great)
            _make_result(sharpe=0.1),   # fold 1 OOS (terrible)
        ]

        analyzer = WalkForwardAnalyzer(cfg, runner=mock_runner)
        result = analyzer.run(end=datetime(2025, 12, 31, tzinfo=UTC))

        assert result.overfitting_detected is True
        assert result.robustness_label == "WEAK"
