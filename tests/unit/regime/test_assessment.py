"""
Unit tests for Regime Assessment module.

Tests forward-looking accuracy, persistence metrics, transition analysis,
and confidence calibration computations.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.regime.assessment import (
    AssessmentMetrics,
    RegimeAssessment,
    RegimeAssessmentConfig,
    compare_detectors,
)
from src.regime.detector import RegimeDetector

TEST_RANDOM_SEED = 42


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Create sample OHLCV DataFrame for testing."""
    np.random.seed(TEST_RANDOM_SEED)
    n = 500

    # Generate trending price data
    base_price = 100
    returns = np.random.normal(0.001, 0.02, n)
    prices = base_price * np.cumprod(1 + returns)

    dates = pd.date_range(start="2023-01-01", periods=n, freq="1h")

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            "high": prices * (1 + np.random.uniform(0, 0.02, n)),
            "low": prices * (1 - np.random.uniform(0, 0.02, n)),
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n),
        },
        index=dates,
    )

    return df


@pytest.fixture
def annotated_df(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Create annotated DataFrame with regime labels."""
    detector = RegimeDetector()
    return detector.annotate(sample_ohlcv_df)


@pytest.fixture
def synthetic_annotated_df() -> pd.DataFrame:
    """
    Create a synthetic annotated DataFrame with predictable regime patterns.

    Regime pattern: 100 bars trend_up, 100 bars trend_down, 100 bars range (repeated)
    """
    np.random.seed(TEST_RANDOM_SEED)
    n = 600

    dates = pd.date_range(start="2023-01-01", periods=n, freq="1h")

    # Create prices that match regimes
    prices = []
    current_price = 100
    for i in range(n):
        regime_idx = (i // 100) % 3
        if regime_idx == 0:  # trend_up
            current_price *= 1.002
        elif regime_idx == 1:  # trend_down
            current_price *= 0.998
        else:  # range
            current_price *= (1 + np.random.uniform(-0.001, 0.001))
        prices.append(current_price)

    prices = np.array(prices)

    # Assign regime labels
    trend_labels = []
    vol_labels = []
    confidences = []

    for i in range(n):
        regime_idx = (i // 100) % 3
        if regime_idx == 0:
            trend_labels.append("trend_up")
            confidences.append(0.8)
        elif regime_idx == 1:
            trend_labels.append("trend_down")
            confidences.append(0.7)
        else:
            trend_labels.append("range")
            confidences.append(0.5)

        vol_labels.append("low_vol" if i % 200 < 100 else "high_vol")

    df = pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n),
            "trend_label": trend_labels,
            "vol_label": vol_labels,
            "regime_confidence": confidences,
        },
        index=dates,
    )

    return df


class TestRegimeAssessmentConfig:
    """Tests for RegimeAssessmentConfig."""

    def test_default_values(self):
        config = RegimeAssessmentConfig()
        assert config.lookahead == 20
        assert config.range_threshold == 0.02
        assert config.confidence_bins == 10
        assert config.min_periods_for_stats == 100

    def test_custom_values(self):
        config = RegimeAssessmentConfig(
            lookahead=50,
            range_threshold=0.03,
            confidence_bins=5,
        )
        assert config.lookahead == 50
        assert config.range_threshold == 0.03
        assert config.confidence_bins == 5


class TestAssessmentMetrics:
    """Tests for AssessmentMetrics dataclass."""

    def test_default_values(self):
        metrics = AssessmentMetrics()
        assert metrics.overall_accuracy == 0.0
        assert metrics.avg_regime_duration == 0.0
        assert metrics.total_transitions == 0
        assert isinstance(metrics.accuracy_by_regime, dict)

    def test_to_dict(self):
        metrics = AssessmentMetrics(
            overall_accuracy=0.65,
            avg_regime_duration=25.0,
            total_transitions=10,
        )
        result = metrics.to_dict()

        assert "forward_accuracy" in result
        assert result["forward_accuracy"]["overall"] == 0.65
        assert result["persistence"]["avg_duration"] == 25.0
        assert result["transitions"]["total"] == 10


class TestRegimeAssessment:
    """Tests for RegimeAssessment class."""

    def test_init_with_valid_data(self, annotated_df: pd.DataFrame):
        assessment = RegimeAssessment(annotated_df)
        assert assessment.df is not None
        assert len(assessment.df) == len(annotated_df)

    def test_init_validates_required_columns(self, sample_ohlcv_df: pd.DataFrame):
        # Missing regime columns should raise error
        with pytest.raises(ValueError, match="Missing required columns"):
            RegimeAssessment(sample_ohlcv_df)

    def test_compute_forward_accuracy(self, synthetic_annotated_df: pd.DataFrame):
        config = RegimeAssessmentConfig(lookahead=10)
        assessment = RegimeAssessment(synthetic_annotated_df, config)
        result = assessment.compute_forward_accuracy()

        assert "overall" in result
        assert "by_regime" in result
        assert "by_volatility" in result
        assert 0 <= result["overall"] <= 1

        # Check that we have accuracy for each regime type
        assert "trend_up" in result["by_regime"]
        assert "trend_down" in result["by_regime"]
        assert "range" in result["by_regime"]

    def test_compute_forward_accuracy_with_short_data(self):
        # Create very short DataFrame
        df = pd.DataFrame(
            {
                "close": [100, 101, 102],
                "trend_label": ["trend_up", "trend_up", "trend_up"],
                "vol_label": ["low_vol", "low_vol", "low_vol"],
                "regime_confidence": [0.8, 0.8, 0.8],
            }
        )
        config = RegimeAssessmentConfig(lookahead=20)
        assessment = RegimeAssessment(df, config)
        result = assessment.compute_forward_accuracy()

        # Should return zero accuracy due to insufficient data
        assert result["overall"] == 0.0

    def test_compute_persistence_metrics(self, synthetic_annotated_df: pd.DataFrame):
        assessment = RegimeAssessment(synthetic_annotated_df)
        result = assessment.compute_persistence_metrics()

        assert "avg_duration" in result
        assert "median_duration" in result
        assert "min_duration" in result
        assert "max_duration" in result
        assert "durations" in result

        # With our synthetic data, regimes last 100 bars each
        assert result["avg_duration"] == pytest.approx(100.0, abs=5)

    def test_compute_transition_analysis(self, synthetic_annotated_df: pd.DataFrame):
        assessment = RegimeAssessment(synthetic_annotated_df)
        result = assessment.compute_transition_analysis()

        assert "total_transitions" in result
        assert "transition_frequency" in result
        assert "transition_matrix" in result
        assert "transition_probabilities" in result

        # 600 bars with regime changes every 100 bars = 5 transitions
        assert result["total_transitions"] == 5

        # Transition frequency = 5/600
        expected_freq = 5 / 600
        assert result["transition_frequency"] == pytest.approx(expected_freq, abs=0.001)

    def test_compute_confidence_calibration(self, synthetic_annotated_df: pd.DataFrame):
        config = RegimeAssessmentConfig(lookahead=10, confidence_bins=5)
        assessment = RegimeAssessment(synthetic_annotated_df, config)
        result = assessment.compute_confidence_calibration()

        assert "expected_calibration_error" in result
        assert "curve" in result

        # ECE should be between 0 and 1
        assert 0 <= result["expected_calibration_error"] <= 1

    def test_compute_distribution(self, synthetic_annotated_df: pd.DataFrame):
        assessment = RegimeAssessment(synthetic_annotated_df)
        result = assessment.compute_distribution()

        assert "regime" in result
        assert "volatility" in result

        # Each regime should appear ~1/3 of the time
        for regime in ["trend_up", "trend_down", "range"]:
            assert regime in result["regime"]
            assert 0.3 <= result["regime"][regime] <= 0.35

    def test_compute_all_metrics(self, annotated_df: pd.DataFrame):
        assessment = RegimeAssessment(annotated_df)
        metrics = assessment.compute_all_metrics()

        assert isinstance(metrics, AssessmentMetrics)
        assert metrics.total_periods == len(annotated_df)
        assert metrics.assessment_timestamp != ""

        # All metric categories should be populated
        assert metrics.overall_accuracy >= 0
        assert metrics.avg_regime_duration >= 0
        assert metrics.total_transitions >= 0

    def test_generate_report(self, annotated_df: pd.DataFrame):
        assessment = RegimeAssessment(annotated_df)
        assessment.compute_all_metrics()
        report = assessment.generate_report()

        assert isinstance(report, str)
        assert "REGIME DETECTOR ASSESSMENT REPORT" in report
        assert "FORWARD-LOOKING ACCURACY" in report
        assert "REGIME PERSISTENCE" in report
        assert "REGIME TRANSITIONS" in report

    def test_save_metrics(self, annotated_df: pd.DataFrame, tmp_path):
        assessment = RegimeAssessment(annotated_df)
        assessment.compute_all_metrics()

        output_path = tmp_path / "metrics.json"
        assessment.save_metrics(output_path)

        assert output_path.exists()

        import json

        with open(output_path) as f:
            loaded = json.load(f)

        assert "forward_accuracy" in loaded
        assert "persistence" in loaded
        assert "transitions" in loaded


class TestCompareDetectors:
    """Tests for compare_detectors function."""

    def test_compare_detectors(self, sample_ohlcv_df: pd.DataFrame):
        detector1 = RegimeDetector()
        detector2 = RegimeDetector()  # Same detector for simplicity

        result = compare_detectors(sample_ohlcv_df, detector1, detector2)

        assert "detector1" in result
        assert "detector2" in result
        assert isinstance(result["detector1"], AssessmentMetrics)
        assert isinstance(result["detector2"], AssessmentMetrics)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self):
        df = pd.DataFrame(
            columns=["close", "trend_label", "vol_label", "regime_confidence"]
        )
        with pytest.raises(ValueError):
            RegimeAssessment(df)

    def test_single_regime(self):
        """Test with data that never changes regime."""
        n = 200
        df = pd.DataFrame(
            {
                "close": np.linspace(100, 110, n),
                "trend_label": ["trend_up"] * n,
                "vol_label": ["low_vol"] * n,
                "regime_confidence": [0.8] * n,
            }
        )
        assessment = RegimeAssessment(df)
        result = assessment.compute_transition_analysis()

        assert result["total_transitions"] == 0
        assert result["transition_frequency"] == 0.0

    def test_nan_handling(self):
        """Test handling of NaN values in data."""
        n = 200
        prices = np.linspace(100, 110, n)
        prices[50:60] = np.nan  # Insert some NaN values

        df = pd.DataFrame(
            {
                "close": prices,
                "trend_label": ["trend_up"] * n,
                "vol_label": ["low_vol"] * n,
                "regime_confidence": [0.8] * n,
            }
        )
        assessment = RegimeAssessment(df)
        result = assessment.compute_forward_accuracy()

        # Should still compute without error
        assert "overall" in result
