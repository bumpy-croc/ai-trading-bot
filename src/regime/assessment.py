"""
Regime Detector Assessment Module

Provides comprehensive metrics for evaluating regime detection accuracy,
including forward-looking validation, persistence analysis, and calibration.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.regime.detector import RegimeDetector, TrendLabel

logger = logging.getLogger(__name__)


@dataclass
class RegimeAssessmentConfig:
    """Configuration for regime assessment."""

    lookahead: int = 20
    range_threshold: float = 0.02
    confidence_bins: int = 10
    min_periods_for_stats: int = 100


@dataclass
class AssessmentMetrics:
    """Container for all assessment metrics."""

    # Forward-looking accuracy
    overall_accuracy: float = 0.0
    accuracy_by_regime: dict[str, float] = field(default_factory=dict)
    accuracy_by_volatility: dict[str, float] = field(default_factory=dict)

    # Persistence metrics
    avg_regime_duration: float = 0.0
    median_regime_duration: float = 0.0
    min_regime_duration: int = 0
    max_regime_duration: int = 0
    duration_std: float = 0.0

    # Transition metrics
    total_transitions: int = 0
    transition_frequency: float = 0.0
    transition_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    transition_probabilities: dict[str, dict[str, float]] = field(default_factory=dict)

    # Calibration metrics
    expected_calibration_error: float = 0.0
    calibration_curve: list[dict[str, float]] = field(default_factory=list)

    # Distribution metrics
    regime_distribution: dict[str, float] = field(default_factory=dict)
    volatility_distribution: dict[str, float] = field(default_factory=dict)

    # Metadata
    total_periods: int = 0
    assessment_timestamp: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "forward_accuracy": {
                "overall": self.overall_accuracy,
                "by_regime": self.accuracy_by_regime,
                "by_volatility": self.accuracy_by_volatility,
            },
            "persistence": {
                "avg_duration": self.avg_regime_duration,
                "median_duration": self.median_regime_duration,
                "min_duration": self.min_regime_duration,
                "max_duration": self.max_regime_duration,
                "std_duration": self.duration_std,
            },
            "transitions": {
                "total": self.total_transitions,
                "frequency": self.transition_frequency,
                "matrix": self.transition_matrix,
                "probabilities": self.transition_probabilities,
            },
            "calibration": {
                "expected_calibration_error": self.expected_calibration_error,
                "curve": self.calibration_curve,
            },
            "distribution": {
                "regime": self.regime_distribution,
                "volatility": self.volatility_distribution,
            },
            "metadata": {
                "total_periods": self.total_periods,
                "timestamp": self.assessment_timestamp,
                "config": self.config,
            },
        }


class RegimeAssessment:
    """
    Comprehensive assessment of regime detector accuracy.

    Computes forward-looking accuracy, persistence metrics, transition analysis,
    and confidence calibration for regime detection evaluation.
    """

    def __init__(
        self,
        annotated_df: pd.DataFrame,
        config: Optional[RegimeAssessmentConfig] = None,
    ):
        """
        Initialize regime assessment.

        Args:
            annotated_df: DataFrame with regime annotations (from RegimeDetector.annotate())
            config: Assessment configuration parameters
        """
        self.df = annotated_df.copy()
        self.config = config or RegimeAssessmentConfig()
        self.metrics = AssessmentMetrics()

        self._validate_data()

    def _validate_data(self) -> None:
        """Validate that required columns exist in the DataFrame."""
        if len(self.df) == 0:
            raise ValueError("DataFrame is empty")

        required_columns = ["close", "trend_label", "vol_label", "regime_confidence"]
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(self.df) < self.config.min_periods_for_stats:
            logger.warning(
                "Data has only %d periods, less than recommended %d for reliable stats",
                len(self.df),
                self.config.min_periods_for_stats,
            )

    def compute_forward_accuracy(self) -> dict[str, Any]:
        """
        Compute forward-looking accuracy of regime predictions.

        Validates whether regime labels correctly predicted future price direction:
        - trend_up: positive future return
        - trend_down: negative future return
        - range: small absolute return (< threshold)

        Returns:
            Dictionary with accuracy metrics
        """
        lookahead = self.config.lookahead
        threshold = self.config.range_threshold

        if len(self.df) <= lookahead:
            logger.warning("Insufficient data for lookahead=%d", lookahead)
            return {"overall": 0.0, "by_regime": {}, "by_volatility": {}}

        # Calculate future returns
        future_returns = self.df["close"].shift(-lookahead) / self.df["close"] - 1
        valid_mask = ~future_returns.isna()

        # Assess accuracy for each regime type
        results = []
        for i in range(len(self.df) - lookahead):
            trend = self.df.iloc[i]["trend_label"]
            vol = self.df.iloc[i]["vol_label"]
            ret = future_returns.iloc[i]

            if pd.isna(ret):
                continue

            # Determine correctness based on regime type
            if trend == TrendLabel.TREND_UP.value or trend == "trend_up":
                correct = ret > 0
            elif trend == TrendLabel.TREND_DOWN.value or trend == "trend_down":
                correct = ret < 0
            else:  # range
                correct = abs(ret) < threshold

            results.append(
                {
                    "trend": str(trend),
                    "vol": str(vol),
                    "return": ret,
                    "correct": correct,
                }
            )

        if not results:
            return {"overall": 0.0, "by_regime": {}, "by_volatility": {}}

        results_df = pd.DataFrame(results)

        # Overall accuracy
        overall = results_df["correct"].mean()

        # Accuracy by trend regime
        by_regime = results_df.groupby("trend")["correct"].mean().to_dict()

        # Accuracy by volatility
        by_volatility = results_df.groupby("vol")["correct"].mean().to_dict()

        # Store in metrics
        self.metrics.overall_accuracy = overall
        self.metrics.accuracy_by_regime = by_regime
        self.metrics.accuracy_by_volatility = by_volatility

        return {
            "overall": overall,
            "by_regime": by_regime,
            "by_volatility": by_volatility,
            "sample_size": len(results),
        }

    def compute_persistence_metrics(self) -> dict[str, Any]:
        """
        Compute regime persistence (duration) metrics.

        Returns:
            Dictionary with persistence statistics
        """
        # Identify regime changes
        regime_col = self.df["trend_label"].astype(str)
        regime_changes = regime_col != regime_col.shift(1)

        # Calculate duration of each regime period
        durations = []
        current_duration = 0
        current_regime = None

        for i, (is_change, regime) in enumerate(zip(regime_changes, regime_col)):
            if is_change or current_regime is None:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 1
                current_regime = regime
            else:
                current_duration += 1

        # Add final regime duration
        if current_duration > 0:
            durations.append(current_duration)

        if not durations:
            return {
                "avg_duration": 0.0,
                "median_duration": 0.0,
                "min_duration": 0,
                "max_duration": 0,
                "std_duration": 0.0,
            }

        durations_arr = np.array(durations)

        self.metrics.avg_regime_duration = float(np.mean(durations_arr))
        self.metrics.median_regime_duration = float(np.median(durations_arr))
        self.metrics.min_regime_duration = int(np.min(durations_arr))
        self.metrics.max_regime_duration = int(np.max(durations_arr))
        self.metrics.duration_std = float(np.std(durations_arr))

        return {
            "avg_duration": self.metrics.avg_regime_duration,
            "median_duration": self.metrics.median_regime_duration,
            "min_duration": self.metrics.min_regime_duration,
            "max_duration": self.metrics.max_regime_duration,
            "std_duration": self.metrics.duration_std,
            "durations": durations,
        }

    def compute_transition_analysis(self) -> dict[str, Any]:
        """
        Analyze regime transitions.

        Returns:
            Dictionary with transition matrix and probabilities
        """
        regime_col = self.df["trend_label"].astype(str)
        regimes = regime_col.unique().tolist()

        # Build transition matrix
        transition_matrix: dict[str, dict[str, int]] = {r: {r2: 0 for r2 in regimes} for r in regimes}
        transitions = []

        prev_regime = None
        for regime in regime_col:
            if prev_regime is not None and regime != prev_regime:
                transition_matrix[prev_regime][regime] += 1
                transitions.append((prev_regime, regime))
            prev_regime = regime

        # Calculate transition probabilities
        transition_probs: dict[str, dict[str, float]] = {}
        for from_regime in regimes:
            total = sum(transition_matrix[from_regime].values())
            if total > 0:
                transition_probs[from_regime] = {
                    to_regime: count / total for to_regime, count in transition_matrix[from_regime].items()
                }
            else:
                transition_probs[from_regime] = {r: 0.0 for r in regimes}

        total_transitions = len(transitions)
        transition_frequency = total_transitions / len(self.df) if len(self.df) > 0 else 0.0

        self.metrics.total_transitions = total_transitions
        self.metrics.transition_frequency = transition_frequency
        self.metrics.transition_matrix = transition_matrix
        self.metrics.transition_probabilities = transition_probs

        return {
            "total_transitions": total_transitions,
            "transition_frequency": transition_frequency,
            "transition_matrix": transition_matrix,
            "transition_probabilities": transition_probs,
        }

    def compute_confidence_calibration(self) -> dict[str, Any]:
        """
        Compute confidence calibration metrics.

        Measures how well the detector's confidence aligns with actual accuracy.
        Expected Calibration Error (ECE) measures the average gap between
        predicted confidence and observed accuracy.

        Returns:
            Dictionary with calibration metrics
        """
        lookahead = self.config.lookahead
        threshold = self.config.range_threshold
        n_bins = self.config.confidence_bins

        if len(self.df) <= lookahead:
            return {"expected_calibration_error": 0.0, "curve": []}

        # Calculate future returns and correctness
        future_returns = self.df["close"].shift(-lookahead) / self.df["close"] - 1

        records = []
        for i in range(len(self.df) - lookahead):
            trend = self.df.iloc[i]["trend_label"]
            confidence = self.df.iloc[i]["regime_confidence"]
            ret = future_returns.iloc[i]

            if pd.isna(ret) or pd.isna(confidence):
                continue

            # Determine correctness
            if trend == TrendLabel.TREND_UP.value or trend == "trend_up":
                correct = ret > 0
            elif trend == TrendLabel.TREND_DOWN.value or trend == "trend_down":
                correct = ret < 0
            else:
                correct = abs(ret) < threshold

            records.append({"confidence": confidence, "correct": correct})

        if not records:
            return {"expected_calibration_error": 0.0, "curve": []}

        records_df = pd.DataFrame(records)

        # Bin by confidence
        records_df["bin"] = pd.cut(records_df["confidence"], bins=n_bins, labels=False)

        # Calculate accuracy per bin
        calibration_curve = []
        ece_sum = 0.0
        total_samples = len(records_df)

        for bin_idx in range(n_bins):
            bin_data = records_df[records_df["bin"] == bin_idx]
            if len(bin_data) == 0:
                continue

            avg_confidence = bin_data["confidence"].mean()
            accuracy = bin_data["correct"].mean()
            bin_size = len(bin_data)

            calibration_curve.append(
                {
                    "bin": bin_idx,
                    "avg_confidence": avg_confidence,
                    "accuracy": accuracy,
                    "sample_size": bin_size,
                }
            )

            # Weighted contribution to ECE
            ece_sum += (bin_size / total_samples) * abs(accuracy - avg_confidence)

        self.metrics.expected_calibration_error = ece_sum
        self.metrics.calibration_curve = calibration_curve

        return {
            "expected_calibration_error": ece_sum,
            "curve": calibration_curve,
        }

    def compute_distribution(self) -> dict[str, Any]:
        """
        Compute regime and volatility distribution.

        Returns:
            Dictionary with distribution percentages
        """
        total = len(self.df)
        if total == 0:
            return {"regime": {}, "volatility": {}}

        # Regime distribution
        regime_counts = self.df["trend_label"].astype(str).value_counts()
        regime_dist = (regime_counts / total).to_dict()

        # Volatility distribution
        vol_counts = self.df["vol_label"].astype(str).value_counts()
        vol_dist = (vol_counts / total).to_dict()

        self.metrics.regime_distribution = regime_dist
        self.metrics.volatility_distribution = vol_dist

        return {
            "regime": regime_dist,
            "volatility": vol_dist,
        }

    def compute_all_metrics(self) -> AssessmentMetrics:
        """
        Compute all assessment metrics.

        Returns:
            AssessmentMetrics object with all computed metrics
        """
        logger.info("Computing forward accuracy...")
        self.compute_forward_accuracy()

        logger.info("Computing persistence metrics...")
        self.compute_persistence_metrics()

        logger.info("Computing transition analysis...")
        self.compute_transition_analysis()

        logger.info("Computing confidence calibration...")
        self.compute_confidence_calibration()

        logger.info("Computing distribution...")
        self.compute_distribution()

        # Set metadata
        self.metrics.total_periods = len(self.df)
        self.metrics.assessment_timestamp = datetime.now().isoformat()
        self.metrics.config = {
            "lookahead": self.config.lookahead,
            "range_threshold": self.config.range_threshold,
            "confidence_bins": self.config.confidence_bins,
        }

        return self.metrics

    def generate_report(self) -> str:
        """
        Generate a formatted console report of assessment results.

        Returns:
            Formatted string report
        """
        m = self.metrics

        report = []
        report.append("=" * 60)
        report.append("REGIME DETECTOR ASSESSMENT REPORT")
        report.append("=" * 60)
        report.append(f"Assessment Date: {m.assessment_timestamp}")
        report.append(f"Total Periods: {m.total_periods:,}")
        report.append(f"Lookahead: {self.config.lookahead} bars")
        report.append("")

        # Forward Accuracy
        report.append("-" * 40)
        report.append("FORWARD-LOOKING ACCURACY")
        report.append("-" * 40)
        report.append(f"Overall Accuracy: {m.overall_accuracy:.1%}")
        report.append("")
        report.append("By Regime:")
        for regime, acc in m.accuracy_by_regime.items():
            report.append(f"  {regime}: {acc:.1%}")
        report.append("")
        report.append("By Volatility:")
        for vol, acc in m.accuracy_by_volatility.items():
            report.append(f"  {vol}: {acc:.1%}")
        report.append("")

        # Persistence
        report.append("-" * 40)
        report.append("REGIME PERSISTENCE")
        report.append("-" * 40)
        report.append(f"Average Duration: {m.avg_regime_duration:.1f} bars")
        report.append(f"Median Duration: {m.median_regime_duration:.1f} bars")
        report.append(f"Min Duration: {m.min_regime_duration} bars")
        report.append(f"Max Duration: {m.max_regime_duration} bars")
        report.append(f"Std Dev: {m.duration_std:.1f} bars")
        report.append("")

        # Transitions
        report.append("-" * 40)
        report.append("REGIME TRANSITIONS")
        report.append("-" * 40)
        report.append(f"Total Transitions: {m.total_transitions}")
        report.append(f"Transition Frequency: {m.transition_frequency:.2%}")
        report.append("")

        # Calibration
        report.append("-" * 40)
        report.append("CONFIDENCE CALIBRATION")
        report.append("-" * 40)
        report.append(f"Expected Calibration Error (ECE): {m.expected_calibration_error:.3f}")
        report.append("")

        # Distribution
        report.append("-" * 40)
        report.append("REGIME DISTRIBUTION")
        report.append("-" * 40)
        for regime, pct in m.regime_distribution.items():
            report.append(f"  {regime}: {pct:.1%}")
        report.append("")
        report.append("Volatility Distribution:")
        for vol, pct in m.volatility_distribution.items():
            report.append(f"  {vol}: {pct:.1%}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    def save_metrics(self, output_path: Path) -> None:
        """
        Save metrics to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
        logger.info("Metrics saved to %s", output_path)


def compare_detectors(
    df: pd.DataFrame,
    detector1: RegimeDetector,
    detector2: RegimeDetector,
    config: Optional[RegimeAssessmentConfig] = None,
) -> dict[str, AssessmentMetrics]:
    """
    Compare two regime detectors on the same data.

    Args:
        df: Raw OHLCV DataFrame
        detector1: First detector to assess
        detector2: Second detector to assess
        config: Assessment configuration

    Returns:
        Dictionary with metrics for each detector
    """
    # Annotate with each detector
    df1 = detector1.annotate(df.copy())
    df2 = detector2.annotate(df.copy())

    # Assess each
    assessment1 = RegimeAssessment(df1, config)
    assessment2 = RegimeAssessment(df2, config)

    metrics1 = assessment1.compute_all_metrics()
    metrics2 = assessment2.compute_all_metrics()

    return {
        "detector1": metrics1,
        "detector2": metrics2,
    }
