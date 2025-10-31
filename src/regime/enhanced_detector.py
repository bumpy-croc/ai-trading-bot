"""Consolidated market regime detection utilities.

This module extends the baseline :class:`~src.regime.detector.RegimeDetector`
with higher level abstractions used across the trading system.  It also
provides calibration and evaluation helpers for quantifying regime detection
accuracy and visualising the results.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .detector import RegimeConfig, RegimeDetector, TrendLabel, VolLabel

__all__ = [
    "RegimeContext",
    "RegimeTransition",
    "EnhancedRegimeDetector",
    "RegimeEvaluationMetrics",
    "RegimeCalibrationResult",
    "calibrate_regime_detector",
    "evaluate_regime_accuracy",
    "plot_regime_accuracy",
]


@dataclass
class RegimeContext:
    """Enhanced regime context used by strategy components."""

    trend: TrendLabel
    volatility: VolLabel
    confidence: float
    duration: int
    strength: float
    timestamp: datetime | None = None
    metadata: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self._validate_regime_context()

    def _validate_regime_context(self) -> None:
        if not isinstance(self.trend, TrendLabel):
            raise ValueError(f"trend must be a TrendLabel enum, got {type(self.trend)}")

        if not isinstance(self.volatility, VolLabel):
            raise ValueError(f"volatility must be a VolLabel enum, got {type(self.volatility)}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

        if self.duration < 0:
            raise ValueError(f"duration must be non-negative, got {self.duration}")

        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be between 0.0 and 1.0, got {self.strength}")

        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError(
                f"metadata must be a dictionary when provided, got {type(self.metadata)}"
            )

    def get_regime_label(self) -> str:
        """Return the combined regime label string."""

        return f"{self.trend.value}:{self.volatility.value}"

    def is_stable(self, min_duration: int = 10) -> bool:
        """Return ``True`` when the regime has lasted at least ``min_duration`` bars."""

        return self.duration >= min_duration

    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Return ``True`` when the regime confidence exceeds ``threshold``."""

        return self.confidence >= threshold

    def is_strong_regime(self, threshold: float = 0.6) -> bool:
        """Return ``True`` when the regime strength exceeds ``threshold``."""

        return self.strength >= threshold

    def get_risk_multiplier(self) -> float:
        """Return a risk multiplier derived from the regime properties."""

        multiplier = 1.0
        if self.volatility == VolLabel.HIGH:
            multiplier *= 0.8
        if self.trend == TrendLabel.TREND_DOWN:
            multiplier *= 0.7
        if self.trend == TrendLabel.RANGE:
            multiplier *= 0.9
        if self.confidence < 0.5:
            multiplier *= 0.8
        if not self.is_stable():
            multiplier *= 0.9
        return max(0.2, multiplier)


@dataclass
class RegimeTransition:
    """Description of a detected regime transition."""

    from_regime: RegimeContext
    to_regime: RegimeContext
    transition_time: datetime
    confidence: float

    def get_transition_type(self) -> str:
        return f"{self.from_regime.get_regime_label()} -> {self.to_regime.get_regime_label()}"

    def is_major_transition(self) -> bool:
        return self.from_regime.trend != self.to_regime.trend


@dataclass(frozen=True)
class RegimeEvaluationMetrics:
    """Summary metrics describing regime detection accuracy."""

    accuracy: float
    trend_accuracy: float
    volatility_accuracy: float
    support: int


@dataclass(frozen=True)
class RegimeCalibrationResult:
    """Result of a calibration run."""

    config: RegimeConfig
    metrics: RegimeEvaluationMetrics
    evaluation_frame: pd.DataFrame
    tried_configs: int


class EnhancedRegimeDetector:
    """Strategy-facing wrapper that augments :class:`RegimeDetector`."""

    def __init__(
        self,
        base_detector: RegimeDetector | None = None,
        stability_threshold: int = 10,
        max_history: int = 1000,
    ) -> None:
        self.base_detector = base_detector or RegimeDetector()
        self.stability_threshold = stability_threshold
        self.max_history = max_history

        self.regime_history: list[RegimeContext] = []
        self.transition_history: list[RegimeTransition] = []
        self.current_regime: RegimeContext | None = None
        self.regime_start_index: int = 0

    @property
    def warmup_period(self) -> int:
        """Return the minimum history required for regime detection."""

        return 0

    def get_feature_generators(self) -> Sequence[object]:  # pragma: no cover - interface hook
        """Return optional feature generator specifications.

        The production system injects feature generators through this hook.  It is
        intentionally lightweight to avoid coupling unit tests to runtime
        infrastructure.
        """

        return []

    def detect_regime(self, df: pd.DataFrame, index: int) -> RegimeContext:
        """Detect the regime at ``index`` within ``df``."""

        if index < 0 or index >= len(df):
            raise IndexError(f"Index {index} is out of bounds for DataFrame of length {len(df)}")

        working_df = df
        if "regime_label" not in working_df.columns:
            working_df = self.base_detector.annotate(working_df.copy())

        current_row = working_df.iloc[index]
        trend_label = TrendLabel(current_row["trend_label"])
        vol_label = VolLabel(current_row["vol_label"])
        base_confidence = float(current_row.get("regime_confidence", 0.5))

        duration = self._calculate_regime_duration(working_df, index, trend_label)
        strength = self._calculate_regime_strength(working_df, index)
        enhanced_confidence = self._enhance_confidence(base_confidence, duration, strength)

        regime_context = RegimeContext(
            trend=trend_label,
            volatility=vol_label,
            confidence=enhanced_confidence,
            duration=duration,
            strength=strength,
            timestamp=datetime.now(),
            metadata={
                "trend_score": float(current_row.get("trend_score", 0.0)),
                "atr_percentile": float(current_row.get("atr_percentile", 0.5)),
                "base_confidence": base_confidence,
                "index": index,
            },
        )

        self._update_regime_tracking(regime_context, index)
        return regime_context

    def apply_calibration(self, result: RegimeCalibrationResult) -> None:
        """Replace the underlying detector with a calibrated configuration."""

        self.base_detector = RegimeDetector(result.config)

    def evaluate_accuracy(
        self,
        df: pd.DataFrame,
        *,
        target_trend_col: str,
        target_vol_col: str,
    ) -> tuple[RegimeEvaluationMetrics, pd.DataFrame]:
        """Evaluate detection accuracy against labelled columns."""

        annotated = self.base_detector.annotate(df.copy())
        return evaluate_regime_accuracy(
            annotated,
            target_trend_col=target_trend_col,
            target_vol_col=target_vol_col,
        )

    def get_regime_history(
        self, df: pd.DataFrame, lookback_periods: int = 100
    ) -> list[RegimeContext]:
        """Return historical regime contexts for ``df``."""

        if df.empty:
            return []

        working_df = df
        if "regime_label" not in working_df.columns:
            working_df = self.base_detector.annotate(working_df.copy())

        history: list[RegimeContext] = []
        start_index = max(0, len(working_df) - lookback_periods)

        for i in range(start_index, len(working_df)):
            try:
                regime_context = self.detect_regime(working_df, i)
            except (IndexError, ValueError):
                continue
            history.append(regime_context)

        return history

    def is_regime_stable(
        self, df: pd.DataFrame, index: int, min_duration: int | None = None
    ) -> bool:
        """Return ``True`` when the current regime has persisted sufficiently."""

        min_dur = min_duration or self.stability_threshold
        try:
            regime_context = self.detect_regime(df, index)
        except (IndexError, ValueError):
            return False
        return regime_context.is_stable(min_dur)

    def detect_regime_transitions(
        self, df: pd.DataFrame, lookback_periods: int = 50
    ) -> list[RegimeTransition]:
        """Return a list of detected regime transitions."""

        if df.empty or len(df) < 2:
            return []

        working_df = df
        if "regime_label" not in working_df.columns:
            working_df = self.base_detector.annotate(working_df.copy())

        transitions: list[RegimeTransition] = []
        start_index = max(1, len(working_df) - lookback_periods)
        prev_regime: RegimeContext | None = None

        for i in range(start_index, len(working_df)):
            try:
                current_regime = self.detect_regime(working_df, i)
            except (IndexError, ValueError):
                continue

            if prev_regime is not None and (
                prev_regime.trend != current_regime.trend
                or prev_regime.volatility != current_regime.volatility
            ):
                transition = RegimeTransition(
                    from_regime=prev_regime,
                    to_regime=current_regime,
                    transition_time=current_regime.timestamp or datetime.now(),
                    confidence=min(prev_regime.confidence, current_regime.confidence),
                )
                transitions.append(transition)

            prev_regime = current_regime

        return transitions

    def get_regime_statistics(
        self, df: pd.DataFrame, lookback_periods: int = 252
    ) -> dict[str, float]:
        """Return aggregate statistics describing regime behaviour."""

        history = self.get_regime_history(df, lookback_periods)
        if not history:
            return {}

        regime_counts: dict[str, int] = {}
        confidence_sum = 0.0
        strength_sum = 0.0
        duration_sum = 0

        for regime in history:
            label = regime.get_regime_label()
            regime_counts[label] = regime_counts.get(label, 0) + 1
            confidence_sum += regime.confidence
            strength_sum += regime.strength
            duration_sum += regime.duration

        total_periods = len(history)
        stats = {
            "total_periods": total_periods,
            "avg_confidence": confidence_sum / total_periods,
            "avg_strength": strength_sum / total_periods,
            "avg_duration": duration_sum / total_periods,
        }

        for label, count in regime_counts.items():
            stats[f"{label}_pct"] = (count / total_periods) * 100

        transitions = self.detect_regime_transitions(df, lookback_periods)
        stats["transition_frequency"] = (
            len(transitions) / total_periods if total_periods > 0 else 0
        )

        return stats

    def get_current_regime(self) -> RegimeContext | None:
        """Return the most recently detected regime."""

        return self.current_regime

    def get_recent_transitions(self, count: int = 5) -> list[RegimeTransition]:
        """Return the last ``count`` regime transitions."""

        return self.transition_history[-count:] if self.transition_history else []

    def reset_tracking(self) -> None:
        """Reset all internal tracking state."""

        self.regime_history.clear()
        self.transition_history.clear()
        self.current_regime = None
        self.regime_start_index = 0

    def _calculate_regime_duration(
        self, df: pd.DataFrame, index: int, current_trend: TrendLabel
    ) -> int:
        duration = 1
        for i in range(index - 1, -1, -1):
            try:
                prev_trend = TrendLabel(df.iloc[i]["trend_label"])
            except (KeyError, ValueError):
                break
            if prev_trend == current_trend:
                duration += 1
            else:
                break
        return duration

    def _calculate_regime_strength(
        self, df: pd.DataFrame, index: int, window: int = 20
    ) -> float:
        if index < window:
            return 0.5

        try:
            start_idx = max(0, index - window + 1)
            trend_scores = df.iloc[start_idx : index + 1]["trend_score"].values
        except (KeyError, IndexError):
            return 0.5

        if len(trend_scores) == 0:
            return 0.5

        valid_scores = trend_scores[~np.isnan(trend_scores)]
        if len(valid_scores) == 0:
            return 0.5

        mean_score = float(np.mean(np.abs(valid_scores)))
        strength = min(1.0, mean_score / 0.05)
        return max(0.0, strength)

    def _enhance_confidence(self, base_confidence: float, duration: int, strength: float) -> float:
        enhanced = base_confidence

        if duration >= self.stability_threshold:
            stability_boost = min(0.2, duration / (self.stability_threshold * 5))
            enhanced += stability_boost

        if strength > 0.7:
            strength_boost = (strength - 0.7) * 0.3
            enhanced += strength_boost

        if duration < 3:
            enhanced *= 0.8

        return max(0.0, min(1.0, enhanced))

    def _update_regime_tracking(self, regime_context: RegimeContext, index: int) -> None:
        if (
            self.current_regime is None
            or self.current_regime.trend != regime_context.trend
            or self.current_regime.volatility != regime_context.volatility
        ):
            if self.current_regime is not None:
                transition = RegimeTransition(
                    from_regime=self.current_regime,
                    to_regime=regime_context,
                    transition_time=regime_context.timestamp or datetime.now(),
                    confidence=min(self.current_regime.confidence, regime_context.confidence),
                )
                self.transition_history.append(transition)
                if len(self.transition_history) > self.max_history // 10:
                    self.transition_history = self.transition_history[-self.max_history // 10 :]
            self.regime_start_index = index

        self.current_regime = regime_context
        self.regime_history.append(regime_context)
        if len(self.regime_history) > self.max_history:
            self.regime_history = self.regime_history[-self.max_history :]


def evaluate_regime_accuracy(
    annotated_df: pd.DataFrame,
    *,
    target_trend_col: str,
    target_vol_col: str,
    predicted_trend_col: str = "trend_label",
    predicted_vol_col: str = "vol_label",
) -> tuple[RegimeEvaluationMetrics, pd.DataFrame]:
    """Compute accuracy metrics for regime detection results."""

    for column in (predicted_trend_col, predicted_vol_col, target_trend_col, target_vol_col):
        if column not in annotated_df.columns:
            raise ValueError(f"Column '{column}' is required for regime accuracy evaluation")

    evaluation = annotated_df[[predicted_trend_col, predicted_vol_col]].copy()
    evaluation.rename(
        columns={
            predicted_trend_col: "predicted_trend",
            predicted_vol_col: "predicted_volatility",
        },
        inplace=True,
    )
    evaluation["target_trend"] = annotated_df[target_trend_col]
    evaluation["target_volatility"] = annotated_df[target_vol_col]

    mask = evaluation[["predicted_trend", "predicted_volatility", "target_trend", "target_volatility"]].notna().all(axis=1)

    evaluation["trend_correct"] = (evaluation["predicted_trend"] == evaluation["target_trend"]).where(mask)
    evaluation["volatility_correct"] = (
        evaluation["predicted_volatility"] == evaluation["target_volatility"]
    ).where(mask)
    evaluation["regime_correct"] = (
        evaluation["trend_correct"] & evaluation["volatility_correct"]
    )

    support = int(mask.sum())
    if support == 0:
        metrics = RegimeEvaluationMetrics(accuracy=np.nan, trend_accuracy=np.nan, volatility_accuracy=np.nan, support=0)
        return metrics, evaluation

    trend_accuracy = float(evaluation.loc[mask, "trend_correct"].mean())
    volatility_accuracy = float(evaluation.loc[mask, "volatility_correct"].mean())
    accuracy = float(evaluation.loc[mask, "regime_correct"].mean())

    evaluation["rolling_accuracy"] = (
        evaluation["regime_correct"].astype(float).rolling(window=20, min_periods=1).mean()
    )

    metrics = RegimeEvaluationMetrics(
        accuracy=accuracy,
        trend_accuracy=trend_accuracy,
        volatility_accuracy=volatility_accuracy,
        support=support,
    )
    return metrics, evaluation


def calibrate_regime_detector(
    df: pd.DataFrame,
    *,
    target_trend_col: str,
    target_vol_col: str,
    slope_windows: Sequence[int] = (30, 40, 50),
    atr_windows: Sequence[int] = (14, 20),
    trend_thresholds: Sequence[float] = (0.0, 0.0005, 0.001),
    r2_mins: Sequence[float] = (0.1, 0.2),
    atr_percentiles: Sequence[float] = (0.6, 0.7, 0.8),
    base_config: RegimeConfig | None = None,
) -> RegimeCalibrationResult:
    """Calibrate the regime detector against labelled data."""

    if df.empty:
        raise ValueError("Calibration requires a non-empty DataFrame")

    base = base_config or RegimeConfig()
    best_result: RegimeCalibrationResult | None = None
    tried = 0

    for slope_window in slope_windows:
        for atr_window in atr_windows:
            for threshold in trend_thresholds:
                for r2_min in r2_mins:
                    for atr_percentile in atr_percentiles:
                        tried += 1
                        candidate = replace(
                            base,
                            slope_window=slope_window,
                            atr_window=atr_window,
                            trend_threshold=threshold,
                            r2_min=r2_min,
                            atr_high_percentile=atr_percentile,
                        )
                        detector = RegimeDetector(candidate)
                        annotated = detector.annotate(df.copy())
                        metrics, evaluation = evaluate_regime_accuracy(
                            annotated,
                            target_trend_col=target_trend_col,
                            target_vol_col=target_vol_col,
                        )
                        if best_result is None or (
                            np.nan_to_num(metrics.accuracy, nan=-1.0)
                            > np.nan_to_num(best_result.metrics.accuracy, nan=-1.0)
                        ):
                            best_result = RegimeCalibrationResult(
                                config=candidate,
                                metrics=metrics,
                                evaluation_frame=evaluation,
                                tried_configs=tried,
                            )

    if best_result is None:
        raise RuntimeError("Calibration failed to evaluate any configuration")

    if best_result.tried_configs != tried:
        best_result = RegimeCalibrationResult(
            config=best_result.config,
            metrics=best_result.metrics,
            evaluation_frame=best_result.evaluation_frame,
            tried_configs=tried,
        )

    return best_result


def plot_regime_accuracy(
    evaluation: pd.DataFrame,
    *,
    window: int = 20,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot rolling accuracy derived from :func:`evaluate_regime_accuracy`."""

    if "regime_correct" not in evaluation.columns:
        raise ValueError("evaluation DataFrame must contain 'regime_correct' column")

    working = evaluation.copy()
    working["rolling_accuracy"] = (
        working["regime_correct"].astype(float).rolling(window, min_periods=1).mean()
    )
    working["rolling_trend_accuracy"] = (
        working["trend_correct"].astype(float).rolling(window, min_periods=1).mean()
    )
    working["rolling_vol_accuracy"] = (
        working["volatility_correct"].astype(float).rolling(window, min_periods=1).mean()
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    x = working.index
    ax.plot(x, working["rolling_accuracy"], label="Regime accuracy", linewidth=2)
    ax.plot(x, working["rolling_trend_accuracy"], label="Trend accuracy", linestyle="--")
    ax.plot(x, working["rolling_vol_accuracy"], label="Volatility accuracy", linestyle=":")

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Time")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    return fig
