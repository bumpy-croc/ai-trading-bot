from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field


class DriftSeverity(enum.Enum):
    """Severity level of strategy performance drift."""

    NONE = "NONE"
    MILD = "MILD"
    SEVERE = "SEVERE"
    CRITICAL = "CRITICAL"


@dataclass
class DriftConfig:
    """Thresholds for drift detection.

    Attributes:
        window_size: Number of recent data points used for the rolling metric.
        mild_z: Z-score threshold for MILD drift.
        severe_z: Z-score threshold for SEVERE drift.
        critical_z: Z-score threshold for CRITICAL drift (recommend pausing).
    """

    window_size: int = 30
    mild_z: float = 1.5
    severe_z: float = 2.0
    critical_z: float = 2.5


@dataclass
class DriftReport:
    """Results of a drift analysis against baseline expectations."""

    severity: DriftSeverity
    sharpe_z: float = 0.0
    win_rate_z: float = 0.0
    drawdown_z: float = 0.0
    details: dict[str, float] = field(default_factory=dict)
    recommendation: str = ""


class StrategyDriftDetector:
    """Monitor live performance against backtested expectations.

    Compares rolling Sharpe ratio, win rate, and maximum drawdown from recent
    live performance against a baseline (e.g. walk-forward OOS results).  Uses
    simple z-score thresholds to classify drift severity.

    At CRITICAL severity the detector recommends pausing trading.
    """

    def __init__(self, config: DriftConfig | None = None):
        self.config = config or DriftConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _z_score(observed: float, baseline_mean: float, baseline_std: float) -> float:
        """Compute z-score of *observed* relative to baseline distribution.

        A negative z-score means observed is worse than baseline (lower Sharpe
        or win rate, or higher drawdown).
        """
        if baseline_std <= 0 or not math.isfinite(baseline_std):
            if math.isfinite(observed) and math.isfinite(baseline_mean):
                diff = observed - baseline_mean
                if abs(diff) < 1e-9:
                    return 0.0
                # No variance — any deviation is noteworthy; return clamped sign
                return max(-5.0, min(5.0, diff * 100))
            return 0.0
        return (observed - baseline_mean) / baseline_std

    def detect(
        self,
        *,
        baseline_sharpe_mean: float,
        baseline_sharpe_std: float,
        baseline_win_rate_mean: float,
        baseline_win_rate_std: float,
        baseline_drawdown_mean: float,
        baseline_drawdown_std: float,
        live_sharpe: float,
        live_win_rate: float,
        live_max_drawdown: float,
    ) -> DriftReport:
        """Compare live metrics against baseline and classify drift severity.

        All rate/percentage inputs should use the same scale (e.g. percent or
        fraction) for both baseline and live values.

        Args:
            baseline_sharpe_mean: Mean Sharpe ratio from baseline (OOS folds).
            baseline_sharpe_std: Std dev of baseline Sharpe.
            baseline_win_rate_mean: Mean win rate from baseline.
            baseline_win_rate_std: Std dev of baseline win rate.
            baseline_drawdown_mean: Mean max drawdown from baseline.
            baseline_drawdown_std: Std dev of baseline max drawdown.
            live_sharpe: Current rolling Sharpe from live trading.
            live_win_rate: Current rolling win rate from live trading.
            live_max_drawdown: Current max drawdown from live trading.

        Returns:
            DriftReport with severity classification and per-metric z-scores.
        """
        sharpe_z = self._z_score(live_sharpe, baseline_sharpe_mean, baseline_sharpe_std)
        win_rate_z = self._z_score(live_win_rate, baseline_win_rate_mean, baseline_win_rate_std)
        # For drawdown, *higher* is worse, so invert the sign
        drawdown_z = -self._z_score(live_max_drawdown, baseline_drawdown_mean, baseline_drawdown_std)

        # Worst (most negative) z-score drives overall severity
        worst_z = min(sharpe_z, win_rate_z, drawdown_z)
        severity = self._classify(worst_z)

        recommendation = self._recommendation(severity)

        self.logger.info(
            "Drift check: sharpe_z=%.2f win_rate_z=%.2f drawdown_z=%.2f → %s",
            sharpe_z,
            win_rate_z,
            drawdown_z,
            severity.value,
        )

        return DriftReport(
            severity=severity,
            sharpe_z=sharpe_z,
            win_rate_z=win_rate_z,
            drawdown_z=drawdown_z,
            details={
                "live_sharpe": live_sharpe,
                "live_win_rate": live_win_rate,
                "live_max_drawdown": live_max_drawdown,
                "baseline_sharpe_mean": baseline_sharpe_mean,
                "baseline_win_rate_mean": baseline_win_rate_mean,
                "baseline_drawdown_mean": baseline_drawdown_mean,
            },
            recommendation=recommendation,
        )

    def _classify(self, worst_z: float) -> DriftSeverity:
        """Map worst z-score to drift severity."""
        cfg = self.config
        # worst_z is negative when performance is worse than baseline
        abs_z = abs(worst_z)
        if worst_z >= -cfg.mild_z:
            return DriftSeverity.NONE
        if abs_z < cfg.severe_z:
            return DriftSeverity.MILD
        if abs_z < cfg.critical_z:
            return DriftSeverity.SEVERE
        return DriftSeverity.CRITICAL

    @staticmethod
    def _recommendation(severity: DriftSeverity) -> str:
        """Return a human-readable recommendation for the given severity."""
        return {
            DriftSeverity.NONE: "Performance within expected range. No action required.",
            DriftSeverity.MILD: "Minor performance deviation detected. Monitor closely.",
            DriftSeverity.SEVERE: (
                "Significant performance drift. Review strategy parameters "
                "and consider reducing position sizes."
            ),
            DriftSeverity.CRITICAL: (
                "Critical performance drift detected. Recommend pausing live trading "
                "and re-running walk-forward analysis to re-validate the strategy."
            ),
        }[severity]
