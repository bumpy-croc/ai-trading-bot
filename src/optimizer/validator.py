from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.optimizer.schemas import ExperimentResult


@dataclass
class ValidationConfig:
    bootstrap_samples: int = 1000
    min_effect_size: float = 0.5  # Cohen's d threshold
    p_value_threshold: float = 0.1  # liberal for MVP


@dataclass
class ValidationReport:
    passed: bool
    p_value: float
    effect_size: float
    baseline_metrics: dict[str, float]
    candidate_metrics: dict[str, float]


class StatisticalValidator:
    """Validate that a candidate configuration improves over baseline using simple bootstrap tests."""

    def __init__(self, config: ValidationConfig | None = None, seed: int | None = None):
        self.cfg = config or ValidationConfig()
        self._rng = np.random.default_rng(seed)

    def _cohens_d(self, x: np.ndarray, y: np.ndarray) -> float:
        nx = len(x)
        ny = len(y)
        if nx < 2 or ny < 2:
            # Not enough samples for pooled std; fallback to signed unit if means differ
            diff = float(np.mean(y) - np.mean(x))
            return 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
        vx = np.var(x, ddof=1)
        vy = np.var(y, ddof=1)
        s = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
        if s == 0:
            diff = float(np.mean(y) - np.mean(x))
            return 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
        return (np.mean(y) - np.mean(x)) / s

    def _bootstrap_pvalue(self, x: np.ndarray, y: np.ndarray) -> float:
        # Test whether mean(y) - mean(x) > 0 using vectorized bootstrap
        obs = np.mean(y) - np.mean(x)
        pooled = np.concatenate([x, y])
        n_x = len(x)
        n_samples = self.cfg.bootstrap_samples
        # Draw all bootstrap samples at once: (n_samples, len(pooled))
        indices = self._rng.integers(0, len(pooled), size=(n_samples, len(pooled)))
        resamples = pooled[indices]
        diffs = resamples[:, n_x:].mean(axis=1) - resamples[:, :n_x].mean(axis=1)
        count = int(np.sum(diffs >= obs))
        # One-sided p-value
        return count / n_samples

    def validate(
        self, baseline: list[ExperimentResult], candidate: list[ExperimentResult]
    ) -> ValidationReport:
        # For MVP, derive per-run metric as annualized_return - max_drawdown_penalty
        bx = np.array([r.annualized_return - 0.5 * r.max_drawdown for r in baseline], dtype=float)
        cx = np.array([r.annualized_return - 0.5 * r.max_drawdown for r in candidate], dtype=float)

        p = self._bootstrap_pvalue(bx, cx)
        d = self._cohens_d(bx, cx)

        passed_expr = (
            (p <= self.cfg.p_value_threshold)
            and (abs(d) >= self.cfg.min_effect_size)
            and (float(np.mean(cx)) > float(np.mean(bx)))
        )

        return ValidationReport(
            passed=bool(passed_expr),
            p_value=float(p),
            effect_size=float(d),
            baseline_metrics={
                "annualized_return": float(
                    np.mean([r.annualized_return for r in baseline]) if baseline else 0.0
                ),
                "max_drawdown": float(
                    np.mean([r.max_drawdown for r in baseline]) if baseline else 0.0
                ),
            },
            candidate_metrics={
                "annualized_return": float(
                    np.mean([r.annualized_return for r in candidate]) if candidate else 0.0
                ),
                "max_drawdown": float(
                    np.mean([r.max_drawdown for r in candidate]) if candidate else 0.0
                ),
            },
        )
