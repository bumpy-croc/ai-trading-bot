"""Rank, compare, and summarize the results of an experiment suite."""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.experiments.schemas import ExperimentResult
from src.experiments.suite import ComparisonSettings, SuiteResult


class Verdict(str, Enum):
    PROMOTE = "PROMOTE"
    HOLD = "HOLD"
    REJECT = "REJECT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


# Each allowed target metric is "higher is better" for ranking purposes.
_SUPPORTED_METRICS = {
    "sharpe_ratio",
    "annualized_return",
    "total_return",
    "calmar",
    "final_balance",
    "win_rate",
}


@dataclass
class VariantReport:
    name: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    final_balance: float
    delta_vs_baseline: float
    # Ranking confidence in [0, 1]. NOT a statistical p-value — it is a
    # monotone heuristic that combines effect size (normalized by baseline
    # magnitude) with trade count. PROMOTE requires this value to be ≥
    # (1 − significance_level). See `_ranking_confidence` for details.
    ranking_confidence: float | None
    verdict: Verdict
    is_baseline: bool = False


@dataclass
class SuiteReport:
    suite_id: str
    description: str
    target_metric: str
    significance_level: float
    min_trades: int
    baseline_name: str
    winner: str | None
    rows: list[VariantReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rows"] = [_row_to_dict(r) for r in self.rows]
        return data


def _row_to_dict(row: VariantReport) -> dict[str, Any]:
    d = asdict(row)
    d["verdict"] = row.verdict.value
    return d


def _metric(result: ExperimentResult, metric: str) -> float:
    """Return a "higher is better" scalar for the named metric.

    Calmar with zero drawdown is treated as ``math.inf`` when the annualized
    return is positive (perfect loss-free profitability ranks best, not
    worst). A zero-return zero-drawdown combination is ``0.0``; negative
    return with zero drawdown is ``-math.inf``.
    """
    if metric == "calmar":
        dd = abs(result.max_drawdown)
        if dd < 1e-9:
            ann = result.annualized_return
            if ann > 0:
                return math.inf
            if ann < 0:
                return -math.inf
            return 0.0
        return result.annualized_return / dd
    if metric not in _SUPPORTED_METRICS:
        raise ValueError(f"unknown metric {metric!r}; supported: {sorted(_SUPPORTED_METRICS)}")
    value = getattr(result, metric, None)
    if value is None:
        raise ValueError(f"ExperimentResult has no attribute {metric!r}")
    return float(value)


def _ranking_confidence(
    baseline: ExperimentResult,
    variant: ExperimentResult,
    metric: str,
) -> float | None:
    """Return a scale-free confidence score in ``[0, 1]`` or ``None``.

    This is NOT a statistical p-value — without per-trade return series we
    cannot run a proper significance test. Instead the score combines:

    * ``effect = |Δmetric| / max(|baseline_metric|, ε)`` — relative effect
      size so the score does not depend on whether the metric is in
      percent, fraction, or Sharpe units.
    * ``sample_confidence = tanh(min_trades / 60)`` — saturates near 1.0
      around 120 trades.

    ``confidence = sample_confidence * min(1.0, effect)``.

    Higher means "variant is more convincingly different from baseline";
    values near 0 mean "effect too small or too few trades to care".
    Returns ``None`` when either run has fewer than 2 trades.
    """
    n_b = baseline.total_trades
    n_v = variant.total_trades
    if n_b < 2 or n_v < 2:
        return None

    baseline_val = _metric(baseline, metric)
    variant_val = _metric(variant, metric)

    # Infinite Calmar edge cases: variant=∞, baseline finite → full confidence.
    if math.isinf(variant_val) and not math.isinf(baseline_val):
        return 1.0
    if math.isinf(baseline_val) and not math.isinf(variant_val):
        return 1.0
    if math.isinf(variant_val) and math.isinf(baseline_val):
        return 0.0  # both ∞ (or both −∞) → indistinguishable
    if not math.isfinite(variant_val) or not math.isfinite(baseline_val):
        return None

    diff = variant_val - baseline_val
    if diff == 0:
        return 0.0  # exact tie — zero confidence in a ranking difference

    denom = max(abs(baseline_val), 1e-6)
    relative_effect = abs(diff) / denom
    effect_component = min(1.0, relative_effect)

    n_eff = min(n_b, n_v)
    sample_confidence = math.tanh(n_eff / 60.0)

    confidence = sample_confidence * effect_component
    return float(max(0.0, min(1.0, confidence)))


def _classify(
    baseline: ExperimentResult,
    variant: ExperimentResult,
    settings: ComparisonSettings,
    confidence: float | None,
) -> Verdict:
    """Classify a variant vs baseline.

    * INSUFFICIENT_DATA — either run has fewer than ``min_trades``.
    * REJECT — variant's target metric is strictly worse than baseline.
    * HOLD — variant is neutral (exact tie), or better but below the
      confidence threshold.
    * PROMOTE — variant is better AND confidence ≥ (1 − α).
    """
    if variant.total_trades < settings.min_trades or baseline.total_trades < settings.min_trades:
        return Verdict.INSUFFICIENT_DATA
    baseline_val = _metric(baseline, settings.target_metric)
    variant_val = _metric(variant, settings.target_metric)
    # Infinities ranked higher than any finite value.
    if math.isinf(variant_val) and math.isinf(baseline_val):
        return Verdict.HOLD
    if math.isinf(variant_val) and variant_val > 0:
        # variant is +inf (e.g. perfect Calmar) and baseline is finite → strictly better
        return (
            Verdict.PROMOTE
            if confidence is not None and confidence >= (1.0 - settings.significance_level)
            else Verdict.HOLD
        )
    delta = variant_val - baseline_val
    if delta < 0:
        return Verdict.REJECT
    if delta == 0:
        return Verdict.HOLD
    if confidence is None:
        return Verdict.HOLD
    if confidence >= (1.0 - settings.significance_level):
        return Verdict.PROMOTE
    return Verdict.HOLD


class ExperimentReporter:
    """Build a :class:`SuiteReport` from a :class:`SuiteResult`."""

    def render(self, suite_result: SuiteResult) -> SuiteReport:
        settings = suite_result.config.comparison
        baseline = suite_result.baseline

        rows: list[VariantReport] = [
            VariantReport(
                name=suite_result.config.baseline.name,
                total_return=baseline.total_return,
                annualized_return=baseline.annualized_return,
                sharpe_ratio=baseline.sharpe_ratio,
                max_drawdown=baseline.max_drawdown,
                win_rate=baseline.win_rate,
                total_trades=baseline.total_trades,
                final_balance=baseline.final_balance,
                delta_vs_baseline=0.0,
                ranking_confidence=None,
                verdict=Verdict.HOLD,
                is_baseline=True,
            )
        ]

        for spec, result in zip(suite_result.config.variants, suite_result.variants, strict=True):
            delta = _metric(result, settings.target_metric) - _metric(
                baseline, settings.target_metric
            )
            # Guard against ±inf leaking into output (JSON serialization breaks).
            if not math.isfinite(delta):
                delta = math.copysign(float("inf"), delta) if delta != 0 else 0.0
            confidence = _ranking_confidence(baseline, result, settings.target_metric)
            verdict = _classify(baseline, result, settings, confidence)
            rows.append(
                VariantReport(
                    name=spec.name,
                    total_return=result.total_return,
                    annualized_return=result.annualized_return,
                    sharpe_ratio=result.sharpe_ratio,
                    max_drawdown=result.max_drawdown,
                    win_rate=result.win_rate,
                    total_trades=result.total_trades,
                    final_balance=result.final_balance,
                    delta_vs_baseline=delta,
                    ranking_confidence=confidence,
                    verdict=verdict,
                )
            )

        ranked = sorted(
            rows,
            key=lambda r: _metric_for_row(r, settings.target_metric),
            reverse=True,
        )
        winner: str | None = None
        for row in ranked:
            if row.is_baseline:
                continue
            if row.verdict == Verdict.PROMOTE:
                winner = row.name
                break

        return SuiteReport(
            suite_id=suite_result.suite_id,
            description=suite_result.config.description,
            target_metric=settings.target_metric,
            significance_level=settings.significance_level,
            min_trades=settings.min_trades,
            baseline_name=suite_result.config.baseline.name,
            winner=winner,
            rows=ranked,
        )

    def render_text(self, report: SuiteReport) -> str:
        header = (
            f"{'Variant':<24} {'Return%':>8} {'Ann%':>8} {'Sharpe':>7} {'MaxDD%':>7} "
            f"{'Win%':>6} {'N':>5} {'Δ':>8} {'Conf':>6} {'Verdict':>18}"
        )
        lines = [
            f"Suite: {report.suite_id}",
            f"Metric: {report.target_metric}  α={report.significance_level}  "
            f"min_trades={report.min_trades}",
            f"Baseline: {report.baseline_name}   Winner: {report.winner or '—'}",
            ("Conf = ranking confidence [0,1] (not a p-value). " "PROMOTE needs Conf ≥ (1 − α)."),
            "",
            header,
            "-" * len(header),
        ]
        for row in report.rows:
            conf_txt = "—" if row.ranking_confidence is None else f"{row.ranking_confidence:0.3f}"
            tag = row.verdict.value + (" *" if row.is_baseline else "")
            delta_str = (
                f"{row.delta_vs_baseline:>+8.3f}"
                if math.isfinite(row.delta_vs_baseline)
                else f"{'+∞' if row.delta_vs_baseline > 0 else '−∞':>8}"
            )
            lines.append(
                f"{row.name:<24} {row.total_return:>8.2f} {row.annualized_return:>8.2f} "
                f"{row.sharpe_ratio:>7.2f} {row.max_drawdown:>7.2f} "
                f"{row.win_rate:>6.2f} {row.total_trades:>5d} "
                f"{delta_str} {conf_txt:>6} {tag:>18}"
            )
        return "\n".join(lines)

    def render_csv(self, report: SuiteReport) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "variant",
                "total_return",
                "annualized_return",
                "sharpe_ratio",
                "max_drawdown",
                "win_rate",
                "total_trades",
                "final_balance",
                "delta_vs_baseline",
                "ranking_confidence",
                "verdict",
                "is_baseline",
            ]
        )
        for row in report.rows:
            writer.writerow(
                [
                    row.name,
                    row.total_return,
                    row.annualized_return,
                    row.sharpe_ratio,
                    row.max_drawdown,
                    row.win_rate,
                    row.total_trades,
                    row.final_balance,
                    row.delta_vs_baseline,
                    "" if row.ranking_confidence is None else row.ranking_confidence,
                    row.verdict.value,
                    row.is_baseline,
                ]
            )
        return buf.getvalue()

    def write_artifacts(self, report: SuiteReport, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        # JSON doesn't serialize ±inf by default — allow_nan is True by default
        # in the stdlib (emits `Infinity`) which is non-portable; write a
        # sanitized dict instead for round-trip safety.
        (out_dir / "report.json").write_text(
            json.dumps(_sanitize_for_json(report.to_dict()), indent=2)
        )
        (out_dir / "report.csv").write_text(self.render_csv(report))
        (out_dir / "report.txt").write_text(self.render_text(report))


def _metric_for_row(row: VariantReport, metric: str) -> float:
    if metric == "calmar":
        dd = abs(row.max_drawdown)
        if dd < 1e-9:
            ann = row.annualized_return
            if ann > 0:
                return math.inf
            if ann < 0:
                return -math.inf
            return 0.0
        return row.annualized_return / dd
    return float(getattr(row, metric))


def _sanitize_for_json(data: Any) -> Any:
    """Replace ±inf / NaN with strings so downstream JSON parsers don't choke."""
    if isinstance(data, dict):
        return {k: _sanitize_for_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_sanitize_for_json(v) for v in data]
    if isinstance(data, float):
        if math.isinf(data):
            return "Infinity" if data > 0 else "-Infinity"
        if math.isnan(data):
            return "NaN"
    return data


__all__ = ["ExperimentReporter", "SuiteReport", "Verdict", "VariantReport"]
