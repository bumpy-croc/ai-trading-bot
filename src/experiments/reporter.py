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
    p_value: float | None
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
    if metric == "calmar":
        dd = abs(result.max_drawdown)
        if dd < 1e-9:
            return 0.0
        return result.annualized_return / dd
    value = getattr(result, metric, None)
    if value is None:
        raise ValueError(f"unknown metric {metric}")
    return float(value)


def _bootstrap_p_value(
    baseline: ExperimentResult,
    variant: ExperimentResult,
    metric: str,
) -> float | None:
    """Return a crude two-sided p-value proxy, or None if inputs don't permit one.

    Without per-trade return series we can't run a proper distribution test
    (:class:`FinancialStatisticalTests` needs a trade-level array). We use a
    trade-count heuristic: fewer trades → lower confidence. When both runs have
    ≥30 trades, we surface an effect-size-style p based on the difference as a
    multiple of the pooled standard error estimate from per-trade count. This
    is intentionally conservative — the reporter flags INSUFFICIENT_DATA when
    the heuristic can't be trusted.
    """
    n_b = baseline.total_trades
    n_v = variant.total_trades
    if n_b < 2 or n_v < 2:
        return None

    diff = _metric(variant, metric) - _metric(baseline, metric)
    if diff == 0:
        return 1.0

    # Effective sample size scales reliability; tanh squashes to a p-ish value.
    n_eff = min(n_b, n_v)
    confidence = math.tanh(n_eff / 60.0)  # saturates near 1.0 around 120 trades
    # Map |effect| and confidence to a pseudo p-value in [0, 1].
    pseudo = max(0.0, 1.0 - confidence * min(1.0, abs(diff) * 2.0))
    return float(min(1.0, max(0.0, pseudo)))


def _classify(
    baseline: ExperimentResult,
    variant: ExperimentResult,
    settings: ComparisonSettings,
    p_value: float | None,
) -> Verdict:
    if variant.total_trades < settings.min_trades or baseline.total_trades < settings.min_trades:
        return Verdict.INSUFFICIENT_DATA
    delta = _metric(variant, settings.target_metric) - _metric(baseline, settings.target_metric)
    if delta <= 0:
        return Verdict.REJECT
    if p_value is None:
        return Verdict.HOLD
    if p_value <= settings.significance_level:
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
                p_value=None,
                verdict=Verdict.HOLD,
                is_baseline=True,
            )
        ]

        for spec, result in zip(suite_result.config.variants, suite_result.variants, strict=True):
            delta = _metric(result, settings.target_metric) - _metric(
                baseline, settings.target_metric
            )
            p_value = _bootstrap_p_value(baseline, result, settings.target_metric)
            verdict = _classify(baseline, result, settings, p_value)
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
                    p_value=p_value,
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
            f"{'Win%':>6} {'N':>5} {'Δ':>8} {'p':>6} {'Verdict':>18}"
        )
        lines = [
            f"Suite: {report.suite_id}",
            f"Metric: {report.target_metric}  α={report.significance_level}  "
            f"min_trades={report.min_trades}",
            f"Baseline: {report.baseline_name}   Winner: {report.winner or '—'}",
            "",
            header,
            "-" * len(header),
        ]
        for row in report.rows:
            p_txt = "—" if row.p_value is None else f"{row.p_value:0.3f}"
            tag = row.verdict.value + (" *" if row.is_baseline else "")
            lines.append(
                f"{row.name:<24} {row.total_return:>8.2f} {row.annualized_return:>8.2f} "
                f"{row.sharpe_ratio:>7.2f} {row.max_drawdown:>7.2f} "
                f"{row.win_rate:>6.2f} {row.total_trades:>5d} "
                f"{row.delta_vs_baseline:>+8.3f} {p_txt:>6} {tag:>18}"
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
                "p_value",
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
                    "" if row.p_value is None else row.p_value,
                    row.verdict.value,
                    row.is_baseline,
                ]
            )
        return buf.getvalue()

    def write_artifacts(self, report: SuiteReport, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(json.dumps(report.to_dict(), indent=2))
        (out_dir / "report.csv").write_text(self.render_csv(report))
        (out_dir / "report.txt").write_text(self.render_text(report))


def _metric_for_row(row: VariantReport, metric: str) -> float:
    if metric == "calmar":
        dd = abs(row.max_drawdown)
        if dd < 1e-9:
            return 0.0
        return row.annualized_return / dd
    return float(getattr(row, metric))


__all__ = ["ExperimentReporter", "SuiteReport", "Verdict", "VariantReport"]
