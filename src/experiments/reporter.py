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
    ERRORED = "ERRORED"  # variant's backtest raised; no verdict possible


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
    # Human-readable diagnostic warnings surfaced by the reporter. The most
    # important case (gap report G6/G7): a variant whose every headline
    # metric and per-trade P&L sequence ties the baseline — almost
    # certainly a dead-code override that didn't actually mutate the
    # strategy. Rendered in the text/CSV/JSON report.
    warnings: list[str] = field(default_factory=list)


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
    # Map of variant_name → exception string for variants whose backtest
    # raised. Persisted to the JSON/text/CSV artifacts so the reason
    # survives overnight runs.
    errors: dict[str, str] = field(default_factory=dict)

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

    NaN inputs are rejected: they propagate silently through sorting and
    comparisons (``nan`` compares False to everything), so a strategy that
    emits NaN Sharpe / annualized return would be classified HOLD and
    rendered as ``−∞``. Fail loudly instead.
    """
    if metric == "calmar":
        dd = abs(result.max_drawdown)
        if math.isnan(result.annualized_return) or math.isnan(result.max_drawdown):
            raise ValueError(
                f"ExperimentResult has NaN inputs for Calmar "
                f"(annualized_return={result.annualized_return!r}, "
                f"max_drawdown={result.max_drawdown!r})"
            )
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
    coerced = float(value)
    if math.isnan(coerced):
        raise ValueError(
            f"ExperimentResult.{metric} is NaN; refusing to rank an experiment "
            "whose metric is not a real number."
        )
    return coerced


def _metric_delta(variant_val: float, baseline_val: float) -> float:
    """Return variant - baseline with explicit handling of ±inf.

    Naive ``inf - inf`` is NaN; coercing NaN into a ±inf placeholder hides
    the fact that the two runs are indistinguishable. The convention here:

    * same-sign infinities (+∞/+∞ or −∞/−∞) → ``0.0`` (a tie for ranking)
    * opposite-sign infinities                → signed ±∞
    * exactly one side infinite               → that side's signed ±∞
    * two finite values                       → ordinary subtraction
    """
    v_inf = math.isinf(variant_val)
    b_inf = math.isinf(baseline_val)
    if v_inf and b_inf:
        if (variant_val > 0) == (baseline_val > 0):
            return 0.0
        return math.inf if variant_val > 0 else -math.inf
    if v_inf:
        return math.inf if variant_val > 0 else -math.inf
    if b_inf:
        # Variant - baseline: if baseline = +inf, delta = -inf; if baseline = -inf, delta = +inf.
        return -math.inf if baseline_val > 0 else math.inf
    return variant_val - baseline_val


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

    # Infinite Calmar edge cases.
    if math.isinf(variant_val) or math.isinf(baseline_val):
        if math.isinf(variant_val) and math.isinf(baseline_val):
            # Both ±∞. If signs match → indistinguishable. Opposite signs →
            # maximally distinguishable.
            if (variant_val > 0) == (baseline_val > 0):
                return 0.0
            return 1.0
        # Exactly one side is ±∞ — maximally distinguishable.
        return 1.0
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
    threshold = 1.0 - settings.significance_level
    passes_confidence = confidence is not None and confidence >= threshold

    # Handle ±inf explicitly for Calmar edge cases.
    if math.isinf(variant_val) or math.isinf(baseline_val):
        if math.isinf(variant_val) and math.isinf(baseline_val):
            # Same-sign infinities are a tie; opposite-sign → strict winner.
            if (variant_val > 0) == (baseline_val > 0):
                return Verdict.HOLD
            if variant_val > 0:
                return Verdict.PROMOTE if passes_confidence else Verdict.HOLD
            return Verdict.REJECT
        # Exactly one side is infinite.
        if math.isinf(variant_val):
            if variant_val > 0:
                return Verdict.PROMOTE if passes_confidence else Verdict.HOLD
            return Verdict.REJECT
        # baseline is infinite
        if baseline_val > 0:
            return Verdict.REJECT
        return Verdict.PROMOTE if passes_confidence else Verdict.HOLD

    delta = variant_val - baseline_val
    if delta < 0:
        return Verdict.REJECT
    if delta == 0:
        return Verdict.HOLD
    if confidence is None:
        return Verdict.HOLD
    if passes_confidence:
        return Verdict.PROMOTE
    return Verdict.HOLD


# Tolerance for "bitwise-identical" comparisons. Floating-point math in the
# backtest engine is deterministic given identical inputs, so two runs with
# effective-noop overrides are typically equal to ULP precision — but we
# allow a tiny slack to survive pandas/numpy version drift.
_IDENTICAL_TOL = 1e-9

# Metrics compared when deciding whether a variant is "identical to
# baseline" for G6/G7 warning purposes. The set deliberately omits
# ``max_drawdown`` derivatives and non-P&L headline numbers — the point
# of the warning is that the VARIANT'S P&L profile is
# indistinguishable from baseline, i.e. the override was a no-op.
_IDENTICAL_METRIC_FIELDS: tuple[str, ...] = (
    "total_return",
    "annualized_return",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "total_trades",
    "final_balance",
)


def _is_identical_to_baseline(
    baseline: ExperimentResult,
    variant: ExperimentResult,
) -> bool:
    """True when every headline metric matches baseline within ``_IDENTICAL_TOL``.

    Used as a trigger for the dead-code-override warning (G6). Integer
    metrics (``total_trades``) require strict equality; floats use the
    tolerance so pandas/numpy FP noise doesn't produce false negatives.
    """
    for field_name in _IDENTICAL_METRIC_FIELDS:
        b = getattr(baseline, field_name)
        v = getattr(variant, field_name)
        if isinstance(b, int) and isinstance(v, int):
            if b != v:
                return False
            continue
        bv = float(b)
        vv = float(v)
        if math.isnan(bv) or math.isnan(vv):
            # NaN inputs are rejected upstream by `_metric`; if one slips
            # through here, treat as non-identical rather than masking it.
            return False
        if abs(bv - vv) > _IDENTICAL_TOL:
            return False
    return True


def _pnl_sequence_identical(
    baseline_trades: list[float],
    variant_trades: list[float],
) -> bool:
    """True when two per-trade P&L sequences match element-wise.

    Empty baseline AND empty variant sequences count as identical (no
    trades is trivially "the same" trades). A single-sequence-empty case
    is treated as different — the tie on aggregate metrics is spurious.
    """
    if len(baseline_trades) != len(variant_trades):
        return False
    for a, b in zip(baseline_trades, variant_trades, strict=True):
        if math.isnan(a) or math.isnan(b):
            # NaN per-trade PnL is invalid upstream; fail the identity
            # check rather than return True for two NaN-filled lists.
            return False
        if abs(float(a) - float(b)) > _IDENTICAL_TOL:
            return False
    return True


def _detect_identical_to_baseline(
    baseline: ExperimentResult,
    variant: ExperimentResult,
    variant_name: str,
) -> list[str]:
    """Return warnings describing baseline-identical variants (G6/G7).

    Emits at most one warning. When the aggregate metrics match, the
    per-trade P&L sequence decides between:
      * "literally the same trades" — the override almost certainly did
        nothing (wrong attribute name, silently dropped, dead-code
        component target)
      * "different trades, same aggregate" — rare but interesting: the
        variant took a different path that happened to tie the baseline
        on every headline metric, worth surfacing so the operator can
        decide whether to dig in or trust the tie.
    """
    if variant_name == "baseline" or variant is baseline:
        return []
    if not _is_identical_to_baseline(baseline, variant):
        return []
    if _pnl_sequence_identical(baseline.trade_pnl_pcts, variant.trade_pnl_pcts):
        return [
            "Every headline metric AND the per-trade P&L sequence match "
            "baseline bitwise. The override almost certainly did not take "
            "effect — verify the attribute name routes to a live component "
            "and the override isn't a no-op on this strategy's risk "
            "manager / sizer."
        ]
    return [
        "Headline metrics match baseline exactly but the per-trade P&L "
        "sequence differs — the variant took different trades that "
        "happened to tie on aggregates. This is rare and may indicate a "
        "degenerate signal path; inspect the diagnostic "
        "(atb experiment diagnose ...) to confirm signal quality."
    ]


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

        errors = getattr(suite_result, "errors", {}) or {}

        for spec, result in zip(suite_result.config.variants, suite_result.variants, strict=True):
            if spec.name in errors:
                # Variant's backtest raised — emit an ERRORED row so the user
                # sees the failure in the report; ranking uses +/-inf = NaN.
                rows.append(
                    VariantReport(
                        name=spec.name,
                        total_return=0.0,
                        annualized_return=0.0,
                        sharpe_ratio=0.0,
                        max_drawdown=0.0,
                        win_rate=0.0,
                        total_trades=0,
                        final_balance=0.0,
                        delta_vs_baseline=0.0,
                        ranking_confidence=None,
                        verdict=Verdict.ERRORED,
                    )
                )
                continue
            delta = _metric_delta(
                _metric(result, settings.target_metric),
                _metric(baseline, settings.target_metric),
            )
            confidence = _ranking_confidence(baseline, result, settings.target_metric)
            verdict = _classify(baseline, result, settings, confidence)
            warnings = _detect_identical_to_baseline(baseline, result, spec.name)
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
                    warnings=warnings,
                )
            )

        def _sort_key(row: VariantReport) -> float:
            # ERRORED variants go to the bottom regardless of their (zero)
            # metrics — otherwise a crashed variant can rank above a losing
            # baseline and mislead a reader.
            if row.verdict == Verdict.ERRORED:
                return -math.inf
            return _metric_for_row(row, settings.target_metric)

        ranked = sorted(rows, key=_sort_key, reverse=True)
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
            errors=dict(errors),
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
            if math.isnan(row.delta_vs_baseline):
                delta_str = f"{'NaN':>8}"
            elif math.isfinite(row.delta_vs_baseline):
                delta_str = f"{row.delta_vs_baseline:>+8.3f}"
            else:
                delta_str = f"{'+∞' if row.delta_vs_baseline > 0 else '−∞':>8}"
            lines.append(
                f"{row.name:<24} {row.total_return:>8.2f} {row.annualized_return:>8.2f} "
                f"{row.sharpe_ratio:>7.2f} {row.max_drawdown:>7.2f} "
                f"{row.win_rate:>6.2f} {row.total_trades:>5d} "
                f"{delta_str} {conf_txt:>6} {tag:>18}"
            )
        rows_with_warnings = [row for row in report.rows if row.warnings]
        if rows_with_warnings:
            lines.append("")
            lines.append(f"Variant warnings ({len(rows_with_warnings)}):")
            for row in rows_with_warnings:
                for msg in row.warnings:
                    lines.append(f"  - {row.name}: {msg}")
        if report.errors:
            lines.append("")
            lines.append(f"Variant errors ({len(report.errors)}):")
            for variant_name, err in report.errors.items():
                lines.append(f"  - {variant_name}: {err}")
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
                "warnings",
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
                    # Join multi-line warnings with " | " so the CSV cell
                    # stays a single line readable in spreadsheets.
                    " | ".join(row.warnings),
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
