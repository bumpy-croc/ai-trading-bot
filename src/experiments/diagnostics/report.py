"""Full signal-quality report aggregating stats + hit-rates for a single run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.experiments.diagnostics.hit_rate import HitRate
from src.experiments.diagnostics.stats import DistributionStats


@dataclass
class DiagnosticReport:
    """Full signal-quality report for a single strategy run.

    Produced by :meth:`SignalDiagnostic.run`. The ``constant_signal_warning``
    field is populated when ``predicted_return`` is effectively constant
    across the sampled window (n>=50, std<1e-9) — the fingerprint of a
    feature-pipeline / model-shape mismatch that returns a sentinel value
    on every bar. Explicitly separated from the decision-mix counts so the
    incident can surface in JSON/text output without the caller having to
    infer it.
    """

    strategy_name: str
    symbol: str
    timeframe: str
    bars_evaluated: int
    buy_count: int
    sell_count: int
    hold_count: int
    predicted_return: DistributionStats
    confidence: DistributionStats
    hit_rates: list[HitRate] = field(default_factory=list)
    # Set when predicted_return is literally constant (n>0 and std≈0) —
    # the single strongest signal that the feature pipeline feeds the
    # model a shape it can't consume and the model is returning a
    # fallback-path sentinel value.
    constant_signal_warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict serialization suitable for JSON output."""
        return {
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bars_evaluated": self.bars_evaluated,
            "decisions": {
                "buy": self.buy_count,
                "sell": self.sell_count,
                "hold": self.hold_count,
            },
            "predicted_return": self.predicted_return.to_dict(),
            "confidence": self.confidence.to_dict(),
            "hit_rates": [hr.to_dict() for hr in self.hit_rates],
            "constant_signal_warning": self.constant_signal_warning,
        }

    def render_text(self) -> str:
        """Return a human-readable text rendering of the report."""
        lines = [
            f"Signal-quality diagnostic: {self.strategy_name} " f"{self.symbol} {self.timeframe}",
            f"  bars evaluated: {self.bars_evaluated}",
            "",
            "Decision mix",
            f"  BUY : {self.buy_count:>6}  ({_pct(self.buy_count, self.bars_evaluated)}%)",
            f"  SELL: {self.sell_count:>6}  ({_pct(self.sell_count, self.bars_evaluated)}%)",
            f"  HOLD: {self.hold_count:>6}  ({_pct(self.hold_count, self.bars_evaluated)}%)",
            "",
            "Predicted return",
            f"  n={self.predicted_return.n}  "
            f"mean={self.predicted_return.mean:+.6f}  "
            f"std={self.predicted_return.std:.6f}  "
            f"min={self.predicted_return.min:+.6f}  "
            f"max={self.predicted_return.max:+.6f}  "
            f"pos_frac={self.predicted_return.positive_fraction:.2%}",
            "",
            "Confidence",
            f"  n={self.confidence.n}  "
            f"mean={self.confidence.mean:.4f}  "
            f"std={self.confidence.std:.4f}",
            "",
            "Direction-conditional hit rate",
        ]
        for hr in self.hit_rates:
            lines.append(
                f"  h={hr.horizon:>3}: "
                f"BUY  acc={hr.buy_accuracy * 100:5.2f}%  n={hr.buy_samples:>5}  "
                f"SELL acc={hr.sell_accuracy * 100:5.2f}%  n={hr.sell_samples:>5}"
            )
        if self.constant_signal_warning:
            lines.append("")
            lines.append(f"WARNING: {self.constant_signal_warning}")
        return "\n".join(lines)


def _pct(n: int, total: int) -> str:
    """Format ``n/total`` as a fixed-width percentage or '  —  ' when total is zero."""
    if total <= 0:
        return "  —  "
    return f"{100.0 * n / total:5.2f}"


__all__ = ["DiagnosticReport"]
