"""Signal-quality diagnostic — measure the raw predictive signal of a strategy.

The ExperimentRunner ranks variants by backtest P&L. That is the right metric
for picking a winner, but it has a critical blind spot: a degenerate signal
(e.g. constant SELL because the model was fed the wrong feature tensor) can
still produce non-zero P&L from stop-loss and trailing-stop mechanics, and
every variant on top of that signal looks bitwise-identical in the reporter.
See ``.claude/reports/hyper_growth_experiment_sweep_2026-04-17.md`` for the
incident that motivated this tool.

The diagnostic walks the strategy's ``SignalGenerator`` bar-by-bar over a
range of history and reports four distributions that, together, tell you
whether the model has any directional edge at all:

* **Decision mix** — counts of BUY / SELL / HOLD. If it's 100% any-one-side
  or 100% HOLD the signal is dead regardless of how good the P&L looks.
* **Predicted return** — ``(prediction - current_price) / current_price``
  from the generator's metadata. A healthy model has a real distribution;
  a broken pipeline returns constants like ``-1.0`` (prediction = 0.0).
* **Confidence** — the generator's per-bar confidence score. A useful
  signal varies between high- and low-conviction bars.
* **Direction-conditional hit rate** — ``P(forward return > 0 | BUY)`` and
  ``P(forward return < 0 | SELL)`` at 1h / 4h / 12h / 24h horizons. These
  are the quantities any confidence-weighted sizer is trying to exploit.

Run via ``atb experiment diagnose --strategy <name> ...`` or programmatically
through :class:`SignalDiagnostic`. The module has no dependencies beyond the
framework that `src/experiments/runner.py` already pulls in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.experiments.runner import ExperimentRunner
from src.experiments.schemas import ExperimentConfig
from src.strategies.components.signal_generator import Signal, SignalDirection

# Horizons (in bars) at which to compute direction-conditional hit rate.
# 1h/4h/12h/24h matches the horizons a typical ML signal generator is
# trained to predict, and matches the horizons the incident report used.
DEFAULT_HORIZONS: tuple[int, ...] = (1, 4, 12, 24)


@dataclass
class DistributionStats:
    """Summary statistics for a numeric series."""

    n: int
    mean: float
    std: float
    min: float
    max: float
    positive_fraction: float

    @classmethod
    def from_series(cls, values: list[float]) -> DistributionStats:
        if not values:
            return cls(n=0, mean=0.0, std=0.0, min=0.0, max=0.0, positive_fraction=0.0)
        n = len(values)
        mean = sum(values) / n
        # Guard n=1 (std is 0 by definition). Use sample std for n≥2.
        if n == 1:
            std = 0.0
        else:
            var = sum((v - mean) ** 2 for v in values) / (n - 1)
            std = math.sqrt(var)
        pos = sum(1 for v in values if v > 0) / n
        return cls(
            n=n,
            mean=float(mean),
            std=float(std),
            min=float(min(values)),
            max=float(max(values)),
            positive_fraction=float(pos),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "positive_fraction": self.positive_fraction,
        }


@dataclass
class HitRate:
    """Direction-conditional accuracy at a specific forward horizon."""

    horizon: int
    buy_samples: int
    buy_accuracy: float
    sell_samples: int
    sell_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "buy_samples": self.buy_samples,
            "buy_accuracy": self.buy_accuracy,
            "sell_samples": self.sell_samples,
            "sell_accuracy": self.sell_accuracy,
        }


@dataclass
class DiagnosticReport:
    """Full signal-quality report for a single strategy run."""

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
    if total <= 0:
        return "  —  "
    return f"{100.0 * n / total:5.2f}"


class SignalDiagnostic:
    """Walk a strategy's signal generator bar-by-bar and report on it.

    Does NOT run a full backtest — the risk manager, position sizer, and
    trade execution are bypassed. This is deliberately signal-only so the
    report answers "is there a real predictive signal here?" without
    confusing that question with "is the risk/sizing/exit logic lucky?".

    The instance reuses :class:`ExperimentRunner` for strategy loading and
    data-provider wiring so factory kwargs, providers, and symbols are
    configured identically to a backtest suite.
    """

    def __init__(self, runner: ExperimentRunner | None = None):
        self.runner = runner or ExperimentRunner()

    def run(
        self,
        strategy_name: str,
        *,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        provider: str = "binance",
        use_cache: bool = True,
        random_seed: int | None = None,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        factory_kwargs: dict[str, Any] | None = None,
    ) -> DiagnosticReport:
        """Evaluate the strategy's signal generator over [start, end]."""
        cfg = ExperimentConfig(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            initial_balance=1000.0,
            provider=provider,
            use_cache=use_cache,
            random_seed=random_seed,
            factory_kwargs=dict(factory_kwargs or {}),
        )

        strategy = self.runner._load_strategy(
            cfg.strategy_name,
            factory_kwargs=cfg.factory_kwargs or None,
        )
        signal_generator = getattr(strategy, "signal_generator", None)
        if signal_generator is None:
            raise ValueError(
                f"Strategy {strategy_name!r} has no signal_generator; diagnostic unavailable."
            )

        provider_obj = self.runner._load_provider(
            cfg.provider,
            cfg.use_cache,
            start=cfg.start,
            end=cfg.end,
            timeframe=cfg.timeframe,
            seed=cfg.random_seed,
        )
        df = provider_obj.get_historical_data(
            symbol=cfg.symbol,
            timeframe=cfg.timeframe,
            start=cfg.start,
            end=cfg.end,
        )

        return self._walk_bars(
            df,
            signal_generator=signal_generator,
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            horizons=horizons,
        )

    def _walk_bars(
        self,
        df: Any,
        *,
        signal_generator: Any,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        horizons: tuple[int, ...],
    ) -> DiagnosticReport:
        if df is None or len(df) == 0:
            raise ValueError(
                f"No historical data returned for diagnostic of {strategy_name!r} "
                f"on {symbol} {timeframe}; cannot walk bars."
            )

        seq_len = int(getattr(signal_generator, "sequence_length", 1) or 1)
        start_idx = max(seq_len, 1)
        # Allocate per-direction forward-return lists used by hit-rate math.
        buy_forward: dict[int, list[float]] = {h: [] for h in horizons}
        sell_forward: dict[int, list[float]] = {h: [] for h in horizons}

        predicted_returns: list[float] = []
        confidences: list[float] = []
        buy_count = sell_count = hold_count = 0

        # ``close`` column is the reference price for forward-return math.
        closes = df["close"].to_numpy() if hasattr(df, "to_numpy") else None
        if closes is None:
            # FixtureProvider / live providers return DataFrames; bail with
            # a clear message rather than crashing deep in the walk.
            raise ValueError(
                "Diagnostic requires a DataFrame with a 'close' column; "
                f"got {type(df).__name__}."
            )

        n_bars = len(df)
        for i in range(start_idx, n_bars):
            try:
                sig: Signal = signal_generator.generate_signal(df, i)
            except Exception as exc:  # pragma: no cover — defensive
                raise RuntimeError(
                    f"Signal generator raised at bar {i} of {n_bars}: {exc}"
                ) from exc

            meta = sig.metadata if isinstance(sig.metadata, dict) else {}
            pr = meta.get("predicted_return")
            if isinstance(pr, int | float) and not isinstance(pr, bool) and math.isfinite(pr):
                predicted_returns.append(float(pr))
            conf = float(sig.confidence)
            if math.isfinite(conf):
                confidences.append(conf)

            current_price = float(closes[i])
            if sig.direction == SignalDirection.BUY:
                buy_count += 1
                for h in horizons:
                    j = i + h
                    if j < n_bars and current_price > 0:
                        buy_forward[h].append((float(closes[j]) - current_price) / current_price)
            elif sig.direction == SignalDirection.SELL:
                sell_count += 1
                for h in horizons:
                    j = i + h
                    if j < n_bars and current_price > 0:
                        sell_forward[h].append((float(closes[j]) - current_price) / current_price)
            else:
                hold_count += 1

        pr_stats = DistributionStats.from_series(predicted_returns)
        conf_stats = DistributionStats.from_series(confidences)

        hit_rates: list[HitRate] = []
        for h in horizons:
            b = buy_forward[h]
            s = sell_forward[h]
            hit_rates.append(
                HitRate(
                    horizon=h,
                    buy_samples=len(b),
                    buy_accuracy=(sum(1 for x in b if x > 0) / len(b)) if b else 0.0,
                    sell_samples=len(s),
                    # SELL "hits" when forward return is NEGATIVE (price drops
                    # after a sell signal), so count x < 0.
                    sell_accuracy=(sum(1 for x in s if x < 0) / len(s)) if s else 0.0,
                )
            )

        warning: str | None = None
        bars_evaluated = buy_count + sell_count + hold_count
        # A truly constant predicted_return (std ~0 across many bars) is
        # the fingerprint of a broken feature pipeline. Only warn when we
        # actually sampled enough bars to say something meaningful.
        if pr_stats.n >= 50 and pr_stats.std < 1e-9:
            warning = (
                f"predicted_return is effectively constant (n={pr_stats.n}, "
                f"std={pr_stats.std:.2e}, value≈{pr_stats.mean:+.6f}). This is "
                "the fingerprint of a feature-pipeline/model-shape mismatch — "
                "the model is likely returning its fallback value on every "
                "bar. Inspect the signal generator's feature pipeline."
            )
        elif pr_stats.n == 0 and bars_evaluated > 0:
            warning = (
                f"signal generator emitted {bars_evaluated} decisions but no "
                "'predicted_return' metadata. The generator may not be an "
                "ML-prediction generator, or its metadata schema changed."
            )

        return DiagnosticReport(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            bars_evaluated=bars_evaluated,
            buy_count=buy_count,
            sell_count=sell_count,
            hold_count=hold_count,
            predicted_return=pr_stats,
            confidence=conf_stats,
            hit_rates=hit_rates,
            constant_signal_warning=warning,
        )


__all__ = [
    "DEFAULT_HORIZONS",
    "DiagnosticReport",
    "DistributionStats",
    "HitRate",
    "SignalDiagnostic",
]
