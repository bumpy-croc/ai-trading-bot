"""SignalDiagnostic — walks a strategy's signal generator bar-by-bar."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from src.experiments.diagnostics.hit_rate import HitRate
from src.experiments.diagnostics.report import DiagnosticReport
from src.experiments.diagnostics.stats import DistributionStats
from src.experiments.runner import ExperimentRunner
from src.experiments.schemas import ExperimentConfig
from src.strategies.components.signal_generator import Signal, SignalDirection

# Horizons (in bars) at which to compute direction-conditional hit rate.
# 1h/4h/12h/24h matches the horizons a typical ML signal generator is
# trained to predict, and matches the horizons the incident report used.
DEFAULT_HORIZONS: tuple[int, ...] = (1, 4, 12, 24)


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
        """Iterate over ``df`` bars and build the diagnostic report."""
        if df is None or len(df) == 0:
            raise ValueError(
                f"No historical data returned for diagnostic of {strategy_name!r} "
                f"on {symbol} {timeframe}; cannot walk bars."
            )

        # Validate the DataFrame shape BEFORE accessing ``df["close"]`` —
        # a missing 'close' column would otherwise raise a bare KeyError
        # deep in the indexer, bypassing the helpful error below.
        if not hasattr(df, "columns") or "close" not in getattr(df, "columns", []):
            raise ValueError(
                "Diagnostic requires a DataFrame with a 'close' column; "
                f"got {type(df).__name__}."
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
        closes = df["close"].to_numpy()

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


__all__ = ["DEFAULT_HORIZONS", "SignalDiagnostic"]
