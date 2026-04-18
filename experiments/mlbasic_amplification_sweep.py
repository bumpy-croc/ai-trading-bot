#!/usr/bin/env python3
"""Control experiment: signal-threshold + confidence-multiplier sweep on ml_basic.

Purpose: prove the PR-602 framework DOES detect signal-quality and
amplification effects on a strategy whose sizer is actually confidence-
sensitive (ml_basic uses ConfidenceWeightedSizer). If these knobs move
returns here but not on hyper_growth, the "zero effect" on hyper_growth is
a property of hyper_growth (FlatRiskManager + FixedFractionSizer by design
ignore confidence), not a framework bug.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import UTC, datetime
from typing import Any

os.environ.setdefault("LOG_LEVEL", "WARNING")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for noisy in (
    "atb",
    "atb.Strategy.MlBasic",
    "atb.src.engines",
    "atb.src.strategies",
    "atb.src.prediction",
    "atb.matplotlib.font_manager",
):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from src.experiments.runner import ExperimentRunner  # noqa: E402
from src.experiments.schemas import ExperimentConfig, ParameterSet  # noqa: E402


def build_cfg(name: str, overrides: dict[str, Any]) -> ExperimentConfig:
    parameters = ParameterSet(name=name, values=overrides) if overrides else None
    return ExperimentConfig(
        strategy_name="ml_basic",
        symbol="BTCUSDT",
        timeframe="1h",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 12, 31, tzinfo=UTC),
        initial_balance=1000.0,
        parameters=parameters,
        use_cache=False,
        provider="fixture",
        random_seed=42,
    )


VARIANTS: list[tuple[str, dict[str, Any]]] = [
    ("baseline (ml_basic defaults)", {}),
    ("long_thr_0.02pct", {"ml_basic.long_entry_threshold": 0.0002}),
    ("long_thr_0.05pct", {"ml_basic.long_entry_threshold": 0.0005}),
    ("long_thr_0.10pct", {"ml_basic.long_entry_threshold": 0.001}),
    ("conf_mult_6", {"ml_basic.confidence_multiplier": 6.0}),
    ("conf_mult_20", {"ml_basic.confidence_multiplier": 20.0}),
    ("conf_mult_30", {"ml_basic.confidence_multiplier": 30.0}),
]


def main() -> int:
    runner = ExperimentRunner()
    print("=== ml_basic amplification sweep (2024) ===")
    header = f"{'variant':<34} {'trades':>6} {'winR%':>6} {'ret%':>8} {'maxDD%':>7} {'sharpe':>7} {'final$':>9}"
    print(header)
    print("-" * len(header))
    baseline_ret = None
    t0 = time.time()
    for name, overrides in VARIANTS:
        cfg = build_cfg(name, overrides)
        t = time.time()
        try:
            r = runner.run(cfg)
        except Exception as exc:
            print(f"{name:<34} ERROR: {type(exc).__name__}: {exc}")
            continue
        dt = time.time() - t
        delta = "" if baseline_ret is None else f"  (Δ{r.total_return - baseline_ret:+.2f})"
        print(
            f"{name:<34} {r.total_trades:>6} {r.win_rate:>6.1f} {r.total_return:>8.2f} "
            f"{r.max_drawdown:>7.2f} {r.sharpe_ratio:>7.3f} {r.final_balance:>9.0f}{delta}  [{dt:.0f}s]"
        )
        if baseline_ret is None:
            baseline_ret = r.total_return
    print(f"\nTotal: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
