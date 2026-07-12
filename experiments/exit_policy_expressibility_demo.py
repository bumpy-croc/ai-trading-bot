#!/usr/bin/env python3
"""Expressibility demo for #971: prove the three previously-inexpressible
trade-management policies are now reachable via config only (no src/ patch).

Runs HyperGrowth ETHUSDT/1h on F1 (2023H1) with:
  1. control            — live prod config (must match the regression baseline)
  2. early_cut arm      — MFE early-cut 1.5% within 18h via factory_kwargs
  3. trailing arm       — wider trailing (activation 5%, distance 3%),
                          breakeven at +8% via factory_kwargs
  4. no_trailing arm    — trailing disabled entirely via factory_kwargs

This is functional evidence for the PR, NOT a preregistered experiment —
no verdicts are drawn from these numbers.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import UTC, datetime

os.environ.setdefault("LOG_LEVEL", "WARNING")
logging.basicConfig(level=logging.WARNING)
for noisy in (
    "atb.Strategy.HyperGrowth",
    "atb.src.engines",
    "atb.src.strategies",
    "atb.src.prediction",
    "atb.src.data_providers",
    "MLBasicSignalGenerator",
):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from src.experiments.runner import ExperimentRunner  # noqa: E402
from src.experiments.schemas import ExperimentConfig  # noqa: E402

F1 = (datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 6, 30, tzinfo=UTC))

ARMS: dict[str, dict] = {
    "control": {},
    "early_cut_1p5_18h": {
        "factory_kwargs": {
            "early_cut_mfe_threshold_pct": 0.015,
            "early_cut_evaluation_window_hours": 18,
        }
    },
    "trail_5_3_be8": {
        "factory_kwargs": {
            "trailing_activation_threshold": 0.05,
            "trailing_distance_pct": 0.03,
            "breakeven_threshold": 0.08,
            "breakeven_buffer": 0.01,
        }
    },
    "no_trailing": {"factory_kwargs": {"enable_trailing_stop": False}},
}


def main() -> int:
    runner = ExperimentRunner()
    start, end = F1
    for arm, extra in ARMS.items():
        cfg = ExperimentConfig(
            strategy_name="hyper_growth",
            symbol="ETHUSDT",
            timeframe="1h",
            start=start,
            end=end,
            initial_balance=85.0,
            use_cache=True,
            provider="binance",
            factory_kwargs=extra.get("factory_kwargs"),
            risk_parameters=extra.get("risk_parameters", {}),
        )
        t0 = time.time()
        result = runner.run(cfg)
        print(
            f"{arm:<20} trades={result.total_trades:>4} "
            f"ret%={result.total_return:>8.4f} maxDD%={result.max_drawdown:>7.4f} "
            f"winR%={result.win_rate:>6.2f} final=${result.final_balance:>7.2f} "
            f"[{time.time() - t0:.0f}s]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
