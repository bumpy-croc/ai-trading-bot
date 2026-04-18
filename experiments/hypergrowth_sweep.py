#!/usr/bin/env python3
"""Hyper-growth parameter sweep driver using the experimentation framework.

Runs baseline + multiple parameter variants against the BTCUSDT 2023-2024
fixture, printing a compact results table. Uses the new src.experiments
framework (PR #602) under the hood.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import UTC, datetime
from typing import Any

os.environ.setdefault("LOG_LEVEL", "WARNING")

# Silence noisy loggers before importing anything that uses them.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for noisy in (
    "atb.Strategy.HyperGrowth",
    "atb.src.engines.backtest",
    "atb.src.engines",
    "atb.src.strategies",
    "atb.src.prediction",
    "atb.matplotlib.font_manager",
    "atb.src.engines.backtest.execution.exit_handler",
):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from src.experiments.runner import ExperimentRunner  # noqa: E402
from src.experiments.schemas import ExperimentConfig, ParameterSet  # noqa: E402


def build_cfg(name: str, overrides: dict[str, Any], start: datetime, end: datetime) -> ExperimentConfig:
    parameters = ParameterSet(name=name, values=overrides) if overrides else None
    return ExperimentConfig(
        strategy_name="hyper_growth",
        symbol="BTCUSDT",
        timeframe="1h",
        start=start,
        end=end,
        initial_balance=1000.0,
        parameters=parameters,
        use_cache=False,
        provider="fixture",
        random_seed=42,
    )


def run_suite(suite_name: str, variants: list[tuple[str, dict[str, Any]]], start: datetime, end: datetime) -> None:
    runner = ExperimentRunner()
    print(f"\n=== {suite_name} ({start.date()} → {end.date()}) ===")
    header = f"{'variant':<28} {'trades':>6} {'winR%':>6} {'return%':>9} {'annual%':>9} {'maxDD%':>7} {'sharpe':>7} {'final$':>9}"
    print(header)
    print("-" * len(header))
    baseline_return = None
    for name, overrides in variants:
        cfg = build_cfg(name, overrides, start, end)
        t0 = time.time()
        try:
            r = runner.run(cfg)
        except Exception as exc:  # pragma: no cover — report instead of crashing the whole sweep
            print(f"{name:<28} ERROR: {type(exc).__name__}: {exc}")
            continue
        dt = time.time() - t0
        # ExperimentResult fields are already in percentage units (win_rate,
        # total_return, annualized_return, max_drawdown) — do not re-scale.
        delta = "" if baseline_return is None else f"  (Δ {r.total_return - baseline_return:+.2f})"
        print(
            f"{name:<28} {r.total_trades:>6} {r.win_rate:>6.1f} "
            f"{r.total_return:>9.2f} {r.annualized_return:>9.2f} "
            f"{r.max_drawdown:>7.2f} {r.sharpe_ratio:>7.3f} {r.final_balance:>9.0f}"
            f"{delta}  [{dt:.0f}s]"
        )
        if baseline_return is None:
            baseline_return = r.total_return


SUITES: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "long_thresholds": [
        ("baseline_defaults", {}),
        ("long_thr_0.02pct", {"hyper_growth.long_entry_threshold": 0.0002}),
        ("long_thr_0.05pct", {"hyper_growth.long_entry_threshold": 0.0005}),
        ("long_thr_0.10pct", {"hyper_growth.long_entry_threshold": 0.001}),
        ("long_thr_0.20pct", {"hyper_growth.long_entry_threshold": 0.002}),
    ],
    "confidence_multiplier": [
        ("baseline_defaults", {}),
        ("conf_mult_6", {"hyper_growth.confidence_multiplier": 6.0}),
        ("conf_mult_20", {"hyper_growth.confidence_multiplier": 20.0}),
        ("conf_mult_30", {"hyper_growth.confidence_multiplier": 30.0}),
    ],
    "short_thresholds": [
        ("baseline_defaults", {}),
        ("no_shorts_in_bull", {"hyper_growth.short_threshold_trend_up": -0.005}),
        ("earlier_bear_shorts", {"hyper_growth.short_threshold_trend_down": -0.0003}),
        (
            "tight_shorts_all",
            {
                "hyper_growth.short_entry_threshold": -0.001,
                "hyper_growth.short_threshold_trend_up": -0.0015,
                "hyper_growth.short_threshold_range": -0.001,
            },
        ),
        ("short_conf_x2.5", {"hyper_growth.short_threshold_confidence_multiplier": 0.5}),
    ],
    "combo": [
        ("baseline_defaults", {}),
        (
            "long_0.05_conf_20",
            {
                "hyper_growth.long_entry_threshold": 0.0005,
                "hyper_growth.confidence_multiplier": 20.0,
            },
        ),
        (
            "long_0.10_no_bull_shorts",
            {
                "hyper_growth.long_entry_threshold": 0.001,
                "hyper_growth.short_threshold_trend_up": -0.005,
            },
        ),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=list(SUITES.keys()) + ["all"], default="all")
    parser.add_argument("--year", type=int, default=2024, help="Fixture year to backtest")
    args = parser.parse_args()

    start = datetime(args.year, 1, 1, tzinfo=UTC)
    end = datetime(args.year, 12, 31, tzinfo=UTC)
    suites_to_run = list(SUITES.keys()) if args.suite == "all" else [args.suite]

    t0 = time.time()
    for name in suites_to_run:
        run_suite(name, SUITES[name], start, end)
    print(f"\nTotal sweep time: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
