#!/usr/bin/env python3
"""Hyper-growth factory-kwarg sweep.

The PR-602 experimentation framework exposes overrides for the signal
generator and confidence-weighted sizer, but for ``hyper_growth`` those
values are not on the P&L critical path (the strategy's returns come from
partial exits + trailing stops, not the ML signal — see
.claude/reports/hyper_growth_strategy_analysis.md).

This driver uses the framework's ``Backtester`` directly with the strategy
factory's kwargs (stop_loss_pct, take_profit_pct, base_fraction,
min_confidence, max_leverage) so we can experimentally validate whether
tuning those ACTUALLY-load-bearing knobs moves returns.

The framework itself is unchanged — we simply swap ``ExperimentRunner``'s
``_load_strategy(name)()`` call for one that accepts kwargs. This is a
legitimate extension of the framework's Python API for knobs that are not
yet exposed via YAML overrides.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("LOG_LEVEL", "WARNING")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for noisy in (
    "atb",
    "atb.Strategy.HyperGrowth",
    "atb.src.engines.backtest",
    "atb.src.engines",
    "atb.src.strategies",
    "atb.src.prediction",
    "atb.matplotlib.font_manager",
    "atb.src.engines.backtest.execution.exit_handler",
):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from src.data_providers.offline import FixtureProvider  # noqa: E402
from src.engines.backtest.engine import Backtester  # noqa: E402
from src.risk.risk_manager import RiskParameters  # noqa: E402
from src.strategies.hyper_growth import create_hyper_growth_strategy  # noqa: E402


def run_variant(name: str, kwargs: dict[str, Any], start: datetime, end: datetime) -> tuple[float, float, float, float, int, float, float]:
    strategy = create_hyper_growth_strategy(**kwargs)
    provider = FixtureProvider(Path("tests/data/BTCUSDT_1h_2023-01-01_2024-12-31.feather"))
    bt = Backtester(
        strategy=strategy,
        data_provider=provider,
        sentiment_provider=None,
        risk_parameters=RiskParameters(),
        initial_balance=1000.0,
        log_to_database=False,
    )
    results = bt.run(symbol="BTCUSDT", timeframe="1h", start=start, end=end)
    return (
        float(results.get("total_return", 0.0)),
        float(results.get("annualized_return", 0.0)),
        float(results.get("max_drawdown", 0.0)),
        float(results.get("sharpe_ratio", 0.0)),
        int(results.get("total_trades", 0)),
        float(results.get("win_rate", 0.0)),
        float(results.get("final_balance", 1000.0)),
    )


VARIANTS: list[tuple[str, dict[str, Any]]] = [
    ("baseline (sl=20 tp=30 f=0.20)", {}),
    # Stop loss tightening
    ("sl_10pct", {"stop_loss_pct": 0.10}),
    ("sl_15pct", {"stop_loss_pct": 0.15}),
    # Take profit variants (interacts with partial exits)
    ("tp_20pct", {"take_profit_pct": 0.20}),
    ("tp_40pct", {"take_profit_pct": 0.40}),
    # Position sizing (move the needle via leverage-of-balance)
    ("frac_10pct", {"base_fraction": 0.10, "risk_fraction": 0.10}),
    ("frac_30pct", {"base_fraction": 0.30, "risk_fraction": 0.30}),
    ("frac_40pct", {"base_fraction": 0.40, "risk_fraction": 0.40}),
    # Min-confidence gate (what the research doc said mattered most)
    ("conf_gate_0.02", {"min_confidence": 0.02}),
    ("conf_gate_0.10", {"min_confidence": 0.10}),
    ("conf_gate_0.20", {"min_confidence": 0.20}),
    # Combo: tighter SL + larger sizing (risk-adjusted aggression)
    ("combo_sl10_frac30", {"stop_loss_pct": 0.10, "base_fraction": 0.30, "risk_fraction": 0.30}),
    ("combo_sl15_frac30_tp40", {"stop_loss_pct": 0.15, "base_fraction": 0.30, "risk_fraction": 0.30, "take_profit_pct": 0.40}),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--full", action="store_true", help="Use full 2023-2024 range")
    p.add_argument("--only", type=str, default=None, help="Comma-separated variant names to run")
    args = p.parse_args()

    if args.full:
        start = datetime(2023, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)
    else:
        start = datetime(args.year, 1, 1, tzinfo=UTC)
        end = datetime(args.year, 12, 31, tzinfo=UTC)

    allow = None
    if args.only:
        allow = {n.strip() for n in args.only.split(",") if n.strip()}

    print(f"=== hyper_growth factory sweep ({start.date()} → {end.date()}) ===")
    header = f"{'variant':<36} {'trades':>6} {'winR%':>6} {'ret%':>8} {'annual%':>8} {'maxDD%':>7} {'sharpe':>7} {'final$':>9}"
    print(header)
    print("-" * len(header))

    t_start = time.time()
    rows = []
    baseline_ret = None
    for name, kwargs in VARIANTS:
        if allow is not None and name not in allow:
            continue
        t0 = time.time()
        try:
            ret, ann, dd, sharpe, trades, wr, final = run_variant(name, kwargs, start, end)
        except Exception as exc:
            print(f"{name:<36} ERROR: {type(exc).__name__}: {exc}")
            continue
        dt = time.time() - t0
        delta = "" if baseline_ret is None else f"  (Δ{ret - baseline_ret:+.2f})"
        print(f"{name:<36} {trades:>6} {wr:>6.1f} {ret:>8.2f} {ann:>8.2f} {dd:>7.2f} {sharpe:>7.3f} {final:>9.0f}{delta}  [{dt:.0f}s]")
        rows.append((name, trades, wr, ret, ann, dd, sharpe, final))
        if baseline_ret is None:
            baseline_ret = ret
    print(f"\nTotal: {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
