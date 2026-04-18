#!/usr/bin/env python3
"""Test hyper_growth with the WORKING basic model (vs. its broken sentiment default).

Diagnostic output showed:
  - hyper_growth uses model_type="sentiment" but feeds 5 OHLCV features to a
    model that was trained on 10 features (incl. sentiment_momentum_scaled).
    Result: prediction=0.0 → predicted_return=-1.0 on every bar.
  - The basic model has a real directional edge (BUY 55-57% at 12-24h horizons).

This experiment swaps model_type to "basic" and rerruns hyper_growth with
the SL variants we found earlier (baseline, sl_10pct). If the basic model
is truly the fix, we should see higher returns AND the signal-threshold
knobs should start showing effects (because predicted_return is no longer
a constant sentinel).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("LOG_LEVEL", "WARNING")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("atb", "atb.src.engines", "atb.src.strategies", "atb.src.prediction", "atb.matplotlib.font_manager"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from src.data_providers.offline import FixtureProvider  # noqa: E402
from src.engines.backtest.engine import Backtester  # noqa: E402
from src.risk.risk_manager import RiskParameters  # noqa: E402
from src.strategies.components import MLBasicSignalGenerator  # noqa: E402
from src.strategies.hyper_growth import create_hyper_growth_strategy  # noqa: E402


def make_strategy_with_basic_model(**kwargs: Any):
    """Build hyper_growth and swap its signal generator to model_type='basic'."""
    strategy = create_hyper_growth_strategy(**kwargs)
    # Replace signal generator with the basic-model variant.
    strategy.signal_generator = MLBasicSignalGenerator(
        name=f"{strategy.name}_signals",
        model_type="basic",
    )
    return strategy


def run_variant(name: str, builder, start: datetime, end: datetime) -> tuple[float, float, float, int, float, float]:
    strategy = builder()
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
        float(results.get("max_drawdown", 0.0)),
        float(results.get("sharpe_ratio", 0.0)),
        int(results.get("total_trades", 0)),
        float(results.get("win_rate", 0.0)),
        float(results.get("final_balance", 1000.0)),
    )


def main() -> int:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 12, 31, tzinfo=UTC)
    print(f"=== hyper_growth model-swap sweep ({start.date()} → {end.date()}) ===")
    header = f"{'variant':<42} {'trades':>6} {'winR%':>6} {'ret%':>8} {'maxDD%':>7} {'sharpe':>7} {'final$':>9}"
    print(header)
    print("-" * len(header))
    t_start = time.time()

    variants = [
        ("sentiment(broken) — baseline", lambda: create_hyper_growth_strategy()),
        ("sentiment(broken) — sl_10pct", lambda: create_hyper_growth_strategy(stop_loss_pct=0.10)),
        ("basic(working) — baseline", lambda: make_strategy_with_basic_model()),
        ("basic(working) — sl_10pct", lambda: make_strategy_with_basic_model(stop_loss_pct=0.10)),
        ("basic(working) — sl_10_frac30", lambda: make_strategy_with_basic_model(stop_loss_pct=0.10, base_fraction=0.30, risk_fraction=0.30)),
    ]

    baseline_ret = None
    for name, builder in variants:
        t0 = time.time()
        try:
            ret, dd, sharpe, trades, wr, final = run_variant(name, builder, start, end)
        except Exception as exc:
            print(f"{name:<42} ERROR: {type(exc).__name__}: {exc}")
            continue
        dt = time.time() - t0
        delta = "" if baseline_ret is None else f"  (Δ{ret - baseline_ret:+.2f})"
        print(f"{name:<42} {trades:>6} {wr:>6.1f} {ret:>8.2f} {dd:>7.2f} {sharpe:>7.3f} {final:>9.0f}{delta}  [{dt:.0f}s]")
        if baseline_ret is None:
            baseline_ret = ret

    print(f"\nTotal: {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
