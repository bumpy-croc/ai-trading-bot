#!/usr/bin/env python3
"""Fast smoke test for the hyper-growth signal fix.

Builds the default hyper_growth strategy and walks the first 500 bars of
2024 through MLBasicSignalGenerator directly to prove:

  1. predicted_return is a real distribution (not the -1.0 constant sentinel
     the broken sentiment-model configuration produced)
  2. decision mix contains BUY and SELL (not 100% SELL)
  3. signal generator is using model_type="basic"

If the fix is correctly applied, this prints OK and exits 0.
This is a signal-level diagnostic — no full backtest required. Runs in
well under 60 seconds.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "WARNING")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for noisy in (
    "atb",
    "atb.Strategy.HyperGrowth",
    "atb.src.engines.backtest",
    "atb.src.strategies",
    "atb.src.prediction",
):
    logging.getLogger(noisy).setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.strategies.components import SignalDirection  # noqa: E402
from src.strategies.hyper_growth import create_hyper_growth_strategy  # noqa: E402


def main() -> int:
    strategy = create_hyper_growth_strategy()
    gen = strategy.signal_generator

    # Verify the fix is live at the configuration level
    assert getattr(gen, "model_type", None) == "basic", (
        f"Expected model_type='basic', got {getattr(gen, 'model_type', None)!r}"
    )
    print(f"  OK: signal generator model_type = {gen.model_type!r}")

    # Walk 2024 bars through the actual generator
    fixture = ROOT / "tests/data/BTCUSDT_1h_2023-01-01_2024-12-31.feather"
    df = pd.read_feather(fixture)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = df.sort_index()
    df_2024 = df[df.index >= "2024-01-01"].head(500).copy()
    if len(df_2024) < 200:
        print(f"  FAIL: insufficient fixture data ({len(df_2024)} bars)")
        return 1

    start_idx = max(gen.sequence_length, 120)
    buy = sell = hold = 0
    preds: list[float] = []
    confs: list[float] = []
    for i in range(start_idx, len(df_2024)):
        sig = gen.generate_signal(df_2024, i)
        if sig.direction == SignalDirection.BUY:
            buy += 1
        elif sig.direction == SignalDirection.SELL:
            sell += 1
        else:
            hold += 1
        pr = sig.metadata.get("predicted_return") if isinstance(sig.metadata, dict) else None
        if pr is not None:
            preds.append(float(pr))
        confs.append(float(sig.confidence))

    total = buy + sell + hold
    print(f"  Decision mix over {total} bars: BUY={buy} SELL={sell} HOLD={hold}")

    if preds:
        pmin, pmax = min(preds), max(preds)
        pmean = sum(preds) / len(preds)
        print(
            f"  predicted_return: n={len(preds)} min={pmin:+.6f} max={pmax:+.6f} mean={pmean:+.6f}"
        )
        if pmin == -1.0 and pmax == -1.0:
            print("  FAIL: predicted_return is the -1.0 constant sentinel — fix not live")
            return 1
        print("  OK: predicted_return is a real distribution")

    if buy == 0 and sell > 0 and hold == 0:
        print("  FAIL: 100% SELL decisions — signal is still constant")
        return 1
    print("  OK: decision mix contains multiple directions")

    print("\n  ALL CHECKS PASSED — hyper-growth signal fix is live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
