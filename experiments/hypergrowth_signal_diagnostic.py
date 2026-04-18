#!/usr/bin/env python3
"""Signal-quality diagnostic for hyper_growth's ML signal generator.

Rather than inferring signal behavior from portfolio P&L, this script
measures it DIRECTLY:

1. Runs the exact MLBasicSignalGenerator that hyper_growth uses over the
   2024 BTCUSDT fixture bar-by-bar.
2. Records the predicted_return, decided direction, and raw confidence.
3. Computes simple hit-rate metrics against N-bar forward returns so we can
   separate signal quality from position sizing / risk-management behavior.
4. Emits the distribution of predicted_returns and the BUY/SELL/HOLD mix so
   we can confirm or falsify the "model only emits SELL" hypothesis.

This does NOT modify the live trading bot. It reuses the same prediction
pipeline as hyper_growth so the numbers reflect what the strategy actually
sees at runtime.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "WARNING")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for noisy in (
    "atb",
    "atb.src.engines.backtest",
    "atb.src.engines",
    "atb.src.strategies",
    "atb.src.prediction",
    "atb.matplotlib.font_manager",
):
    logging.getLogger(noisy).setLevel(logging.ERROR)

import numpy as np  # noqa: E402

from src.data_providers.offline import FixtureProvider  # noqa: E402
from src.strategies.components.ml_signal_generator import MLBasicSignalGenerator  # noqa: E402
from src.strategies.components.signal_generator import SignalDirection  # noqa: E402


def main() -> int:
    provider = FixtureProvider(Path("tests/data/BTCUSDT_1h_2023-01-01_2024-12-31.feather"))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 12, 31, tzinfo=UTC)
    df = provider.get_historical_data("BTCUSDT", "1h", start, end)
    if df.empty:
        print("Fixture returned empty data.")
        return 1
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    gen = MLBasicSignalGenerator(name="diag", model_type="sentiment")
    print(
        "Signal generator config: "
        f"long_thr={gen.long_entry_threshold:.4f}, short_thr={gen.short_entry_threshold:.4f}, "
        f"conf_mult={gen.confidence_multiplier:.2f}"
    )

    start_idx = max(gen.sequence_length, 120) + 1
    horizons = (1, 4, 12, 24)
    preds: list[float] = []
    decisions = {SignalDirection.BUY: 0, SignalDirection.SELL: 0, SignalDirection.HOLD: 0}
    confidences: list[float] = []
    # Hit counts keyed by horizon → (buy_right, buy_wrong, sell_right, sell_wrong)
    hits = {h: [0, 0, 0, 0] for h in horizons}

    step = 1  # every bar — full walk
    indices = range(start_idx, len(df) - max(horizons) - 1, step)
    total = len(list(indices))
    print(f"Walking {total} bars (every bar from idx {start_idx})...")

    last_logged_pct = -1
    for n, idx in enumerate(range(start_idx, len(df) - max(horizons) - 1, step)):
        signal = gen.generate_signal(df, idx)
        pred = getattr(signal, "predicted_return", None)
        # generate_signal sets predicted_return via metadata; fall back to parsing metadata.
        meta = getattr(signal, "metadata", {}) or {}
        if pred is None:
            pred = meta.get("predicted_return")
        if pred is None:
            # Skip if no prediction (warmup bars)
            continue
        preds.append(float(pred))
        decisions[signal.direction] += 1
        confidences.append(float(signal.confidence))

        entry_price = float(df["close"].iloc[idx])
        for h in horizons:
            fwd_price = float(df["close"].iloc[idx + h])
            fwd_ret = (fwd_price - entry_price) / entry_price
            if signal.direction == SignalDirection.BUY:
                if fwd_ret > 0:
                    hits[h][0] += 1
                else:
                    hits[h][1] += 1
            elif signal.direction == SignalDirection.SELL:
                if fwd_ret < 0:
                    hits[h][2] += 1
                else:
                    hits[h][3] += 1

        pct = int((n + 1) * 100 / total)
        if pct != last_logged_pct and pct % 10 == 0:
            print(f"  progress: {pct}%  (bars processed: {n + 1}/{total})")
            last_logged_pct = pct

    if not preds:
        print("No predictions generated — check model availability.")
        return 1

    arr = np.array(preds)
    conf_arr = np.array(confidences)
    print("\n=== Predicted-return distribution ===")
    print(f"  n={len(arr)}  mean={arr.mean():+.6f}  median={np.median(arr):+.6f}  std={arr.std():.6f}")
    print(f"  min={arr.min():+.6f}  max={arr.max():+.6f}")
    for q in (1, 5, 25, 50, 75, 95, 99):
        print(f"  p{q:02d} = {np.percentile(arr, q):+.6f}")
    pos_pct = 100 * (arr > 0).mean()
    pos_beyond_long_thr = 100 * (arr > gen.long_entry_threshold).mean()
    neg_beyond_short_thr = 100 * (arr < gen.short_entry_threshold).mean()
    print(f"  fraction positive: {pos_pct:.2f}%")
    print(f"  fraction > long_entry_threshold ({gen.long_entry_threshold}): {pos_beyond_long_thr:.2f}%")
    print(f"  fraction < short_entry_threshold ({gen.short_entry_threshold}): {neg_beyond_short_thr:.2f}%")

    print("\n=== Decision mix ===")
    total_dec = sum(decisions.values())
    for d, c in decisions.items():
        print(f"  {d.value:>5}: {c} ({100*c/total_dec:.2f}%)")

    print("\n=== Confidence distribution ===")
    print(f"  mean={conf_arr.mean():.4f}  median={np.median(conf_arr):.4f}  std={conf_arr.std():.4f}")
    for q in (1, 25, 50, 75, 95, 99):
        print(f"  p{q:02d} = {np.percentile(conf_arr, q):.4f}")
    # Hyper_growth's FlatRiskManager gate
    above_gate_pct = 100 * (conf_arr >= 0.05).mean()
    print(f"  fraction >= 0.05 (FlatRiskManager gate): {above_gate_pct:.2f}%")

    print("\n=== Directional hit rate vs. forward return ===")
    for h in horizons:
        br, bw, sr, sw = hits[h]
        if br + bw:
            buy_acc = br / (br + bw)
            print(f"  h={h:>3}: BUY  acc={buy_acc:6.2%}  (wins={br}, losses={bw})")
        if sr + sw:
            sell_acc = sr / (sr + sw)
            print(f"       SELL acc={sell_acc:6.2%}  (wins={sr}, losses={sw})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
