#!/usr/bin/env python3
"""Signal diagnostic for ml_basic (basic model, not sentiment).

Mirrors hypergrowth_signal_diagnostic but uses model_type='basic' so we can
compare whether the -1.0 sentinel issue is specific to the sentiment-model
mismatch or systemic to the prediction pipeline.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "WARNING")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("atb", "atb.src.engines", "atb.src.strategies", "atb.src.prediction", "atb.matplotlib.font_manager"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

import numpy as np  # noqa: E402

from src.data_providers.offline import FixtureProvider  # noqa: E402
from src.strategies.components.ml_signal_generator import MLBasicSignalGenerator  # noqa: E402
from src.strategies.components.signal_generator import SignalDirection  # noqa: E402


def main() -> int:
    provider = FixtureProvider(Path("tests/data/BTCUSDT_1h_2023-01-01_2024-12-31.feather"))
    df = provider.get_historical_data("BTCUSDT", "1h", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 12, 31, tzinfo=UTC))
    print(f"Loaded {len(df)} bars")

    gen = MLBasicSignalGenerator(name="diag_basic", model_type="basic")
    print(f"long_thr={gen.long_entry_threshold}, short_thr={gen.short_entry_threshold}, conf_mult={gen.confidence_multiplier}")

    start_idx = max(gen.sequence_length, 120) + 1
    horizons = (1, 4, 12, 24)
    preds: list[float] = []
    decisions = {SignalDirection.BUY: 0, SignalDirection.SELL: 0, SignalDirection.HOLD: 0}
    confidences: list[float] = []
    hits = {h: [0, 0, 0, 0] for h in horizons}

    # Stride to keep runtime bounded — 1/4 of all bars
    step = 4
    for n, idx in enumerate(range(start_idx, len(df) - max(horizons) - 1, step)):
        signal = gen.generate_signal(df, idx)
        meta = getattr(signal, "metadata", {}) or {}
        pred = meta.get("predicted_return")
        if pred is None:
            continue
        preds.append(float(pred))
        decisions[signal.direction] += 1
        confidences.append(float(signal.confidence))

        entry_price = float(df["close"].iloc[idx])
        for h in horizons:
            if idx + h >= len(df):
                continue
            fwd_price = float(df["close"].iloc[idx + h])
            fwd_ret = (fwd_price - entry_price) / entry_price
            if signal.direction == SignalDirection.BUY:
                hits[h][0 if fwd_ret > 0 else 1] += 1
            elif signal.direction == SignalDirection.SELL:
                hits[h][2 if fwd_ret < 0 else 3] += 1

    if not preds:
        print("No predictions.")
        return 1

    arr = np.array(preds)
    conf_arr = np.array(confidences)
    print("\n=== Predicted-return distribution (basic model) ===")
    print(f"  n={len(arr)}  mean={arr.mean():+.6f}  median={np.median(arr):+.6f}  std={arr.std():.6f}")
    print(f"  min={arr.min():+.6f}  max={arr.max():+.6f}")
    for q in (1, 5, 25, 50, 75, 95, 99):
        print(f"  p{q:02d}={np.percentile(arr, q):+.6f}")
    print(f"  fraction positive: {100*(arr > 0).mean():.2f}%")

    print("\n=== Decision mix ===")
    total_dec = sum(decisions.values())
    for d, c in decisions.items():
        print(f"  {d.value:>5}: {c} ({100*c/total_dec:.2f}%)")

    print("\n=== Confidence distribution ===")
    print(f"  mean={conf_arr.mean():.4f}  std={conf_arr.std():.4f}  p50={np.percentile(conf_arr, 50):.4f}")

    print("\n=== Directional hit rate vs. forward return ===")
    for h in horizons:
        br, bw, sr, sw = hits[h]
        if br + bw:
            print(f"  h={h:>3}: BUY  acc={br/(br+bw):6.2%}  (wins={br}, losses={bw})")
        if sr + sw:
            print(f"       SELL acc={sr/(sr+sw):6.2%}  (wins={sr}, losses={sw})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
