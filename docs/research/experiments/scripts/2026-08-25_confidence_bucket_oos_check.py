"""
Phase 1 check for GH #1067 (conviction-blind FlatRiskManager).

Reuses MLBasicSignalGenerator.generate_signal() directly (the exact production
inference path -- no hand-rolled scoring) to score every bar of a genuinely
out-of-sample window for the CURRENTLY DEPLOYED ETHUSDT model
(src/ml/models/ETHUSDT/basic/2026-07-04_22h_v1, training end_date=2026-07-04),
then buckets by |predicted_return| and by the model's own confidence score and
checks whether either separates realized next-bar hit rate or magnitude.

No backtest engine involved -- this is a pure scoring/statistics script, cheap
and side-effect-free. Does not touch RiskParameters, CostCalculator, or any
live/paper trading path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_providers.cached_data_provider import CachedDataProvider  # noqa: E402
from src.data_providers.binance_provider import BinanceProvider  # noqa: E402
from src.strategies.components.ml_signal_generator import MLBasicSignalGenerator  # noqa: E402

SYMBOL = "ETHUSDT"
TIMEFRAME = "1h"
MODEL_VERSION = "2026-07-04_22h_v1"  # currently deployed `latest` for ETHUSDT/basic
TRAIN_END = pd.Timestamp("2026-07-04T22:44:32+00:00")  # metadata created_at / training end_date=2026-07-04

# Genuinely fresh OOS window: bars strictly after the model's training cutoff,
# up to the latest cached data. Never seen by #912's exam either (which ran
# 2026-01-01 -> 2026-07-04 against a DIFFERENT retrain).
OOS_START = "2026-07-06"  # 1 clear day of buffer past training end_date
OOS_END = "2026-08-24"

CACHE_DIR = Path(__file__).resolve().parents[4] / "cache" / "market_data"


def load_data() -> pd.DataFrame:
    provider = CachedDataProvider(BinanceProvider(), cache_dir=str(CACHE_DIR))
    df = provider.get_historical_data(SYMBOL, TIMEFRAME, start=pd.Timestamp("2026-01-01"), end=pd.Timestamp("2026-08-25"))
    df = df.sort_index()
    return df


def score(df: pd.DataFrame) -> pd.DataFrame:
    gen = MLBasicSignalGenerator(symbol=SYMBOL, model_version=MODEL_VERSION)
    rows = []
    oos_start_ts = pd.Timestamp(OOS_START, tz="UTC")
    oos_end_ts = pd.Timestamp(OOS_END, tz="UTC")
    for i in range(len(df) - 1):  # need i+1 for realized forward return
        ts = df.index[i]
        if ts < oos_start_ts or ts > oos_end_ts:
            continue
        sig = gen.generate_signal(df, i)
        meta = sig.metadata
        if meta.get("reason") in ("insufficient_history", "prediction_failed", "invalid_prediction_or_price"):
            continue
        predicted_return = meta.get("predicted_return")
        if predicted_return is None:
            continue
        current_price = df["close"].iloc[i]
        next_price = df["close"].iloc[i + 1]
        realized_return = (next_price - current_price) / current_price
        rows.append(
            {
                "timestamp": ts,
                "predicted_return": predicted_return,
                "confidence": sig.confidence,
                "direction": sig.direction.value if hasattr(sig.direction, "value") else str(sig.direction),
                "realized_return": realized_return,
                "predicted_sign": 1 if predicted_return > 0 else (-1 if predicted_return < 0 else 0),
                "realized_sign": 1 if realized_return > 0 else (-1 if realized_return < 0 else 0),
            }
        )
    return pd.DataFrame(rows)


def bucket_stats(scored: pd.DataFrame, bucket_col: str, n_buckets: int) -> pd.DataFrame:
    df = scored.copy()
    df["hit"] = (df["predicted_sign"] == df["realized_sign"]).astype(int)
    df = df[df["predicted_sign"] != 0]  # exclude zero-predicted-return bars from hit-rate framing
    try:
        df["bucket"] = pd.qcut(df[bucket_col].abs(), n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    rows = []
    for b, g in df.groupby("bucket"):
        n = len(g)
        hits = g["hit"].sum()
        rate = hits / n if n else float("nan")
        ci_low, ci_high = wilson_ci(hits, n)
        rows.append(
            {
                "bucket": int(b),
                "n": n,
                "hit_rate": rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mean_abs_val": g[bucket_col].abs().mean(),
                "mean_abs_realized_return": g["realized_return"].abs().mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)


def wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((center - margin) / denom, (center + margin) / denom)


def cochran_armitage(bucket_df: pd.DataFrame) -> tuple[float, float]:
    # scores = bucket index; trend test on binomial counts across ordered groups
    scores = bucket_df["bucket"].values.astype(float)
    n = bucket_df["n"].values.astype(float)
    hits = (bucket_df["hit_rate"].values * n).round()
    N = n.sum()
    R = hits.sum()
    pbar = R / N
    s_mean = (n * scores).sum() / N
    num = (hits - n * pbar) @ (scores - s_mean)
    denom = pbar * (1 - pbar) * ((n * (scores - s_mean) ** 2).sum())
    if denom <= 0:
        return (float("nan"), float("nan"))
    z = num / np.sqrt(denom)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return (z, p)


def magnitude_correlation(scored: pd.DataFrame) -> dict:
    # Does |predicted_return| correlate with |realized_return| (magnitude prediction)?
    x = scored["predicted_return"].abs()
    y = scored["realized_return"].abs()
    rho, p = stats.spearmanr(x, y)
    return {"spearman_rho": rho, "spearman_p": p, "n": len(scored)}


def main():
    df = load_data()
    print(f"Loaded {len(df)} bars, {df.index.min()} -> {df.index.max()}")
    scored = score(df)
    print(f"Scored {len(scored)} bars in OOS window {OOS_START} -> {OOS_END}")
    print(f"Bars with predicted_sign != 0: {(scored['predicted_sign'] != 0).sum()}")
    print(f"Overall hit rate (direction): {(scored['predicted_sign'] == scored['realized_sign'])[scored['predicted_sign'] != 0].mean():.4f}")
    print(f"Confidence distribution: {scored['confidence'].describe()}")
    print(f"Fraction >= 0.05 gate: {(scored['confidence'] >= 0.05).mean():.4f}")

    results = {}
    for bucket_col, n_buckets in [("predicted_return", 5), ("predicted_return", 10), ("confidence", 5)]:
        key = f"{bucket_col}_q{n_buckets}"
        bdf = bucket_stats(scored, bucket_col, n_buckets)
        if bdf.empty:
            results[key] = {"error": "insufficient distinct values for qcut"}
            continue
        z, p = cochran_armitage(bdf)
        rho, sp_p = stats.spearmanr(bdf["bucket"], bdf["hit_rate"])
        results[key] = {
            "table": bdf.to_dict(orient="records"),
            "cochran_armitage_z": z,
            "cochran_armitage_p": p,
            "spearman_rho": rho,
            "spearman_p": sp_p,
        }
        print(f"\n=== {key} ===")
        print(bdf.to_string(index=False))
        print(f"Cochran-Armitage Z={z:.3f} p={p:.4f}; Spearman rho={rho:.3f} p={sp_p:.4f}")

    mag = magnitude_correlation(scored)
    results["magnitude_correlation"] = mag
    print(f"\n=== magnitude correlation (|pred| vs |realized|) ===\n{mag}")

    out_path = Path(__file__).resolve().parent / "2026-08-25_confidence_bucket_oos_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    scored.to_csv(Path(__file__).resolve().parent / "2026-08-25_confidence_bucket_oos_scored.csv", index=False)


if __name__ == "__main__":
    main()
