#!/usr/bin/env python3
"""EV-CONDITIONING (GH #984 part 2) -- assemble entry-time observables for
each control-arm trade emitted by ev_conditioning_control_trades.py.

For each trade's entry_time, independently computes (no engine/src change,
no re-run of the backtest):

  - predicted_return / confidence / strength: recomputed via a standalone
    MLBasicSignalGenerator.generate_signal(df, idx) call at the trade's
    entry bar -- the exact live/backtest-parity signal-generation path
    (same method the 2026-07-05 confidence-calibration study used), against
    the same cached ETHUSDT 1h OHLCV the control backtest ran on. BaseTrade
    (src/engines/shared/models.py) does not persist confidence/predicted
    return on the in-memory Trade object, so this is the correct way to get
    it without touching src/.
  - realized_vol_entry: trailing 24-bar stdev of 1h log returns ending at
    the entry bar (plain descriptive statistic on cached OHLCV, not a
    duplicated trading-decision calculation).
  - regime label: EnhancedRegimeDetector's own RegimeDetector.annotate()
    (src/regime/detector.py), run ONCE over the full continuous OHLCV
    history (required -- its hysteresis loop is sequential and stateful),
    then read per-trade at the entry bar. This is the same detector class
    hyper_growth's LeveragedPositionSizer consumes at runtime.
  - hour_of_day (UTC) and a coarse session bucket, from entry_time itself.

GH #997/#998 note: symbol is threaded explicitly everywhere below --
scoring ETHUSDT candles with the BTCUSDT model (the bug those issues
describe) would silently corrupt every predicted_return value here.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("LOG_LEVEL", "WARNING")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("atb", "Strategy.HyperGrowth", "MLBasicSignalGenerator"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data_providers.binance_provider import BinanceProvider  # noqa: E402
from src.data_providers.cached_data_provider import CachedDataProvider  # noqa: E402
from src.regime.detector import RegimeDetector  # noqa: E402
from src.strategies.components.ml_signal_generator import MLBasicSignalGenerator  # noqa: E402

SYMBOL = "ETHUSDT"
TIMEFRAME = "1h"
# Wide enough to cover F1-F3 (2023-01 -> 2025-06-30) plus lookback for the
# model's 120-bar sequence length and the regime detector's rolling windows.
HISTORY_START = datetime(2022, 6, 1, tzinfo=UTC)
HISTORY_END = datetime(2025, 7, 5, tzinfo=UTC)

SESSION_BOUNDS = [
    (0, 8, "asia"),
    (8, 14, "europe"),
    (14, 22, "us"),
    (22, 24, "late_us_early_asia"),
]


def session_for_hour(hour: int) -> str:
    for lo, hi, name in SESSION_BOUNDS:
        if lo <= hour < hi:
            return name
    return "unknown"


def nearest_index(df: pd.DataFrame, ts: pd.Timestamp) -> int | None:
    """Locate the bar index whose timestamp matches ts (entry bar).

    Trades enter on the candle whose index the strategy evaluated; backtest
    entry_time should match a bar exactly. Falls back to the nearest bar at
    or before ts (searchsorted) if not an exact hit, and rejects gaps > 1
    bar (2h) as a data-alignment problem rather than silently miscoding it.
    """
    idx = df.index.searchsorted(ts, side="right") - 1
    if idx < 0 or idx >= len(df):
        return None
    gap_hours = (ts - df.index[idx]).total_seconds() / 3600.0
    if gap_hours > 2.0:
        return None
    return int(idx)


def main() -> int:
    trades_path = Path("experiments/ev_conditioning_control_trades.jsonl")
    trades = [json.loads(line) for line in trades_path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(trades)} control trades from {trades_path}")

    provider = CachedDataProvider(BinanceProvider(), cache_ttl_hours=24 * 365)
    df = provider.get_historical_data(SYMBOL, TIMEFRAME, HISTORY_START, HISTORY_END)
    print(f"Loaded {len(df)} bars {df.index.min()} .. {df.index.max()}")

    # Regime: one full-history annotate() pass (sequential hysteresis state
    # inside RegimeDetector.annotate -- must run once over the whole
    # continuous series, not per-trade).
    regime_df = RegimeDetector().annotate(df.copy())

    # Realized vol: trailing 24-bar stdev of 1h log returns, computed once
    # vectorized over the whole series.
    log_ret = np.log(df["close"]).diff()
    realized_vol_24 = log_ret.rolling(24).std()

    gen = MLBasicSignalGenerator(name="ev_conditioning", model_type="basic", symbol=SYMBOL)
    print(
        f"Signal generator: symbol={gen.symbol} conf_mult={gen.confidence_multiplier} "
        f"seq_len={gen.sequence_length}"
    )

    out = []
    skipped = 0
    for t in trades:
        entry_ts = pd.Timestamp(t["entry_time"])
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize(UTC)
        idx = nearest_index(df, entry_ts)
        if idx is None or idx < gen.sequence_length + 1:
            skipped += 1
            continue

        signal = gen.generate_signal(df, idx)
        meta = getattr(signal, "metadata", {}) or {}
        predicted_return = meta.get("predicted_return")
        if predicted_return is None:
            predicted_return = getattr(signal, "predicted_return", None)
        if predicted_return is None:
            skipped += 1
            continue

        row = regime_df.iloc[idx]
        rec = dict(t)
        rec["bar_index"] = idx
        rec["bar_timestamp"] = df.index[idx].isoformat()
        rec["predicted_return"] = float(predicted_return)
        rec["abs_predicted_return"] = abs(float(predicted_return))
        rec["confidence"] = float(signal.confidence)
        rec["strength"] = float(getattr(signal, "strength", float("nan")))
        rec["realized_vol_entry"] = (
            float(realized_vol_24.iloc[idx]) if not pd.isna(realized_vol_24.iloc[idx]) else None
        )
        rec["trend_label"] = str(row["trend_label"])
        rec["vol_label"] = str(row["vol_label"])
        rec["hour_utc"] = entry_ts.hour
        rec["session"] = session_for_hour(entry_ts.hour)
        rec["win"] = bool((rec.get("pnl_percent") or 0) > 0)
        out.append(rec)

    out_path = Path("experiments/ev_conditioning_features.jsonl")
    with open(out_path, "w") as f:
        for rec in out:
            f.write(json.dumps(rec, default=str) + "\n")
    print(f"Wrote {len(out)} feature rows to {out_path} ({skipped} trades skipped)")

    # Quick sanity distributions (full stats/quantile testing done in the
    # separate analysis notebook/script, not here).
    conf = np.array([r["confidence"] for r in out])
    vol = np.array([r["realized_vol_entry"] for r in out if r["realized_vol_entry"] is not None])
    print(
        f"confidence: n={len(conf)} mean={conf.mean():.4f} median={np.median(conf):.4f} "
        f"p95={np.percentile(conf,95):.4f}"
    )
    print(f"realized_vol_entry: n={len(vol)} mean={vol.mean():.5f} median={np.median(vol):.5f}")
    trend_counts = pd.Series([r["trend_label"] for r in out]).value_counts()
    print("trend_label counts:\n", trend_counts)
    session_counts = pd.Series([r["session"] for r in out]).value_counts()
    print("session counts:\n", session_counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
