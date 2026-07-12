#!/usr/bin/env python3
"""GH #984 part 1 -- measure real slippage from our own prod ETHUSDT fills.

Reads the fill-level export from prod Postgres `orders`
(experiments/fills_scoped.csv -- entries+exits for the 12 closed trades in
`trades` plus position #22's still-open entry; position #13's exit is
excluded, see note below) and the cached ETHUSDT 1h OHLCV covering the same
dates, and computes two independent slippage references per fill:

  (a) same-bar reference: the 1h candle CONTAINING the fill timestamp, using
      its `open` (the market state at the start of the execution bar) --
      plus an OHLC sanity check that the fill price actually falls within
      [low, high] of that bar.
  (b) decision-time reference: the CLOSE of the most recently closed 1h
      candle strictly before the order's `created_at` timestamp -- i.e. the
      price the strategy's decision was implicitly anchored to (the last
      candle the model/signal saw before submitting the order).

In most cases (a) and (b) are nearly the same value (bar-open ~= prior
bar-close), so they double as a cross-check on each other rather than two
independent hypotheses.

Sign convention: slippage_bps > 0 means adverse (we paid more / received
less than the reference); < 0 means favorable execution (price improvement).
Trade direction (BUY vs SELL) is derived from (position side, order_type):
  LONG+ENTRY=BUY, LONG+EXIT=SELL, SHORT+ENTRY=SELL, SHORT+EXIT=BUY.

Fee/slippage separation: Binance's `actual_fill_price` is gross of
commission (commission is a separate deduction denominated in the received
asset, per `orders.actual_commission` -- base asset on buys, quote asset on
sells). Slippage here is computed purely from price deviation and never
nets out or double-counts the commission line -- verified below by
reporting both quantities side by side, never combined.
"""

from __future__ import annotations

import json
from datetime import UTC, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.cached_data_provider import CachedDataProvider

SYMBOL = "ETHUSDT"
TIMEFRAME = "1h"


def action_for(position_side: str, order_type: str) -> str:
    is_entry = order_type == "ENTRY"
    if position_side == "LONG":
        return "BUY" if is_entry else "SELL"
    if position_side == "SHORT":
        return "SELL" if is_entry else "BUY"
    raise ValueError(f"Unknown position side {position_side!r}")


def signed_slippage_bps(fill_price: float, ref_price: float, action: str) -> float:
    if action == "BUY":
        return (fill_price - ref_price) / ref_price * 10_000
    if action == "SELL":
        return (ref_price - fill_price) / ref_price * 10_000
    raise ValueError(action)


def containing_bar(df: pd.DataFrame, ts: pd.Timestamp) -> tuple[int, pd.Series] | tuple[None, None]:
    idx = df.index.searchsorted(ts, side="right") - 1
    if idx < 0 or idx >= len(df):
        return None, None
    bar_start = df.index[idx]
    if ts < bar_start or ts >= bar_start + timedelta(hours=1):
        return None, None
    return int(idx), df.iloc[idx]


def prior_closed_bar(
    df: pd.DataFrame, ts: pd.Timestamp
) -> tuple[int, pd.Series] | tuple[None, None]:
    """Most recently CLOSED 1h bar strictly before ts (bar close = bar_start + 1h)."""
    idx = df.index.searchsorted(ts, side="right") - 1
    # If ts falls inside bar idx (still forming), the last CLOSED bar is idx-1.
    if idx < 0:
        return None, None
    bar_start = df.index[idx]
    if ts >= bar_start + timedelta(hours=1):
        closed_idx = idx
    else:
        closed_idx = idx - 1
    if closed_idx < 0 or closed_idx >= len(df):
        return None, None
    return int(closed_idx), df.iloc[closed_idx]


def main() -> int:
    fills_path = Path("experiments/fills_scoped.csv")
    fills = pd.read_csv(fills_path, parse_dates=["created_at", "filled_at"])
    for col in ("created_at", "filled_at"):
        fills[col] = fills[col].dt.tz_localize(UTC)

    # position side lookup (order.side in this schema stores the POSITION
    # side, not the buy/sell action -- confirmed against prod `positions`).
    position_side = {
        5: "LONG",
        8: "LONG",
        10: "LONG",
        11: "LONG",
        13: "LONG",
        15: "SHORT",
        16: "LONG",
        17: "LONG",
        18: "LONG",
        19: "SHORT",
        20: "SHORT",
        21: "LONG",
        22: "SHORT",
    }

    provider = CachedDataProvider(BinanceProvider(), cache_ttl_hours=24 * 365)
    start = fills["created_at"].min() - timedelta(days=2)
    end = fills["filled_at"].max() + timedelta(days=1)
    df = provider.get_historical_data(SYMBOL, TIMEFRAME, start, end)
    print(f"Loaded {len(df)} ETHUSDT 1h bars {df.index.min()} .. {df.index.max()}")

    # CachedDataProvider silently returns empty for "1m" (unsupported by the
    # cache layer) -- fetch minute bars directly per-fill, live, in narrow
    # +/-15min windows. This is what makes reference (b) genuinely "closest
    # available" rather than up-to-59-minutes stale like the 1h bar boundary
    # (see the 1h-only first pass of this script, kept as reference (a)).
    raw_provider = BinanceProvider()
    minute_cache: dict[tuple, pd.DataFrame] = {}

    def minute_price_at_or_before(ts: pd.Timestamp) -> float | None:
        """Close of the last FULLY ELAPSED 1-minute bar strictly before ts.

        Deliberately NOT "the bar containing ts" -- during a fast move a
        still-forming minute bar's `close` reflects the price up to 60s
        *after* ts (look-ahead). Found this the hard way: the first version
        of this function used the containing bar's close and produced a
        +137.98bps outlier for order 7418 (position 17's exit, filled 2s
        after creation) that traced to a genuine ~4% 3-minute price spike
        (2026-06-07 22:11-22:14, volume 223->10077) -- the reference was
        reading 25s into the future relative to the actual order-creation
        instant. Using the prior CLOSED bar's close removes that lookahead.
        """
        window_key = (ts.floor("30min"),)
        if window_key not in minute_cache:
            wstart = ts.floor("30min") - timedelta(minutes=20)
            wend = ts.ceil("30min") + timedelta(minutes=5)
            try:
                minute_cache[window_key] = raw_provider.get_historical_data(
                    SYMBOL, "1m", wstart, wend
                )
            except Exception as exc:  # pragma: no cover - network
                print(f"  1m fetch failed for {ts}: {exc}")
                minute_cache[window_key] = pd.DataFrame()
        mdf = minute_cache[window_key]
        if mdf.empty:
            return None
        if mdf.index.tz is None:
            mdf.index = mdf.index.tz_localize(UTC)
        idx = mdf.index.searchsorted(ts, side="right") - 1
        if idx < 0 or idx >= len(mdf):
            return None
        bar_start = mdf.index[idx]
        if ts < bar_start + timedelta(minutes=1):
            idx -= 1  # ts falls inside a still-forming bar; step back one bar
        if idx < 0 or idx >= len(mdf):
            return None
        return float(mdf.iloc[idx]["close"])

    rows = []
    for _, f in fills.iterrows():
        pos_id = int(f["position_id"])
        side = position_side[pos_id]
        action = action_for(side, f["order_type"])
        fill_price = float(f["actual_fill_price"])

        bar_idx, bar = containing_bar(df, f["filled_at"])
        decision_idx, decision_bar = prior_closed_bar(df, f["created_at"])

        rec = {
            "order_id": int(f["id"]),
            "position_id": pos_id,
            "order_type": f["order_type"],
            "position_side": side,
            "action": action,
            "fill_price": fill_price,
            "fill_qty": float(f["actual_fill_quantity"]),
            "actual_commission": float(f["actual_commission"]),
            "created_at": f["created_at"].isoformat(),
            "filled_at": f["filled_at"].isoformat(),
        }

        if bar is not None:
            rec["bar_open"] = float(bar["open"])
            rec["bar_high"] = float(bar["high"])
            rec["bar_low"] = float(bar["low"])
            rec["bar_close"] = float(bar["close"])
            rec["within_bar_range"] = bool(bar["low"] <= fill_price <= bar["high"])
            rec["slippage_bps_bar_open"] = signed_slippage_bps(
                fill_price, float(bar["open"]), action
            )
        else:
            rec["within_bar_range"] = None
            rec["slippage_bps_bar_open"] = None

        if decision_bar is not None:
            rec["decision_bar_close"] = float(decision_bar["close"])
            rec["decision_bar_ts"] = df.index[decision_idx].isoformat()
            rec["decision_to_fill_minutes"] = (
                f["filled_at"] - df.index[decision_idx] - timedelta(hours=1)
            ).total_seconds() / 60.0
            rec["slippage_bps_decision_close_1h"] = signed_slippage_bps(
                fill_price, float(decision_bar["close"]), action
            )
        else:
            rec["decision_bar_close"] = None
            rec["slippage_bps_decision_close_1h"] = None

        # Tight reference: closest 1-minute close at-or-before order creation
        # (seconds-scale staleness, not up to 59 minutes).
        ref_1m = minute_price_at_or_before(f["created_at"])
        if ref_1m is not None:
            rec["ref_1m_price"] = ref_1m
            rec["slippage_bps_1m"] = signed_slippage_bps(fill_price, ref_1m, action)
        else:
            rec["ref_1m_price"] = None
            rec["slippage_bps_1m"] = None

        rows.append(rec)

    out = pd.DataFrame(rows)
    out_path = Path("experiments/slippage_fills.jsonl")
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    print(f"Wrote {len(rows)} fill records to {out_path}")

    print("\n=== Sanity: fills within containing-bar OHLC range ===")
    print(out["within_bar_range"].value_counts(dropna=False))

    for col, label in [
        (
            "slippage_bps_bar_open",
            "1h-bar-open reference (a) -- WARNING: includes up to 59min drift, see caveat",
        ),
        (
            "slippage_bps_decision_close_1h",
            "1h decision-close reference -- same 1h-staleness caveat",
        ),
        (
            "slippage_bps_1m",
            "1-MINUTE closest-price reference (b), tight -- PRIMARY slippage estimate",
        ),
    ]:
        vals = out[col].dropna().values
        print(f"\n=== Slippage bps, {label} (n={len(vals)}) ===")
        print(f"  mean={vals.mean():.3f}  median={np.median(vals):.3f}  std={vals.std():.3f}")
        print(f"  p10={np.percentile(vals,10):.3f}  p90={np.percentile(vals,90):.3f}")
        print(f"  min={vals.min():.3f}  max={vals.max():.3f} (worst adverse = max)")
        print(
            f"  n adverse (>0)={int((vals>0).sum())}  n favorable (<0)={int((vals<0).sum())}  n zero={int((vals==0).sum())}"
        )

    print("\n=== Per order_type breakdown (1-minute reference, primary) ===")
    print(out.groupby("order_type")["slippage_bps_1m"].agg(["count", "mean", "median", "std"]))

    vals_1m = out["slippage_bps_1m"].dropna().values
    print(
        f"\n=== Is the 1-minute signed slippage distinguishable from zero? (n={len(vals_1m)}) ==="
    )
    t_stat, t_p = stats.ttest_1samp(vals_1m, 0.0)
    w_stat, w_p = stats.wilcoxon(vals_1m)
    print(f"  one-sample t-test vs 0: t={t_stat:.3f} p={t_p:.4f}")
    print(f"  Wilcoxon signed-rank vs 0: W={w_stat:.3f} p={w_p:.4f}")
    print(f"  median |slippage| (per fill, per side) = {np.median(np.abs(vals_1m)):.3f} bps")
    print(f"  mean |slippage| (per fill, per side) = {np.mean(np.abs(vals_1m)):.3f} bps")
    print(f"  p90 |slippage| = {np.percentile(np.abs(vals_1m), 90):.3f} bps")

    print("\n=== Real order latency (filled_at - created_at), seconds ===")
    lat = (
        out["filled_at"].apply(lambda x: pd.Timestamp(x))
        - out["created_at"].apply(lambda x: pd.Timestamp(x))
    ).dt.total_seconds()
    print(lat.describe())

    print("\n=== Commission (separate cost, NOT slippage) ===")
    print(
        out[["order_id", "position_id", "order_type", "fill_qty", "actual_commission"]].to_string(
            index=False
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
