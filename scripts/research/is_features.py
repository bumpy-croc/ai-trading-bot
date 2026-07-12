#!/usr/bin/env python3
"""Feature construction for the LINEAR INPUT-SCREENING experiment (Lane A, Phase 1).

Research script only (CODE.md Backtest-Live Parity: nothing here counts as
backtest-ready until it goes through src/engines/shared/). Every function's
docstring states its own alignment rule; these match the prereg's §4/§6 tables
verbatim — do not change the rule here without amending the prereg first.

Feature-contract convention: every "extra" feature (arms 1/3/4/5/6) is computed on
the FULL historical frame using only backward-looking rolling windows, then sampled
at row `idx-1` (the last bar in the price-only 120-bar sequence, already closed) when
assembling a training/eval sample for target bar `idx`. Arm 2 (calendar) is the sole
exception — it uses bar `idx`'s own timestamp, which is a deterministic function of
the exchange clock and carries no lookahead risk by construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.prediction.features.price_only import PriceOnlyFeatureExtractor  # noqa: E402

SEQUENCE_LENGTH = 120
PRICE_ONLY_COLS = [
    "close_normalized",
    "volume_normalized",
    "high_normalized",
    "low_normalized",
    "open_normalized",
]


def build_price_only_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """PriceOnlyFeatureExtractor(120) applied to the full OHLCV frame — the exact
    feature contract the target-redesign tournament's linear baseline used."""
    extractor = PriceOnlyFeatureExtractor(normalization_window=SEQUENCE_LENGTH)
    return extractor.extract(ohlcv.copy())


# ---------------------------------------------------------------------------
# Arm 1: multi-scale realized vol + range dynamics (own OHLCV)
# ---------------------------------------------------------------------------


def realized_vol_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Alignment rule: every rolling window ends at the row's own index (that bar,
    once closed). Sampled at `t-1` when scoring target `t` — this frame's row `t-1`
    never uses bar `t`'s still-forming high/low/close."""
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    log_ret = np.log(close / close.shift(1))

    out = pd.DataFrame(index=ohlcv.index)
    out["rv_6h"] = log_ret.rolling(6, min_periods=6).std()
    out["rv_24h"] = log_ret.rolling(24, min_periods=24).std()
    out["rv_168h"] = log_ret.rolling(168, min_periods=168).std()
    # Parkinson range-based volatility estimator, 24h rolling window.
    log_hl2 = (np.log(high / low)) ** 2
    out["park_vol_24h"] = np.sqrt(log_hl2.rolling(24, min_periods=24).mean() / (4 * np.log(2)))
    out["hl_range_pct"] = (high - low) / close
    out["hl_range_pct_ma24"] = out["hl_range_pct"].rolling(24, min_periods=24).mean()
    return out


REALIZED_VOL_COLS = [
    "rv_6h",
    "rv_24h",
    "rv_168h",
    "park_vol_24h",
    "hl_range_pct",
    "hl_range_pct_ma24",
]


# ---------------------------------------------------------------------------
# Arm 2: calendar/session features
# ---------------------------------------------------------------------------


def calendar_features_for_timestamps(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Alignment rule: uses bar `t`'s OWN timestamp directly (not `t-1`) — a
    deterministic function of the exchange clock, known arbitrarily far in advance.
    No lookahead surface."""
    hour = timestamps.hour + timestamps.minute / 60.0
    dow = timestamps.dayofweek
    out = pd.DataFrame(index=timestamps)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    # Funding settles at 00:00/08:00/16:00 UTC — hours until the next settlement.
    hours_since_midnight = hour
    hours_to_next = (8 - (hours_since_midnight % 8)) % 8
    hours_to_next = np.where(hours_to_next == 0, 8.0, hours_to_next)
    out["hours_to_funding"] = hours_to_next
    return out


CALENDAR_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "hours_to_funding"]


# ---------------------------------------------------------------------------
# Arm 3: BTC -> ETH cross-asset
# ---------------------------------------------------------------------------


def btc_cross_frame(btc_ohlcv: pd.DataFrame, eth_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Alignment rule: BTC features are computed on BTC's own closed bars, then
    reindexed onto ETH's timeline with forward-fill only (never a future BTC bar).
    When a sample for ETH target `t` is assembled, this frame is sampled at ETH's
    `t-1` timestamp — i.e. BTC's bar at (or the last BTC bar at/before) ETH's `t-1`,
    which by construction closed before ETH's `t` prediction is made."""
    close = btc_ohlcv["close"]
    log_ret = np.log(close / close.shift(1))
    btc = pd.DataFrame(index=btc_ohlcv.index)
    btc["btc_ret_1h"] = log_ret
    btc["btc_ret_6h"] = np.log(close / close.shift(6))
    btc["btc_ret_24h"] = np.log(close / close.shift(24))
    btc["btc_rv_24h"] = log_ret.rolling(24, min_periods=24).std()
    # Reindex onto ETH's own timeline; forward-fill carries the last KNOWN (past)
    # BTC value forward across any gap — never pulls a later BTC bar backward.
    out = btc.reindex(eth_index).ffill()
    return out


BTC_COLS = ["btc_ret_1h", "btc_ret_6h", "btc_ret_24h", "btc_rv_24h"]


# ---------------------------------------------------------------------------
# Arm 4: funding rate (ETHUSDT perp)
# ---------------------------------------------------------------------------


def funding_frame(
    funding_raw: pd.DataFrame, eth_index: pd.DatetimeIndex, train_cutoff
) -> pd.DataFrame:
    """Alignment rule: forward-filled from the last SETTLED print as of the queried
    timestamp (merge_asof, direction='backward' — never a settlement that hasn't
    printed yet). `markPrice`'s empty pre-2021 values are NaN (never coerced to 0.0)
    and excluded from anything that would use them. The 30-day z-score's mean/std are
    fit on the TRAINING split only (rows with timestamp <= train_cutoff) and applied
    unchanged to the eval window — the harness-wide frozen-statistic rule from the
    target-redesign tournament, applied here so eval-window bars never leak into
    their own normalization."""
    fr = funding_raw.sort_index()
    settled = fr["fundingRate"]

    eth_df = pd.DataFrame(index=eth_index)
    merged = pd.merge_asof(
        eth_df.rename_axis("ts").reset_index(),
        settled.rename_axis("ts").reset_index().rename(columns={"fundingRate": "funding_rate"}),
        on="ts",
        direction="backward",
    ).set_index("ts")

    out = pd.DataFrame(index=eth_index)
    out["funding_rate"] = merged["funding_rate"]
    # rate-of-change vs the previous *settlement* (not previous hourly row, which
    # would mostly be a repeat of the same settlement between prints)
    settled_unique = settled[~settled.index.duplicated(keep="first")]
    roc = settled_unique.diff()
    roc_merged = pd.merge_asof(
        eth_df.rename_axis("ts").reset_index(),
        roc.rename_axis("ts").reset_index().rename(columns={"fundingRate": "funding_roc"}),
        on="ts",
        direction="backward",
    ).set_index("ts")
    out["funding_roc"] = roc_merged["funding_roc"]

    train_mask = out.index <= pd.Timestamp(train_cutoff)
    train_vals = out.loc[train_mask, "funding_rate"].dropna()
    mu, sigma = train_vals.mean(), train_vals.std()
    if not sigma or np.isnan(sigma):
        sigma = 1.0
    out["funding_z"] = (out["funding_rate"] - mu) / sigma
    out["funding_extreme"] = (out["funding_z"].abs() > 2.0).astype(float)
    return out


FUNDING_COLS = ["funding_rate", "funding_roc", "funding_z", "funding_extreme"]
FUNDING_FIRST_SETTLEMENT = pd.Timestamp("2019-11-27", tz="UTC")


# ---------------------------------------------------------------------------
# Arm 5: basis / perp-spot premium proxy
# ---------------------------------------------------------------------------


def premium_frame(premium_raw: pd.DataFrame, eth_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Alignment rule: uses only the closed bar's own `close`-of-premium value —
    never that same bar's `high`/`low` (not known until the bar itself closes)."""
    close = premium_raw["close"].sort_index()
    close = close[~close.index.duplicated(keep="first")]
    out = pd.DataFrame(index=eth_index)
    out["premium_close"] = close.reindex(eth_index).ffill()
    out["premium_vol_24h"] = out["premium_close"].rolling(24, min_periods=24).std()
    return out


PREMIUM_COLS = ["premium_close", "premium_vol_24h"]
PREMIUM_FIRST_BAR = pd.Timestamp("2019-12-24", tz="UTC")


# ---------------------------------------------------------------------------
# Arm 6: Fear & Greed
# ---------------------------------------------------------------------------


def fear_greed_frame(fng_raw: pd.DataFrame, eth_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Alignment rule: a daily value printed for calendar day D is used only for
    bars on or after D+1 — one full day of conservatism past the print (the audit's
    explicit lag rule; alternative.me's "today" value is a running/settling read,
    not confirmed final until the day closes)."""
    value = fng_raw["value"].sort_index()
    lagged = value.copy()
    lagged.index = lagged.index + pd.Timedelta(days=1)
    lagged = lagged[~lagged.index.duplicated(keep="first")]

    out = pd.DataFrame(index=eth_index)
    merged = pd.merge_asof(
        pd.DataFrame(index=eth_index).rename_axis("ts").reset_index(),
        lagged.rename_axis("ts").reset_index().rename(columns={"value": "fng_value"}),
        on="ts",
        direction="backward",
    ).set_index("ts")
    out["fng_value"] = merged["fng_value"]
    out["fng_momentum_7d"] = out["fng_value"] - out["fng_value"].shift(7 * 24)
    out["fng_extreme"] = ((out["fng_value"] < 20) | (out["fng_value"] > 80)).astype(float)
    return out


FNG_COLS = ["fng_value", "fng_momentum_7d", "fng_extreme"]
FNG_FIRST_DAY = pd.Timestamp("2018-02-01", tz="UTC") + pd.Timedelta(days=1)


# ---------------------------------------------------------------------------
# Sample assembly
# ---------------------------------------------------------------------------


def assemble_samples(
    price_only: pd.DataFrame,
    target_indices: np.ndarray,
    extra_frame: pd.DataFrame | None,
    extra_cols: list[str] | None,
    calendar_for_targets: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for the given target row positions (integer positions into
    `price_only`, each >= SEQUENCE_LENGTH and >= 1).

    X = flattened 120-bar price-only sequence [t-120 .. t-1] (600 dims) plus, if
    given, `extra_frame`'s row at position `t-1` (arms 1/3/4/5) and/or
    `calendar_for_targets`'s row at position `t` (arm 2 only).
    y = 1 if close[t] > close[t-1] else 0.
    """
    values = price_only[PRICE_ONLY_COLS].to_numpy(dtype=np.float64)
    close = price_only["close"].to_numpy(dtype=np.float64)

    extra_vals = (
        extra_frame[extra_cols].to_numpy(dtype=np.float64) if extra_frame is not None else None
    )
    cal_vals = (
        calendar_for_targets.to_numpy(dtype=np.float64)
        if calendar_for_targets is not None
        else None
    )

    X_rows = []
    y_rows = []
    for i, t in enumerate(target_indices):
        seq = values[t - SEQUENCE_LENGTH : t].reshape(-1)
        row = [seq]
        if extra_vals is not None:
            row.append(extra_vals[t - 1])
        if cal_vals is not None:
            row.append(cal_vals[i])
        X_rows.append(np.concatenate(row))
        y_rows.append(1.0 if close[t] > close[t - 1] else 0.0)

    return np.vstack(X_rows), np.array(y_rows)
