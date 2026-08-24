"""Model-free predictability/inefficiency scan across symbols x timeframes.

Preregistration: docs/research/experiments/2026-08-13_model-free-predictability-scan.md

No model training. Pulls cached OHLCV via CachedDataProvider, computes a battery of
model-free statistics (variance ratio, autocorrelation/Ljung-Box, ARCH/vol-clustering,
Hurst exponent, regime persistence, naive directional-accuracy baselines) against an
i.i.d.-shuffle bootstrap null (or analytic null cross-validated against bootstrap),
applies BH-FDR correction across the full grid, and translates surviving statistics into
an approximate bps edge vs measured round-trip transaction cost.

Run: python docs/research/experiments/scripts/2026-08-13_model_free_scan.py
Output: docs/research/experiments/scripts/output/2026-08-13_results.json
        docs/research/experiments/scripts/output/2026-08-13_results.csv
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.data_providers.cached_data_provider import CachedDataProvider  # noqa: E402
from src.data_providers.binance_provider import BinanceProvider  # noqa: E402

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "DOGEUSDT"]
TIMEFRAMES = ["15m", "1h", "4h", "1d"]
YEARS = 2
N_SHUFFLE_FULL = 500       # Hurst, regime persistence, DA baselines
N_SHUFFLE_CROSSCHECK = 200  # VR(2) and lag-1 autocorr analytic cross-validation
SEED = 42
ROUND_TRIP_COST_BPS = 11.0  # 2 x 5.5bps measured (docs cite 5.1-5.8bps/side)
FDR_Q = 0.05

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

TIMEFRAME_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


def load_returns(provider: CachedDataProvider, symbol: str, timeframe: str) -> tuple[pd.DataFrame | None, str | None]:
    """Load OHLCV and validate bar spacing matches the declared timeframe.

    Returns (df, reject_reason). df is None if data is missing or fails the
    spacing-integrity check (a provider fallback silently returned a different
    resolution than requested -- seen live for CoinGecko fallback on 15m bars,
    which returns daily bars under the same column schema). Never silently
    compute statistics on a mixed-frequency series.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * YEARS + 5)
    df = provider.get_historical_data(symbol, timeframe, start=start, end=end)
    if df is None or len(df) < 200:
        return None, "insufficient_data"
    df = df.sort_index()

    expected_seconds = TIMEFRAME_SECONDS[timeframe]
    diffs = df.index.to_series().diff().dropna().dt.total_seconds()
    if len(diffs) == 0:
        return None, "no_spacing_data"
    median_spacing = diffs.median()
    # allow slack for the odd missing bar, but a >2x deviation means the
    # series is a different resolution than declared (fallback-provider drift)
    if median_spacing > expected_seconds * 2 or median_spacing < expected_seconds * 0.5:
        return None, (
            f"spacing_mismatch: median={median_spacing:.0f}s expected={expected_seconds}s "
            f"(likely provider-fallback resolution drift)"
        )
    # flag (but don't reject) a meaningful minority of irregular gaps
    irregular_frac = float(np.mean(np.abs(diffs - expected_seconds) > expected_seconds * 0.5))

    df["log_ret"] = np.log(df["close"]).diff()
    df = df.dropna(subset=["log_ret"])
    df.attrs["irregular_gap_frac"] = irregular_frac
    df.attrs["median_spacing_s"] = float(median_spacing)
    return df, None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def variance_ratio(returns: np.ndarray, q: int) -> tuple[float, float]:
    """Lo-MacKinlay heteroskedasticity-robust variance ratio test.

    Returns (VR(q), z-stat). VR=1 under random walk; z-stat is asymptotically N(0,1).
    """
    n = len(returns)
    mu = returns.mean()
    var_1 = np.sum((returns - mu) ** 2) / (n - 1)
    if var_1 == 0 or n <= q:
        return np.nan, np.nan
    # q-period overlapping differences
    cum = np.concatenate([[0.0], np.cumsum(returns)])
    idx = np.arange(0, n - q + 1)
    q_sums = cum[idx + q] - cum[idx]
    m = n - q + 1
    var_q = np.sum((q_sums - q * mu) ** 2) / (q * m)
    vr = var_q / var_1 if var_1 else np.nan

    # Heteroskedasticity-robust asymptotic variance (Lo & MacKinlay 1988), vectorized
    dev_sq = (returns - mu) ** 2
    denom = (np.sum(dev_sq)) ** 2
    delta_sum = 0.0
    for j in range(1, q):
        weight = (2 * (q - j) / q) ** 2
        num = np.sum(dev_sq[j:] * dev_sq[:-j])
        delta_j = (n * num) / denom if denom else 0.0
        delta_sum += weight * delta_j
    theta = delta_sum
    if theta <= 0:
        return vr, np.nan
    z = (vr - 1) / np.sqrt(theta)
    return float(vr), float(z)


def ljung_box(series: np.ndarray, lags: list[int]) -> dict:
    """Ljung-Box Q-test, implemented directly (no external dependency)."""
    n = len(series)
    x = series - series.mean()
    var = np.sum(x ** 2)
    acf = {}
    for lag in lags:
        num = np.sum(x[lag:] * x[:-lag])
        acf[lag] = num / var if var else np.nan
    max_lag = max(lags)
    q_stat = n * (n + 2) * sum(
        (acf[lag] ** 2) / (n - lag) for lag in lags if not np.isnan(acf[lag])
    )
    # chi-squared survival function via regularized upper incomplete gamma,
    # approximated with a series-free method (Wilson-Hilferty) to avoid a scipy dependency
    k = len(lags)
    p_value = _chi2_sf(q_stat, k)
    return {"acf": acf, "Q": float(q_stat), "p": float(p_value), "dof": k}


def _chi2_sf(x: float, k: int) -> float:
    """Survival function of chi-squared(k) via Wilson-Hilferty approximation."""
    if x <= 0 or k <= 0:
        return 1.0
    # Wilson-Hilferty: (x/k)^(1/3) approx normal with mean 1-2/(9k), var 2/(9k)
    mean = 1 - 2.0 / (9 * k)
    sd = np.sqrt(2.0 / (9 * k))
    z = ((x / k) ** (1.0 / 3.0) - mean) / sd
    return float(_norm_sf(z))


def _norm_sf(z: float) -> float:
    """Standard normal survival function via erfc (no scipy dependency)."""
    import math
    return 0.5 * math.erfc(z / math.sqrt(2))


def hurst_rs(returns: np.ndarray, min_chunk: int = 16) -> float:
    """Hurst exponent via rescaled-range (R/S) analysis, log-log slope."""
    n = len(returns)
    chunk_sizes = np.unique(
        np.floor(np.logspace(np.log10(min_chunk), np.log10(n // 2), num=12)).astype(int)
    )
    chunk_sizes = chunk_sizes[chunk_sizes >= min_chunk]
    logs_n, logs_rs = [], []
    for size in chunk_sizes:
        n_chunks = n // size
        if n_chunks < 1:
            continue
        rs_vals = []
        for i in range(n_chunks):
            chunk = returns[i * size:(i + 1) * size]
            mean = chunk.mean()
            dev = np.cumsum(chunk - mean)
            r = dev.max() - dev.min()
            s = chunk.std(ddof=0)
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            logs_n.append(np.log(size))
            logs_rs.append(np.log(np.mean(rs_vals)))
    if len(logs_n) < 3:
        return np.nan
    slope, _ = np.polyfit(logs_n, logs_rs, 1)
    return float(slope)


def regime_persistence(returns: np.ndarray, window: int = 20) -> float:
    """P(stay in same realized-vol regime | in that regime), trailing rolling vol only."""
    s = pd.Series(returns)
    roll_vol = s.rolling(window).std()
    roll_vol = roll_vol.dropna()
    if len(roll_vol) < window * 2:
        return np.nan
    median = roll_vol.median()
    regime = (roll_vol > median).astype(int).values
    stay = np.sum(regime[1:] == regime[:-1])
    return float(stay / (len(regime) - 1)) if len(regime) > 1 else np.nan


def directional_baselines(returns: np.ndarray) -> dict:
    """Momentum and mean-reversion naive baselines: predict sign from previous bar's sign."""
    sign = np.sign(returns)
    sign = sign[sign != 0]  # drop exact-zero bars
    if len(sign) < 30:
        return {"n": len(sign), "momentum_acc": np.nan, "reversion_acc": np.nan}
    prev = sign[:-1]
    cur = sign[1:]
    momentum_correct = np.sum(prev == cur)
    n = len(cur)
    momentum_acc = momentum_correct / n
    reversion_acc = 1 - momentum_acc
    return {"n": int(n), "momentum_acc": float(momentum_acc), "reversion_acc": float(reversion_acc)}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half = (z * np.sqrt((phat * (1 - phat) + z ** 2 / (4 * n)) / n)) / denom
    return (center - half, center + half)


def binom_p_two_sided(k: int, n: int, p0: float = 0.5) -> float:
    """Normal-approximation two-sided p-value for a binomial proportion test vs p0."""
    if n == 0:
        return np.nan
    phat = k / n
    se = np.sqrt(p0 * (1 - p0) / n)
    if se == 0:
        return np.nan
    z = (phat - p0) / se
    return 2 * _norm_sf(abs(z))


# ---------------------------------------------------------------------------
# Bootstrap null helpers
# ---------------------------------------------------------------------------

def bootstrap_p_value(observed: float, statistic_fn, returns: np.ndarray, n_iter: int, rng) -> float:
    if np.isnan(observed):
        return np.nan
    surrogate_vals = np.empty(n_iter)
    for i in range(n_iter):
        shuffled = rng.permutation(returns)
        surrogate_vals[i] = statistic_fn(shuffled)
    count = np.sum(np.abs(surrogate_vals) >= np.abs(observed))
    return float((1 + count) / (n_iter + 1))


# ---------------------------------------------------------------------------
# Multiple comparisons: BH-FDR
# ---------------------------------------------------------------------------

def bh_fdr(pvalues: list[float], q: float = FDR_Q) -> list[bool]:
    m = len(pvalues)
    idx = np.argsort(pvalues)
    sorted_p = np.array(pvalues)[idx]
    thresh = (np.arange(1, m + 1) / m) * q
    below = sorted_p <= thresh
    if not np.any(below):
        return [False] * m
    max_i = np.max(np.where(below)[0])
    cutoff = sorted_p[max_i]
    return [p <= cutoff for p in pvalues]


# ---------------------------------------------------------------------------
# Per-cell computation
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    symbol: str
    timeframe: str
    n_bars: int
    start: str
    end: str
    stats: dict = field(default_factory=dict)


def compute_cell(symbol: str, timeframe: str, df: pd.DataFrame, rng) -> CellResult:
    returns = df["log_ret"].values.astype(float)
    n = len(returns)
    stats: dict = {}

    # 1. Variance ratio, analytic + bootstrap cross-check on q=2
    vr_results = {}
    for q in (2, 4, 8, 16):
        vr, z = variance_ratio(returns, q)
        p_analytic = 2 * _norm_sf(abs(z)) if not np.isnan(z) else np.nan
        vr_results[q] = {"VR": vr, "z": z, "p_analytic": p_analytic}
    vr2_boot_p = bootstrap_p_value(
        vr_results[2]["VR"] - 1.0,
        lambda r: variance_ratio(r, 2)[0] - 1.0,
        returns, N_SHUFFLE_CROSSCHECK, rng,
    )
    stats["variance_ratio"] = vr_results
    stats["vr2_bootstrap_p"] = vr2_boot_p

    # 2. Autocorrelation / Ljung-Box on raw returns
    lb_ret = ljung_box(returns, [1, 5, 10])
    lag1_boot_p = bootstrap_p_value(
        lb_ret["acf"][1],
        lambda r: (lambda x: np.sum((x[1:] - x.mean()) * (x[:-1] - x.mean())) / np.sum((x - x.mean()) ** 2))(r),
        returns, N_SHUFFLE_CROSSCHECK, rng,
    )
    stats["autocorrelation"] = lb_ret
    stats["lag1_bootstrap_p"] = lag1_boot_p

    # 3. ARCH / vol clustering on squared returns
    sq = returns ** 2
    lb_sq = ljung_box(sq, [1, 5, 10])
    stats["arch_vol_clustering"] = lb_sq

    # 4. Hurst exponent (bootstrap null only)
    hurst_obs = hurst_rs(returns)
    hurst_p = bootstrap_p_value(hurst_obs - 0.5, lambda r: hurst_rs(r) - 0.5, returns, N_SHUFFLE_FULL, rng)
    stats["hurst"] = {"H": hurst_obs, "p_bootstrap": hurst_p}

    # 5. Regime persistence (bootstrap null only)
    persist_obs = regime_persistence(returns)
    persist_p = bootstrap_p_value(
        persist_obs - 0.5, lambda r: regime_persistence(r) - 0.5, returns, N_SHUFFLE_FULL, rng
    )
    stats["regime_persistence"] = {"P_stay": persist_obs, "p_bootstrap": persist_p}

    # 6. Naive directional-accuracy baselines
    da = directional_baselines(returns)
    if da["n"] > 0:
        k_mom = int(round(da["momentum_acc"] * da["n"]))
        ci_lo, ci_hi = wilson_ci(k_mom, da["n"])
        p_binom = binom_p_two_sided(k_mom, da["n"], 0.5)
        # bootstrap cross-check on momentum accuracy
        da_boot_p = bootstrap_p_value(
            da["momentum_acc"] - 0.5,
            lambda r: directional_baselines(r)["momentum_acc"] - 0.5,
            returns, N_SHUFFLE_FULL, rng,
        )
        da["ci95"] = [ci_lo, ci_hi]
        da["p_binom_analytic"] = p_binom
        da["p_bootstrap"] = da_boot_p
    stats["directional_baseline"] = da

    # 7. Seasonality (secondary, cheap, ANOVA F-test across hour-of-day buckets)
    try:
        hours = df.index.hour
        groups = [returns[hours == h] for h in range(24) if np.sum(hours == h) > 5]
        if len(groups) >= 2:
            grand_mean = returns.mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
            ss_within = sum(np.sum((g - g.mean()) ** 2) for g in groups)
            df_between = len(groups) - 1
            df_within = n - len(groups)
            if ss_within > 0 and df_within > 0:
                f_stat = (ss_between / df_between) / (ss_within / df_within)
            else:
                f_stat = np.nan
        else:
            f_stat = np.nan
        stats["seasonality_hour_f_stat"] = float(f_stat) if not np.isnan(f_stat) else None
    except Exception:
        stats["seasonality_hour_f_stat"] = None

    return CellResult(
        symbol=symbol,
        timeframe=timeframe,
        n_bars=n,
        start=str(df.index.min()),
        end=str(df.index.max()),
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Cost-reality translation
# ---------------------------------------------------------------------------

def translate_da_edge_bps(accuracy: float, avg_abs_return_bps: float) -> float:
    return (2 * accuracy - 1) * avg_abs_return_bps - ROUND_TRIP_COST_BPS


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(SEED)
    api_provider = BinanceProvider()
    provider = CachedDataProvider(api_provider, cache_dir=str(REPO_ROOT / "cache" / "market_data"))

    results: list[CellResult] = []
    skipped: list[dict] = []

    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            print(f"[scan] {symbol} {timeframe} ...", flush=True)
            df, reject_reason = load_returns(provider, symbol, timeframe)
            if df is None:
                print(f"[scan]   SKIPPED - {reject_reason}", flush=True)
                skipped.append({"symbol": symbol, "timeframe": timeframe, "reason": reject_reason})
                continue
            avg_abs_ret_bps = float(np.mean(np.abs(df["log_ret"])) * 10000)
            cell = compute_cell(symbol, timeframe, df, rng)
            cell.stats["avg_abs_return_bps"] = avg_abs_ret_bps
            cell.stats["irregular_gap_frac"] = df.attrs.get("irregular_gap_frac")
            results.append(cell)
            print(f"[scan]   n={cell.n_bars} avg|ret|={avg_abs_ret_bps:.2f}bps "
                  f"irregular_gaps={df.attrs.get('irregular_gap_frac', 0):.3f} done", flush=True)

    # ---- Assemble the correction grid: 1 representative p-value per (cell, statistic) ----
    grid_rows = []
    for cell in results:
        s = cell.stats
        # representative p per statistic (lowest analytic VR order as primary; joint LB p for others)
        vr_p = min(v["p_analytic"] for v in s["variance_ratio"].values() if not np.isnan(v["p_analytic"]))
        entries = {
            "variance_ratio": vr_p,
            "autocorrelation": s["autocorrelation"]["p"],
            "arch_vol_clustering": s["arch_vol_clustering"]["p"],
            "hurst": s["hurst"]["p_bootstrap"],
            "regime_persistence": s["regime_persistence"]["p_bootstrap"],
            "directional_baseline": s["directional_baseline"].get("p_bootstrap", np.nan),
        }
        for stat_name, p in entries.items():
            grid_rows.append({
                "symbol": cell.symbol, "timeframe": cell.timeframe,
                "statistic": stat_name, "p_value": p,
            })

    valid_rows = [r for r in grid_rows if not (r["p_value"] is None or np.isnan(r["p_value"]))]
    pvals = [r["p_value"] for r in valid_rows]
    sig_flags = bh_fdr(pvals, FDR_Q)
    for r, sig in zip(valid_rows, sig_flags):
        r["bh_fdr_significant"] = bool(sig)
    invalid_rows = [r for r in grid_rows if r not in valid_rows]
    for r in invalid_rows:
        r["bh_fdr_significant"] = None

    grid_df = pd.DataFrame(valid_rows + invalid_rows)

    # ---- Cost-reality translation for directional_baseline + note VR/Hurst as order-of-magnitude ----
    econ_rows = []
    for cell in results:
        da = cell.stats["directional_baseline"]
        avg_abs = cell.stats["avg_abs_return_bps"]
        if da.get("n", 0) > 0 and not np.isnan(da.get("momentum_acc", np.nan)):
            ev_bps = translate_da_edge_bps(da["momentum_acc"], avg_abs)
            econ_rows.append({
                "symbol": cell.symbol, "timeframe": cell.timeframe,
                "momentum_acc": da["momentum_acc"], "ci95": da.get("ci95"),
                "avg_abs_return_bps": avg_abs, "ev_bps_per_trade": ev_bps,
                "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
                "clears_cost": bool(ev_bps > 0),
            })
    econ_df = pd.DataFrame(econ_rows)

    # ---- Write outputs ----
    out_json = {
        "config": {
            "symbols": SYMBOLS, "timeframes": TIMEFRAMES, "years": YEARS,
            "n_shuffle_full": N_SHUFFLE_FULL, "n_shuffle_crosscheck": N_SHUFFLE_CROSSCHECK,
            "seed": SEED, "round_trip_cost_bps": ROUND_TRIP_COST_BPS, "fdr_q": FDR_Q,
        },
        "skipped": skipped,
        "cells": [
            {"symbol": c.symbol, "timeframe": c.timeframe, "n_bars": c.n_bars,
             "start": c.start, "end": c.end, "stats": c.stats}
            for c in results
        ],
    }
    (OUT_DIR / "2026-08-13_results.json").write_text(json.dumps(out_json, indent=2, default=str))
    grid_df.to_csv(OUT_DIR / "2026-08-13_fdr_grid.csv", index=False)
    econ_df.to_csv(OUT_DIR / "2026-08-13_econ_translation.csv", index=False)

    n_sig = grid_df["bh_fdr_significant"].sum()
    print(f"\n[scan] DONE. {len(results)} cells computed, {len(skipped)} skipped.")
    print(f"[scan] {n_sig}/{len(valid_rows)} tests BH-FDR significant at q={FDR_Q}.")
    print(f"[scan] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
