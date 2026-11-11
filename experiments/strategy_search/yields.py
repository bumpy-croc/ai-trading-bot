"""Descriptive survey: what passive dollar yields actually paid, 2023-2026
(section 8 of docs/research/experiments/2026-11-11_event-driven-intraday-prereg.md).

Realised track records, not backtests:
  - Hyperliquid's HLP vault, from its account-value and cumulative-P&L history (Hyperliquid API)
  - stablecoin supply rates from DefiLlama's yield history (Aave v3 USDT / USDC on Ethereum, Ethena sUSDe)

Usage: python experiments/strategy_search/yields.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RESULTS = Path(__file__).resolve().parent / "results"
HLP = "0xdfc24b077bc1425ad1dea75bcb6f8158e10df303"
END = pd.Timestamp("2026-10-31", tz="UTC")
LLAMA_TARGETS = {
    "Aave v3 USDT (Ethereum)": {"project": "aave-v3", "chain": "Ethereum", "symbol": "USDT"},
    "Aave v3 USDC (Ethereum)": {"project": "aave-v3", "chain": "Ethereum", "symbol": "USDC"},
    "Ethena sUSDe": {"project": "ethena-usde", "chain": "Ethereum", "symbol": "SUSDE"},
}


def hlp() -> dict:
    d = requests.post("https://api.hyperliquid.xyz/info", json={"type": "vaultDetails", "vaultAddress": HLP}, timeout=60).json()
    port = dict(d["portfolio"])["allTime"]
    av = pd.Series({int(t): float(v) for t, v in port["accountValueHistory"]})
    pnl = pd.Series({int(t): float(v) for t, v in port["pnlHistory"]})
    df = pd.DataFrame({"value": av, "pnl": pnl}).sort_index()
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    df = df.loc[:END + pd.Timedelta(days=1)]
    # Return over each interval = P&L earned in it / average capital in it (modified Dietz with flows at the
    # midpoint). Deposits and withdrawals move the vault's value but not its P&L, so dividing by the starting
    # value alone would overstate returns while the vault was growing.
    dpnl = df["pnl"].diff()
    start = df["value"].shift(1)
    flows = df["value"] - start - dpnl
    ret = (dpnl / (start + flows / 2.0)).replace([np.inf, -np.inf], np.nan).dropna()
    ret = ret[start.reindex(ret.index) > 1e6]  # skip the first weeks, when the vault was tiny
    naive = (dpnl / start).reindex(ret.index)
    index = (1.0 + ret).cumprod()
    years = (ret.index[-1] - ret.index[0]).days / 365.0
    step_days = float(np.median(np.diff(ret.index.values).astype("timedelta64[h]").astype(float)) / 24.0)
    return {
        "name": d.get("name"),
        "first_sample": str(ret.index[0].date()),
        "last_sample": str(ret.index[-1].date()),
        "samples": int(len(ret)),
        "median_days_between_samples": step_days,
        "cagr": float(index.iloc[-1] ** (1.0 / years) - 1.0),
        "cagr_if_divided_by_starting_value_only": float((1.0 + naive).prod() ** (1.0 / years) - 1.0),
        "by_year": {int(y): float((1.0 + r).prod() - 1.0) for y, r in ret.groupby(ret.index.year)},
        "max_drawdown_at_sample_resolution": float(-(index / index.cummax() - 1.0).min()),
        "worst_interval": float(ret.min()),
        "negative_intervals_share": float((ret < 0).mean()),
        "vault_value_latest_usd": float(df["value"].iloc[-1]),
        "current_apr_reported_by_venue": d.get("apr"),
    }


def llama() -> dict:
    pools = requests.get("https://yields.llama.fi/pools", timeout=120).json()["data"]
    out = {}
    for label, want in LLAMA_TARGETS.items():
        match = [p for p in pools if p["project"] == want["project"] and p["chain"] == want["chain"] and p["symbol"].upper() == want["symbol"]]
        if not match:
            out[label] = {"error": "pool not found"}
            continue
        pool = max(match, key=lambda p: p.get("tvlUsd") or 0)
        rows = requests.get(f"https://yields.llama.fi/chart/{pool['pool']}", timeout=120).json()["data"]
        s = pd.Series({pd.Timestamp(r["timestamp"]).tz_convert("UTC").normalize(): r["apy"] for r in rows if r.get("apy") is not None}) / 100.0
        s = s[~s.index.duplicated()].sort_index().loc[: END]
        out[label] = {
            "pool": pool["pool"],
            "tvl_usd": pool.get("tvlUsd"),
            "first": str(s.index[0].date()),
            "mean_apy_by_year": {int(y): float(v.mean()) for y, v in s.groupby(s.index.year)},
            "mean_apy_2025_01_to_2026_10": float(s.loc["2025-01-01":].mean()),
            "mean_apy_2023_07_to_2024_12": float(s.loc["2023-07-01":"2024-12-31"].mean()) if len(s.loc["2023-07-01":"2024-12-31"]) else None,
            "latest_apy": float(s.iloc[-1]),
        }
    return out


if __name__ == "__main__":
    report = {"hlp": hlp(), "stablecoin_lending": llama()}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "yields.json").write_text(json.dumps(report, indent=1, default=float, sort_keys=True))
    print(json.dumps(report, indent=1, default=float))
