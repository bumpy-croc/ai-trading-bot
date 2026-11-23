"""Descriptive survey: what passive dollar yields actually paid, 2023-2026
(section 8 of docs/research/experiments/2026-11-11_event-driven-intraday-prereg.md).

Realised track records, not backtests:
  - Hyperliquid's HLP vault, from its account-value and cumulative-P&L history (Hyperliquid API)
  - stablecoin supply rates from DefiLlama's yield history (Aave v3 USDT / USDC on Ethereum, Ethena sUSDe)

Usage: python experiments/strategy_search/yields.py            (vault and lending survey)
       python experiments/strategy_search/yields.py hedged     (staking with a hedge, prereg 2026-11-23)
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


HEDGED = {
    "ETH": {"pool": "747c1d2a-c668-4682-b9f9-296708a3dd90", "perp": "ETHUSDT", "token": "Lido stETH"},
    "SOL": {"pool": "0e7d0722-9054-4907-8593-567b353c0900", "perp": "SOLUSDT", "token": "Jito JitoSOL"},
}
HEDGE_COST_PER_YEAR = 2 * (0.0010 + 0.0005)  # one round trip a year on both legs at taker fees
MARGIN = 0.25  # idle dollars per hedged dollar in the "separate margin" version


def hedged_staking() -> dict:
    """Staking token long, same coin's Binance perp short: realised yield by year.
    Definitions: docs/research/experiments/2026-11-23_hedged-staking-yield-prereg.md."""
    import data as data_mod

    m = data_mod.build()
    out: dict = {}
    for coin, spec in HEDGED.items():
        chart = requests.get(f"https://yields.llama.fi/chart/{spec['pool']}", timeout=60).json()["data"]
        apy = pd.Series({pd.Timestamp(r["timestamp"]).normalize(): r["apy"] / 100.0 for r in chart if r["apy"] is not None})
        apy.index = apy.index.tz_convert("UTC") if apy.index.tz is not None else apy.index.tz_localize("UTC")
        apy = apy[~apy.index.duplicated(keep="last")].sort_index()
        perp, spot, funding = m.close[spec["perp"]], m.spot_close[spec["perp"]], m.funding[spec["perp"]]
        df = pd.DataFrame({
            "staking": apy.reindex(perp.index).ffill(limit=7) / 365.0,
            "funding": funding,
            "basis": spot.pct_change(fill_method=None) - perp.pct_change(fill_method=None),
        }).loc[:END].dropna()
        first_year = df.index[0].year if df.index[0].dayofyear == 1 else df.index[0].year + 1
        df = df.loc[f"{first_year}-01-01":]
        df["net"] = df["staking"] + df["funding"] + df["basis"] - HEDGE_COST_PER_YEAR / 365.0
        years = {}
        for y, g in df.groupby(df.index.year):
            ann = 365.0 / len(g)
            years[int(y)] = {
                "days": int(len(g)),
                "staking": float(g["staking"].sum() * ann), "funding": float(g["funding"].sum() * ann),
                "basis_change": float(g["basis"].sum() * ann), "cost": HEDGE_COST_PER_YEAR,
                "net_token_as_collateral": float(((1 + g["net"]).prod() - 1) * ann),
                "net_separate_margin": float(((1 + g["net"] / (1 + MARGIN)).prod() - 1) * ann),
                "share_of_days_funding_negative": float((g["funding"] < 0).mean()),
            }
        recent = df.loc["2025-01-01":]
        curve = (1 + recent["net"] / (1 + MARGIN)).cumprod()
        gate = {
            "2025 at least 7%": years.get(2025, {}).get("net_separate_margin", -1) >= 0.07,
            "2026 at least 7%": years.get(2026, {}).get("net_separate_margin", -1) >= 0.07,
            "drawdown 2025-26 at most 5%": float(-(curve / curve.cummax() - 1).min()) <= 0.05,
        }
        out[coin] = {
            "token": spec["token"], "perp": spec["perp"], "first_day": str(df.index[0].date()), "last_day": str(df.index[-1].date()),
            "latest_staking_apy": float(apy.iloc[-1]), "years": years,
            "max_drawdown_2025_26_separate_margin": float(-(curve / curve.cummax() - 1).min()),
            "worst_day_2025_26": float((recent["net"] / (1 + MARGIN)).min()),
            "gate": gate, "h1_supported": all(gate.values()),
        }
    return out


if __name__ == "__main__":
    import sys

    RESULTS.mkdir(exist_ok=True)
    if sys.argv[1:] == ["hedged"]:
        report = hedged_staking()
        (RESULTS / "hedged_staking.json").write_text(json.dumps(report, indent=1, default=float, sort_keys=True))
    else:
        report = {"hlp": hlp(), "stablecoin_lending": llama()}
        (RESULTS / "yields.json").write_text(json.dumps(report, indent=1, default=float, sort_keys=True))
    print(json.dumps(report, indent=1, default=float))
