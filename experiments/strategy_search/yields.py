"""Descriptive survey: what passive dollar yields actually paid, 2023-2026
(section 8 of docs/research/experiments/2026-11-11_event-driven-intraday-prereg.md).

Realised track records, not backtests:
  - Hyperliquid's HLP vault, from its account-value and cumulative-P&L history (Hyperliquid API)
  - stablecoin supply rates from DefiLlama's yield history (Aave v3 USDT / USDC on Ethereum, Ethena sUSDe)

Usage: python experiments/strategy_search/yields.py            (vault and lending survey)
       python experiments/strategy_search/yields.py hedged     (staking with a hedge, prereg 2026-11-23)
       python experiments/strategy_search/yields.py hedged-wide  (16 coins and two mixes, prereg 2026-11-24)
       python experiments/strategy_search/yields.py dollar       (dollar-yield pools and rotations, prereg 2026-11-24)
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


# Wider measurement (docs/research/experiments/2026-11-24_hedged-staking-wide-prereg.md). One DefiLlama
# liquid staking pool per coin: the largest by value locked on 2026-11-23 whose coin has a Binance USDT perp
# and at least 180 days of yield history inside 2025-01-01..2026-10-31. ETH and SOL keep the pools above.
WIDE = {
    "ETH": ("747c1d2a-c668-4682-b9f9-296708a3dd90", "Lido stETH"),
    "SOL": ("0e7d0722-9054-4907-8593-567b353c0900", "Jito JitoSOL"),
    "HYPE": ("9f25a954-db87-4bb2-a8b2-4be0b843a44c", "Kinetiq kHYPE"),
    "BNB": ("50bb5f69-85ea-4f70-81da-3661a1633fc4", "Lista slisBNB"),
    "AVAX": ("3790c3e5-8644-4f6b-8feb-12434d8b99f9", "Benqi sAVAX"),
    "LINK": ("4f5deb26-5f61-45a3-8f61-0edca3a86809", "stake.link stLINK"),
    "FIL": ("53048f5b-7b0a-4917-b283-c49bef77e580", "GLIF iFIL"),
    "TON": ("691a2aaa-cef5-4a6a-a3f3-86170fada8f4", "Hipo hTON"),
    "DOT": ("ff05ab26-971e-4e68-b1c6-c61a4c12c364", "Bifrost vDOT"),
    "BERA": ("2b9e97cd-15f2-406c-8e18-a426e451d09d", "Infrared iBERA"),
    "ICP": ("4450335c-bad3-4293-947b-64896b893a5b", "WaterNeuron nICP"),
    "STRK": ("1ce1d44a-0bff-48ef-80c9-f8b6bd78bd17", "Endur xSTRK"),
    "MON": ("ee40513c-9356-4c53-9f26-446b484a8ae2", "shMonad shMON"),
    "FLOW": ("f1246ee7-9797-46b6-bd87-f9b90bc37e36", "Ankr ankrFLOW"),
    "POL": ("5b1fe146-7cbd-448d-bf53-8df9c3501016", "Stader MaticX"),
    "APT": ("490006e3-284b-4107-9bb9-99906b7880b2", "Amnis amAPT"),
}
WIDE_WINDOW = ("2025-01-01", "2026-10-31")
APY_CAP = 0.25  # published yields above this are treated as reporting artefacts
APY_MEDIAN_DAYS = 7
LIQ_MOVE = 0.20  # a one-day rise this large against a short with 25% margin is treated as a liquidation
LIQ_PENALTY = 0.015
SWITCH_COST = 0.0025  # per dollar moved in or out of a hedged position: spot 0.10% + perp 0.05% + staking token swap 0.10%
MIX_TOP = 3
MIX_LOOKBACK = 30
MIX_MIN_HISTORY = 25  # days of the lookback a coin needs to be eligible
IDLE = 0.04
GATE_YIELD = 0.07
GATE_DD = 0.05


def _wide_carry(m, coin: str, pool: str) -> pd.DataFrame:
    """Daily pieces of one hedged dollar for a coin: staking, funding, basis change, liquidation penalty."""
    chart = requests.get(f"https://yields.llama.fi/chart/{pool}", timeout=60).json()["data"]
    apy = pd.Series({pd.Timestamp(r["timestamp"]).normalize(): r["apy"] / 100.0 for r in chart if r["apy"] is not None})
    apy.index = apy.index.tz_convert("UTC") if apy.index.tz is not None else apy.index.tz_localize("UTC")
    apy = apy[~apy.index.duplicated(keep="last")].sort_index()
    perp = coin + "USDT"
    close, spot = m.close[perp], m.spot_close[perp]
    staking = apy.reindex(close.index).ffill(limit=7).rolling(APY_MEDIAN_DAYS, min_periods=1).median().clip(0.0, APY_CAP) / 365.0
    basis = (spot.pct_change(fill_method=None) - close.pct_change(fill_method=None)) if spot.notna().any() else close * 0.0
    liq = (m.high[perp] / close.shift(1) - 1.0 >= LIQ_MOVE).astype(float) * LIQ_PENALTY
    df = pd.DataFrame({"staking": staking, "funding": m.funding[perp], "basis": basis.fillna(0.0), "liq": liq})
    df = df[close.notna()].dropna(subset=["staking", "funding"]).loc[:END]
    df["carry"] = df["staking"] + df["funding"] + df["basis"] - df["liq"]
    return df


def _year_table(r: pd.Series) -> dict:
    return {int(y): {"days": int(len(g)), "net": float(((1 + g).prod() - 1) * 365.0 / len(g))} for y, g in r.groupby(r.index.year)}


def _mix(carry: pd.DataFrame, rotate: bool) -> tuple[pd.Series, dict]:
    """Monthly mixes on total capital. `carry` holds each coin's daily return per hedged dollar (NaN before
    its series starts). Equal weight over every eligible coin, or the top MIX_TOP by trailing carry with
    idle dollars in any slot that would not beat them."""
    days = carry.loc[WIDE_WINDOW[0]:WIDE_WINDOW[1]].index
    month_starts = [d for d in days if d.day == 1]
    weights = pd.DataFrame(0.0, index=days, columns=carry.columns)
    prev = pd.Series(0.0, index=carry.columns)
    cost = pd.Series(0.0, index=days)
    held_log = {}
    for i, start in enumerate(month_starts):
        end = month_starts[i + 1] if i + 1 < len(month_starts) else days[-1] + pd.Timedelta(days=1)
        hist = carry.loc[start - pd.Timedelta(days=MIX_LOOKBACK): start - pd.Timedelta(days=1)]
        eligible = [c for c in carry.columns if hist[c].notna().sum() >= MIX_MIN_HISTORY and pd.notna(carry.at[start, c])]
        w = pd.Series(0.0, index=carry.columns)
        if rotate:
            trailing = (hist[eligible].mean() * 365.0 / (1 + MARGIN)).sort_values(ascending=False)
            picks = [c for c in trailing.index[:MIX_TOP] if trailing[c] > IDLE]
            w[picks] = 1.0 / MIX_TOP
        elif eligible:
            w[eligible] = 1.0 / len(eligible)
        weights.loc[start: end - pd.Timedelta(days=1)] = w.values
        cost[start] = float((w - prev).abs().sum()) * SWITCH_COST
        held_log[str(start.date())] = sorted(w[w > 0].index)
        prev = w
    c = carry.reindex(days)
    hedged = (weights * c.fillna(0.0)).sum(axis=1) / (1 + MARGIN)
    idle = (1.0 - weights.sum(axis=1)) * ((1 + IDLE) ** (1 / 365.0) - 1)
    return hedged + idle - cost, {"holdings_by_month": held_log, "switch_cost_total": float(cost.sum()),
                                 "average_share_hedged": float(weights.sum(axis=1).mean())}


def hedged_staking_wide() -> dict:
    import data as data_mod

    m = data_mod.build()
    pieces = {coin: _wide_carry(m, coin, pool) for coin, (pool, _) in WIDE.items()}
    out: dict = {"coins": {}, "mixes": {}}
    for coin, df in pieces.items():
        w = df.loc[WIDE_WINDOW[0]:WIDE_WINDOW[1]]
        net = (w["carry"] - HEDGE_COST_PER_YEAR / 365.0) / (1 + MARGIN)
        curve = (1 + net).cumprod()
        years = {}
        for y, g in w.groupby(w.index.year):
            ann = 365.0 / len(g)
            years[int(y)] = {"days": int(len(g)), "staking": float(g["staking"].sum() * ann), "funding": float(g["funding"].sum() * ann),
                             "basis_change": float(g["basis"].sum() * ann), "liquidation_days": int((g["liq"] > 0).sum()),
                             "net_on_total_capital": float(((1 + net.loc[g.index]).prod() - 1) * ann)}
        both = all(years.get(y, {}).get("days", 0) >= 180 and years[y]["net_on_total_capital"] >= GATE_YIELD for y in (2025, 2026))
        out["coins"][coin] = {"token": WIDE[coin][1], "first_day": str(w.index[0].date()), "days": int(len(w)), "years": years,
                              "max_drawdown": float(-(curve / curve.cummax() - 1).min()), "worst_day": float(net.min()),
                              "at_least_7pct_in_both_years": bool(both)}
    carry = pd.DataFrame({c: df["carry"] for c, df in pieces.items()})
    for name, rotate in (("M1 equal weight", False), ("M2 top three by trailing yield", True)):
        r, info = _mix(carry, rotate)
        curve = (1 + r).cumprod()
        years = _year_table(r)
        dd = float(-(curve / curve.cummax() - 1).min())
        gate = {"2025 at least 7%": years[2025]["net"] >= GATE_YIELD, "2026 at least 7%": years[2026]["net"] >= GATE_YIELD,
                "drawdown at most 5%": dd <= GATE_DD}
        out["mixes"][name] = {"years": years, "total_return": float(curve.iloc[-1] - 1), "max_drawdown": dd, "worst_day": float(r.min()),
                              "gate": gate, "h1_supported": all(gate.values()), **info}
    return out


# Dollar yields (docs/research/experiments/2026-11-24_dollar-yield-prereg.md). Universe: every pool DefiLlama
# flags as stablecoin with no impermanent-loss risk and at least 20 million dollars locked on 2026-11-23
# (results/dollar_yield_universe.json). Histories are cached under data/llama_stable/.
DOLLAR_MIN_TVL = 50e6  # point in time, the day before a month starts
DOLLAR_SWITCH_COST = 0.0010  # per dollar moved between pools
DOLLAR_TOP = 3
DOLLAR_GROUPS = {
    "lending": {"Lending", "CDP", "Uncollateralized Lending"},
    "exchange pools": {"Dexs"},
    "yield tokens": {"Basis Trading", "RWA", "Yield", "Onchain Capital Allocator", "Risk Curators"},
}


def _dollar_series() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    universe = json.loads((RESULTS / "dollar_yield_universe.json").read_text())
    cache = Path(__file__).resolve().parent / "data" / "llama_stable"
    cache.mkdir(parents=True, exist_ok=True)
    base, total, tvl = {}, {}, {}
    for p in universe:
        f = cache / f"{p['pool']}.json"
        if not f.exists():
            f.write_text(json.dumps(requests.get(f"https://yields.llama.fi/chart/{p['pool']}", timeout=60).json()))
        df = pd.DataFrame(json.loads(f.read_text())["data"])
        day = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
        df = df.assign(day=day).drop_duplicates("day", keep="last").set_index("day").sort_index().loc[:END]
        apy = df["apy"].astype(float)
        reward = df["apyReward"].astype(float).fillna(0.0)
        b = df["apyBase"].astype(float).where(df["apyBase"].notna(), apy - reward)
        base[p["pool"]], total[p["pool"]], tvl[p["pool"]] = b / 100.0, apy / 100.0, df["tvlUsd"].astype(float)
    idx = pd.date_range("2024-11-01", "2026-10-31", freq="D", tz="UTC")
    smooth = lambda d: pd.DataFrame(d).reindex(idx).ffill(limit=7).rolling(APY_MEDIAN_DAYS, min_periods=1).median().clip(0.0, APY_CAP)  # noqa: E731
    return smooth(base), smooth(total), pd.DataFrame(tvl).reindex(idx).ffill(limit=7), {p["pool"]: p for p in universe}


def _dollar_mix(y: pd.DataFrame, tvl: pd.DataFrame, pools: list[str], top: int | None) -> tuple[pd.Series, dict]:
    """Monthly: equal weight over eligible pools, or the `top` by trailing 30-day yield. Idle dollars when none."""
    days = y.loc[WIDE_WINDOW[0]:WIDE_WINDOW[1]].index
    starts = [d for d in days if d.day == 1]
    weights = pd.DataFrame(0.0, index=days, columns=y.columns)
    prev = pd.Series(0.0, index=y.columns)
    cost = pd.Series(0.0, index=days)
    log = {}
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else days[-1] + pd.Timedelta(days=1)
        hist = y.loc[start - pd.Timedelta(days=MIX_LOOKBACK): start - pd.Timedelta(days=1)]
        size = tvl.loc[start - pd.Timedelta(days=1)]
        eligible = [c for c in pools if hist[c].notna().sum() >= MIX_MIN_HISTORY and size.get(c, 0) >= DOLLAR_MIN_TVL]
        w = pd.Series(0.0, index=y.columns)
        if eligible:
            picks = list(hist[eligible].mean().sort_values(ascending=False).index[:top]) if top else eligible
            w[picks] = 1.0 / len(picks)
        weights.loc[start: end - pd.Timedelta(days=1)] = w.values
        cost[start] = float((w - prev).abs().sum()) * DOLLAR_SWITCH_COST
        log[str(start.date())] = sorted(w[w > 0].index)
        prev = w
    held = (weights * ((1.0 + y.reindex(days).fillna(0.0)) ** (1 / 365.0) - 1.0)).sum(axis=1)  # APY to a daily rate
    idle = (1.0 - weights.sum(axis=1)) * ((1 + IDLE) ** (1 / 365.0) - 1)
    return held + idle - cost, {"holdings_by_month": log, "switch_cost_total": float(cost.sum()),
                               "pools_eligible_first_and_last_month": [len(log[str(starts[0].date())]) if not top else None,
                                                                       len(log[str(starts[-1].date())]) if not top else None]}


def dollar_yields() -> dict:
    base, total, tvl, meta = _dollar_series()
    groups = {"all": list(base.columns)}
    for name, cats in DOLLAR_GROUPS.items():
        groups[name] = [c for c in base.columns if meta[c]["category"] in cats]
    candidates = {
        "D1 all large pools, equal weight": ("all", None),
        "D2 lending, top three by trailing yield": ("lending", DOLLAR_TOP),
        "D3 any pool, top three by trailing yield": ("all", DOLLAR_TOP),
        "D4 exchange pools, equal weight": ("exchange pools", None),
        "D5 yield tokens, equal weight": ("yield tokens", None),
    }
    label = lambda c: f"{meta[c]['project']} {meta[c]['symbol']} ({meta[c]['chain']})"  # noqa: E731
    out: dict = {"candidates": {}, "pools_in_universe": len(meta), "pools_by_group": {k: len(v) for k, v in groups.items()}}
    for name, (group, top) in candidates.items():
        r, info = _dollar_mix(base, tvl, groups[group], top)
        r_total, _ = _dollar_mix(total, tvl, groups[group], top)
        years = _year_table(r)
        gate = {"2025 at least 7%": years[2025]["net"] >= GATE_YIELD, "2026 at least 7%": years[2026]["net"] >= GATE_YIELD}
        info["holdings_by_month"] = {k: [label(c) for c in v] for k, v in info["holdings_by_month"].items()}
        out["candidates"][name] = {"years": years, "years_including_reward_tokens": _year_table(r_total),
                                   "total_return": float((1 + r).prod() - 1), "worst_day": float(r.min()),
                                   "gate": gate, "h1_supported": all(gate.values()), **info}
    window = base.loc[WIDE_WINDOW[0]:WIDE_WINDOW[1]]
    big = tvl.loc[WIDE_WINDOW[0]:WIDE_WINDOW[1]] >= DOLLAR_MIN_TVL
    per_pool = []
    for c in base.columns:
        w = window[c].where(big[c]).dropna()
        if len(w) >= 300:
            per_pool.append({"pool": label(c), "category": meta[c]["category"], "days_large": int(len(w)),
                             "mean_base_yield_2025": float(w.loc[:"2025-12-31"].mean()) if len(w.loc[:"2025-12-31"]) >= 150 else None,
                             "mean_base_yield_2026": float(w.loc["2026-01-01":].mean()) if len(w.loc["2026-01-01":]) >= 150 else None})
    out["pools_large_for_300_days"] = sorted(per_pool, key=lambda x: -(x["mean_base_yield_2026"] or 0))
    return out


if __name__ == "__main__":
    import sys

    RESULTS.mkdir(exist_ok=True)
    if sys.argv[1:] == ["hedged"]:
        report = hedged_staking()
        (RESULTS / "hedged_staking.json").write_text(json.dumps(report, indent=1, default=float, sort_keys=True))
    elif sys.argv[1:] == ["dollar"]:
        report = dollar_yields()
        (RESULTS / "dollar_yields.json").write_text(json.dumps(report, indent=1, default=float, sort_keys=True))
    elif sys.argv[1:] == ["hedged-wide"]:
        report = hedged_staking_wide()
        (RESULTS / "hedged_staking_wide.json").write_text(json.dumps(report, indent=1, default=float, sort_keys=True))
    else:
        report = {"hlp": hlp(), "stablecoin_lending": llama()}
        (RESULTS / "yields.json").write_text(json.dumps(report, indent=1, default=float, sort_keys=True))
    print(json.dumps(report, indent=1, default=float))
