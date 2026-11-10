"""Cross-venue funding arbitrage, Binance vs Hyperliquid
(docs/research/experiments/2026-11-10_cross-venue-funding-arb-prereg.md).

  python experiments/strategy_search/xvenue.py download   # candles for all HL perps; funding to 2024-12-31 only
  python experiments/strategy_search/xvenue.py sanity     # data truncated at 2024-12-31
  python experiments/strategy_search/xvenue.py holdout    # refuses unless sanity passed; only then fetches 2025-26 funding
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as data_mod  # noqa: E402
import engine as eng  # noqa: E402

HL_URL = "https://api.hyperliquid.xyz/info"
HL_DIR = data_mod.DATA_DIR / "hyperliquid"
RESULTS = Path(__file__).resolve().parent / "results"
HL_START = pd.Timestamp("2023-05-01", tz="UTC")
END = data_mod.END
SANITY = ("2023-07-01", "2024-12-31")
HOLDOUT = ("2025-01-01", "2026-10-31")
CUTOFF = "2024-12-31"

HL_FEE, HL_SLIP = 0.00045, 0.0010
MIN_CLOSES = 30
HL_TOP_N = 30
BINANCE_MAX_RANK = 100
SLOTS = 5
MARGIN_MOVE = 0.30
MARGIN_COST = 0.0010
FORCED_PENALTY = 0.02
MISSING_FUNDING_PAY = 0.0003

_session = requests.Session()


def _post(body: dict) -> list | dict | None:
    for attempt in range(8):
        try:
            resp = _session.post(HL_URL, json=body, timeout=30)
        except requests.RequestException:
            time.sleep(2 + attempt)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            time.sleep(5 + 5 * attempt)
            continue
        time.sleep(1 + attempt)
    raise RuntimeError(f"Hyperliquid request failed: {body}")


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def binance_symbol(coin: str) -> str:
    return f"1000{coin[1:]}USDT" if coin.startswith("k") else f"{coin}USDT"


# -------------------------------------------------------------------- download


def fetch_candles(coin: str) -> pd.DataFrame:
    path = HL_DIR / f"candles_{coin}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    rows = _post({"type": "candleSnapshot", "req": {"coin": coin, "interval": "1d", "startTime": _ms(HL_START),
                                                    "endTime": _ms(END + pd.Timedelta(days=1)) - 1}}) or []
    df = pd.DataFrame(
        [(int(r["t"]), float(r["h"]), float(r["l"]), float(r["c"]), float(r["v"]) * float(r["c"]), int(r["n"])) for r in rows],
        columns=["t", "high", "low", "close", "dollar_volume", "trades"],
    )
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.normalize()
    df = df[df["trades"] > 0].drop_duplicates("date").set_index("date").sort_index()
    df = df[["high", "low", "close", "dollar_volume"]]
    df.to_pickle(path)
    return df


def fetch_funding(coin: str, start: pd.Timestamp, end: pd.Timestamp, segment: str) -> pd.Series:
    """Daily funding sums for one coin over [start, end]; cached per segment so the holdout
    segment is only ever downloaded after the sanity check has passed."""
    path = HL_DIR / f"funding_{coin}_{segment}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    prints: dict[int, float] = {}
    # a print stamped 00:00 on day d+1 settles at day d's close, so each segment runs to 00:30 after its last day
    cursor, stop = _ms(start) + 30 * 60_000, _ms(end + pd.Timedelta(days=1)) + 30 * 60_000
    while cursor < stop:
        rows = _post({"type": "fundingHistory", "coin": coin, "startTime": cursor, "endTime": stop}) or []
        if not rows:
            break
        for r in rows:
            prints[int(r["time"])] = float(r["fundingRate"])
        cursor = int(rows[-1]["time"]) + 1
        if len(rows) < 500:
            break
    s = pd.Series(prints, dtype=float)
    s.index = pd.to_datetime(s.index, unit="ms", utc=True)
    daily = s.groupby((s.index - pd.Timedelta(minutes=30)).floor("D")).sum() if len(s) else pd.Series(dtype=float)
    daily.to_pickle(path)
    return daily


@dataclass
class Pair:
    """Aligned daily frames, columns = Hyperliquid coin names."""

    hl_close: pd.DataFrame
    hl_high: pd.DataFrame
    hl_low: pd.DataFrame
    hl_fund: pd.DataFrame
    bn_close: pd.DataFrame
    bn_high: pd.DataFrame
    bn_low: pd.DataFrame
    bn_fund: pd.DataFrame
    bn_rank: pd.DataFrame
    eligible: pd.DataFrame

    def truncate(self, end: str) -> Pair:
        cut = pd.Timestamp(end, tz="UTC")
        return Pair(**{k: v.loc[:cut] for k, v in self.__dict__.items()})


def load(download: bool = False, with_holdout: bool = False) -> Pair:
    HL_DIR.mkdir(parents=True, exist_ok=True)
    m = data_mod.build()
    universe = [u["name"] for u in _post({"type": "meta"})["universe"]] if download else json.loads((HL_DIR / "universe.json").read_text())
    if download:
        (HL_DIR / "universe.json").write_text(json.dumps(universe))
    idx = pd.date_range(HL_START, END, freq="D")
    # Candles for every Hyperliquid perp, so the liquidity rank is taken across the whole venue.
    candles = {}
    for i, c in enumerate(universe):
        if not download and not (HL_DIR / f"candles_{c}.pkl").exists():
            raise RuntimeError(f"candles for {c} not downloaded; run `download` first")
        candles[c] = fetch_candles(c)
        if download and i % 40 == 0:
            print(f"  candles {i}/{len(universe)}", flush=True)
    listed = [c for c in universe if len(candles[c])]
    all_close = pd.DataFrame({c: candles[c]["close"] for c in listed}).reindex(idx)
    all_vol = pd.DataFrame({c: candles[c]["dollar_volume"] for c in listed}).reindex(idx)
    hl_seasoned = (all_close.notna().cumsum() >= MIN_CLOSES) & all_close.notna()
    hl_rank_all = all_vol.rolling(30, min_periods=30).median().where(hl_seasoned).rank(axis=1, ascending=False, method="first")

    coins = [c for c in listed if binance_symbol(c) in m.close.columns and binance_symbol(c) not in eng.EXCLUDE]
    wide = lambda col: pd.DataFrame({c: candles[c][col] for c in coins}).reindex(idx)  # noqa: E731
    hl_close, hl_high, hl_low = wide("close"), wide("high"), wide("low")
    bn_cols = [binance_symbol(c) for c in coins]
    pick = lambda df: df[bn_cols].set_axis(coins, axis=1).reindex(idx)  # noqa: E731
    bn_close = pick(m.close)
    bn_seasoned = (bn_close.notna().cumsum() >= MIN_CLOSES) & bn_close.notna()
    bn_rank = pick(eng.liquidity_rank(m.close, m.qvol))
    eligible = (hl_rank_all[coins] <= HL_TOP_N) & bn_seasoned & (bn_rank <= BINANCE_MAX_RANK)

    cut = pd.Timestamp(CUTOFF, tz="UTC")
    segments = [("sanity", HL_START, cut)] + ([("holdout", cut + pd.Timedelta(days=1), END)] if with_holdout else [])
    scope = eligible if with_holdout else eligible.loc[:cut]
    ever = [c for c in coins if scope[c].any()]
    print(f"HL perps: {len(listed)}, paired with Binance: {len(coins)}, ever eligible in scope: {len(ever)}", flush=True)
    fund = {}
    for i, c in enumerate(ever):
        parts = []
        for seg, seg_start, seg_end in segments:
            if not download and not (HL_DIR / f"funding_{c}_{seg}.pkl").exists():
                raise RuntimeError(f"funding for {c} ({seg}) not downloaded; run the download step first")
            first = max(seg_start, candles[c].index[0])
            parts.append(fetch_funding(c, first, seg_end, seg) if first <= seg_end else pd.Series(dtype=float))
        fund[c] = pd.concat(parts) if parts else pd.Series(dtype=float)
        fund[c] = fund[c][~fund[c].index.duplicated()]
        if download and i % 10 == 0:
            print(f"  funding {i}/{len(ever)}", flush=True)
    hl_fund = pd.DataFrame(fund).reindex(idx).reindex(columns=coins)
    return Pair(hl_close, hl_high, hl_low, hl_fund, bn_close, pick(m.high), pick(m.low), pick(m.funding), bn_rank, eligible)


# ------------------------------------------------------------------- simulation


def simulate(p: Pair, lookback: int = 7, enter: float = 0.20, exit_: float = 0.05, leg_leverage: float = 1.0,
             forced_move: float = 0.60) -> eng.Result:
    idx = p.hl_close.index
    coins = list(p.hl_close.columns)
    col = {c: i for i, c in enumerate(coins)}
    hl_c, bn_c = p.hl_close.to_numpy(), p.bn_close.to_numpy()
    hl_r = p.hl_close.pct_change(fill_method=None).to_numpy()
    bn_r = p.bn_close.pct_change(fill_method=None).to_numpy()
    hl_f, bn_f = p.hl_fund.to_numpy(), p.bn_fund.to_numpy()
    hl_hi, hl_lo, bn_hi, bn_lo = (x.to_numpy() for x in (p.hl_high, p.hl_low, p.bn_high, p.bn_low))
    spread = (p.hl_fund - p.bn_fund).rolling(lookback, min_periods=lookback).mean().to_numpy() * eng.ANN
    elig = p.eligible.to_numpy()
    bn_cost = eng.PERP_FEE + eng.slippage_for_rank(p.bn_rank.fillna(10**6).to_numpy())
    hl_cost = HL_FEE + HL_SLIP
    notional = (1.0 / SLOTS) * 0.5 * leg_leverage  # per leg, as a fraction of equity

    held: dict[str, dict] = {}  # coin -> {"dir": +1 short HL / -1 short Binance, "ref": price at last margin balance}
    pending_exit: set[str] = set()
    pending_entry: list[tuple[str, int]] = []
    n = len(idx)
    o = {k: np.zeros(n) for k in ("ret", "fund", "basis", "cost", "turn", "gross", "trips", "forced", "slots")}
    for d in range(n):
        cost = 0.0
        # 1. carry the positions held since yesterday's close through today
        for c in list(held):
            j, pos = col[c], held[c]
            alive = not (np.isnan(hl_c[d, j]) or np.isnan(bn_c[d, j]))
            rh = 0.0 if np.isnan(hl_r[d, j]) else hl_r[d, j]
            rb = 0.0 if np.isnan(bn_r[d, j]) else bn_r[d, j]
            fh, fb = hl_f[d, j], bn_f[d, j]
            if pos["dir"] > 0:  # short Hyperliquid (receive fh), long Binance (pay fb)
                recv = 0.0 if np.isnan(fh) else fh
                pay = MISSING_FUNDING_PAY if np.isnan(fb) else fb
                basis = rb - rh
            else:  # long Hyperliquid (pay fh), short Binance (receive fb)
                recv = 0.0 if np.isnan(fb) else fb
                pay = MISSING_FUNDING_PAY if np.isnan(fh) else fh
                basis = rh - rb
            o["basis"][d] += notional * basis
            o["fund"][d] += notional * (recv - pay)
            forced = False
            if alive and d > 0:
                for hi, lo, cl in ((hl_hi, hl_lo, hl_c), (bn_hi, bn_lo, bn_c)):
                    prev = cl[d - 1, j]
                    if not np.isnan(prev) and (hi[d, j] / prev - 1.0 > forced_move or lo[d, j] / prev - 1.0 < -forced_move):
                        forced = True
            if forced or not alive:
                cost += notional * (bn_cost[d, j] + hl_cost) + (notional * FORCED_PENALTY if forced else 0.0)
                o["turn"][d] += 2 * notional
                o["forced"][d] += int(forced)
                del held[c]
                pending_exit.discard(c)
                continue
            if abs(bn_c[d, j] / pos["ref"] - 1.0) > MARGIN_MOVE:
                cost += notional * MARGIN_COST
                pos["ref"] = bn_c[d, j]
        # 2. execute yesterday's decisions at today's close
        for c in list(pending_exit):
            if c in held:
                j = col[c]
                cost += notional * (bn_cost[d, j] + hl_cost)
                o["turn"][d] += 2 * notional
                o["trips"][d] += 1
                del held[c]
        pending_exit.clear()
        for c, direction in pending_entry:
            j = col[c]
            if len(held) < SLOTS and c not in held and not (np.isnan(hl_c[d, j]) or np.isnan(bn_c[d, j])):
                cost += notional * (bn_cost[d, j] + hl_cost)
                o["turn"][d] += 2 * notional
                held[c] = {"dir": direction, "ref": bn_c[d, j]}
        pending_entry = []
        # 3. decide at today's close, for execution tomorrow
        for c, pos in held.items():
            j = col[c]
            m = spread[d, j]
            if np.isnan(m) or not elig[d, j] or np.sign(m) != pos["dir"] or abs(m) < exit_:
                pending_exit.add(c)
        free = SLOTS - (len(held) - len(pending_exit))
        if free > 0:
            row = spread[d]
            order = np.argsort(-np.abs(np.where(np.isnan(row), 0.0, row)))
            for j in order:
                if free <= 0 or np.isnan(row[j]) or abs(row[j]) < enter:
                    break
                c = coins[j]
                if elig[d, j] and c not in held:
                    pending_entry.append((c, int(np.sign(row[j]))))
                    free -= 1
        o["cost"][d] = cost
        o["ret"][d] = o["basis"][d] + o["fund"][d] - cost
        o["gross"][d] = 2 * notional * len(held)
        o["slots"][d] = len(held)
    s = lambda a: pd.Series(a, index=idx)  # noqa: E731
    res = eng.Result(s(o["ret"]), s(o["basis"]), s(o["cost"]), -s(o["fund"]), s(o["turn"]), s(o["gross"]), s(o["gross"]) * 0)
    res.extra.update({"round_trips": s(o["trips"]), "forced": s(o["forced"]), "basis": s(o["basis"]), "funding_pnl": s(o["fund"]),
                      "slots": s(o["slots"])})
    return res


def summarise(res: eng.Result, start: str, end: str) -> dict:
    mt = eng.metrics(res, *(start, end))
    w = lambda x: eng.window(x, start, end)  # noqa: E731
    years = len(w(res.returns)) / eng.ANN
    mt.update({
        "funding_pnl_per_year": float(w(res.extra["funding_pnl"]).sum() / years),
        "basis_pnl_per_year": float(w(res.extra["basis"]).sum() / years),
        "round_trips": int(w(res.extra["round_trips"]).sum()),
        "forced_closes": int(w(res.extra["forced"]).sum()),
        "active_share": float((w(res.gross) > 0).mean()),
        "avg_slots": float(w(res.extra["slots"]).mean()),
    })
    return mt


VARIANTS = {
    "primary (7d, enter 20%, exit 5%, 1x legs)": {},
    "secondary: 2x legs, forced close at 40%": {"leg_leverage": 2.0, "forced_move": 0.40},
    "secondary: lookback 3d": {"lookback": 3},
    "secondary: lookback 14d": {"lookback": 14},
    "secondary: enter 10%": {"enter": 0.10},
    "secondary: enter 30%": {"enter": 0.30},
}


def describe_gap(p: Pair, start: str, end: str) -> dict:
    sp = eng.window_frame(p.hl_fund - p.bn_fund, start, end).where(eng.window_frame(p.eligible, start, end)) * eng.ANN
    hl = eng.window_frame(p.hl_fund, start, end).where(eng.window_frame(p.eligible, start, end)) * eng.ANN
    bn = eng.window_frame(p.bn_fund, start, end).where(eng.window_frame(p.eligible, start, end)) * eng.ANN
    return {
        "mean_gap": float(np.nanmean(sp.to_numpy())),
        "median_abs_gap": float(np.nanmedian(np.abs(sp.to_numpy()))),
        "mean_hl_funding": float(np.nanmean(hl.to_numpy())),
        "mean_binance_funding": float(np.nanmean(bn.to_numpy())),
        "btc_mean_gap": float(sp["BTC"].mean()) if "BTC" in sp else None,
        "eligible_coins_per_day": float(eng.window_frame(p.eligible, start, end).sum(axis=1).mean()),
        "hl_funding_missing_share": float((eng.window_frame(p.hl_fund, start, end).isna() & eng.window_frame(p.eligible, start, end)).sum().sum()
                                          / max(1, eng.window_frame(p.eligible, start, end).sum().sum())),
    }


def run(mode: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    if mode == "sanity":
        p = load().truncate(CUTOFF)
        win, out_path = SANITY, RESULTS / "xvenue_sanity.json"
    else:
        sanity = json.loads((RESULTS / "xvenue_sanity.json").read_text())
        if not sanity["sanity_pass"]:
            print("Sanity check failed: the holdout stays unread (its funding data is not even downloaded).")
            return
        p = load(download=True, with_holdout=True)
        win, out_path = HOLDOUT, RESULTS / "xvenue_holdout.json"
    report: dict = {"window": win, "variants": {}, "gap": describe_gap(p, *win)}
    for name, kw in VARIANTS.items():
        res = simulate(p, **kw)
        report["variants"][name] = summarise(res, *win)
        report["variants"][name]["years"] = eng.yearly(res)
    prim = report["variants"][next(iter(VARIANTS))]
    if mode == "sanity":
        report["sanity_pass"] = bool(prim["cagr"] > eng.RF_ANNUAL)
    else:
        yrs = prim["years"]
        gates = {
            "G1 CAGR >= 8%": prim["cagr"] >= 0.08,
            "G2 excess Sharpe >= 1.0": prim["sharpe"] >= 1.0,
            "G3 max DD <= 10%": prim["max_dd"] <= 0.10,
            "G4 2025 and 2026 both positive": yrs.get(2025, 0) > 0 and yrs.get(2026, 0) > 0,
            "G5 active >= 50% of days": prim["active_share"] >= 0.5,
        }
        report["gates"], report["pass"] = gates, all(gates.values())
    out_path.write_text(json.dumps(report, indent=1, default=float, sort_keys=True))
    pct = lambda x: f"{x * 100:6.2f}%"  # noqa: E731
    print("gap:", json.dumps({k: (round(v, 4) if v is not None else None) for k, v in report["gap"].items()}))
    for name, v in report["variants"].items():
        print(f"{name:45} CAGR {pct(v['cagr'])} Sharpe {v['sharpe']:6.2f} DD {pct(v['max_dd'])} funding {pct(v['funding_pnl_per_year'])} "
              f"basis {pct(v['basis_pnl_per_year'])} cost {pct(v['cost_per_year'])} trips {v['round_trips']:4d} forced {v['forced_closes']} "
              f"active {v['active_share']:.2f} slots {v['avg_slots']:.2f} years {({y: round(r * 100, 1) for y, r in v['years'].items() if r})}")
    print({k: report[k] for k in ("sanity_pass", "gates", "pass") if k in report})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["download", "sanity", "holdout"])
    args = ap.parse_args()
    if args.mode == "download":
        pair = load(download=True)
        print("eligible coins per day by year:", pair.eligible.sum(axis=1).groupby(pair.eligible.index.year).mean().round(1).to_dict())
        print("HL funding coverage on eligible days:",
              1 - float((pair.hl_fund.isna() & pair.eligible).sum().sum() / max(1, pair.eligible.sum().sum())))
    else:
        run(args.mode)
