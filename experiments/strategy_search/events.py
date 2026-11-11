"""Event-driven intraday trades (docs/research/experiments/2026-11-11_event-driven-intraday-prereg.md).

  python experiments/strategy_search/events.py download   # hourly klines for the symbol-months the rules need
  python experiments/strategy_search/events.py run        # E1, E2, E3 once each, gates applied

E1 funding-settlement capture, E2 buy capitulation, E3 follow a volume breakout.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as data_mod  # noqa: E402
import engine as eng  # noqa: E402

HOURLY_DIR = data_mod.DATA_DIR / "hourly"
RESULTS = Path(__file__).resolve().parent / "results"
END = data_mod.END
HOUR = pd.Timedelta(hours=1)

E1_THRESHOLD = 0.005
E1_MAX_RANK = 100
E23_MAX_RANK = 50
MOVE_FLOOR = 0.03
MOVE_SIGMAS = 4.0
VOLUME_MULT = 3.0
LOOKBACK_H = 168
MIN_LOOKBACK_H = 120
HOLD_H = 4
TRADE_SIZE = 0.10
MAX_OPEN = 5
SUBPERIODS = {"2020-22": ("2020-01-01", "2022-12-31"), "2023-24": ("2023-01-01", "2024-12-31"), "2025-26": ("2025-01-01", "2026-10-31")}


# -------------------------------------------------------------------- data


def load_funding_prints(symbol: str) -> pd.Series:
    """Per-settlement funding prints, index rounded to the settlement hour."""
    f = pd.read_pickle(data_mod.DATA_DIR / "symbols" / f"{symbol}.pkl")["funding"]
    if len(f) == 0:
        return f
    f = f.copy()
    f.index = f.index.round("h")
    return f[~f.index.duplicated()].loc[: END + pd.Timedelta(days=1)]


def needed_symbol_months(rank_prev: pd.DataFrame) -> dict[str, set[str]]:
    need: dict[str, set[str]] = {}

    def add(sym: str, stamp: pd.Timestamp) -> None:
        for ts in (stamp, stamp - pd.Timedelta(days=8), stamp + pd.Timedelta(days=2)):  # lookback and exit spill-over
            need.setdefault(sym, set()).add(ts.strftime("%Y-%m"))

    in50 = rank_prev <= E23_MAX_RANK
    for sym in in50.columns[in50.any()]:
        days = in50.index[in50[sym]]
        for month_start in sorted({d.replace(day=1) for d in days}):
            add(sym, month_start)
            add(sym, month_start + pd.offsets.MonthEnd(0))
    for sym in rank_prev.columns:
        if sym in eng.EXCLUDE:
            continue
        f = load_funding_prints(sym)
        if len(f) < 2:
            continue
        prev = f.shift(1)
        ev = f.index[(prev.abs() >= E1_THRESHOLD).to_numpy()]
        for t in ev:
            if rank_prev.at[t.normalize(), sym] <= E1_MAX_RANK if t.normalize() in rank_prev.index else False:
                add(sym, t)
    first = pd.Timestamp("2020-01-01", tz="UTC").strftime("%Y-%m")
    last = END.strftime("%Y-%m")
    return {s: {m for m in months if first <= m <= last} for s, months in need.items()}


def _fetch_month(args: tuple[str, str]) -> tuple[str, str, pd.DataFrame | None]:
    sym, month = args
    key = f"data/futures/um/monthly/klines/{sym}/1h/{sym}-1h-{month}.zip"
    try:
        df = data_mod._read_zip_csv(key).iloc[:, :8]
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return sym, month, None
        raise
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume"]
    out = pd.DataFrame({"open": df["open"].astype(float).to_numpy(), "close": df["close"].astype(float).to_numpy(),
                        "qvol": df["quote_volume"].astype(float).to_numpy()}, index=data_mod._to_ts(df["open_time"]))
    return sym, month, out


def download(rank_prev: pd.DataFrame) -> None:
    HOURLY_DIR.mkdir(parents=True, exist_ok=True)
    need = needed_symbol_months(rank_prev)
    jobs = []
    for sym, months in need.items():
        done_path = HOURLY_DIR / f"{sym}.months.json"
        done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
        jobs += [(sym, m) for m in sorted(months - done)]
    print(f"{len(need)} symbols, {sum(len(v) for v in need.values())} symbol-months needed, {len(jobs)} to download", flush=True)
    fetched: dict[str, list] = {}
    missing = 0
    with ThreadPoolExecutor(data_mod.WORKERS) as pool:
        for i, (sym, month, df) in enumerate(pool.map(_fetch_month, jobs), 1):
            fetched.setdefault(sym, []).append((month, df))
            missing += df is None
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    for sym, parts in fetched.items():
        path, done_path = HOURLY_DIR / f"{sym}.pkl", HOURLY_DIR / f"{sym}.months.json"
        frames = [pd.read_pickle(path)] if path.exists() else []
        frames += [df for _, df in parts if df is not None]
        if frames:
            allh = pd.concat(frames)
            allh = allh[~allh.index.duplicated()].sort_index()
            allh.to_pickle(path)
        done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
        done_path.write_text(json.dumps(sorted(done | {m for m, _ in parts})))
    print(f"done; {missing} symbol-months not in the archive", flush=True)


def load_hourly(symbol: str) -> pd.DataFrame | None:
    path = HOURLY_DIR / f"{symbol}.pkl"
    return pd.read_pickle(path) if path.exists() else None


# -------------------------------------------------------------------- rules


def _cost(rank: float, slip_mult: float) -> float:
    """Round-trip cost: fee + slippage, both sides."""
    slip = float(eng.slippage_for_rank(np.array([rank if not np.isnan(rank) else 10**6]))[0])
    return 2.0 * (eng.PERP_FEE + slip * slip_mult)


def trades_e1(symbol: str, hourly: pd.DataFrame, funding: pd.Series, rank_prev: pd.Series, slip_mult: float = 1.0) -> list[dict]:
    """Take the receiving side one hour before a settlement whose previous print was extreme; close one hour after."""
    out: list[dict] = []
    if len(funding) < 2:
        return out
    opens, closes = hourly["open"], hourly["close"]
    prev = funding.shift(1)
    busy_until = pd.Timestamp.min.tz_localize("UTC")
    for t in funding.index[(prev.abs() >= E1_THRESHOLD).to_numpy()]:
        entry_bar, exit_bar = t - HOUR, t
        if entry_bar < busy_until or entry_bar not in opens.index or exit_bar not in closes.index:
            continue
        day = t.normalize()
        rank = rank_prev.get(day, np.nan)
        if not rank <= E1_MAX_RANK:
            continue
        side = -float(np.sign(prev[t]))
        price_ret = side * (closes[exit_bar] / opens[entry_bar] - 1.0)
        fund = -side * float(funding[t])
        cost = _cost(rank, slip_mult)
        out.append({"symbol": symbol, "entry": entry_bar, "exit": exit_bar + HOUR, "side": side, "rank": float(rank),
                    "price_ret": float(price_ret), "funding": fund, "cost": cost, "net": float(price_ret + fund - cost)})
        busy_until = exit_bar + HOUR
    return out


def trades_e23(symbol: str, hourly: pd.DataFrame, funding: pd.Series, rank_prev: pd.Series, direction: int,
               hold: int = HOLD_H, slip_mult: float = 2.0, btc: pd.DataFrame | None = None) -> list[dict]:
    """direction -1: buy after an extreme down hour (E2). direction +1: buy after an extreme up hour (E3)."""
    out: list[dict] = []
    full = hourly.reindex(pd.date_range(hourly.index[0], hourly.index[-1], freq="h"))
    r = full["close"] / full["close"].shift(1) - 1.0
    sigma = r.shift(1).rolling(LOOKBACK_H, min_periods=MIN_LOOKBACK_H).std()
    vavg = full["qvol"].shift(1).rolling(LOOKBACK_H, min_periods=MIN_LOOKBACK_H).mean()
    bar = np.maximum(MOVE_FLOOR, MOVE_SIGMAS * sigma)
    move = (r >= bar) if direction > 0 else (r <= -bar)
    trig = move & (full["qvol"] >= VOLUME_MULT * vavg)
    opens, closes = full["open"], full["close"]
    busy_until = pd.Timestamp.min.tz_localize("UTC")
    for h in full.index[trig.fillna(False).to_numpy()]:
        entry_bar, exit_bar = h + HOUR, h + hold * HOUR
        if entry_bar < busy_until or exit_bar > full.index[-1]:
            continue
        rank = rank_prev.get(h.normalize(), np.nan)
        if not rank <= E23_MAX_RANK:
            continue
        p_in, p_out = opens.get(entry_bar, np.nan), closes.get(exit_bar, np.nan)
        if np.isnan(p_in) or np.isnan(p_out):
            continue
        price_ret = p_out / p_in - 1.0
        paid = float(funding.loc[(funding.index > entry_bar) & (funding.index <= exit_bar + HOUR)].sum()) if len(funding) else 0.0
        cost = _cost(rank, slip_mult)
        row = {"symbol": symbol, "entry": entry_bar, "exit": exit_bar + HOUR, "side": 1.0, "rank": float(rank),
               "trigger_ret": float(r[h]), "price_ret": float(price_ret), "funding": -paid, "cost": cost,
               "net": float(price_ret - paid - cost)}
        if btc is not None:
            b_in, b_out = btc["open"].get(entry_bar, np.nan), btc["close"].get(exit_bar, np.nan)
            row["btc_ret"] = float(b_out / b_in - 1.0) if not (np.isnan(b_in) or np.isnan(b_out)) else np.nan
        out.append(row)
        busy_until = exit_bar + HOUR
    return out


# -------------------------------------------------------------------- evaluation


def clustered_t(trades: pd.DataFrame, col: str = "net") -> float:
    daily = trades.groupby(trades["exit"].dt.normalize())[col].sum()
    if len(daily) < 3 or daily.std() == 0:
        return float("nan")
    return float(daily.mean() / (daily.std() / np.sqrt(len(daily))))


def portfolio(trades: pd.DataFrame) -> dict:
    """10% of equity per trade, at most MAX_OPEN open at once; P&L booked on the exit day."""
    taken, open_exits = [], []
    for row in trades.sort_values("entry").itertuples():
        open_exits = [x for x in open_exits if x > row.entry]
        if len(open_exits) >= MAX_OPEN:
            continue
        open_exits.append(row.exit)
        taken.append((row.exit.normalize(), row.net))
    if not taken:
        return {"trades_taken": 0}
    pnl = pd.Series([n for _, n in taken], index=[d for d, _ in taken]).groupby(level=0).sum() * TRADE_SIZE
    days = pd.date_range(trades["entry"].min().normalize(), END, freq="D")
    daily = pnl.reindex(days).fillna(0.0)
    eq = (1.0 + daily).cumprod()
    years = len(days) / eng.ANN
    return {
        "trades_taken": len(taken),
        "cagr": float(eq.iloc[-1] ** (1.0 / years) - 1.0) if eq.iloc[-1] > 0 else -1.0,
        "max_dd": float(-(eq / eq.cummax().clip(lower=1.0) - 1.0).min()),
        "worst_day": float(daily.min()),
        "years": {int(y): float((1.0 + r).prod() - 1.0) for y, r in daily.groupby(daily.index.year)},
    }


def summarise(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0}
    out = {
        "trades": int(len(trades)),
        "mean_net": float(trades["net"].mean()),
        "median_net": float(trades["net"].median()),
        "t_clustered": clustered_t(trades),
        "win_rate": float((trades["net"] > 0).mean()),
        "worst_trade": float(trades["net"].min()),
        "mean_price_ret": float(trades["price_ret"].mean()),
        "mean_funding": float(trades["funding"].mean()),
        "mean_cost": float(trades["cost"].mean()),
        "by_subperiod": {},
        "by_year": {int(y): {"n": int(len(g)), "mean_net": float(g["net"].mean())} for y, g in trades.groupby(trades["exit"].dt.year)},
        "by_rank_tier": {},
    }
    for name, (a, b) in SUBPERIODS.items():
        g = trades[(trades["exit"] >= pd.Timestamp(a, tz="UTC")) & (trades["exit"] <= pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))]
        out["by_subperiod"][name] = {"n": int(len(g)), "mean_net": float(g["net"].mean()) if len(g) else None}
    for name, lo, hi in (("1-10", 0, 10), ("11-30", 10, 30), ("31+", 30, 10**9)):
        g = trades[(trades["rank"] > lo) & (trades["rank"] <= hi)]
        out["by_rank_tier"][name] = {"n": int(len(g)), "mean_net": float(g["net"].mean()) if len(g) else None}
    return out


def gates(summary: dict, port: dict, stress_mean: float) -> dict:
    sub = summary.get("by_subperiod", {})
    qualifying = {k: v for k, v in sub.items() if v["n"] >= 50}
    g = {
        "G1 >= 300 trades": summary.get("trades", 0) >= 300,
        "G2 mean net > 0 with clustered t >= 3": summary.get("mean_net", -1) > 0 and summary.get("t_clustered", 0) >= 3.0,
        "G3 positive in every sub-period with >= 50 trades, incl. 2025-26": bool(qualifying) and "2025-26" in qualifying
        and all(v["mean_net"] > 0 for v in qualifying.values()),
        "G4 portfolio CAGR >= 10% and max DD <= 20%": port.get("cagr", -1) >= 0.10 and port.get("max_dd", 1) <= 0.20,
        "G5 mean net > 0 with slippage doubled again": stress_mean > 0,
    }
    return {"gates": g, "pass": all(g.values())}


def run() -> None:
    m = data_mod.build()
    rank_prev = eng.liquidity_rank(m.close, m.qvol).shift(1)
    btc = load_hourly("BTCUSDT")
    symbols = [s for s in m.close.columns if s not in eng.EXCLUDE and (HOURLY_DIR / f"{s}.pkl").exists()]
    sets: dict[str, list] = {k: [] for k in ("E1", "E1_stress", "E2", "E2_stress", "E3", "E3_stress")}
    holds: dict[str, list] = {f"{e}_hold{h}": [] for e in ("E2", "E3") for h in (1, 2, 8, 24)}
    for sym in symbols:
        hourly = load_hourly(sym)
        funding = load_funding_prints(sym)
        rp = rank_prev[sym]
        sets["E1"] += trades_e1(sym, hourly, funding, rp, 1.0)
        sets["E1_stress"] += trades_e1(sym, hourly, funding, rp, 2.0)
        if not (rp <= E23_MAX_RANK).any():
            continue
        for name, direction in (("E2", -1), ("E3", +1)):
            sets[name] += trades_e23(sym, hourly, funding, rp, direction, HOLD_H, 2.0, btc)
            sets[f"{name}_stress"] += trades_e23(sym, hourly, funding, rp, direction, HOLD_H, 4.0)
            for h in (1, 2, 8, 24):
                holds[f"{name}_hold{h}"] += trades_e23(sym, hourly, funding, rp, direction, h, 2.0)
    report: dict = {}
    frames = {k: pd.DataFrame(v) for k, v in sets.items()}
    for idea in ("E1", "E2", "E3"):
        tr = frames[idea]
        summ = summarise(tr)
        port = portfolio(tr) if len(tr) else {}
        stress = frames[f"{idea}_stress"]
        stress_mean = float(stress["net"].mean()) if len(stress) else -1.0
        entry = {"summary": summ, "portfolio": port, "stress_mean_net": stress_mean, **gates(summ, port, stress_mean)}
        if idea in ("E2", "E3") and len(tr):
            hedged = tr.dropna(subset=["btc_ret"]).copy()
            hedged["net"] = hedged["net"] - hedged["btc_ret"] - 2.0 * (eng.PERP_FEE + 0.0005)
            mirror = tr.copy()
            mirror["net"] = -tr["price_ret"] - tr["funding"] - tr["cost"]
            entry["secondary"] = {
                "btc_hedged": {"mean_net": float(hedged["net"].mean()), "t_clustered": clustered_t(hedged)},
                "mirror_short": {"mean_net": float(mirror["net"].mean()), "t_clustered": clustered_t(mirror)},
                "other_holds": {k.split("_")[1]: {"n": len(v), "mean_net": float(pd.DataFrame(v)["net"].mean()) if v else None,
                                                  "t_clustered": clustered_t(pd.DataFrame(v)) if v else None}
                                for k, v in holds.items() if k.startswith(idea)},
            }
        if idea == "E1" and len(tr):
            mirror = tr.copy()
            mirror["net"] = -tr["price_ret"] - tr["funding"] - tr["cost"]
            entry["secondary"] = {
                "mirror_pay_funding": {"mean_net": float(mirror["net"].mean()), "t_clustered": clustered_t(mirror)},
                "by_side": {("short (funding positive)" if s < 0 else "long (funding negative)"):
                            {"n": int(len(g)), "mean_net": float(g["net"].mean()), "mean_price_ret": float(g["price_ret"].mean()),
                             "mean_funding": float(g["funding"].mean())} for s, g in tr.groupby("side")},
            }
        report[idea] = entry
        if len(tr):
            tr.assign(entry=tr["entry"].astype(str), exit=tr["exit"].astype(str)).to_csv(RESULTS / f"events_{idea}_trades.csv", index=False)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "events.json").write_text(json.dumps(report, indent=1, default=float, sort_keys=True))
    bp = lambda x: f"{x * 1e4:7.1f}bp" if x is not None else "    n/a"  # noqa: E731
    for idea, e in report.items():
        s, p = e["summary"], e["portfolio"]
        print(f"\n{idea}: trades {s.get('trades', 0)}  mean net {bp(s.get('mean_net'))}  median {bp(s.get('median_net'))}  "
              f"t {s.get('t_clustered', float('nan')):.2f}  win {s.get('win_rate', 0):.2f}  worst {s.get('worst_trade', 0) * 100:.1f}%")
        print(f"   price {bp(s.get('mean_price_ret'))}  funding {bp(s.get('mean_funding'))}  cost {bp(s.get('mean_cost'))}  "
              f"stress mean {bp(e['stress_mean_net'])}")
        print("   sub-periods:", {k: (v["n"], round(v["mean_net"] * 1e4, 1) if v["mean_net"] is not None else None) for k, v in s.get("by_subperiod", {}).items()})
        print("   rank tiers:", {k: (v["n"], round(v["mean_net"] * 1e4, 1) if v["mean_net"] is not None else None) for k, v in s.get("by_rank_tier", {}).items()})
        print(f"   portfolio: taken {p.get('trades_taken')}  CAGR {p.get('cagr', 0) * 100:.1f}%  DD {p.get('max_dd', 0) * 100:.1f}%  "
              f"years {({y: round(v * 100, 1) for y, v in p.get('years', {}).items()})}")
        print("   secondary:", json.dumps(e.get("secondary", {}), default=float)[:900])
        print("   ", {k.split()[0]: v for k, v in e["gates"].items()}, "PASS" if e["pass"] else "FAIL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["download", "run"])
    args = ap.parse_args()
    if args.mode == "download":
        market = data_mod.build()
        download(eng.liquidity_rank(market.close, market.qvol).shift(1))
    else:
        run()
