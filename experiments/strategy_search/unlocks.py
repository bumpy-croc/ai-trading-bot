"""Shorting coins around large scheduled token unlocks, hedged with a basket of similar coins
(docs/research/experiments/2026-11-24_token-unlock-short-prereg.md).

  python experiments/strategy_search/unlocks.py events --raw DIR
      Build results/unlock_events.csv from DefiLlama's per-token emission files (one JSON per token in DIR,
      as served at https://defillama-datasets.llama.fi/emissions/<slug>), matched to Binance perps.

  python experiments/strategy_search/unlocks.py design
      Data and events truncated at 2024-12-31. Runs the three pre-specified windows, applies the
      shortlist rule and writes results/unlock_design.json and results/unlock_shortlist.json.

  python experiments/strategy_search/unlocks.py exam
      Runs only what the committed shortlist names, once, on 2025-01-01..2026-10-31. Writes
      results/unlock_exam.json.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as data_mod  # noqa: E402
import engine as eng  # noqa: E402

import strategies as strat  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
EVENTS_CSV = RESULTS / "unlock_events.csv"
FIT = ("2022-01-01", "2023-12-31")
VAL = ("2024-01-01", "2024-12-31")
EXAM = ("2025-01-01", "2026-10-31")
DESIGN_CUTOFF = "2024-12-31"

CATEGORIES = ("insiders", "privateSale")  # team and investors
MIN_SHARE, MAX_SHARE = 0.01, 1.0  # of the supply already unlocked the day before
MIN_LISTED_DAYS = 45  # the perp must have traded this long before the unlock day
WINDOWS = {"U1": (14, 0), "U2": (7, 7), "U3": (30, 14)}  # days before the unlock to enter, days after to exit

SLOT = 0.10  # capital per position; scaled down when more than 1 / SLOT positions are open
STOP = 0.30  # a position is closed inside the day once it has lost this much since entry
STOP_SLIP = 0.02  # extra loss taken on a stopped exit
HEDGE_RANKS = (31, 100)  # liquidity ranks of the hedge basket
LEG_COST = 0.0005 + 0.0020  # fee + slippage per side, each leg
SHORTLIST_SHARPE = 0.7
SHORTLIST_DD = 0.25
EXAM_T = 2.69  # one-sided 5% shared over 14 candidates on the exam window (normal quantile of 1 - 0.05/14)


# ------------------------------------------------------------------------ events


def build_events(raw_dir: Path, protocols_json: Path, chains_json: Path) -> pd.DataFrame:
    m = data_mod.build()
    first = m.close.apply(lambda s: s.first_valid_index())
    prot = json.loads(protocols_json.read_text())
    by_id = {str(p["id"]): p for p in prot}
    gecko_sym = {p["gecko_id"]: p.get("symbol") for p in prot if p.get("gecko_id")}
    chains = json.loads(chains_json.read_text())
    chain_gecko = {c["gecko_id"]: c.get("tokenSymbol") for c in chains if c.get("gecko_id")}
    chain_name = {c["name"].lower(): c.get("tokenSymbol") for c in chains}
    rows = []
    for f in sorted(raw_dir.glob("*.json")):
        d = json.loads(f.read_text())
        sym = None
        for pid in d.get("defillamaIds") or []:
            p = by_id.get(str(pid))
            if p and p.get("symbol") and p["symbol"] != "-":
                sym = p["symbol"]
                break
        if not sym and d.get("gecko_id"):
            sym = gecko_sym.get(d["gecko_id"]) or chain_gecko.get(d["gecko_id"])
        if not sym:
            sym = chain_name.get((d.get("name") or "").lower())
        if not sym:
            continue
        perp = next((p for p in (sym.upper() + "USDT", "1000" + sym.upper() + "USDT") if p in m.close.columns), None)
        if perp is None:
            continue
        noncirc = set((d.get("categories") or {}).get("noncirculating", []))
        unlocked: dict[int, float] = {}
        for sec in (d.get("documentedData") or {}).get("data", []):
            if sec["label"] in noncirc:
                continue
            for pt in sec["data"]:
                unlocked[pt["timestamp"]] = unlocked.get(pt["timestamp"], 0.0) + (pt.get("unlocked") or 0.0)
        if not unlocked:
            continue
        ts = sorted(unlocked)
        for e in (d.get("metadata") or {}).get("unlockEvents") or []:
            amount = sum(a.get("amount") or 0.0 for a in e.get("cliffAllocations", []) if a.get("category") in CATEGORIES)
            if amount <= 0:
                continue
            i = bisect.bisect_left(ts, e["timestamp"] - 86400) - 1
            before = unlocked[ts[i]] if i >= 0 else 0.0
            if before <= 0:
                continue
            day = pd.Timestamp(e["timestamp"], unit="s", tz="UTC").normalize()
            share = amount / before
            if MIN_SHARE <= share <= MAX_SHARE and (day - first[perp]).days >= MIN_LISTED_DAYS:
                rows.append({"slug": f.stem, "perp": perp, "date": day.date().isoformat(), "share": round(share, 6)})
    ev = pd.DataFrame(rows).drop_duplicates(["perp", "date"]).sort_values(["date", "perp"])
    ev = ev[(ev["date"] >= "2021-01-01") & (ev["date"] <= EXAM[1])]
    RESULTS.mkdir(exist_ok=True)
    ev.to_csv(EVENTS_CSV, index=False)
    print(f"{len(ev)} events, {ev['perp'].nunique()} coins;", dict(ev.groupby(ev['date'].str[:4]).size()))
    return ev


def load_events(end: str) -> pd.DataFrame:
    ev = pd.read_csv(EVENTS_CSV)
    ev["date"] = pd.to_datetime(ev["date"], utc=True)
    return ev[ev["date"] <= pd.Timestamp(end, tz="UTC")]


# ------------------------------------------------------------------------ simulation


def intervals(ev: pd.DataFrame, days_in: int, days_out: int, index: pd.DatetimeIndex) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Per coin, merged [entry close, exit close] intervals. An unlock whose entry falls inside an open
    interval extends it."""
    out: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for perp, g in ev.groupby("perp"):
        merged: list[list[pd.Timestamp]] = []
        for day in sorted(g["date"]):
            a, b = day - pd.Timedelta(days=days_in), day + pd.Timedelta(days=days_out)
            if a < index[0] or a > index[-1]:
                continue
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        out[perp] = [(a, min(b, index[-1])) for a, b in merged]
    return out


def simulate(m: data_mod.Market, ev: pd.DataFrame, days_in: int, days_out: int, hedge: str = "basket") -> dict:
    """Daily portfolio of short-coin / long-hedge pairs. Returns accrue from the day after the entry close
    to the exit close. `hedge` is "basket" (ranks 31-100, coins with an open position left out), "btc" or "none"."""
    idx = m.close.index
    r = m.close.pct_change(fill_method=None)
    fund = m.funding.reindex(idx).fillna(0.0)
    rank, _ = strat.prepare(m)
    spans = intervals(ev, days_in, days_out, idx)

    # pass 1: which coins are scheduled to be open on each day (used to keep them out of the basket)
    scheduled = pd.DataFrame(False, index=idx, columns=m.close.columns)
    for perp, lst in spans.items():
        for a, b in lst:
            scheduled.loc[a + pd.Timedelta(days=1): b, perp] = True
    if hedge == "basket":
        member = ((rank >= HEDGE_RANKS[0]) & (rank <= HEDGE_RANKS[1])).shift(1, fill_value=False) & ~scheduled
        hedge_r = r.where(member).mean(axis=1).fillna(0.0)
        hedge_f = fund.where(member).mean(axis=1).fillna(0.0)
    elif hedge == "btc":
        hedge_r, hedge_f = r["BTCUSDT"].fillna(0.0), fund["BTCUSDT"]
    else:
        hedge_r, hedge_f = pd.Series(0.0, index=idx), pd.Series(0.0, index=idx)
    legs = 2 if hedge != "none" else 1

    # pass 2: each position's daily pair return, with the stop
    pair = pd.DataFrame(np.nan, index=idx, columns=sorted(spans))
    trades = []
    for perp, lst in spans.items():
        for a, b in lst:
            days = idx[(idx > a) & (idx <= b)]
            if len(days) == 0 or pd.isna(m.close.at[a, perp]):
                continue
            daily = (-r.loc[days, perp] + fund.loc[days, perp] + hedge_r.loc[days] - hedge_f.loc[days]).fillna(0.0)
            # Stop, checked inside the day on the coin's high. The exit is at the stop level or at the open if the
            # coin gapped through it, less STOP_SLIP; the hedge leg is taken at its close-to-close return that day.
            prev_close = m.close[perp].shift(1).loc[days]
            up_high = (m.high.loc[days, perp] / prev_close - 1.0).fillna(0.0).values
            up_open = (m.open.loc[days, perp] / prev_close - 1.0).fillna(0.0).values
            rest = (fund.loc[days, perp] + hedge_r.loc[days] - hedge_f.loc[days]).fillna(0.0).values
            vals, cum, stopped = daily.values.copy(), 1.0, False
            for i in range(len(vals)):
                if cum * (1.0 - up_high[i] + rest[i]) - 1.0 <= -STOP:
                    exit_level = min(1.0 - STOP, cum * (1.0 - up_open[i] + rest[i])) - STOP_SLIP
                    vals[i] = exit_level / cum - 1.0
                    vals, stopped = vals[: i + 1], True
                    break
                cum *= 1.0 + vals[i]
            daily = pd.Series(vals, index=days[: len(vals)])
            pair.loc[daily.index, perp] = daily.values
            gross = float((1.0 + daily).prod() - 1.0)
            trades.append({"perp": perp, "entry": a.date().isoformat(), "exit": daily.index[-1].date().isoformat(),
                           "days": int(len(daily)), "gross": gross, "net": gross - 2 * legs * LEG_COST, "stopped": bool(stopped)})
    is_open = pair.notna()
    n_open = is_open.sum(axis=1)
    weight = np.minimum(SLOT, 1.0 / n_open.where(n_open > 0)).fillna(0.0)
    opened = (is_open & ~is_open.shift(1, fill_value=False)).sum(axis=1)
    closed = (is_open & ~is_open.shift(-1, fill_value=False)).sum(axis=1)
    cost = (opened + closed) * weight * legs * LEG_COST
    port = pair.fillna(0.0).sum(axis=1) * weight - cost
    return {"returns": port, "trades": pd.DataFrame(trades), "n_open": n_open, "gross_exposure": n_open * weight * legs}


def port_metrics(sim: dict, start: str, end: str) -> dict:
    r = sim["returns"]
    res = eng.Result(r, r, r * 0, r * 0, r * 0, sim["gross_exposure"], r * 0)
    mt = eng.metrics(res, start, end)
    w = eng.window(r, start, end)
    ex = w - eng.RF_DAILY
    t = sim["trades"]
    if len(t):
        t = t[(t["entry"] >= start) & (t["entry"] <= end)]
    by_month = t.groupby(t["entry"].str[:7])["net"].mean() if len(t) else pd.Series(dtype=float)
    return {
        **{k: mt.get(k) for k in ("cagr", "vol", "sharpe", "max_dd", "worst_day", "avg_gross", "days")},
        "t_stat": float(ex.mean() / ex.std() * np.sqrt(len(ex))) if ex.std() > 0 else float("nan"),
        "total_return": float((1.0 + w).prod() - 1.0),
        "positions": int(len(t)),
        "mean_net_per_position": float(t["net"].mean()) if len(t) else None,
        "mean_gross_per_position": float(t["gross"].mean()) if len(t) else None,
        "median_net_per_position": float(t["net"].median()) if len(t) else None,
        "share_positive": float((t["net"] > 0).mean()) if len(t) else None,
        "worst_position": float(t["net"].min()) if len(t) else None,
        "stopped": int(t["stopped"].sum()) if len(t) else 0,
        "t_by_entry_month": float(by_month.mean() / by_month.std() * np.sqrt(len(by_month))) if len(by_month) > 2 else None,
        "avg_open_positions": float(eng.window(sim["n_open"], start, end).mean()),
    }


# ------------------------------------------------------------------------ stages


def design() -> dict:
    m = data_mod.build().truncate(DESIGN_CUTOFF)
    assert m.close.index.max() <= pd.Timestamp(DESIGN_CUTOFF, tz="UTC")
    ev = load_events(DESIGN_CUTOFF)
    report: dict = {"windows": {"fit": FIT, "validation": VAL}, "events": int(len(ev)), "variants": {}}
    for name, (k_in, k_out) in WINDOWS.items():
        sim = simulate(m, ev, k_in, k_out)
        fit, val = port_metrics(sim, *FIT), port_metrics(sim, *VAL)
        checks = {
            f"fit Sharpe above {SHORTLIST_SHARPE}": (fit["sharpe"] or -9) > SHORTLIST_SHARPE,
            f"validation Sharpe above {SHORTLIST_SHARPE}": (val["sharpe"] or -9) > SHORTLIST_SHARPE,
            f"drawdown under {SHORTLIST_DD:.0%} in both": max(fit["max_dd"], val["max_dd"]) < SHORTLIST_DD,
            "mean net per position positive in both": (fit["mean_net_per_position"] or -1) > 0 and (val["mean_net_per_position"] or -1) > 0,
        }
        report["variants"][name] = {"enter_days_before": k_in, "exit_days_after": k_out, "fit": fit, "validation": val,
                                    "btc_hedge": {"fit": port_metrics(simulate(m, ev, k_in, k_out, "btc"), *FIT),
                                                  "validation": port_metrics(simulate(m, ev, k_in, k_out, "btc"), *VAL)},
                                    "checks": checks, "qualifies": all(checks.values())}
    ok = [n for n, v in report["variants"].items() if v["qualifies"]]
    pick = max(ok, key=lambda n: report["variants"][n]["validation"]["sharpe"]) if ok else None
    report["shortlist"] = pick
    (RESULTS / "unlock_design.json").write_text(json.dumps(report, indent=1, default=float, sort_keys=True))
    (RESULTS / "unlock_shortlist.json").write_text(json.dumps({"variant": pick}, indent=1))
    _print(report["variants"], ("fit", "validation"))
    print("shortlist:", pick)
    return report


def exam() -> dict:
    pick = json.loads((RESULTS / "unlock_shortlist.json").read_text())["variant"]
    if pick is None:
        print("Shortlist is empty: nothing may touch the exam window.")
        return {}
    m = data_mod.build()
    ev = load_events(EXAM[1])
    k_in, k_out = WINDOWS[pick]
    sim = simulate(m, ev, k_in, k_out)
    mt = port_metrics(sim, *EXAM)
    years = {int(y): float((1.0 + g).prod() - 1.0) for y, g in eng.window(sim["returns"], *EXAM).groupby(lambda d: d.year)}
    gates = {
        f"t-statistic of daily excess return at least {EXAM_T}": mt["t_stat"] >= EXAM_T,
        "return at least 10% a year": mt["cagr"] >= 0.10,
        "drawdown at most 20%": mt["max_dd"] <= 0.20,
        "2025 and 2026 both positive": years.get(2025, 0) > 0 and years.get(2026, 0) > 0,
        "mean net per position positive": (mt["mean_net_per_position"] or -1) > 0,
    }
    report = {"variant": pick, "window": EXAM, "exam": mt, "years": years, "gates": gates, "pass": all(gates.values()),
              "btc_hedge": port_metrics(simulate(m, ev, k_in, k_out, "btc"), *EXAM),
              "no_hedge": port_metrics(simulate(m, ev, k_in, k_out, "none"), *EXAM),
              "daily_returns_sha256": hashlib.sha256(sim["returns"].round(12).to_csv().encode()).hexdigest()}
    (RESULTS / "unlock_exam.json").write_text(json.dumps(report, indent=1, default=float, sort_keys=True))
    t = sim["trades"]
    t[(t["entry"] >= EXAM[0])].to_csv(RESULTS / "unlock_exam_trades.csv", index=False)
    _print({pick: {"exam": mt}}, ("exam",))
    print("years", years, "| gates", gates, "| PASS" if report["pass"] else "| FAIL")
    print("BTC hedge:", {k: report["btc_hedge"][k] for k in ("cagr", "sharpe", "max_dd", "mean_net_per_position")})
    print("no hedge:", {k: report["no_hedge"][k] for k in ("cagr", "sharpe", "max_dd", "mean_net_per_position")})
    print("daily returns sha256:", report["daily_returns_sha256"])
    return report


def _print(variants: dict, periods: tuple[str, ...]) -> None:
    for name, v in variants.items():
        for p in periods:
            x = v[p]
            print(f"{name} {p:10} Sharpe {x['sharpe']:.2f} t {x['t_stat']:.2f} return/yr {x['cagr'] * 100:6.1f}% DD {x['max_dd'] * 100:5.1f}% "
                  f"positions {x['positions']:4d} mean net {(x['mean_net_per_position'] or 0) * 1e4:7.0f} bp gross {(x['mean_gross_per_position'] or 0) * 1e4:7.0f} bp "
                  f"win {(x['share_positive'] or 0) * 100:4.0f}% worst {(x['worst_position'] or 0) * 100:6.1f}% stopped {x['stopped']} "
                  f"open {x['avg_open_positions']:.1f} month-t {x['t_by_entry_month'] if x['t_by_entry_month'] is None else round(x['t_by_entry_month'], 2)}")
        if "checks" in v:
            print("   qualifies:", v["qualifies"], [k for k, ok in v["checks"].items() if not ok])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["events", "design", "exam"])
    ap.add_argument("--raw", type=Path, help="directory of DefiLlama emission JSON files (events mode)")
    ap.add_argument("--protocols", type=Path, help="saved https://api.llama.fi/protocols (events mode)")
    ap.add_argument("--chains", type=Path, help="saved https://api.llama.fi/v2/chains (events mode)")
    args = ap.parse_args()
    if args.mode == "events":
        build_events(args.raw, args.protocols, args.chains)
    elif args.mode == "design":
        design()
    else:
        exam()
