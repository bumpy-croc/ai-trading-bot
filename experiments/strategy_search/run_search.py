"""Runner for the wide strategy search
(docs/research/experiments/2026-11-10_strategy-search-prereg.md).

  python experiments/strategy_search/run_search.py discover   # data truncated at 2024-12-31
  python experiments/strategy_search/run_search.py holdout    # reads the committed shortlist

`discover` applies the fit -> validation selection rules and writes results/discovery.json and
results/shortlist.json. `holdout` runs only what the shortlist names and writes results/holdout.json.
"""

from __future__ import annotations

import argparse
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
FIT = ("2020-07-01", "2022-12-31")
VAL = ("2023-01-01", "2024-12-31")
FITVAL = (FIT[0], VAL[1])
HOLDOUT = ("2025-01-01", "2026-10-31")
DISCOVERY_CUTOFF = "2024-12-31"
MAX_SHORTLIST = 5
MIX_REBALANCE_COST = 0.0015


def run_variant(m: data_mod.Market, rank: pd.DataFrame, univ: pd.DataFrame, spec: tuple, lag: int = 1) -> eng.Result:
    if spec[0] == "cash_and_carry":
        return strat.cash_and_carry(m, univ, rank, spec[1], lag=lag)
    targets = strat.build_targets(m, univ, spec)
    return eng.vol_targeted(m.close, m.funding, rank, targets, lag=lag)


def total_return(res: eng.Result, start: str, end: str) -> float:
    return float((1.0 + eng.window(res.returns, start, end)).prod() - 1.0)


def benchmarks(m: data_mod.Market, univ: pd.DataFrame) -> dict[str, pd.Series]:
    r = m.close.pct_change(fill_method=None)
    w = univ.shift(1, fill_value=False)
    ew = (r.where(w).mean(axis=1)).fillna(0.0)
    return {"BTC buy&hold": r["BTCUSDT"].fillna(0.0), "Top-30 equal weight": ew}


def series_metrics(r: pd.Series, start: str, end: str) -> dict:
    dummy = eng.Result(r, r, r * 0, r * 0, r * 0, r * 0 + 1, r * 0 + 1)
    return eng.metrics(dummy, start, end)


def build_mix(results: dict[str, eng.Result], weights: dict[str, float]) -> eng.Result:
    """Fixed-weight combination of sleeves, rescaled weekly to the vol target."""
    names = list(weights)
    r = sum(results[n].returns * weights[n] for n in names)
    g = sum(results[n].gross * weights[n] for n in names)
    vol = r.rolling(eng.VOL_LOOKBACK, min_periods=eng.VOL_LOOKBACK).std() * np.sqrt(eng.ANN)
    raw = (eng.VOL_TARGET / vol.replace(0.0, np.nan)).fillna(0.0)
    raw = np.minimum(raw, eng.MAX_GROSS / g.replace(0.0, np.nan)).fillna(0.0)
    mondays = eng.weekly_days(r.index)
    # scale decided at Monday's close, in force from the close of the next day (same delay as the sleeves)
    scale = raw.where(r.index.isin(mondays)).ffill().shift(2).fillna(0.0)
    cost = scale.diff().abs().fillna(0.0) * g * MIX_REBALANCE_COST
    net = (1.0 + scale * r) * (1.0 - cost) - 1.0
    costs = sum(results[n].costs * weights[n] for n in names) * scale + cost
    fund = sum(results[n].funding * weights[n] for n in names) * scale
    turn = sum(results[n].turnover * weights[n] for n in names) * scale
    nets = sum(results[n].net * weights[n] for n in names) * scale
    return eng.Result(net, scale * r, costs, fund, turn, g * scale, nets)


def discover() -> None:
    full = data_mod.build()
    m = full.truncate(DISCOVERY_CUTOFF)
    assert m.close.index.max() <= pd.Timestamp(DISCOVERY_CUTOFF, tz="UTC")
    rank, univ = strat.prepare(m)
    report: dict = {"families": {}, "windows": {"fit": FIT, "validation": VAL}}
    results: dict[str, eng.Result] = {}
    for fam, variants in strat.FAMILIES.items():
        fam_rep: dict = {"variants": {}}
        for name, spec in variants.items():
            res = run_variant(m, rank, univ, spec)
            same_close = run_variant(m, rank, univ, spec, lag=0)
            results[name] = res
            fam_rep["variants"][name] = {
                "fit": eng.metrics(res, *FIT),
                "validation": eng.metrics(res, *VAL),
                "fit_plus_validation_return": total_return(res, *FITVAL),
                "years": eng.yearly(res),
                "same_close_fill_sharpe": {
                    "fit": eng.metrics(same_close, *FIT).get("sharpe"),
                    "validation": eng.metrics(same_close, *VAL).get("sharpe"),
                },
            }
        v = fam_rep["variants"]
        rep = max(v, key=lambda n: v[n]["fit"]["sharpe"])
        checks = {
            "fit Sharpe > 0.7": v[rep]["fit"]["sharpe"] > 0.7,
            "validation Sharpe > 0.7": v[rep]["validation"]["sharpe"] > 0.7,
            "validation max DD < 25%": v[rep]["validation"]["max_dd"] < 0.25,
            "all variants positive over fit+validation": all(x["fit_plus_validation_return"] > 0 for x in v.values()),
        }
        fam_rep.update({"representative": rep, "checks": checks, "shortlisted": all(checks.values())})
        report["families"][fam] = fam_rep

    passed = [f for f, r in report["families"].items() if r["shortlisted"]]
    passed.sort(key=lambda f: -report["families"][f]["variants"][report["families"][f]["representative"]]["validation"]["sharpe"])
    passed = passed[:MAX_SHORTLIST]
    reps = [report["families"][f]["representative"] for f in passed]
    mix_weights: dict[str, float] = {}
    if len(reps) >= 2:
        inv = {n: 1.0 / eng.metrics(results[n], *FITVAL)["vol"] for n in reps}
        tot = sum(inv.values())
        mix_weights = {n: inv[n] / tot for n in reps}
        mix = build_mix(results, mix_weights)
        report["mix"] = {"weights": mix_weights, "fit": eng.metrics(mix, *FIT), "validation": eng.metrics(mix, *VAL),
                         "years": eng.yearly(mix)}
        rets = pd.DataFrame({n: eng.window(results[n].returns, *FITVAL) for n in reps})
        report["mix"]["sleeve_correlations"] = rets.corr().round(3).to_dict()

    report["benchmarks"] = {
        k: {"fit": series_metrics(s, *FIT), "validation": series_metrics(s, *VAL)} for k, s in benchmarks(m, univ).items()
    }
    btc = m.close["BTCUSDT"].pct_change(fill_method=None)
    report["btc_correlation_fitval"] = {
        n: float(eng.window(r.returns, *FITVAL).corr(eng.window(btc, *FITVAL))) for n, r in results.items()
    }
    report["universe"] = {
        "symbols_in_archive": int(m.close.shape[1]),
        "universe_size_by_year": {int(y): float(s.mean()) for y, s in univ.sum(axis=1).groupby(univ.index.year)},
        "distinct_universe_members": int(univ.any().sum()),
        "funding_missing_share_in_universe": float((m.funding.isna() & univ).sum().sum() / max(1, univ.sum().sum())),
    }
    shortlist = {"candidates": reps, "families": passed, "mix_weights": mix_weights}

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "discovery.json").write_text(json.dumps(report, indent=1, default=float, sort_keys=True))
    (RESULTS / "shortlist.json").write_text(json.dumps(shortlist, indent=1, sort_keys=True))
    print_discovery(report, shortlist)


def print_discovery(report: dict, shortlist: dict) -> None:
    pct = lambda x: f"{x * 100:6.1f}%"  # noqa: E731
    print(f"{'variant':20} {'fitShp':>7} {'fitCAGR':>8} {'fitDD':>7} | {'valShp':>7} {'valCAGR':>8} {'valDD':>7} | "
          f"{'cost/y':>7} {'fund/y':>7} {'turn/y':>7} {'gross':>6} | same-close fit/val")
    for fam, fr in report["families"].items():
        for name, v in fr["variants"].items():
            f, va, sc = v["fit"], v["validation"], v["same_close_fill_sharpe"]
            star = "*" if name == fr["representative"] else " "
            print(f"{star}{name:19} {f['sharpe']:7.2f} {pct(f['cagr'])}  {pct(f['max_dd'])} | {va['sharpe']:7.2f} "
                  f"{pct(va['cagr'])}  {pct(va['max_dd'])} | {pct(va['cost_per_year'])} {pct(va['funding_per_year'])} "
                  f"{va['turnover_per_year']:6.1f}x {va['avg_gross']:6.2f} | {sc['fit']:5.2f} / {sc['validation']:5.2f}")
        print(f"   -> {fam}: {'SHORTLISTED' if fr['shortlisted'] else 'dropped'}  "
              + ", ".join(f"{k}: {'ok' if ok else 'FAIL'}" for k, ok in fr["checks"].items()))
    for k, b in report["benchmarks"].items():
        print(f" {k:19} {b['fit']['sharpe']:7.2f} {pct(b['fit']['cagr'])}  {pct(b['fit']['max_dd'])} | "
              f"{b['validation']['sharpe']:7.2f} {pct(b['validation']['cagr'])}  {pct(b['validation']['max_dd'])}")
    print("shortlist:", json.dumps(shortlist))
    if "mix" in report:
        mx = report["mix"]
        print(f" MIX fit Sharpe {mx['fit']['sharpe']:.2f} CAGR {pct(mx['fit']['cagr'])} DD {pct(mx['fit']['max_dd'])} | "
              f"val Sharpe {mx['validation']['sharpe']:.2f} CAGR {pct(mx['validation']['cagr'])} DD {pct(mx['validation']['max_dd'])}")
    print("universe:", json.dumps(report["universe"]))


def holdout() -> None:
    shortlist = json.loads((RESULTS / "shortlist.json").read_text())
    if not shortlist["candidates"]:
        print("Shortlist is empty: nothing may touch the holdout.")
        return
    m = data_mod.build()
    rank, univ = strat.prepare(m)
    specs = {n: s for fam in strat.FAMILIES.values() for n, s in fam.items()}
    results = {n: run_variant(m, rank, univ, specs[n]) for n in shortlist["candidates"]}
    same_close = {n: run_variant(m, rank, univ, specs[n], lag=0) for n in shortlist["candidates"]}
    if shortlist["mix_weights"]:
        results["MIX"] = build_mix(results, shortlist["mix_weights"])
    btc = m.close["BTCUSDT"].pct_change(fill_method=None)
    report: dict = {"window": HOLDOUT, "candidates": {}}
    for name, res in results.items():
        mt = eng.metrics(res, *HOLDOUT)
        yrs = eng.yearly(res)
        is_cc = name.startswith("CC_")
        active = float((eng.window(res.gross, *HOLDOUT) > 0).mean())
        gates = {
            "G1 excess Sharpe >= 1.0": mt["sharpe"] >= 1.0,
            "G2 CAGR >= floor": mt["cagr"] >= (0.08 if is_cc else 0.10),
            "G3 max DD <= cap": mt["max_dd"] <= (0.10 if is_cc else 0.20),
            "G4 2025 and 2026 both positive": yrs.get(2025, 0) > 0 and yrs.get(2026, 0) > 0,
            "G5 activity floor": active >= (0.5 if is_cc else 0.8),
        }
        report["candidates"][name] = {
            "holdout": mt,
            "fit": eng.metrics(res, *FIT),
            "validation": eng.metrics(res, *VAL),
            "years": yrs,
            "active_share": active,
            "btc_correlation": float(eng.window(res.returns, *HOLDOUT).corr(eng.window(btc, *HOLDOUT))),
            "same_close_fill_sharpe": eng.metrics(same_close[name], *HOLDOUT).get("sharpe") if name in same_close else None,
            "gates": gates,
            "pass": all(gates.values()),
            "strong_pass": all(gates.values()) and mt["sharpe"] >= 1.8,
        }
    report["benchmarks"] = {k: series_metrics(s, *HOLDOUT) for k, s in benchmarks(m, univ).items()}
    (RESULTS / "holdout.json").write_text(json.dumps(report, indent=1, default=float, sort_keys=True))
    pd.DataFrame({n: r.returns for n, r in results.items()}).to_csv(RESULTS / "holdout_daily_returns.csv")
    pct = lambda x: f"{x * 100:6.1f}%"  # noqa: E731
    for name, c in report["candidates"].items():
        h = c["holdout"]
        print(f"{name:20} Sharpe {h['sharpe']:5.2f} CAGR {pct(h['cagr'])} DD {pct(h['max_dd'])} worst day {pct(h['worst_day'])} "
              f"2025 {pct(c['years'].get(2025, 0))} 2026 {pct(c['years'].get(2026, 0))} corrBTC {c['btc_correlation']:5.2f} "
              f"-> {'STRONG PASS' if c['strong_pass'] else 'PASS' if c['pass'] else 'FAIL'} "
              + ", ".join(k.split()[0] for k, ok in c["gates"].items() if not ok))
    for k, b in report["benchmarks"].items():
        print(f"{k:20} Sharpe {b['sharpe']:5.2f} CAGR {pct(b['cagr'])} DD {pct(b['max_dd'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["discover", "holdout"])
    args = ap.parse_args()
    discover() if args.mode == "discover" else holdout()
