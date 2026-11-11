"""EXPLORATORY, post-hoc. Not part of the preregistered gates.

Run after XC_3 failed the holdout on one short squeeze (MYX, September 2025). It asks two
diagnostic questions and cannot turn the FAIL into a pass, because every choice here was made
after the holdout was read:

  1. Is the funding-carry premium still there in 2025-26 once the squeeze days are set aside?
  2. How much of the damage would a per-position loss cap have removed? On Binance that cap is
     what isolated margin does: a short carrying margin equal to `cap` x its notional is
     liquidated when the price has risen about `cap` from entry, and the loss stops there.

The capped simulator checks each short against the day's HIGH relative to its price at the
last rebalance. When the cap is hit, the position is closed at the cap level plus an extra
slippage haircut, and stays closed until the next rebalance.

Usage: python experiments/strategy_search/explore_tail.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as data_mod  # noqa: E402
import engine as eng  # noqa: E402

import strategies as strat  # noqa: E402

FIT = ("2020-07-01", "2022-12-31")
VAL = ("2023-01-01", "2024-12-31")
HOLDOUT = ("2025-01-01", "2026-10-31")
STOP_SLIPPAGE = 0.10  # extra loss, as a fraction of the cap move, for getting out in a squeeze


def simulate_capped(m, rank, targets, cap: float, lag: int = 1) -> eng.Result:
    idx = m.close.index
    close = m.close.to_numpy()
    high = m.high.to_numpy()
    rets = m.close.pct_change(fill_method=None).to_numpy()
    alive = m.close.notna().to_numpy()
    fund = m.funding.to_numpy()
    tg = targets.shift(lag).to_numpy()
    cost_rate = eng.PERP_FEE + eng.slippage_for_rank(rank.shift(lag).fillna(10**6).to_numpy())
    n_days, n_assets = rets.shape
    w = np.zeros(n_assets)
    ref = np.full(n_assets, np.nan)  # price at the last rebalance, the cap's reference
    out = {k: np.zeros(n_days) for k in ("ret", "gret", "cost", "fund", "turn", "gross", "net", "stops")}
    for d in range(n_days):
        held = w != 0
        cost = fund_paid = port = 0.0
        if held.any():
            r = np.where(np.isnan(rets[d]), 0.0, rets[d])
            prev = close[d - 1]
            # shorts whose intraday high breached the cap are closed there
            hit = (w < 0) & (high[d] >= ref * (1.0 + cap))
            if hit.any():
                stop_px = ref * (1.0 + cap * (1.0 + STOP_SLIPPAGE))
                # a gap through the level is filled no better than the day's open-to-high path allows
                stop_px = np.minimum(np.maximum(stop_px, prev), high[d])
                r = np.where(hit, stop_px / prev - 1.0, r)
            f = fund[d]
            f_eff = np.where(np.isnan(f), np.where(w > 0, eng.MISSING_FUNDING_LONG, 0.0), f)
            port = float((w * r).sum())
            fund_paid = float((w * f_eff)[held].sum())
            w = w * (1.0 + r) / (1.0 + port - fund_paid)
            gone = (held & ~alive[d]) | hit
            if gone.any():
                cost += float((np.abs(w[gone]) * cost_rate[d][gone]).sum())
                w = np.where(gone, 0.0, w)
                out["stops"][d] = int(hit.sum())
        turn = 0.0
        row = tg[d]
        if not np.isnan(row).all():
            tgt = np.where(np.isnan(row) | ~alive[d], 0.0, row)
            delta = np.abs(tgt - w)
            turn = float(delta.sum())
            cost += float((delta * cost_rate[d]).sum())
            w = tgt
            ref = np.where(w != 0, close[d], np.nan)
        out["gret"][d], out["fund"][d], out["cost"][d] = port, fund_paid, cost
        out["ret"][d] = (1.0 + port - fund_paid) * (1.0 - cost) - 1.0
        out["turn"][d], out["gross"][d], out["net"][d] = turn, np.abs(w).sum(), w.sum()
    s = lambda a: pd.Series(a, index=idx)  # noqa: E731
    res = eng.Result(s(out["ret"]), s(out["gret"]), s(out["cost"]), s(out["fund"]), s(out["turn"]), s(out["gross"]), s(out["net"]))
    res.extra["stops"] = s(out["stops"])
    return res


def vol_targeted_capped(m, rank, targets, cap: float) -> eng.Result:
    base = simulate_capped(m, rank, targets, cap)
    vol = base.returns.rolling(eng.VOL_LOOKBACK, min_periods=eng.VOL_LOOKBACK).std() * np.sqrt(eng.ANN)
    lev = (eng.VOL_TARGET / vol.replace(0.0, np.nan)).fillna(0.0)
    gross = targets.abs().sum(axis=1).replace(0.0, np.nan)
    lev = np.minimum(lev, eng.MAX_GROSS / gross).fillna(0.0)
    out = simulate_capped(m, rank, targets.mul(lev, axis=0), cap)
    out.extra["unlevered"] = base
    return out


def row(name: str, res: eng.Result) -> dict:
    out = {"name": name}
    for label, win in (("fit", FIT), ("val", VAL), ("holdout", HOLDOUT)):
        mt = eng.metrics(res, *win)
        out[label] = {k: mt[k] for k in ("sharpe", "cagr", "max_dd", "worst_day", "funding_per_year", "cost_per_year", "avg_gross")}
    out["years"] = eng.yearly(res)
    if "stops" in res.extra:
        out["stops_by_year"] = {int(y): int(v) for y, v in res.extra["stops"].groupby(res.extra["stops"].index.year).sum().items()}
    return out


def main() -> None:
    m = data_mod.build()
    rank, univ = strat.prepare(m)
    targets = strat.build_targets(m, univ, strat.FAMILIES["XC"]["XC_3"])
    base = eng.vol_targeted(m.close, m.funding, rank, targets)
    rows = [row("XC_3 as preregistered (no cap)", base)]

    # 1. premium with the squeeze set aside: unlevered sleeve, drop the three MYX days
    unlev = base.extra["unlevered"]
    r = eng.window(unlev.returns, *HOLDOUT)
    squeeze = pd.to_datetime(["2025-09-07", "2025-09-08", "2025-09-09"], utc=True)
    ex = r.drop(squeeze)
    diag = {
        "unlevered_holdout_sharpe_rf0": float(r.mean() / r.std() * np.sqrt(eng.ANN)),
        "unlevered_holdout_sharpe_rf0_ex_squeeze_days": float(ex.mean() / ex.std() * np.sqrt(eng.ANN)),
        "unlevered_holdout_total_return": float((1 + r).prod() - 1),
        "unlevered_holdout_total_return_ex_squeeze_days": float((1 + ex).prod() - 1),
        "unlevered_funding_collected_per_year": {
            "fit": -eng.metrics(unlev, *FIT)["funding_per_year"],
            "val": -eng.metrics(unlev, *VAL)["funding_per_year"],
            "holdout": -eng.metrics(unlev, *HOLDOUT)["funding_per_year"],
        },
        "unlevered_avg_gross": eng.metrics(unlev, *HOLDOUT)["avg_gross"],
    }

    # 2. per-position loss caps on shorts
    for cap in (1.0, 0.5, 0.25):
        rows.append(row(f"XC_3 with shorts capped at +{int(cap * 100)}% from rebalance price", vol_targeted_capped(m, rank, targets, cap)))

    out = {"WARNING": "exploratory and post-hoc; not a gate result", "diagnostics": diag, "rows": rows}
    path = Path(__file__).resolve().parent / "results" / "explore_tail.json"
    path.write_text(json.dumps(out, indent=1, default=float, sort_keys=True))
    pct = lambda x: f"{x * 100:6.1f}%"  # noqa: E731
    print(json.dumps(diag, indent=1, default=float))
    for rw in rows:
        print(rw["name"])
        for label in ("fit", "val", "holdout"):
            x = rw[label]
            print(f"   {label:8} Sharpe {x['sharpe']:5.2f} CAGR {pct(x['cagr'])} DD {pct(x['max_dd'])} worst day {pct(x['worst_day'])} "
                  f"funding {pct(-x['funding_per_year'])} cost {pct(x['cost_per_year'])} gross {x['avg_gross']:.2f}")
        print("   years:", {y: round(v * 100, 1) for y, v in rw["years"].items() if y >= 2020}, rw.get("stops_by_year", ""))


if __name__ == "__main__":
    main()
