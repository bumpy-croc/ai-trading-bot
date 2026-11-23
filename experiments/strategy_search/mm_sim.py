"""Naive market maker on BTCUSDT perp trades: quote both sides around the last trade, refresh on a clock.

Context: docs/research/experiments/2026-11-23_naive-market-maker-prereg.md. Every rule here is
fixed there. Data comes from mm_data.py.

The bot reads the last trade price every T ms. Its orders go live L ms later and rest until the
next pair replaces them: bid = price * (1 - d) rounded down a tick, ask = price * (1 + d) rounded
up. Conservative fills need a trade strictly through the order's price; optimistic fills also
count an aggressor trade exactly at it. Order size and the position limit are fixed dollar
amounts, so fees can be applied afterwards to the dollars traded.

Needs numba. Usage:
  python mm_sim.py selftest
  python mm_sim.py run --dir /path/used/by/mm_data [--tag run1]
  python mm_sim.py report
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

RESULTS = Path(__file__).resolve().parent / "results"
CHUNKS = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
          "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10", "2026-11"]
WINDOW = ("2025-11-01", "2026-11-01")  # [start, end)
HALF = "2026-05-01"
TICK = 0.1
CAPITAL, ORDER_USD, CAP_USD = 10_000.0, 1_000.0, 5_000.0
FEE = 0.0002  # Binance USDT-M maker, standard tier
IDLE = 0.04
LATENCY_MS = 50
CLOCKS_MS = (100, 1_000, 15_000)
DISTANCES_BP = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0)
VOL_K = (0.25, 0.5, 1.0, 2.0)
VOL_FLOOR_BP = 0.25
VOL_WINDOW_MIN = 60
PRIMARY = "d5_T1000"
OVERLAP_MS = 60_000
HOUR_MS, MIN_MS = 3_600_000, 60_000
# state columns
CASH, INV, NOTIONAL, FILLS, SLOT, BID, ASK = range(7)


def configs() -> list[dict]:
    """The 25 preregistered cells with both fill rules, plus the latency sensitivity of the primary cell."""
    cells = [{"name": f"d{d:g}_T{t}", "d_bp": d, "k": 0.0, "T": t, "L": LATENCY_MS} for t in CLOCKS_MS for d in DISTANCES_BP]
    cells += [{"name": f"vol{k:g}_T1000", "d_bp": 0.0, "k": k, "T": 1_000, "L": LATENCY_MS} for k in VOL_K]
    out = [{**c, "touch": touch, "key": f"{c['name']}|{'touch' if touch else 'through'}"} for c in cells for touch in (False, True)]
    for lat in (5, 250):
        out.append({"name": f"{PRIMARY}_L{lat}", "d_bp": 5.0, "k": 0.0, "T": 1_000, "L": lat, "touch": False,
                    "key": f"{PRIMARY}_L{lat}|through"})
    return out


@njit(parallel=True, cache=True)
def run_chunk(ts, px, bm, start, sig, sig_min0, cT, cL, cd, ck, ctouch, state, marks, hour0, order_usd, cap_usd, tick, floor_d):
    """Advance every config over trades[start:]. trades[:start] are the previous chunk's tail (for the reference price)."""
    n_trades = ts.shape[0]
    for c in prange(cT.shape[0]):
        T, L, d_fixed, k, touch = cT[c], cL[c], cd[c], ck[c], ctouch[c]
        cash, inv, notional, fills = state[c, CASH], state[c, INV], state[c, NOTIONAL], state[c, FILLS]
        slot, bid, ask = np.int64(state[c, SLOT]), np.int64(state[c, BID]), np.int64(state[c, ASK])
        j = 0
        cur_h = ts[start] // HOUR_MS
        for i in range(start, n_trades):
            t = ts[i]
            h = t // HOUR_MS
            if h != cur_h:
                last = px[i - 1] * tick
                marks[c, cur_h - hour0, 0] = cash + inv * last
                marks[c, cur_h - hour0, 1] = notional
                marks[c, cur_h - hour0, 2] = inv * last
                marks[c, cur_h - hour0, 3] = fills
                cur_h = h
            n = (t - L) // T
            if n != slot:
                slot = n
                grid = n * T
                while j + 1 < n_trades and ts[j + 1] <= grid:
                    j += 1
                bid, ask = -1, -1
                if ts[j] <= grid and i > 0:
                    ref = px[j]
                    d = d_fixed
                    if k > 0.0:
                        s = sig[grid // MIN_MS - sig_min0]
                        d = max(k * s, floor_d) if s == s else -1.0
                    if d > 0.0:
                        prev = px[i - 1]  # the market when the new orders arrive
                        inv_usd = inv * ref * tick
                        b = np.int64(np.floor(ref * (1.0 - d) + 1e-9))
                        a = np.int64(np.ceil(ref * (1.0 + d) - 1e-9))
                        if b < prev and inv_usd + order_usd <= cap_usd + 1e-6:
                            bid = b
                        if a > prev and inv_usd - order_usd >= -cap_usd - 1e-6:
                            ask = a
            p = px[i]
            if bid >= 0 and (p < bid or (touch and p == bid and bm[i])):
                price = bid * tick
                inv += order_usd / price
                cash -= order_usd
                notional += order_usd
                fills += 1.0
                bid = -1
            elif ask >= 0 and (p > ask or (touch and p == ask and not bm[i])):
                price = ask * tick
                inv -= order_usd / price
                cash += order_usd
                notional += order_usd
                fills += 1.0
                ask = -1
        last = px[n_trades - 1] * tick
        marks[c, cur_h - hour0, 0] = cash + inv * last
        marks[c, cur_h - hour0, 1] = notional
        marks[c, cur_h - hour0, 2] = inv * last
        marks[c, cur_h - hour0, 3] = fills
        state[c, CASH], state[c, INV], state[c, NOTIONAL], state[c, FILLS] = cash, inv, notional, fills
        state[c, SLOT], state[c, BID], state[c, ASK] = slot, bid, ask


def minute_sigma(directory: Path, chunks: list[str]) -> tuple[np.ndarray, int]:
    """Std of one-minute log returns over the previous 60 complete minutes, indexed by minute since the first."""
    closes = []
    for chunk in chunks:
        ts = np.load(directory / f"{chunk}_ts.npy", mmap_mode="r")
        px = np.load(directory / f"{chunk}_px.npy", mmap_mode="r")
        minute = np.asarray(ts) // MIN_MS
        last = np.flatnonzero(np.diff(minute, append=minute[-1] + 1))
        closes.append(pd.Series(np.asarray(px)[last].astype(np.float64), index=minute[last]))
    s = pd.concat(closes)
    s = s[~s.index.duplicated(keep="last")]
    full = s.reindex(range(int(s.index[0]), int(s.index[-1]) + 1)).ffill()
    sigma = np.log(full).diff().rolling(VOL_WINDOW_MIN).std().shift(1)  # known at the start of the minute
    return sigma.to_numpy(), int(full.index[0])


def simulate(directory: Path, chunks: list[str]) -> tuple[list[dict], np.ndarray, int]:
    cfgs = configs()
    cT = np.array([c["T"] for c in cfgs], dtype=np.int64)
    cL = np.array([c["L"] for c in cfgs], dtype=np.int64)
    cd = np.array([c["d_bp"] / 1e4 for c in cfgs])
    ck = np.array([c["k"] for c in cfgs])
    ctouch = np.array([c["touch"] for c in cfgs], dtype=np.bool_)
    sig, sig_min0 = minute_sigma(directory, chunks)
    first = np.load(directory / f"{chunks[0]}_ts.npy", mmap_mode="r")
    final = np.load(directory / f"{chunks[-1]}_ts.npy", mmap_mode="r")
    hour0 = int(first[0] // HOUR_MS)
    marks = np.full((len(cfgs), int(final[-1] // HOUR_MS) - hour0 + 1, 4), np.nan)
    state = np.zeros((len(cfgs), 7))
    state[:, SLOT:] = -1
    tail = None
    for chunk in chunks:
        ts, px, bm = (np.load(directory / f"{chunk}_{k}.npy") for k in ("ts", "px", "bm"))
        start = 0
        if tail is not None:
            start = len(tail[0])
            ts, px, bm = (np.concatenate([a, b]) for a, b in zip(tail, (ts, px, bm)))
        run_chunk(ts, px, bm, start, sig, sig_min0, cT, cL, cd, ck, ctouch, state, marks, hour0,
                  ORDER_USD, CAP_USD, TICK, VOL_FLOOR_BP / 1e4)
        keep = np.searchsorted(ts, ts[-1] - OVERLAP_MS)
        tail = (ts[keep:].copy(), px[keep:].copy(), bm[keep:].copy())
        print(f"{chunk}: {len(ts) - start:,} trades done", flush=True)
    return cfgs, marks, hour0


def summarise(cfg: dict, marks: np.ndarray, hour0: int) -> dict:
    """Metrics for one config from its hourly marks (gross P&L, dollars traded, position, fills)."""
    idx = pd.to_datetime((np.arange(marks.shape[0]) + hour0 + 1) * HOUR_MS, unit="ms", utc=True)  # end of each hour
    df = pd.DataFrame(marks, index=idx, columns=["gross", "notional", "inv_usd", "fills"]).ffill().fillna(0.0)
    start, end, half = (pd.Timestamp(x, tz="UTC") for x in (*WINDOW, HALF))
    main = df[(df.index > start) & (df.index <= end)]
    net = main["gross"] - FEE * main["notional"]
    equity = CAPITAL + net
    last = main.iloc[-1]
    daily = equity[equity.index.hour == 0]
    rets = daily.diff().dropna() / CAPITAL
    peak = equity.cummax()
    years = (end - start).days / 365.0
    ruined = equity[equity <= 0]
    at_half = main[main.index <= half].iloc[-1]
    net_half1 = at_half["gross"] - FEE * at_half["notional"]
    after = df[df.index > end]
    fresh = None
    if len(after):
        z = after.iloc[-1]
        fresh = {"days": round(len(after) / 24, 1), "gross_usd": z["gross"] - last["gross"],
                 "net_usd": (z["gross"] - last["gross"]) - FEE * (z["notional"] - last["notional"]),
                 "traded_usd": z["notional"] - last["notional"]}
    traded = last["notional"]
    return {
        "key": cfg["key"], "name": cfg["name"], "fill_rule": "touch" if cfg["touch"] else "through",
        "d_bp": cfg["d_bp"], "vol_k": cfg["k"], "refresh_ms": cfg["T"], "latency_ms": cfg["L"],
        "fills": int(last["fills"]), "traded_usd": traded,
        "gross_usd": last["gross"], "fees_usd": FEE * traded, "net_usd": net.iloc[-1],
        "net_return": net.iloc[-1] / CAPITAL, "net_return_per_year": net.iloc[-1] / CAPITAL / years,
        "gross_bp_per_usd_traded": last["gross"] / traded * 1e4 if traded else None,
        "net_bp_per_usd_traded": net.iloc[-1] / traded * 1e4 if traded else None,
        "break_even_fee_bp": max(last["gross"] / traded * 1e4, 0.0) if traded else None,
        "max_drawdown": float(((peak - equity) / peak).max()),
        "sharpe_daily_excess": float((rets.mean() - IDLE / 365) / rets.std() * np.sqrt(365)) if rets.std() > 0 else None,
        "net_usd_first_half": net_half1, "net_usd_second_half": net.iloc[-1] - net_half1,
        "account_at_zero_on": f"{ruined.index[0]:%Y-%m-%d}" if len(ruined) else None,
        "mean_abs_position_usd": float(main["inv_usd"].abs().mean()),
        "worst_day": float(rets.min()) if len(rets) else None,
        "fresh_slice": fresh,
        "passes": bool(net.iloc[-1] / CAPITAL > IDLE * years and ((peak - equity) / peak).max() <= 0.20
                       and net_half1 > 0 and net.iloc[-1] - net_half1 > 0 and not len(ruined)),
    }


def run(directory: Path, tag: str) -> None:
    chunks = [c for c in CHUNKS if (directory / "npy" / f"{c}_ts.npy").exists()]
    missing = [c for c in CHUNKS[:-1] if c not in chunks]
    if missing:
        raise SystemExit(f"missing chunks: {missing}")
    cfgs, marks, hour0 = simulate(directory / "npy", chunks)
    cells = [summarise(c, marks[i], hour0) for i, c in enumerate(cfgs)]
    primary = [c for c in cells if c["fill_rule"] == "through" and c["latency_ms"] == LATENCY_MS]
    n_pass = sum(c["passes"] for c in primary)
    primary_pass = next(c["passes"] for c in primary if c["name"] == PRIMARY)
    report = {
        "window": WINDOW, "chunks": chunks, "capital": CAPITAL, "order_usd": ORDER_USD, "cap_usd": CAP_USD, "fee": FEE,
        "cells_passing": n_pass, "primary_cell_passes": primary_pass,
        "verdict": "H1 supported" if primary_pass or n_pass >= 5 else "H0 supported",
        "cells": cells,
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"mm_naive{('_' + tag) if tag else ''}.json"
    out.write_text(json.dumps(report, indent=1, default=float, sort_keys=True))
    i = next(k for k, c in enumerate(cfgs) if c["key"] == f"{PRIMARY}|through")
    idx = pd.to_datetime((np.arange(marks.shape[1]) + hour0 + 1) * HOUR_MS, unit="ms", utc=True)
    eq = pd.DataFrame(marks[i], index=idx, columns=["gross_usd", "traded_usd", "position_usd", "fills"]).ffill()
    eq["net_equity"] = CAPITAL + eq["gross_usd"] - FEE * eq["traded_usd"]
    eq[eq.index.hour == 0].round(4).to_csv(RESULTS / f"mm_naive_primary_daily{('_' + tag) if tag else ''}.csv")
    print(f"wrote {out} sha256 {hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")
    show(report)


def show(report: dict) -> None:
    print(f"verdict: {report['verdict']} ({report['cells_passing']} of 25 cells pass; primary passes: {report['primary_cell_passes']})")
    print(f"{'cell':22} {'fills':>10} {'traded $m':>10} {'gross $':>10} {'net $':>11} {'net %':>8} {'gross bp':>9} {'net bp':>7} {'maxDD':>7} {'zero on':>11}")
    for c in report["cells"]:
        gb = c["gross_bp_per_usd_traded"]
        print(f"{c['key']:22} {c['fills']:>10,} {c['traded_usd'] / 1e6:>10.1f} {c['gross_usd']:>10,.0f} {c['net_usd']:>11,.0f} "
              f"{c['net_return'] * 100:>7.1f}% {gb if gb is not None else float('nan'):>9.2f} "
              f"{c['net_bp_per_usd_traded'] if gb is not None else float('nan'):>7.2f} {c['max_drawdown'] * 100:>6.1f}% {c['account_at_zero_on'] or '':>11}")


def selftest() -> None:
    """Hand-checkable tapes: fill timing, one fill per refresh, the two fill rules, the position limit, post-only."""
    def tape(rows, d_bp=10.0, T=1000, L=50, touch=False, cap=CAP_USD):
        ts = np.array([r[0] for r in rows], dtype=np.int64)
        px = np.array([r[1] for r in rows], dtype=np.int32)
        bm = np.array([r[2] for r in rows], dtype=np.bool_)
        state = np.zeros((1, 7)); state[:, SLOT:] = -1
        marks = np.full((1, 2, 4), np.nan)
        run_chunk(ts, px, bm, 0, np.array([np.nan]), 0, np.array([T]), np.array([L]), np.array([d_bp / 1e4]), np.array([0.0]),
                  np.array([touch]), state, marks, 0, ORDER_USD, cap, TICK, VOL_FLOOR_BP / 1e4)
        return state[0]

    base = [(10, 10000, True), (500, 10000, False)]  # nothing at or before 0 ms, so no quotes in the first second
    # slot 1 (orders live from 1050 ms) quotes around the 1000.0 seen at 1000 ms: bid 999.0, ask 1001.0
    s = tape(base + [(1100, 9989, True), (1200, 9980, True)])
    assert s[FILLS] == 1 and abs(s[INV] - 1000 / 999.0) < 1e-12 and s[CASH] == -1000, "one buy at 999.0, once per refresh"
    s = tape(base + [(1040, 9989, True), (1100, 9995, True)])
    assert s[FILLS] == 0, "no fill before the orders are live or inside the quotes"
    s = tape(base + [(1100, 9990, True)])
    assert s[FILLS] == 0, "conservative rule: a trade at the bid is not a fill"
    s = tape(base + [(1100, 9990, True)], touch=True)
    assert s[FILLS] == 1, "optimistic rule: a seller-initiated trade at the bid is a fill"
    s = tape(base + [(1100, 9990, False)], touch=True)
    assert s[FILLS] == 0, "optimistic rule needs the aggressor on the right side"
    # buy at 999.0, then slot 2 quotes around 998.0 (ask 999.0 rounded up from 998.998): sell when a trade prints above it
    s = tape(base + [(1100, 9989, True), (1900, 9980, True), (2100, 9991, False)])
    assert s[FILLS] == 2 and abs(s[INV]) < 1e-12 and abs(s[CASH]) < 1e-9, "round trip at the same price earns nothing before fees"
    s = tape(base + [(1100, 9989, True), (2100, 9970, True), (3100, 9950, True)], cap=1000.0)
    assert s[FILLS] == 1, "no second buy when the position limit is reached"
    s = tape(base + [(1020, 9985, True), (1100, 9980, True)])
    assert s[FILLS] == 0, "post-only: the bid is rejected when the market is already below it on arrival"
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["selftest", "run", "report"])
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.mode == "selftest":
        selftest()
    elif args.mode == "run":
        run(args.dir, args.tag)
    else:
        show(json.loads((RESULTS / "mm_naive.json").read_text()))
