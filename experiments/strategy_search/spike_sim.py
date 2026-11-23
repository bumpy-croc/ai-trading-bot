"""Spike fade: far resting orders that fill only in sudden jumps, closed after a fixed time.

Context: docs/research/experiments/2026-11-23_spike-fade-prereg.md. Every rule here is fixed there.
Data comes from mm_data.py (one directory of packed arrays per test set).

Quote logic is the market-maker simulator's (mm_sim.py): every T ms read the last trade, orders go
live L ms later, a fill needs a trade strictly through the order. Here a fill opens a position of
the whole account, both orders are pulled, and the position is closed `hold` ms later at the last
trade price. One position at a time.

Needs numba. Usage:
  python spike_sim.py selftest
  python spike_sim.py run --dir /path/used/by/mm_data --set ETHUSDT
  python spike_sim.py report
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
SETS = {  # name: (directory under --dir, distance in bp, role, months)
    "BTCUSDT_prev": ("npy_BTCUSDT_prev", 25.0, "control", ("2024-11", "2025-10")),
    "ETHUSDT": ("npy_ETHUSDT", 43.0, "candidate", ("2025-11", "2026-10")),
    "SOLUSDT": ("npy_SOLUSDT", 50.0, "candidate", ("2025-11", "2026-10")),
    "XRPUSDT": ("npy_XRPUSDT", 55.0, "candidate", ("2025-11", "2026-10")),
}
FRESH = "2026-11"
CLOCK_MS = 100
# (label, latency ms, hold ms); the first is the preregistered rule, the rest are sensitivities
VARIANTS = [("primary", 250, 300_000), ("age50", 50, 300_000), ("age1000", 1_000, 300_000),
            ("hold60", 250, 60_000), ("hold1800", 250, 1_800_000)]
FEE_IN, FEE_OUT, SLIP_OUT = 0.0002, 0.0005, 0.0001
COST = FEE_IN + FEE_OUT + SLIP_OUT
COST_MAKER_EXIT = 2 * FEE_IN
OVERLAP_MS = 60_000
MAX_TRADES = 200_000
DAY_MS = 86_400_000
# state columns
SLOT, BID, ASK, POS, ENTRY_PX, ENTRY_T, EXIT_T = range(7)


@njit(parallel=True, cache=True)
def run_chunk(ts, px, start, cT, cL, cd, chold, state, out, out_n):
    """Advance every variant over trades[start:]. out[c, k] = (entry time, side, entry ticks, exit ticks)."""
    n_trades = ts.shape[0]
    for c in prange(cT.shape[0]):
        T, L, d, hold = cT[c], cL[c], cd[c], chold[c]
        slot, bid, ask = state[c, SLOT], state[c, BID], state[c, ASK]
        pos, entry_px, entry_t, exit_t = state[c, POS], state[c, ENTRY_PX], state[c, ENTRY_T], state[c, EXIT_T]
        j = 0
        k = 0
        for i in range(start, n_trades):
            t = ts[i]
            if pos != 0:
                if t <= exit_t:
                    continue
                if k < MAX_TRADES:
                    out[c, k, 0], out[c, k, 1], out[c, k, 2], out[c, k, 3] = entry_t, pos, entry_px, px[i - 1]
                    k += 1
                pos = 0
                slot = (t - L) // T  # no orders until the next refresh
                bid, ask = -1, -1
            n = (t - L) // T
            if n != slot:
                slot = n
                grid = n * T
                while j + 1 < n_trades and ts[j + 1] <= grid:
                    j += 1
                bid, ask = -1, -1
                if ts[j] <= grid and i > 0:
                    ref = px[j]
                    prev = px[i - 1]
                    b = np.int64(np.floor(ref * (1.0 - d) + 1e-9))
                    a = np.int64(np.ceil(ref * (1.0 + d) - 1e-9))
                    if b < prev:
                        bid = b
                    if a > prev:
                        ask = a
            p = px[i]
            if bid >= 0 and p < bid:
                pos, entry_px, entry_t, exit_t = 1, bid, t, t + hold
                bid, ask = -1, -1
            elif ask >= 0 and p > ask:
                pos, entry_px, entry_t, exit_t = -1, ask, t, t + hold
                bid, ask = -1, -1
        out_n[c] = k
        state[c, SLOT], state[c, BID], state[c, ASK] = slot, bid, ask
        state[c, POS], state[c, ENTRY_PX], state[c, ENTRY_T], state[c, EXIT_T] = pos, entry_px, entry_t, exit_t


def chunks_of(directory: Path) -> list[str]:
    return sorted(p.name[:7] for p in directory.glob("*_ts.npy"))


def simulate(directory: Path, d_bp: float) -> dict[str, pd.DataFrame]:
    cT = np.full(len(VARIANTS), CLOCK_MS, dtype=np.int64)
    cL = np.array([v[1] for v in VARIANTS], dtype=np.int64)
    chold = np.array([v[2] for v in VARIANTS], dtype=np.int64)
    cd = np.full(len(VARIANTS), d_bp / 1e4)
    state = np.zeros((len(VARIANTS), 7), dtype=np.int64)
    state[:, SLOT:ASK + 1] = -1
    rows: dict[str, list] = {v[0]: [] for v in VARIANTS}
    tail = None
    for chunk in chunks_of(directory):
        ts, px = np.load(directory / f"{chunk}_ts.npy"), np.load(directory / f"{chunk}_px.npy")
        start = 0
        if tail is not None:
            start = len(tail[0])
            ts, px = np.concatenate([tail[0], ts]), np.concatenate([tail[1], px])
        out = np.zeros((len(VARIANTS), MAX_TRADES, 4), dtype=np.int64)
        out_n = np.zeros(len(VARIANTS), dtype=np.int64)
        run_chunk(ts, px, start, cT, cL, cd, chold, state, out, out_n)
        for c, v in enumerate(VARIANTS):
            if out_n[c] >= MAX_TRADES:
                raise RuntimeError(f"trade buffer full in {chunk} for {v[0]}")
            rows[v[0]].append(out[c, :out_n[c]].copy())
        keep = np.searchsorted(ts, ts[-1] - OVERLAP_MS)
        tail = (ts[keep:].copy(), px[keep:].copy())
        print(f"{directory.name} {chunk}: {len(ts) - start:,} trades done", flush=True)
    frames = {}
    for name, parts in rows.items():
        a = np.concatenate(parts) if parts else np.zeros((0, 4), dtype=np.int64)
        df = pd.DataFrame({"entry_ms": a[:, 0], "side": a[:, 1], "entry_ticks": a[:, 2], "exit_ticks": a[:, 3]})
        df["gross"] = df["side"] * (df["exit_ticks"] / df["entry_ticks"] - 1.0)
        df["net"] = df["gross"] - COST
        df["month"] = pd.to_datetime(df["entry_ms"], unit="ms", utc=True).dt.strftime("%Y-%m")
        frames[name] = df
    return frames


def day_t(daily: pd.Series) -> float | None:
    return float(daily.mean() / (daily.std() / np.sqrt(len(daily)))) if len(daily) > 2 and daily.std() > 0 else None


def daily_sums(trades: pd.DataFrame, col: str, months: tuple[str, str]) -> pd.Series:
    """Sum of `col` per UTC day over every calendar day of the window (days without trades are zero)."""
    days = pd.date_range(f"{months[0]}-01", pd.Period(months[1], freq="M").end_time.normalize(), freq="D", tz="UTC")
    by_day = trades.groupby(trades["entry_ms"] // DAY_MS)[col].sum()
    return by_day.reindex(days.asi8 // 10**6 // DAY_MS, fill_value=0.0)


def stats(trades: pd.DataFrame, months: tuple[str, str]) -> dict:
    main = trades[(trades["month"] >= months[0]) & (trades["month"] <= months[1])]
    if not len(main):
        return {"trades": 0}
    years = len(pd.period_range(months[0], months[1], freq="M")) / 12.0
    curve = 1.0 + main["net"].cumsum()
    mid = sorted(main["month"].unique())
    half = pd.period_range(months[0], months[1], freq="M")[5].strftime("%Y-%m")
    return {
        "trades": int(len(main)), "days_with_trades": int((main["entry_ms"] // DAY_MS).nunique()), "months_with_trades": len(mid),
        "long_share": float((main["side"] > 0).mean()),
        "gross_mean_bp": float(main["gross"].mean() * 1e4), "net_mean_bp": float(main["net"].mean() * 1e4),
        "net_mean_bp_maker_exit": float((main["gross"].mean() - COST_MAKER_EXIT) * 1e4),
        "gross_t_by_day": day_t(daily_sums(main, "gross", months)), "net_t_by_day": day_t(daily_sums(main, "net", months)),
        "net_return_per_year": float(main["net"].sum() / years), "gross_return_per_year": float(main["gross"].sum() / years),
        "win_rate_net": float((main["net"] > 0).mean()), "worst_trade": float(main["net"].min()), "best_trade": float(main["net"].max()),
        "max_drawdown": float(((curve.cummax().clip(lower=1.0) - curve) / curve.cummax().clip(lower=1.0)).max()),
        "net_mean_bp_first_half": float(main.loc[main["month"] <= half, "net"].mean() * 1e4),
        "net_mean_bp_second_half": float(main.loc[main["month"] > half, "net"].mean() * 1e4),
        "net_mean_bp_longs": float(main.loc[main["side"] > 0, "net"].mean() * 1e4),
        "net_mean_bp_shorts": float(main.loc[main["side"] < 0, "net"].mean() * 1e4),
    }


def run(directory: Path, name: str) -> None:
    sub, d_bp, role, months = SETS[name]
    frames = simulate(directory / sub, d_bp)
    report = {"set": name, "role": role, "distance_bp": d_bp, "months": months, "cost": COST, "variants": {}}
    for label, trades in frames.items():
        report["variants"][label] = stats(trades, months)
        fresh = trades[trades["month"] == FRESH]
        if role == "candidate":
            report["variants"][label]["fresh_slice"] = {"trades": int(len(fresh)), "net_mean_bp": float(fresh["net"].mean() * 1e4) if len(fresh) else None,
                                                        "net_sum": float(fresh["net"].sum())}
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"spike_{name}.json"
    out.write_text(json.dumps(report, indent=1, default=float, sort_keys=True))
    frames["primary"].drop(columns="month").to_csv(RESULTS / f"spike_{name}_trades.csv", index=False)
    print(f"wrote {out} sha256 {hashlib.sha256(out.read_bytes()).hexdigest()[:16]} "
          f"trades csv {hashlib.sha256((RESULTS / f'spike_{name}_trades.csv').read_bytes()).hexdigest()[:16]}")
    for label, s in report["variants"].items():
        if s["trades"]:
            print(f"  {label:9} trades {s['trades']:5d} gross {s['gross_mean_bp']:+7.2f} bp net {s['net_mean_bp']:+7.2f} bp t {s['net_t_by_day'] or float('nan'):+5.2f} "
                  f"net/yr {s['net_return_per_year'] * 100:+6.1f}% worst {s['worst_trade'] * 100:+5.2f}% DD {s['max_drawdown'] * 100:4.1f}%")


def report() -> None:
    """Pool the three candidates' primary trades and apply the gates."""
    per, pooled_net, pooled_gross = {}, [], []
    for name, (_, _, role, months) in SETS.items():
        r = json.loads((RESULTS / f"spike_{name}.json").read_text())
        per[name] = r["variants"]["primary"]
        if role == "candidate":
            tr = pd.read_csv(RESULTS / f"spike_{name}_trades.csv")
            tr["net"] = tr["gross"] - COST
            tr["month"] = pd.to_datetime(tr["entry_ms"], unit="ms", utc=True).dt.strftime("%Y-%m")
            tr = tr[(tr["month"] >= months[0]) & (tr["month"] <= months[1])]
            pooled_net.append(daily_sums(tr, "net", months))
            pooled_gross.append(tr)
    cands = [n for n, s in SETS.items() if s[2] == "candidate"]
    daily = sum(pooled_net)
    alltr = pd.concat(pooled_gross)
    gates = {
        "1 pooled mean net > 0 with t >= 2.5": bool(alltr["net"].mean() > 0 and (day_t(daily) or 0) >= 2.5),
        "2 net positive in each coin": all(per[n].get("net_mean_bp", -1) > 0 for n in cands),
        "3 average net return >= 10% a year": float(np.mean([per[n].get("net_return_per_year", 0) for n in cands])) >= 0.10,
        "4 worst trade >= -5% and drawdown <= 15% in each": all(per[n].get("worst_trade", -1) >= -0.05 and per[n].get("max_drawdown", 1) <= 0.15 for n in cands),
        "5 at least 100 trades in each": all(per[n].get("trades", 0) >= 100 for n in cands),
    }
    out = {"pooled": {"trades": int(len(alltr)), "net_mean_bp": float(alltr["net"].mean() * 1e4), "gross_mean_bp": float(alltr["gross"].mean() * 1e4),
                      "net_t_by_day": day_t(daily)}, "gates": gates,
           "verdict": "H1 supported" if all(gates.values()) else ("mixed" if gates["1 pooled mean net > 0 with t >= 2.5"] and gates["2 net positive in each coin"] else "H0 supported")}
    (RESULTS / "spike_pooled.json").write_text(json.dumps(out, indent=1, default=float, sort_keys=True))
    print(json.dumps(out, indent=1, default=float))


def selftest() -> None:
    def tape(rows, d_bp=10.0, L=50, hold=300_000):
        ts = np.array([r[0] for r in rows], dtype=np.int64)
        px = np.array([r[1] for r in rows], dtype=np.int32)
        state = np.zeros((1, 7), dtype=np.int64); state[:, SLOT:ASK + 1] = -1
        out = np.zeros((1, MAX_TRADES, 4), dtype=np.int64); n = np.zeros(1, dtype=np.int64)
        run_chunk(ts, px, 0, np.array([1000]), np.array([L]), np.array([d_bp / 1e4]), np.array([hold]), state, out, n)
        return out[0, :n[0]], state[0]

    base = [(10, 10000), (500, 10000)]  # slot 1 (live from 1050 ms) quotes bid 9990 / ask 10010
    tr, st = tape(base + [(1100, 9989), (1200, 9950), (301_000, 9995), (301_101, 9996), (302_000, 9996)])
    assert len(tr) == 1 and tuple(tr[0]) == (1100, 1, 9990, 9995), "long at the bid, closed at the last price 300 s after the fill"
    tr, st = tape(base + [(1100, 9989), (2100, 9900), (3100, 9800)])
    assert len(tr) == 0 and st[POS] == 1, "no second entry while a position is open"
    tr, st = tape(base + [(1100, 10011), (200_000, 10050), (301_200, 10020)])
    assert len(tr) == 1 and tuple(tr[0]) == (1100, -1, 10010, 10050), "short at the ask; exit uses the last trade at or before the exit time"
    tr, st = tape(base + [(1100, 9989), (301_101, 9996), (301_150, 9000)])
    assert len(tr) == 1 and st[POS] == 0, "after an exit, no order until the next refresh (the 9000 print is not a fill)"
    tr, st = tape(base + [(1100, 9989), (301_101, 9996), (302_100, 9996), (303_100, 9980)])
    assert len(tr) == 1 and st[POS] == 1, "orders return on the refresh after an exit"
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["selftest", "run", "report"])
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--set", choices=list(SETS))
    args = ap.parse_args()
    if args.mode == "selftest":
        selftest()
    elif args.mode == "run":
        run(args.dir, args.set)
    else:
        report()
