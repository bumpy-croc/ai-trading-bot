"""Score the forward paper test from the recorded weights
(docs/research/experiments/2026-11-11_slow-adaptive-trend-prereg.md, Stage B and section 6).

  git fetch origin data/paper-trend-recorder && git worktree add /tmp/paper-data FETCH_HEAD
  python experiments/strategy_search/paper_score.py --records /tmp/paper-data [--as-of YYYY-MM-DD]

What it does:
  1. Reads every record written by tools/paper_trend_recorder.py. A signal-day record counts only if it
     was written before its fill price existed (before the close of the next UTC day). A Monday with no
     valid record leaves the book holding its previous weights for that week, as a failed job would.
  2. Downloads prices and funding again from the public archive (not what the recorder saw) and runs the
     recorded weights through the Phase 1b simulator: fills at the next day's close, fee 5 bp a side,
     slippage by liquidity rank, actual funding.
  3. Recomputes each book's weights from the archive data and reports the difference from what was
     recorded. That is the check that the recorder does what the design run did.
  4. Prints where each book stands against the fixed checkpoints: stop above 20% drawdown, plumbing at
     week 8, returns at weeks 26 and 52.

`--accept-late` scores records written after their fill price existed. It is for dry runs of this script
only: such a result is a backtest, not a forward test, and the output says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adaptive  # noqa: E402
import data as data_mod  # noqa: E402
import engine as eng  # noqa: E402
import forward_data  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
HISTORY_DAYS = 420
STOP_DRAWDOWN = 0.20
MAX_MISSED_DAYS = 3
MAX_WEIGHT_DIFF = 0.01
CHECKPOINTS = {26: (0.05, 0.15), 52: (0.10, 0.20)}  # week: (minimum net return, maximum drawdown)


def load_records(root: Path) -> list[dict]:
    files = sorted((root / "records").glob("*/record_*.json"))
    return [json.loads(f.read_text()) for f in files]


def on_time(record: dict) -> bool:
    """Written before the fill price existed: the fill is the close of the UTC day after the data day."""
    fill_close = pd.Timestamp(record["data_day"], tz="UTC") + pd.Timedelta(days=2)
    return pd.Timestamp(record["recorded_at"]) < fill_close


def recorded_targets(records: list[dict], book: str, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """Target rows on the signal days that have a usable record; every other row is 'no rebalance'."""
    targets = pd.DataFrame(np.nan, index=index, columns=columns)
    for rec in records:
        if not rec["signal_day"] or book not in rec["books"]:
            continue
        day = pd.Timestamp(rec["data_day"], tz="UTC")
        weights = pd.Series(rec["books"][book]["weights"], dtype=float)
        unknown = weights.index.difference(columns)
        if len(unknown):
            raise RuntimeError(f"{book} {rec['data_day']}: recorded symbols missing from the archive: {list(unknown)}")
        targets.loc[day] = 0.0
        targets.loc[day, weights.index] = weights.to_numpy()
    return targets


def summarise(returns: pd.Series) -> dict:
    eq = (1.0 + returns).cumprod()
    return {"net_return": float(eq.iloc[-1] - 1.0),
            "max_drawdown": float(-(eq / eq.cummax().clip(lower=1.0) - 1.0).min()),
            "days": int(len(returns))}


def at_checkpoint(returns: pd.Series, day0: pd.Timestamp, weeks: int) -> pd.Series:
    """The checkpoints are fixed dates. Past week 26 (or 52) the bar is judged on returns up to that date,
    not up to whenever the script happens to be run."""
    week = 52 if weeks >= 52 else 26 if weeks >= 26 else None
    return returns if week is None else returns.loc[: day0 + pd.Timedelta(weeks=week)]


def checkpoint(weeks: int, stats: dict, idle: float, plumbing_ok: bool, drawdown_to_date: float = 0.0) -> str:
    """`stats` and `idle` are measured at the fixed checkpoint date; the stop rule applies at any time."""
    worst = max(stats["max_drawdown"], drawdown_to_date)
    if worst > STOP_DRAWDOWN:
        return f"STOP: drawdown {worst:.1%} is above {STOP_DRAWDOWN:.0%}. The book is recorded as failed."
    if weeks < 8:
        return f"week {weeks}: before the first checkpoint (week 8, plumbing only)."
    if weeks < 26:
        return (f"week {weeks}: plumbing {'passes' if plumbing_ok else 'FAILS and needs fixing'}. "
                "No claim about returns before week 26.")
    week = 52 if weeks >= 52 else 26
    floor, max_dd = CHECKPOINTS[week]
    ret, dd = stats["net_return"], stats["max_drawdown"]
    if ret >= floor and dd <= max_dd:
        what = "a small live trial" if week == 26 else "a live allocation"
        return f"week {week} bar met ({ret:+.1%}, drawdown {dd:.1%}): proposal to Alex for {what}."
    if week == 26:
        return (f"week 26: {ret:+.1%} leans against; recording continues to week 52." if ret < 0
                else f"week 26: {ret:+.1%} is between the bars; recording continues.")
    if ret < idle:
        return f"week 52: {ret:+.1%} is below idle dollars ({idle:+.1%}). Stop, recorded as failed."
    return f"week 52: {ret:+.1%} is between the bars. Alex decides whether to keep recording."


def score(records_dir: Path, as_of: str | None, accept_late: bool) -> dict:
    records = load_records(records_dir)
    if not records:
        raise SystemExit("no records found")
    usable = [r for r in records if accept_late or on_time(r)]
    signals = [r for r in usable if r["signal_day"] and r["books"]]
    if not signals:
        raise SystemExit("no signal-day record written before its fill price: nothing to score yet")
    day0 = pd.Timestamp(signals[0]["data_day"], tz="UTC")
    end = pd.Timestamp(as_of or records[-1]["data_day"], tz="UTC")
    books = sorted({b for r in signals for b in r["books"]})

    market, info = forward_data.fetch_market(end, HISTORY_DAYS + (end - day0).days)
    # Same universe filter as the recorder: the committed non-crypto list, plus what the exchange's own
    # contract list added on the days the recorder could reach it, plus what it says today.
    fresh, source = forward_data.extra_exclusions(list(market.close.columns))
    recorded_filters = [r["universe_filter"] for r in records if "universe_filter" in r]
    extra = {s for f in recorded_filters if f["source"] == "exchange" for s in f["excluded_beyond_committed_list"]}
    extra |= set(fresh) if source == "exchange" else set()
    eng.EXCLUDE.update(extra)
    rank = eng.liquidity_rank(market.close, market.qvol)
    held = sorted({s for r in signals for b in r["books"].values() for s in b["weights"]})
    ranked = rank.loc[day0 - pd.Timedelta(days=120):].le(adaptive.BROAD_N).any()
    need = sorted((set(held) | set(ranked.index[ranked])) & set(market.close.columns))
    fund, fund_info = forward_data.fetch_funding(need, day0 - pd.Timedelta(days=120), end)
    funding = market.funding.copy()  # zero for coins never near the universe; they are never held
    funding.loc[fund.index, fund.columns] = fund.where(market.close.loc[fund.index, fund.columns].notna())
    market = data_mod.Market(market.open, market.high, market.low, market.close, market.qvol, funding, market.spot_close)

    recomputed = adaptive.build_books(market, tuple(books))
    days = pd.date_range(day0, end, freq="D")
    have = {r["data_day"] for r in records}
    late = [r["data_day"] for r in records if not on_time(r) and day0 <= pd.Timestamp(r["data_day"], tz="UTC") <= end]
    missed = [f"{d:%Y-%m-%d}" for d in days if f"{d:%Y-%m-%d}" not in have]
    weeks = (end - day0).days // 7
    idle = float((1.0 + eng.RF_ANNUAL) ** ((end - day0).days / eng.ANN) - 1.0)

    r_all = market.close.pct_change(fill_method=None)
    univ = eng.universe(rank)
    bench = {
        "BTC buy and hold": r_all["BTCUSDT"].fillna(0.0),
        "Top-30 equal weight": r_all.where(univ.shift(1, fill_value=False)).mean(axis=1).fillna(0.0),
    }
    after = slice(day0 + pd.Timedelta(days=2), end)  # the first fill is at the close of day0 + 1
    out: dict = {
        "day0": f"{day0:%Y-%m-%d}", "as_of": f"{end:%Y-%m-%d}", "weeks": int(weeks),
        "forward_test": not accept_late,
        "records": {"total": len(records), "signal_days_scored": len(signals), "missed_days": missed, "late_records": late},
        "data": {**info, **fund_info, "funding_missing_coin_days_held": 0},
        "idle_dollars": idle,
        "universe_filter": {
            "excluded_beyond_committed_list": sorted(extra), "source_today": source,
            "days_recorded_on_fallback": [r["data_day"] for r in records
                                          if r.get("universe_filter", {}).get("source") == "fallback"],
        },
        "benchmarks": {k: summarise(v.loc[after]) for k, v in bench.items() if len(v.loc[after])},
        "books": {},
    }
    for book in books:
        targets = recorded_targets(signals, book, market.close.index, market.close.columns)
        res = eng.simulate(market.close, market.funding, rank, targets)
        forward = res.returns.loc[day0:]
        stats = summarise(forward)
        fixed = at_checkpoint(forward, day0, weeks)
        idle_fixed = float((1.0 + eng.RF_ANNUAL) ** ((fixed.index[-1] - day0).days / eng.ANN) - 1.0)
        diffs = {}
        for rec in signals:
            if book not in rec["books"]:
                continue
            day = pd.Timestamp(rec["data_day"], tz="UTC")
            mine = recomputed[book].weights.loc[day].fillna(0.0)
            theirs = targets.loc[day].fillna(0.0)
            gross = float(theirs.abs().sum())
            diffs[rec["data_day"]] = float((mine - theirs).abs().sum() / gross) if gross > 0 else float(mine.abs().sum())
        mean_diff = float(np.mean(list(diffs.values())))
        plumbing_ok = len(missed) + len(late) <= MAX_MISSED_DAYS and mean_diff < MAX_WEIGHT_DIFF
        position = targets.ffill().shift(1).fillna(0.0)
        out["data"]["funding_missing_coin_days_held"] += int(
            (market.funding.isna() & (position != 0)).loc[day0:].sum().sum())
        out["books"][book] = {
            **stats,
            "cost": float(res.costs.loc[day0:].sum()), "funding_paid": float(res.funding.loc[day0:].sum()),
            "avg_gross": float(res.gross.loc[day0:].mean()),
            "weight_difference_vs_recomputed": {"mean": mean_diff, "max": float(max(diffs.values())), "by_day": diffs},
            "plumbing_ok": plumbing_ok,
            "at_checkpoint_date": {"date": f"{fixed.index[-1]:%Y-%m-%d}", **summarise(fixed)},
            "checkpoint": checkpoint(weeks, summarise(fixed), idle_fixed, plumbing_ok, stats["max_drawdown"]),
        }
    RESULTS.mkdir(exist_ok=True)
    name = f"paper_score_{end:%Y-%m-%d}{'_dryrun' if accept_late else ''}.json"
    (RESULTS / name).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the forward paper test from the recorded weights")
    ap.add_argument("--records", type=Path, required=True, help="checkout of the data/paper-trend-recorder branch")
    ap.add_argument("--as-of", help="last UTC day to score (default: the last recorded day)")
    ap.add_argument("--accept-late", action="store_true", help="dry run only: also score records written after their fill")
    args = ap.parse_args()
    out = score(args.records, args.as_of, args.accept_late)
    if not out["forward_test"]:
        print("DRY RUN: late records accepted. This is a backtest of the plumbing, not a forward result.")
    rec = out["records"]
    print(f"day 0 {out['day0']}, scored to {out['as_of']} (week {out['weeks']}); {rec['signal_days_scored']} signal days, "
          f"{len(rec['missed_days'])} missed days, {len(rec['late_records'])} late records")
    print(f"idle dollars {out['idle_dollars']:+.2%}; " + "; ".join(
        f"{k} {v['net_return']:+.2%} (drawdown {v['max_drawdown']:.1%})" for k, v in out["benchmarks"].items()))
    for book, b in out["books"].items():
        print(f"{book}: net {b['net_return']:+.2%}, drawdown {b['max_drawdown']:.1%}, costs {b['cost']:.2%}, "
              f"funding paid {b['funding_paid']:+.2%}, recorded vs recomputed weights differ by "
              f"{b['weight_difference_vs_recomputed']['mean']:.2%} of gross on average")
        print(f"   {b['checkpoint']}")


if __name__ == "__main__":
    main()
