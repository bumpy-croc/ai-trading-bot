#!/usr/bin/env python3
"""Analysis pass for EXIT-GEOMETRY ROUND 2 (docs/research/experiments/2026-07-12_exit-geometry-round2.md).

Applies Sec. 1's four-bar test and Sec. 13's promotion rule EXACTLY as
pre-committed (copy-pasted logic, not reinterpreted after seeing results):

1. Bonferroni-significant return improvement (two-sided bootstrap
   diff-in-means on trade_pnl_pcts, 10,000 resamples, alpha=0.05/6=0.0083)
   on >= 2 of the 3 PRIMARY folds (F1/F2/F3).
2. Aggregate PF AND aggregate return improve vs control, where "aggregate"
   is defined in Sec. 6:
     - aggregate return delta = mean, across all 5 folds, of
       (arm.total_return[fold] - control.total_return[fold])
     - aggregate PF = pool every trade's pnl_percent across all 5 folds
       into one list (per arm and separately control), PF = sum(pos)/abs(sum(neg))
3. No fold (any of 5) has MaxDD worse than control's MaxDD on that SAME
   fold by more than 2.0 percentage points.
4. No fabrication signature (manual/descriptive check, reported not computed).

An arm clearing all four bars is a STAGING-TRIAL CANDIDATE ONLY (Sec. 13) --
never a straight-to-prod recommendation.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MAXDD_WORSENING_CAP_PP = 2.0
BONFERRONI_N_ARMS = 6
ALPHA = 0.05 / BONFERRONI_N_ARMS
N_BOOTSTRAP = 10_000
PRIMARY_FOLDS = ["F1_2023H1", "F2_2024H1", "F3_2025H1"]
ALL_FOLDS = ["F1_2023H1", "F2_2024H1", "F3_2025H1", "F0a_2021H1", "F0b_2022H1"]


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "determinism_check" in rec:
                continue
            records.append(rec)
    return records


def bootstrap_diff_pvalue(a: list[float], b: list[float], n: int, rng: random.Random) -> float:
    """Two-sided bootstrap p-value for difference in means (a - b), null = 0."""
    if not a or not b:
        return 1.0
    observed = statistics.fmean(a) - statistics.fmean(b)
    pooled_mean = statistics.fmean(a + b)
    a_shifted = [x - statistics.fmean(a) + pooled_mean for x in a]
    b_shifted = [x - statistics.fmean(b) + pooled_mean for x in b]
    count_extreme = 0
    for _ in range(n):
        ra = [rng.choice(a_shifted) for _ in range(len(a))]
        rb = [rng.choice(b_shifted) for _ in range(len(b))]
        diff = statistics.fmean(ra) - statistics.fmean(rb)
        if abs(diff) >= abs(observed):
            count_extreme += 1
    return count_extreme / n


def pooled_pf(fold_records: list[dict]) -> float:
    pos = 0.0
    neg = 0.0
    for rec in fold_records:
        for p in rec.get("trade_pnl_pcts") or []:
            if p > 0:
                pos += p
            elif p < 0:
                neg += p
    return (pos / abs(neg)) if neg != 0 else float("inf") if pos > 0 else 0.0


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/exit_geometry_round2_results.jsonl")
    records = load_records(path)

    by_fold_arm: dict[str, dict[str, dict]] = defaultdict(dict)
    for rec in records:
        by_fold_arm[rec["fold"]][rec["arm"]] = rec

    arms = sorted({rec["arm"] for rec in records if rec["arm"] != "control"})
    rng = random.Random(1234567)

    print("=" * 110)
    print("PER-FOLD RESULTS (all 5 folds)")
    print("=" * 110)
    for fold in ALL_FOLDS:
        c = by_fold_arm[fold]["control"]
        print(f"\n--- {fold} --- control: trades={c['total_trades']} ret%={c['total_return']:.2f} "
              f"PF={c['profit_factor']:.3f} maxDD%={c['max_drawdown']:.2f}")
        for arm in arms:
            r = by_fold_arm[fold][arm]
            dd_delta = r["max_drawdown"] - c["max_drawdown"]
            print(f"  {arm:<20} trades={r['total_trades']:>4} ret%={r['total_return']:>8.2f} "
                  f"(D{r['total_return']-c['total_return']:+7.2f}) PF={r['profit_factor']:>6.3f} "
                  f"maxDD%={r['max_drawdown']:>6.2f} (D{dd_delta:+5.2f}pp) "
                  f"capture={r['capture']['capture_ratio']} mae_ride={r['capture']['mae_ride_fraction']}")

    print("\n" + "=" * 110)
    print("BAR 1: Bonferroni-significant return improvement on >= 2 of 3 PRIMARY folds (F1/F2/F3)")
    print(f"(bootstrap diff-in-means on trade_pnl_pcts, {N_BOOTSTRAP} resamples, alpha={ALPHA:.4f})")
    print("=" * 110)
    bar1_pass: dict[str, int] = {}
    for arm in arms:
        wins = 0
        details = []
        for fold in PRIMARY_FOLDS:
            c = by_fold_arm[fold]["control"]
            r = by_fold_arm[fold][arm]
            a = r.get("trade_pnl_pcts") or []
            b = c.get("trade_pnl_pcts") or []
            pval = bootstrap_diff_pvalue(a, b, N_BOOTSTRAP, rng)
            ret_improves = r["total_return"] > c["total_return"]
            sig = pval < ALPHA
            win = ret_improves and sig
            if win:
                wins += 1
            details.append(f"{fold}: p={pval:.4f} ({'SIG' if sig else 'ns'}, {'up' if ret_improves else 'down'})")
        bar1_pass[arm] = wins
        print(f"  {arm:<20} folds-won={wins}/3  [{'; '.join(details)}]  => {'BAR1 PASS' if wins >= 2 else 'bar1 fail'}")

    print("\n" + "=" * 110)
    print("BAR 2: aggregate PF AND aggregate return improve vs control (pooled across all 5 folds, Sec. 6 method)")
    print("=" * 110)
    bar2_pass: dict[str, bool] = {}
    control_fold_records = {fold: by_fold_arm[fold]["control"] for fold in ALL_FOLDS}
    control_pooled_pf = pooled_pf(list(control_fold_records.values()))
    for arm in arms:
        arm_fold_records = {fold: by_fold_arm[fold][arm] for fold in ALL_FOLDS}
        deltas = [
            arm_fold_records[fold]["total_return"] - control_fold_records[fold]["total_return"]
            for fold in ALL_FOLDS
        ]
        agg_return_delta = statistics.fmean(deltas)
        arm_pooled_pf = pooled_pf(list(arm_fold_records.values()))
        pf_improves = arm_pooled_pf > control_pooled_pf
        ret_improves = agg_return_delta > 0
        win = pf_improves and ret_improves
        bar2_pass[arm] = win
        print(f"  {arm:<20} agg_return_delta={agg_return_delta:+7.2f}pp ({'up' if ret_improves else 'down'})  "
              f"pooled_PF={arm_pooled_pf:.4f} vs control={control_pooled_pf:.4f} ({'up' if pf_improves else 'down'})  "
              f"=> {'BAR2 PASS' if win else 'bar2 fail'}")

    print("\n" + "=" * 110)
    print("BAR 3: no fold's MaxDD worse than control's MaxDD on that fold by > 2.0pp")
    print("=" * 110)
    bar3_pass: dict[str, bool] = {}
    for arm in arms:
        worst_delta = max(
            by_fold_arm[fold][arm]["max_drawdown"] - by_fold_arm[fold]["control"]["max_drawdown"]
            for fold in ALL_FOLDS
        )
        win = worst_delta <= MAXDD_WORSENING_CAP_PP
        bar3_pass[arm] = win
        print(f"  {arm:<20} worst MaxDD delta vs control (any fold) = {worst_delta:+.2f}pp  "
              f"=> {'BAR3 PASS' if win else 'BAR3 FAIL (breach)'}")

    print("\n" + "=" * 110)
    print("FINAL VERDICT (Sec. 1 all-four-bars test; Sec. 13 promotion rule)")
    print("=" * 110)
    for arm in arms:
        b1, b2, b3 = bar1_pass[arm] >= 2, bar2_pass[arm], bar3_pass[arm]
        all_pass = b1 and b2 and b3
        status = "STAGING-TRIAL CANDIDATE (bar 4 fabrication check still required, see report)" if all_pass else "NO-GO"
        print(f"  {arm:<20} bar1={'PASS' if b1 else 'fail'}({bar1_pass[arm]}/3) bar2={'PASS' if b2 else 'fail'} "
              f"bar3={'PASS' if b3 else 'FAIL'}  => {status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
