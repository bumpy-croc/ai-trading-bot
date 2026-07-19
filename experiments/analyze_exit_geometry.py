#!/usr/bin/env python3
"""Analysis pass for the EXIT-GEOMETRY prereg (2026-07-12_exit-geometry-honest.md).

Reads experiments/exit_geometry_results.jsonl (produced by exit_geometry_sweep.py),
applies the pre-committed decision table (Sec. 1 of the prereg): an arm is a
"promotion candidate" only if, on EVERY primary fold (F1/F2/F3):
  1. total_return improves vs control
  2. profit_factor improves vs control
  3. max_drawdown <= 20% (risk-limits.json portfolio.max_drawdown_pct)
  4. bootstrap two-sided p-value on trade_pnl_pcts difference-in-means < 0.05/6 (Bonferroni)

Prints a decision table and a mechanism-metrics table (MFE capture / MAE ride).
No thresholds here were chosen after seeing results -- they are copy-pasted
from the prereg document, locked before the sweep ran.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MAXDD_CAP_PCT = 20.0  # risk-limits.json portfolio.max_drawdown_pct, as a percentage
BONFERRONI_N_ARMS = 6
ALPHA = 0.05 / BONFERRONI_N_ARMS
N_BOOTSTRAP = 10_000
PRIMARY_FOLDS = ["F1_2023H1", "F2_2024H1", "F3_2025H1"]


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
    """Two-sided bootstrap p-value for difference in means (a - b), null = 0.

    Resamples each group independently (not paired -- trade counts differ
    across arms since exit width changes when/whether a position closes).
    """
    if not a or not b:
        return 1.0
    observed = statistics.fmean(a) - statistics.fmean(b)
    # Pool-and-recenter approach: shift both samples to share the pooled mean,
    # then resample under the null and see how extreme `observed` is.
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


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/exit_geometry_results.jsonl")
    records = load_records(path)

    by_fold_arm: dict[str, dict[str, dict]] = defaultdict(dict)
    for rec in records:
        by_fold_arm[rec["fold"]][rec["arm"]] = rec

    arms = sorted({rec["arm"] for rec in records if rec["arm"] != "control"})
    rng = random.Random(1234567)  # fixed seed for reproducible p-values across reruns of this analysis

    print("=" * 100)
    print("DECISION TABLE (per prereg Sec. 1 falsifiable statement, thresholds locked before this ran)")
    print("=" * 100)

    verdicts: dict[str, dict[str, bool]] = defaultdict(dict)

    for fold in PRIMARY_FOLDS:
        if fold not in by_fold_arm or "control" not in by_fold_arm[fold]:
            print(f"[{fold}] MISSING control result -- skipped")
            continue
        control = by_fold_arm[fold]["control"]
        print(f"\n--- Fold {fold} --- control: ret%={control['total_return']:.2f} PF={control['profit_factor']:.3f} "
              f"maxDD%={control['max_drawdown']:.2f} trades={control['total_trades']}")
        for arm in arms:
            if arm not in by_fold_arm[fold]:
                print(f"  {arm:<18} MISSING")
                continue
            rec = by_fold_arm[fold][arm]
            ret_improves = rec["total_return"] > control["total_return"]
            pf_improves = rec["profit_factor"] > control["profit_factor"]
            dd_ok = rec["max_drawdown"] <= MAXDD_CAP_PCT
            a = rec.get("trade_pnl_pcts") or []
            b = control.get("trade_pnl_pcts") or []
            pval = bootstrap_diff_pvalue(a, b, N_BOOTSTRAP, rng)
            significant = pval < ALPHA
            win = ret_improves and pf_improves and dd_ok and significant
            verdicts[arm][fold] = win
            print(
                f"  {arm:<18} ret%={rec['total_return']:>8.2f} (Δ{rec['total_return']-control['total_return']:+7.2f}) "
                f"PF={rec['profit_factor']:>6.3f} ({'UP' if pf_improves else 'down'}) "
                f"maxDD%={rec['max_drawdown']:>6.2f} ({'OK' if dd_ok else 'BREACH'}) "
                f"p={pval:.4f} ({'SIG' if significant else 'ns'}, α={ALPHA:.4f}) "
                f"trades={rec['total_trades']:>4} => {'WIN' if win else 'no'}"
            )

    print("\n" + "=" * 100)
    print("MULTI-FOLD VERDICT (must WIN on every primary fold to be a promotion candidate)")
    print("=" * 100)
    for arm in arms:
        folds_seen = [f for f in PRIMARY_FOLDS if f in verdicts[arm]]
        wins = [verdicts[arm][f] for f in folds_seen]
        all_win = len(folds_seen) == len(PRIMARY_FOLDS) and all(wins)
        any_win = any(wins)
        status = "PROMOTION CANDIDATE" if all_win else ("promising but not ready (partial)" if any_win else "NO-GO")
        print(f"  {arm:<18} folds won: {sum(wins)}/{len(folds_seen)}  => {status}")

    print("\n" + "=" * 100)
    print("MECHANISM METRICS (MFE capture / MAE ride) -- descriptive, not gating")
    print("=" * 100)
    for fold in PRIMARY_FOLDS:
        if fold not in by_fold_arm:
            continue
        print(f"\n--- {fold} ---")
        for arm in ["control"] + arms:
            if arm not in by_fold_arm[fold]:
                continue
            c = by_fold_arm[fold][arm]["capture"]
            print(
                f"  {arm:<18} capture_ratio={c['capture_ratio']} (n={c['n_winners_with_mfe']}) "
                f"mae_ride={c['mae_ride_fraction']} (n={c['n_losers_with_mae']})"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
