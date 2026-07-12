#!/usr/bin/env python3
"""LINEAR INPUT-SCREENING experiment runner (Lane A, Phase 1).

Preregistered at docs/research/experiments/2026-07-12_input-screening-linear.md —
read that file for the hypothesis, arm definitions, thresholds, and graduation
rule. This script implements exactly what's pre-committed there; do not add or
remove arms/thresholds here without amending the prereg first.

Usage: python scripts/research/run_input_screening.py [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from is_data import load_fear_greed, load_funding_rate, load_ohlcv, load_premium_index  # noqa: E402
from is_features import (  # noqa: E402
    BTC_COLS,
    FNG_COLS,
    FNG_FIRST_DAY,
    FUNDING_COLS,
    FUNDING_FIRST_SETTLEMENT,
    PREMIUM_COLS,
    PREMIUM_FIRST_BAR,
    REALIZED_VOL_COLS,
    SEQUENCE_LENGTH,
    assemble_samples,
    btc_cross_frame,
    build_price_only_frame,
    calendar_features_for_timestamps,
    fear_greed_frame,
    funding_frame,
    premium_frame,
    realized_vol_frame,
)

SYMBOL = "ETHUSDT"
BTC_SYMBOL = "BTCUSDT"
TIMEFRAME = "1h"
HISTORY_START = datetime(2017, 8, 17, tzinfo=UTC)
HISTORY_END = datetime(2025, 7, 2, tzinfo=UTC)  # buffer past F3 eval end

FOLDS = [
    {
        "name": "F1",
        "train_cutoff": pd.Timestamp("2022-12-31 23:00", tz="UTC"),
        "eval_start": pd.Timestamp("2023-01-03 00:00", tz="UTC"),
        "eval_end": pd.Timestamp("2023-06-30 23:00", tz="UTC"),
    },
    {
        "name": "F2",
        "train_cutoff": pd.Timestamp("2023-12-31 23:00", tz="UTC"),
        "eval_start": pd.Timestamp("2024-01-03 00:00", tz="UTC"),
        "eval_end": pd.Timestamp("2024-06-30 23:00", tz="UTC"),
    },
    {
        "name": "F3",
        "train_cutoff": pd.Timestamp("2024-12-31 23:00", tz="UTC"),
        "eval_start": pd.Timestamp("2025-01-03 00:00", tz="UTC"),
        "eval_end": pd.Timestamp("2025-06-30 23:00", tz="UTC"),
    },
]

# Tournament's reported linear-baseline DA (results.json aggregate_stats_CORRECTED),
# used only for the arm-0 validity check (prereg §2).
TOURNAMENT_LINEAR_BASELINE_DA = {"F1": 53.24, "F2": 53.61, "F3": 53.18}
VALIDITY_TOLERANCE_PP = 2.0

BONFERRONI_ALPHA = 0.05 / 7  # 7 arms tested against the control, prereg §5
GRADUATION_MIN_SIGNIFICANT_FOLDS = 2
GRADUATION_MIN_AVG_DELTA_PP = 0.5

ARMS = [
    {"id": 0, "name": "price_only_control", "extra": None},
    {"id": 1, "name": "realized_vol_range", "extra": "vol"},
    {"id": 2, "name": "calendar", "extra": "calendar"},
    {"id": 3, "name": "btc_cross", "extra": "btc"},
    {"id": 4, "name": "funding_rate", "extra": "funding"},
    {"id": 5, "name": "basis_premium", "extra": "premium"},
    {"id": 6, "name": "fear_greed", "extra": "fng"},
    {"id": 7, "name": "all_combined", "extra": "all"},
]


def _mcnemar_p(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Exact two-sided McNemar test (binomial on discordant pairs). a=control, b=arm."""
    b_wins = int(np.sum((~correct_a) & correct_b))  # control wrong, arm right
    a_wins = int(np.sum(correct_a & (~correct_b)))  # control right, arm wrong
    n_discordant = b_wins + a_wins
    if n_discordant == 0:
        return 1.0
    result = binomtest(min(b_wins, a_wins), n_discordant, 0.5, alternative="two-sided")
    return float(result.pvalue)


def build_all_frames():
    print("Loading OHLCV (ETH, BTC) ...")
    eth = load_ohlcv(SYMBOL, TIMEFRAME, HISTORY_START, HISTORY_END)
    btc = load_ohlcv(BTC_SYMBOL, TIMEFRAME, HISTORY_START, HISTORY_END)

    print("Loading funding / premium / fear-greed raw data ...")
    funding_raw = load_funding_rate(SYMBOL, HISTORY_START, HISTORY_END)
    premium_raw = load_premium_index(SYMBOL, HISTORY_START, HISTORY_END)
    fng_raw = load_fear_greed()

    print("Building price-only feature contract ...")
    price_only = build_price_only_frame(eth)

    print("Building arm feature frames ...")
    vol_frame = realized_vol_frame(eth)
    cal_frame = calendar_features_for_timestamps(eth.index)
    btc_frame = btc_cross_frame(btc, eth.index)
    premium_feat = premium_frame(premium_raw, eth.index)
    fng_feat = fear_greed_frame(fng_raw, eth.index)

    return {
        "eth": eth,
        "price_only": price_only,
        "vol": vol_frame,
        "calendar": cal_frame,
        "btc": btc_frame,
        "premium": premium_feat,
        "fng": fng_feat,
        "funding_raw": funding_raw,
    }


def positions_for(index: pd.DatetimeIndex, start_ts, end_ts) -> np.ndarray:
    mask = (index >= start_ts) & (index <= end_ts)
    return np.nonzero(np.asarray(mask))[0]


def valid_positions(
    positions: np.ndarray, index: pd.DatetimeIndex, min_ts: pd.Timestamp
) -> np.ndarray:
    """Drop target positions whose `t-1` (last input bar) predates an extra
    feature's earliest real coverage — never backfilled/coerced, per the prereg."""
    ts_prev = index[positions - 1]
    keep = ts_prev >= min_ts
    return positions[keep]


def drop_nan_rows(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


def run_arm_fold(arm: dict, fold: dict, frames: dict) -> dict:
    eth_index = frames["eth"].index
    price_only = frames["price_only"]

    # Training positions: SEQUENCE_LENGTH..N-1 with timestamp <= train_cutoff.
    train_positions = positions_for(eth_index, eth_index[SEQUENCE_LENGTH], fold["train_cutoff"])
    train_positions = train_positions[train_positions >= SEQUENCE_LENGTH]
    eval_positions = positions_for(eth_index, fold["eval_start"], fold["eval_end"])

    extra_key = arm["extra"]
    extra_frame = None
    extra_cols: list[str] | None = None

    min_coverage_ts = None
    frames_needed: list[tuple[str, list[str]]] = []
    if extra_key == "vol":
        frames_needed = [("vol", REALIZED_VOL_COLS)]
    elif extra_key == "calendar":
        pass  # handled below via cal_train/cal_eval
    elif extra_key == "btc":
        frames_needed = [("btc", BTC_COLS)]
    elif extra_key == "funding":
        funding_feat = funding_frame(frames["funding_raw"], eth_index, fold["train_cutoff"])
        frames["_funding_this_fold"] = funding_feat
        frames_needed = [("_funding_this_fold", FUNDING_COLS)]
        min_coverage_ts = FUNDING_FIRST_SETTLEMENT
    elif extra_key == "premium":
        frames_needed = [("premium", PREMIUM_COLS)]
        min_coverage_ts = PREMIUM_FIRST_BAR
    elif extra_key == "fng":
        frames_needed = [("fng", FNG_COLS)]
        min_coverage_ts = FNG_FIRST_DAY
    elif extra_key == "all":
        funding_feat = funding_frame(frames["funding_raw"], eth_index, fold["train_cutoff"])
        frames["_funding_this_fold"] = funding_feat
        combined = pd.concat(
            [
                frames["vol"][REALIZED_VOL_COLS],
                frames["btc"][BTC_COLS],
                frames["_funding_this_fold"][FUNDING_COLS],
                frames["premium"][PREMIUM_COLS],
                frames["fng"][FNG_COLS],
            ],
            axis=1,
        )
        frames["_all_this_fold"] = combined
        extra_cols = list(combined.columns)
        extra_frame = combined
        min_coverage_ts = PREMIUM_FIRST_BAR

    if frames_needed:
        name, cols = frames_needed[0]
        extra_frame = frames[name]
        extra_cols = cols

    if min_coverage_ts is not None:
        train_positions = valid_positions(train_positions, eth_index, min_coverage_ts)
        eval_positions = valid_positions(eval_positions, eth_index, min_coverage_ts)

    cal_train = cal_eval = None
    if extra_key == "calendar":
        cal_train = frames["calendar"].loc[eth_index[train_positions]]
        cal_eval = frames["calendar"].loc[eth_index[eval_positions]]
        extra_frame = None
        extra_cols = None
    elif extra_key == "all":
        cal_train = frames["calendar"].loc[eth_index[train_positions]]
        cal_eval = frames["calendar"].loc[eth_index[eval_positions]]

    X_train, y_train = assemble_samples(
        price_only, train_positions, extra_frame, extra_cols, cal_train
    )
    X_eval, y_eval = assemble_samples(price_only, eval_positions, extra_frame, extra_cols, cal_eval)

    X_train, y_train = drop_nan_rows(X_train, y_train)
    eval_keep_mask = ~np.isnan(X_eval).any(axis=1)
    X_eval, y_eval = X_eval[eval_keep_mask], y_eval[eval_keep_mask]
    eval_positions_kept = eval_positions[eval_keep_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_eval_s = scaler.transform(X_eval)

    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=3000)
    clf.fit(X_train_s, y_train)

    proba = clf.predict_proba(X_eval_s)[:, 1]
    pred = (proba >= 0.5).astype(float)
    correct = pred == y_eval
    da = float(correct.mean() * 100.0)
    brier = float(np.mean((proba - y_eval) ** 2))

    # naive persistence over the SAME eval positions (secondary, never ranked)
    close = price_only["close"].to_numpy(dtype=np.float64)
    naive_pred = np.array(
        [1.0 if close[t - 1] > close[t - 2] else 0.0 for t in eval_positions_kept]
    )
    naive_correct = naive_pred == y_eval
    naive_da = float(naive_correct.mean() * 100.0)
    disagree_mask = pred != naive_pred
    da_on_disagree = (
        float(correct[disagree_mask].mean() * 100.0) if disagree_mask.sum() > 0 else float("nan")
    )

    return {
        "arm": arm["name"],
        "fold": fold["name"],
        "n_train": int(len(y_train)),
        "n_eval": int(len(y_eval)),
        "da": da,
        "brier": brier,
        "naive_da": naive_da,
        "da_on_naive_disagree_subset": da_on_disagree,
        "n_naive_disagree": int(disagree_mask.sum()),
        "correct": correct,  # per-bar vector, used for McNemar; stripped before JSON dump
        "eval_positions": eval_positions_kept,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "input_screening_results.json"))
    args = ap.parse_args()

    t0 = time.time()
    frames = build_all_frames()
    print(f"Frames built in {time.time() - t0:.1f}s")

    all_results: dict[str, dict[str, dict]] = {}
    for fold in FOLDS:
        print(f"\n=== Fold {fold['name']} ===")
        fold_results = {}
        control_result = None
        for arm in ARMS:
            t1 = time.time()
            res = run_arm_fold(arm, fold, frames)
            dt = time.time() - t1
            print(
                f"  arm={arm['name']:20s} n_train={res['n_train']:6d} n_eval={res['n_eval']:5d} "
                f"DA={res['da']:.2f}% brier={res['brier']:.4f}  ({dt:.1f}s)"
            )
            if arm["id"] == 0:
                control_result = res
            fold_results[arm["name"]] = res
        # McNemar vs control, only valid where the eval position sets match exactly
        for name, res in fold_results.items():
            if name == "price_only_control":
                res["mcnemar_p_vs_control"] = None
                res["delta_da_vs_control_pp"] = 0.0
                continue
            ctrl_pos = control_result["eval_positions"]
            arm_pos = res["eval_positions"]
            common = np.intersect1d(ctrl_pos, arm_pos)
            ctrl_map = dict(
                zip(
                    control_result["eval_positions"].tolist(),
                    control_result["correct"].tolist(),
                    strict=False,
                )
            )
            arm_map = dict(
                zip(res["eval_positions"].tolist(), res["correct"].tolist(), strict=False)
            )
            ctrl_common = np.array([ctrl_map[p] for p in common])
            arm_common = np.array([arm_map[p] for p in common])
            res["n_common_with_control"] = int(len(common))
            res["mcnemar_p_vs_control"] = _mcnemar_p(ctrl_common, arm_common)
            res["delta_da_vs_control_pp"] = res["da"] - control_result["da"]
        all_results[fold["name"]] = fold_results

    # ---- Aggregation & graduation verdict ----
    verdicts = {}
    for arm in ARMS:
        if arm["id"] == 0:
            continue
        name = arm["name"]
        deltas = [all_results[f["name"]][name]["delta_da_vs_control_pp"] for f in FOLDS]
        pvals = [all_results[f["name"]][name]["mcnemar_p_vs_control"] for f in FOLDS]
        n_sig = sum(1 for p in pvals if p is not None and p < BONFERRONI_ALPHA)
        avg_delta = float(np.mean(deltas))
        graduates = (
            n_sig >= GRADUATION_MIN_SIGNIFICANT_FOLDS and avg_delta >= GRADUATION_MIN_AVG_DELTA_PP
        )
        verdicts[name] = {
            "per_fold_delta_pp": dict(zip([f["name"] for f in FOLDS], deltas, strict=False)),
            "per_fold_p": dict(zip([f["name"] for f in FOLDS], pvals, strict=False)),
            "n_significant_folds": n_sig,
            "avg_delta_pp": avg_delta,
            "graduates": graduates,
        }

    # ---- Validity check (arm 0 vs tournament baseline) ----
    validity = {}
    for f in FOLDS:
        ours = all_results[f["name"]]["price_only_control"]["da"]
        theirs = TOURNAMENT_LINEAR_BASELINE_DA[f["name"]]
        diff = ours - theirs
        validity[f["name"]] = {
            "ours": ours,
            "tournament": theirs,
            "diff_pp": diff,
            "within_tolerance": abs(diff) <= VALIDITY_TOLERANCE_PP,
        }

    # Strip non-JSON-serializable per-bar arrays before dumping.
    dump = {}
    for fold_name, fold_results in all_results.items():
        dump[fold_name] = {}
        for arm_name, res in fold_results.items():
            r = {k: v for k, v in res.items() if k not in ("correct", "eval_positions")}
            dump[fold_name][arm_name] = r

    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "results_by_fold": dump,
        "verdicts": verdicts,
        "validity_check": validity,
        "graduation_rule": {
            "min_significant_folds": GRADUATION_MIN_SIGNIFICANT_FOLDS,
            "min_avg_delta_pp": GRADUATION_MIN_AVG_DELTA_PP,
            "bonferroni_alpha": BONFERRONI_ALPHA,
        },
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nWrote results to {out_path}")

    print("\n=== Validity check (arm 0 vs tournament) ===")
    for f, v in validity.items():
        status = "PASS" if v["within_tolerance"] else "FAIL"
        print(
            f"  {f}: ours={v['ours']:.2f}% tournament={v['tournament']:.2f}% diff={v['diff_pp']:+.2f}pp [{status}]"
        )

    print("\n=== Graduation verdicts ===")
    for name, v in verdicts.items():
        print(
            f"  {name:20s} avg_delta={v['avg_delta_pp']:+.2f}pp n_sig={v['n_significant_folds']}/3 "
            f"-> {'GRADUATES' if v['graduates'] else 'does not graduate'}"
        )


if __name__ == "__main__":
    main()
