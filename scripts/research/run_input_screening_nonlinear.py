#!/usr/bin/env python3
"""NONLINEAR INPUT-SCREENING re-screen runner (follow-up to Lane A Phase 1).

Preregistered at docs/research/experiments/2026-07-12_input-screening-nonlinear.md
— read that file for the hypothesis, arm definitions, thresholds, and graduation
rule. Reuses the identical feature contract, folds, and arms as
run_input_screening.py (the linear screen); swaps only the model family to a
single fixed LightGBM configuration (no hyperparameter search).

Requires `pip install lightgbm` (research-only dependency, not added to
requirements.txt — not used by any production/serving code path). On macOS,
LightGBM also needs the OpenMP runtime: `brew install libomp`.

Usage: python scripts/research/run_input_screening_nonlinear.py [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
from is_features import (  # noqa: E402
    BTC_COLS,
    FNG_COLS,
    FNG_FIRST_DAY,
    FUNDING_COLS,
    FUNDING_FIRST_SETTLEMENT,
    PREMIUM_COLS,
    PREMIUM_FIRST_BAR,
    PRICE_ONLY_COLS,
    REALIZED_VOL_COLS,
    SEQUENCE_LENGTH,
)  # noqa: E402
from run_input_screening import (  # noqa: E402
    ARMS,
    BONFERRONI_ALPHA,
    FOLDS,
    GRADUATION_MIN_AVG_DELTA_PP,
    GRADUATION_MIN_SIGNIFICANT_FOLDS,
    build_all_frames,
    drop_nan_rows,
    positions_for,
    valid_positions,
)
from run_input_screening import (
    run_arm_fold as _linear_run_arm_fold,  # noqa: F401  (imported for reference only, not called)
)

# ---------------------------------------------------------------------------
# LightGBM fixed configuration — pre-committed in the prereg, not tuned.
# ---------------------------------------------------------------------------
LGBM_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=50,
    random_state=42,
    n_jobs=4,
    importance_type="gain",
    verbose=-1,
)
EARLY_STOPPING_ROUNDS = 20
TRAIN_TAIL_VALIDATION_FRACTION = 0.10


def assemble_samples_with_names(
    price_only: pd.DataFrame,
    target_indices: np.ndarray,
    extra_frame: pd.DataFrame | None,
    extra_cols: list[str] | None,
    calendar_for_targets: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Identical construction to is_features.assemble_samples, but also returns
    a feature-name-per-column list so gain-based importance can be grouped."""
    values = price_only[PRICE_ONLY_COLS].to_numpy(dtype=np.float64)
    close = price_only["close"].to_numpy(dtype=np.float64)

    extra_vals = (
        extra_frame[extra_cols].to_numpy(dtype=np.float64) if extra_frame is not None else None
    )
    cal_vals = (
        calendar_for_targets.to_numpy(dtype=np.float64)
        if calendar_for_targets is not None
        else None
    )

    names: list[str] = [
        f"price_only[{ts}][{col}]" for ts in range(SEQUENCE_LENGTH) for col in PRICE_ONLY_COLS
    ]
    if extra_cols is not None:
        names += list(extra_cols)
    if calendar_for_targets is not None:
        names += list(calendar_for_targets.columns)

    X_rows = []
    y_rows = []
    for i, t in enumerate(target_indices):
        seq = values[t - SEQUENCE_LENGTH : t].reshape(-1)
        row = [seq]
        if extra_vals is not None:
            row.append(extra_vals[t - 1])
        if cal_vals is not None:
            row.append(cal_vals[i])
        X_rows.append(np.concatenate(row))
        y_rows.append(1.0 if close[t] > close[t - 1] else 0.0)

    return np.vstack(X_rows), np.array(y_rows), names


def _mcnemar_p(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    b_wins = int(np.sum((~correct_a) & correct_b))
    a_wins = int(np.sum(correct_a & (~correct_b)))
    n_discordant = b_wins + a_wins
    if n_discordant == 0:
        return 1.0
    result = binomtest(min(b_wins, a_wins), n_discordant, 0.5, alternative="two-sided")
    return float(result.pvalue)


def _group_importance(importances: np.ndarray, names: list[str]) -> dict:
    groups: dict[str, float] = {}
    for imp, name in zip(importances, names, strict=True):
        group = "price_only" if name.startswith("price_only[") else name
        groups[group] = groups.get(group, 0.0) + float(imp)
    total = sum(groups.values()) or 1.0
    return {
        k: {"gain": v, "gain_pct": 100.0 * v / total}
        for k, v in sorted(groups.items(), key=lambda kv: -kv[1])
    }


def run_arm_fold_nonlinear(arm: dict, fold: dict, frames: dict) -> dict:
    eth_index = frames["eth"].index
    price_only = frames["price_only"]

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
        pass
    elif extra_key == "btc":
        frames_needed = [("btc", BTC_COLS)]
    elif extra_key == "funding":
        from is_features import funding_frame

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
        from is_features import funding_frame

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

    X_train, y_train, feat_names = assemble_samples_with_names(
        price_only, train_positions, extra_frame, extra_cols, cal_train
    )
    X_eval, y_eval, _ = assemble_samples_with_names(
        price_only, eval_positions, extra_frame, extra_cols, cal_eval
    )

    X_train, y_train = drop_nan_rows(X_train, y_train)
    eval_keep_mask = ~np.isnan(X_eval).any(axis=1)
    X_eval, y_eval = X_eval[eval_keep_mask], y_eval[eval_keep_mask]
    eval_positions_kept = eval_positions[eval_keep_mask]

    # Train-tail validation split for early stopping (last 10% of TRAINING rows
    # by timestamp — still <= train cutoff, never touches embargo/eval).
    n_val = max(50, int(len(X_train) * TRAIN_TAIL_VALIDATION_FRACTION))
    X_fit, y_fit = X_train[:-n_val], y_train[:-n_val]
    X_val, y_val = X_train[-n_val:], y_train[-n_val:]

    clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    proba = clf.predict_proba(X_eval)[:, 1]
    pred = (proba >= 0.5).astype(float)
    correct = pred == y_eval
    da = float(correct.mean() * 100.0)
    brier = float(np.mean((proba - y_eval) ** 2))

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

    importance = _group_importance(clf.feature_importances_, feat_names)

    return {
        "arm": arm["name"],
        "fold": fold["name"],
        "n_train": int(len(y_fit)),
        "n_val": int(len(y_val)),
        "n_eval": int(len(y_eval)),
        "best_iteration": (
            int(clf.best_iteration_) if clf.best_iteration_ else LGBM_PARAMS["n_estimators"]
        ),
        "da": da,
        "brier": brier,
        "naive_da": naive_da,
        "da_on_naive_disagree_subset": da_on_disagree,
        "n_naive_disagree": int(disagree_mask.sum()),
        "feature_importance_gain": importance,
        "correct": correct,
        "eval_positions": eval_positions_kept,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default=str(Path(__file__).parent / "input_screening_nonlinear_results.json")
    )
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
            res = run_arm_fold_nonlinear(arm, fold, frames)
            dt = time.time() - t1
            print(
                f"  arm={arm['name']:20s} n_train={res['n_train']:6d} n_eval={res['n_eval']:5d} "
                f"DA={res['da']:.2f}% brier={res['brier']:.4f} best_iter={res['best_iteration']:4d} ({dt:.1f}s)"
            )
            if arm["id"] == 0:
                control_result = res
            fold_results[arm["name"]] = res

        for name, res in fold_results.items():
            if name == "price_only_control":
                res["mcnemar_p_vs_control"] = None
                res["delta_da_vs_control_pp"] = 0.0
                continue
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
            common = np.intersect1d(control_result["eval_positions"], res["eval_positions"])
            ctrl_common = np.array([ctrl_map[p] for p in common])
            arm_common = np.array([arm_map[p] for p in common])
            res["n_common_with_control"] = int(len(common))
            res["mcnemar_p_vs_control"] = _mcnemar_p(ctrl_common, arm_common)
            res["delta_da_vs_control_pp"] = res["da"] - control_result["da"]
        all_results[fold["name"]] = fold_results

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
            "per_fold_delta_pp": dict(zip([f["name"] for f in FOLDS], deltas, strict=True)),
            "per_fold_p": dict(zip([f["name"] for f in FOLDS], pvals, strict=True)),
            "n_significant_folds": n_sig,
            "avg_delta_pp": avg_delta,
            "graduates": graduates,
        }

    dump = {}
    for fold_name, fold_results in all_results.items():
        dump[fold_name] = {}
        for arm_name, res in fold_results.items():
            r = {k: v for k, v in res.items() if k not in ("correct", "eval_positions")}
            dump[fold_name][arm_name] = r

    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": "LGBMClassifier",
        "params": {k: v for k, v in LGBM_PARAMS.items()},
        "results_by_fold": dump,
        "verdicts": verdicts,
        "graduation_rule": {
            "min_significant_folds": GRADUATION_MIN_SIGNIFICANT_FOLDS,
            "min_avg_delta_pp": GRADUATION_MIN_AVG_DELTA_PP,
            "bonferroni_alpha": BONFERRONI_ALPHA,
        },
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nWrote results to {out_path}")

    print("\n=== Graduation verdicts ===")
    for name, v in verdicts.items():
        print(
            f"  {name:20s} avg_delta={v['avg_delta_pp']:+.2f}pp n_sig={v['n_significant_folds']}/3 "
            f"-> {'GRADUATES' if v['graduates'] else 'does not graduate'}"
        )


if __name__ == "__main__":
    main()
