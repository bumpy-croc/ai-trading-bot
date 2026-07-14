#!/usr/bin/env python3
"""EV-CONDITIONING (GH #984 part 2) -- does per-trade EV vary with anything
observable at entry?

Reads experiments/ev_conditioning_features.jsonl (136 control-arm HyperGrowth
trades, ETHUSDT/1h, F1-F3, correct-symbol config per GH #997/#998) and tests,
per conditioner, whether win-rate and/or pnl_percent vary significantly
across it. Four conditioners (not five): predicted_return magnitude,
confidence, and strength are a single perfectly-collinear quantity in this
population (corr=1.0, verified separately) so they count as ONE test, not
three, in the multiple-comparison correction.

Pre-committed test design (written before reading any p-value below):
  - Continuous/ordinal conditioners (abs_predicted_return, realized_vol_entry):
    Spearman rank correlation vs pnl_percent (EV-trend test) and a
    Cochran-Armitage trend test of win-rate across quartile buckets
    (win-rate trend test), matching the 2026-07-05 confidence-calibration
    study's method for comparability (statsmodels' Logit is unavailable in
    this environment -- scipy/statsmodels ABI mismatch -- so Cochran-Armitage
    plus a Spearman cross-check substitutes for that study's three-way
    CA/logit/Spearman redundancy check).
  - Categorical conditioners (trend_label regime, session): Kruskal-Wallis
    across groups for pnl_percent, chi-square test of independence for win
    vs group.
  - 4 conditioners x 2 outcome tests = 8 tests. Bonferroni alpha = 0.05/8 =
    0.00625, two-sided.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ALPHA_RAW = 0.05
N_TESTS = 8
ALPHA_BONF = ALPHA_RAW / N_TESTS


def cochran_armitage(bucket_idx: np.ndarray, win: np.ndarray) -> tuple[float, float]:
    """Cochran-Armitage trend test (equally-spaced integer scores per bucket).

    Same method (and score convention) the 2026-07-05 confidence-calibration
    study used for its decile trend test. statsmodels is unavailable in this
    environment (scipy/statsmodels ABI mismatch), so this is a direct,
    dependency-free implementation rather than a Logit-based slope test --
    cross-checked below against Spearman, exactly as that study did with its
    three redundant methods.
    """
    buckets = np.unique(bucket_idx)
    n_i = np.array([(bucket_idx == b).sum() for b in buckets], dtype=float)
    r_i = np.array([win[bucket_idx == b].sum() for b in buckets], dtype=float)
    t_i = np.arange(1, len(buckets) + 1, dtype=float)  # equally spaced scores

    n_total = n_i.sum()
    r_total = r_i.sum()
    p_bar = r_total / n_total

    t_bar_weighted = (n_i * t_i).sum() / n_total
    numerator = (t_i * r_i).sum() - p_bar * (n_i * t_i).sum()
    variance = p_bar * (1 - p_bar) * ((n_i * (t_i - t_bar_weighted) ** 2).sum())
    z = numerator / np.sqrt(variance)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def load() -> pd.DataFrame:
    lines = Path("experiments/ev_conditioning_features.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    df = pd.DataFrame(rows)
    df["win_int"] = df["win"].astype(int)
    return df


def quantile_table(df: pd.DataFrame, col: str, n_q: int = 4) -> pd.DataFrame:
    labels = [f"Q{i+1}" for i in range(n_q)]
    df = df.copy()
    df["_bucket"] = pd.qcut(df[col], n_q, labels=labels, duplicates="drop")
    g = df.groupby("_bucket", observed=True).agg(
        n=("pnl_percent", "size"),
        win_rate=("win_int", "mean"),
        mean_pnl_pct=("pnl_percent", "mean"),
        mean_value=(col, "mean"),
    )
    return g


def continuous_tests(df: pd.DataFrame, col: str, label: str) -> dict:
    print(f"\n=== {label} ({col}) ===")
    table = quantile_table(df, col)
    print(table.to_string(float_format=lambda x: f"{x:.5f}"))

    # EV trend: Spearman(value, pnl_percent)
    rho_pnl, p_pnl = stats.spearmanr(df[col], df["pnl_percent"])
    print(f"Spearman(value, pnl_percent): rho={rho_pnl:.4f} p={p_pnl:.4f}")

    # Win-rate trend: Cochran-Armitage on quartile buckets (primary test,
    # matches the calibration study's method) + Spearman(value, win) cross-check.
    bucket_idx = pd.qcut(df[col], 4, labels=False, duplicates="drop").values
    z_ca, p_ca = cochran_armitage(bucket_idx, df["win_int"].values)
    rho_win, p_win_spearman = stats.spearmanr(df[col], df["win_int"])
    print(
        f"Cochran-Armitage (win trend across quartiles): Z={z_ca:.4f} p={p_ca:.4f} | "
        f"Spearman(value, win) cross-check: rho={rho_win:.4f} p={p_win_spearman:.4f}"
    )
    return {
        "conditioner": label,
        "pnl_test": "spearman",
        "pnl_stat": rho_pnl,
        "pnl_p": p_pnl,
        "win_test": "cochran_armitage",
        "win_stat": z_ca,
        "win_p": p_ca,
        "win_p_spearman_crosscheck": p_win_spearman,
    }


def categorical_tests(df: pd.DataFrame, col: str, label: str) -> dict:
    print(f"\n=== {label} ({col}) ===")
    g = df.groupby(col, observed=True).agg(
        n=("pnl_percent", "size"),
        win_rate=("win_int", "mean"),
        mean_pnl_pct=("pnl_percent", "mean"),
    )
    print(g.to_string(float_format=lambda x: f"{x:.5f}"))

    groups = [grp["pnl_percent"].values for _, grp in df.groupby(col, observed=True)]
    h_stat, p_kw = stats.kruskal(*groups)
    print(f"Kruskal-Wallis (pnl_percent across {col}): H={h_stat:.4f} p={p_kw:.4f}")

    ct = pd.crosstab(df[col], df["win_int"])
    chi2, p_chi2, dof, _ = stats.chi2_contingency(ct)
    print(f"Chi-square (win vs {col}): chi2={chi2:.4f} dof={dof} p={p_chi2:.4f}")
    return {
        "conditioner": label,
        "pnl_test": "kruskal_wallis",
        "pnl_stat": h_stat,
        "pnl_p": p_kw,
        "win_test": "chi_square",
        "win_stat": chi2,
        "win_p": p_chi2,
    }


def main() -> int:
    df = load()
    print(f"Loaded {len(df)} trades")
    print(
        f"Overall win rate: {df['win_int'].mean():.4f}  mean pnl_percent: {df['pnl_percent'].mean():.5f}"
    )

    print("\n--- Collinearity check: predicted_return magnitude / confidence / strength ---")
    print(
        "corr(confidence, strength) =",
        np.corrcoef(df["confidence"], df["strength"])[0, 1],
        "-- treated as ONE conditioner (using confidence as the representative value).",
    )

    results = []
    results.append(
        continuous_tests(
            df, "confidence", "predicted-return magnitude / confidence / strength (collinear)"
        )
    )
    results.append(continuous_tests(df, "realized_vol_entry", "realized volatility at entry"))
    results.append(categorical_tests(df, "trend_label", "regime (trend_label)"))
    results.append(categorical_tests(df, "session", "hour-of-day session bucket"))

    print("\n" + "=" * 70)
    print(
        f"DECISION TABLE (Bonferroni alpha = {ALPHA_RAW}/{N_TESTS} = {ALPHA_BONF:.5f}, two-sided)"
    )
    print("=" * 70)
    any_significant = False
    for r in results:
        pnl_sig = r["pnl_p"] < ALPHA_BONF
        win_sig = r["win_p"] < ALPHA_BONF
        any_significant = any_significant or pnl_sig or win_sig
        print(
            f"{r['conditioner']:<55} pnl_p={r['pnl_p']:.4f} ({'SIG' if pnl_sig else 'ns'})  "
            f"win_p={r['win_p']:.4f} ({'SIG' if win_sig else 'ns'})"
        )
    print("\nAny conditioner significant at Bonferroni threshold:", any_significant)

    out_path = Path("experiments/ev_conditioning_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "n_trades": len(df),
                "alpha_bonferroni": ALPHA_BONF,
                "n_tests": N_TESTS,
                "results": results,
                "any_significant": any_significant,
            },
            f,
            indent=2,
            default=float,
        )
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
