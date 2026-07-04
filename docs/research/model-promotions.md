# Model promotions

Log of every `latest` symlink change in `src/ml/models/`. One entry per change: date, symbol,
old version, new version, reason, eval numbers. Append-only.

---

## 2026-07-05 — ETHUSDT/basic

- **Old version**: none (first `basic` bundle for ETHUSDT; only a `sentiment` bundle existed previously)
- **New version**: `2026-07-04_22h_v1`
- **Reason**: HyperGrowth trades ETHUSDT live but had no native price model, so every ETHUSDT bar was scored with the BTCUSDT/basic model (cross-symbol substitution, see #867/#872). This ships the first native ETHUSDT basic model so the #867 fail-fast guard can arm without `FEATURE_ALLOW_CROSS_SYMBOL_MODEL`.
- **Eval numbers**: test_rmse 0.065141 (train 0.063904, vs BTCUSDT/basic bar of 0.0665), directional_accuracy 0.5312 on temporal holdout (2024-09-25 → 2026-07-04). Validation backtest (hyper_growth, ETHUSDT, 1h, 90d, prod-matched risk params): native -1.31% vs cross-symbol baseline -3.29% vs buy-and-hold -15.49%; 11 trades, 72.73% win rate, MaxDD 3.05%, Sharpe 0.04.
- **Scope of this symlink change**: registry-only, in this PR's worktree/branch (`feat/ethusdt-basic-model`, PR #886). Does **not** affect any live-trading process — no production `latest` was touched. Per ml-engineer operating rules, this must run in paper for at least 48h after merging to `develop`/staging before any promotion proposal to live, which requires pm + human sign-off.
- **Refs**: PR #886, issue #887, #867, #872

