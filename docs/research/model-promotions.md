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

---

## 2026-08-09 — ETHUSDT/basic — NO CHANGE (weekly retrain evaluated, incumbent retained)

No symlink moved. Recorded here so the weekly retrain leaves a trail even when it declines to promote.

- **Incumbent (retained)**: `basic/2026-07-04_22h_v1`
- **Challenger (rejected)**: `price/2026-08-09_07h29m50s_v1`, SageMaker job `atb-ethusdt-1h-20260809-071153`,
  full-history price-only retrain (2017-08-17 → 2026-08-09), hyperparameters matched to the incumbent
  (cnn_lstm, 50 epochs, batch 256, sequence length 120) so fresh data was the only changed variable.
  645 billable seconds on spot `ml.g4dn.xlarge` (~$0.04).

| Metric | Incumbent `2026-07-04_22h_v1` | Challenger `2026-08-09_07h29m50s_v1` | Winner |
|---|---|---|---|
| Test RMSE (temporal holdout) | 0.065141 | 0.065983 | incumbent (+1.29% worse) |
| Train RMSE | 0.063904 | 0.063527 | challenger |
| OOS profit factor | 0.4814 | not measurable — see below | — |
| OOS total return | -1.33% | not measurable — see below | — |
| OOS max drawdown | 3.26% | not measurable — see below | — |
| OOS win rate | 83.33% (6 trades) | not measurable — see below | — |

Incumbent backtest: hyper_growth, ETHUSDT 1h, 2026-06-10 → 2026-08-09, `--initial-balance 85
--risk-per-trade 0.02 --max-risk-per-trade 0.03 --max-position-size 0.20`, model pinned with
`--model-version`. Sharpe 0.03, final balance $83.86, buy-and-hold over the same window +18.34%.

- **Decision**: incumbent retained. The challenger loses on test RMSE, and test RMSE was the only
  metric in the gate that could be measured honestly this week.
- **Why the backtest column is empty** — two independent blockers, either one sufficient:
  1. **Look-ahead contamination.** A model trained through 2026-08-09 has the entire 2026-06-10 →
     2026-08-09 evaluation window inside its training set. The incumbent has ~24 of those 60 days
     in-sample; the challenger has 60 of 60. A challenger win on return/PF would measure memorisation,
     not edge. Fixing this needs a second job with `--end-date` set 60 days back so the eval window is
     genuinely held out (roughly doubles run cost).
  2. **Cloud bundles are missing `price_normalization`.** See the defect note below — the backtest
     would have produced silently wrong numbers regardless.
- **Refs**: weekly-model-retrain scheduled task; branch `claude/weekly-retrain-2026-08-09`

### Defect found: `atb train cloud` bundles omit `price_normalization`

The synced cloud bundle's `metadata.json` carries neither `price_normalization` nor `model_file`/
`framework`, all of which the locally-trained incumbent has (`cli/commands/train_commands.py:280-281`
writes them; the cloud path does not). Feature names and `feature_schema.json` are byte-identical
between the two, so the gap is metadata-only — but it is load-bearing:

- `src/prediction/models/onnx_runner.py:534` gates denormalization on `metadata.get("price_normalization")`
- `src/prediction/engine.py:1183` returns the normalized value unchanged when `method != "rolling_minmax"`

Neither path raises. A bundle without that key keeps its output in normalized ~[0,1] space while the
strategy compares it against real ETH prices, so every prediction is wrong and **nothing fails loudly**.
Any `atb train cloud-promote SYMBOL VERSION --to basic --set-latest` would therefore point a live
strategy at silently garbage predictions. This is the same silent-fabrication class as the pre-#838
partial-exit bug: wrong numbers that look plausible.

