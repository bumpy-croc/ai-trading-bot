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


---

## 2026-08-23 — ETHUSDT/basic — NO CHANGE (weekly retrain evaluated, incumbent retained)

No symlink moved. Second consecutive weekly retrain to decline promotion.

- **Incumbent (retained)**: `basic/2026-07-04_22h_v1`
- **Challenger (rejected)**: `price/2026-08-23_07h40m55s_v1`, SageMaker job `atb-ethusdt-1h-20260823-071439`,
  full-history price-only retrain (2017-08-17 → 2026-08-23), hyperparameters matched to the incumbent
  (cnn_lstm, 50 epochs, batch 256, sequence length 120). 655 billable seconds vs 927s wall clock
  (~29% managed-spot saving) on `ml.g4dn.xlarge` ≈ **$0.13**.
- **Training image**: ECR `latest`, provenance label `bcedb26c`. Verified current — `git log
  bcedb26c..origin/develop` over `src/ml/training_pipeline`, `src/ml/cloud`, `cli`, `pyproject.toml`
  is empty, so no develop commit is missing from the image.

| Metric | Incumbent `2026-07-04_22h_v1` | Challenger as-shipped | Challenger + metadata patch |
|---|---|---|---|
| Test RMSE (temporal holdout) | **0.065141** | 0.066355 | 0.066355 |
| Train RMSE | 0.063904 | 0.063776 | 0.063776 |
| OOS profit factor | **999.0** (sentinel: zero losing trades) | 0.0 | 999.0 |
| OOS total return | **4.82%** | 0.00% | 4.63% |
| OOS max drawdown | **1.23%** | 0.00% | 1.34% |
| OOS win rate | 100% (9 trades) | — (0 trades) | 100% (8 trades) |
| Sharpe | 0.068 | 0.000 | 0.067 |
| Final balance | $89.17 | $85.00 | $89.01 |

Backtests: hyper_growth, ETHUSDT 1h, 2026-06-24 → 2026-08-23, `--initial-balance 85
--risk-per-trade 0.02 --max-risk-per-trade 0.03 --max-position-size 0.20`. Buy-and-hold over the
same window: +45.45% — both models underperform hold by ~41pp.

- **Decision**: incumbent retained. Gate is >= incumbent on 2 of 3 (test RMSE, OOS PF, OOS return);
  the challenger scores **0 of 3** as shipped, and **1 of 3** even when credited with the metadata
  patch and a tied profit factor. It is worse on both RMSE and return.
- **Refs**: weekly-model-retrain scheduled task; branch `claude/weekly-retrain-2026-08-23`; #1049

### #1049 reproduced — still unfixed, now with measured downstream impact

The 2026-08-09 entry flagged that cloud bundles omit `price_normalization` but could not measure the
consequence. This run did. The challenger as-shipped produced **zero trades**, and the cause is not
the weights — raw ONNX output is healthy and near-identical to the incumbent (pred-minus-close mean
+0.0014, 41.2% of predictions below current price vs the incumbent's 47.9%).

With denormalization silently skipped, a normalized ~0.55 is compared against a ~$2400 price, which
reads as a catastrophic crash on every bar:

| | Incumbent | Challenger as-shipped | Challenger + patch |
|---|---|---|---|
| Decision mix | 623 BUY / 607 SELL / 91 HOLD | **1321 SELL / 0 BUY / 0 HOLD** | 646 BUY / 581 SELL / 94 HOLD |
| Confidence | spread 0.00–1.00, mode 0.01 | **1.00 on all 1321 bars** | spread |

ETHUSDT is long-only (#1020), so every one of those signals was filtered and the damage surfaced as
zero trades rather than as losses. **On a symbol that allows shorts, this bundle would have shorted
every candle at maximum confidence.** That the long-only guard is what contained it is luck, not
defence in depth.

Injecting `price_normalization` + `feature_strategy` into the bundle metadata restored a normal
decision mix and 8 trades — confirming both the diagnosis and that the trained weights are sound.
`#1049` should be treated as blocking for the whole cloud retraining path: until it lands, no cloud
bundle can be honestly evaluated against an incumbent, let alone promoted.

### Caveats on this comparison

1. **The backtest legs are not out-of-sample for the challenger.** It trains through 2026-08-23, so
   all 60 evaluation days are in-sample; the incumbent has ~50 of 60 in-sample. Both backtest legs are
   biased *toward* the challenger — and it lost anyway, which only strengthens the retain decision.
   A clean read needs a second `--end-date`-shifted job (~$0.13 more).
2. **RMSE is not like-for-like** — each model's holdout is its own chronological split.
3. **Not a pure data refresh.** The incumbent's ONNX is a pure LSTM (2 `Loop` ops, no `Conv`); the
   pipeline's price-only path now builds `cnn_lstm` (Conv1D 48 + Conv1D 96 + GRU). Architecture
   changed alongside the data, so this comparison cannot attribute the difference to either one.
4. **n=9 vs n=8 trades, both 100% win rate** — far too small to separate two models. The profit-factor
   leg is near-unwinnable by construction against a zero-loss incumbent run.

### Two operational notes

- `src/ml/cloud/verify-image.sh` (on unmerged branch `fix/ecr-training-image-1041`) fails on macOS:
  `xargs -I{} curl -s {}` rejects the pre-signed layer URL with "command line cannot be assembled,
  too long". Read the config blob with a direct `curl "$url"` instead.
- Issue #1041 ("ECR image stale — blocks weekly retrain") is still open, but the rebuild it asks for
  was done on 2026-08-13. Looks closeable.
