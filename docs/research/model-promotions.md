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

---

## 2026-09-06 — ETHUSDT/basic — NO CHANGE (weekly retrain evaluated, promotion NOT recommended)

No symlink moved. Third consecutive weekly retrain; the first in which the challenger technically
clears the 2-of-3 gate, and the first in which clearing it means nothing.

- **Incumbent (retained)**: `basic/2026-07-04_22h_v1`
- **Challenger**: `price/2026-09-06_07h21m47s_v1`, SageMaker job `atb-ethusdt-1h-20260906-071131`,
  full-history price-only retrain (2017-08-17 → 2026-09-06), 50 epochs, batch 256, sequence
  length 120. **345 billable seconds** vs 591 training seconds (~42% managed-spot saving) on
  `ml.g4dn.xlarge` ≈ **$0.07**.
- **Training image**: ECR `latest`, OCI provenance label `bcedb26c`
  (branch `fix/ecr-training-image-1041`). `git diff bcedb26c origin/develop --
  src/ml/training_pipeline` is **empty**, so the baked-in pipeline is byte-identical to develop.

| Metric | Incumbent `2026-07-04_22h_v1` | Challenger as-shipped | Challenger + #1049 patch | Gate |
|---|---|---|---|---|
| Test RMSE (temporal holdout) | **0.0651406** | 0.0676403 | 0.0676403 | incumbent (+3.84% worse) — **FAIL** |
| Train RMSE | 0.0639041 | 0.0654346 | 0.0654346 | incumbent |
| OOS profit factor | 999.0 (sentinel) | 0.0 | 999.0 (sentinel) | **tie — PASS** |
| OOS total return | 3.279229% | 0.00% | 3.284429% | challenger by **+0.0052pp** — **PASS** |
| OOS max drawdown | 1.34693% | 0.00% | 1.34700% | incumbent (negligible) |
| OOS win rate | 100% (7 trades) | — (0 trades) | 100% (7 trades) | tie |
| Sharpe | 0.051208 | 0.000 | 0.051290 | challenger (negligible) |
| Final balance | $87.6855 | $85.00 | $87.6879 | challenger (+$0.0024) |

Backtests: hyper_growth, ETHUSDT 1h, 2026-07-08 → 2026-09-06, `--initial-balance 85
--risk-per-trade 0.02 --max-risk-per-trade 0.03 --max-position-size 0.20`, each model pinned with
`--model-version` (no symlink was ever moved). Buy-and-hold over the same window: **+44.23%** —
both models underperform hold by ~41pp.

- **Gate arithmetic**: challenger is >= incumbent on 2 of 3 (PF tie, return by 0.0052pp), so the
  mechanical gate opens a PR.
- **Recommendation: do NOT merge / do NOT promote.** The gate pass is an artifact, not evidence:
  - the profit-factor "win" is a tie between two sentinel 999.0 values (both runs had zero losing
    trades), so that criterion carries no information and is unwinnable-but-untieable by design;
  - the return "win" is +0.0052 percentage points — $0.0024 on an $85 book, across 7 trades that
    are otherwise identical in count, direction and duration to the incumbent's;
  - the only criterion with discriminating power, test RMSE, the challenger **loses** by 3.84%.
  Two of three criteria are degenerate this week, so "2 of 3" reduces to "lost the only real one".

### #1049 reproduced for the third consecutive week — still unfixed

`price/2026-09-06_07h21m47s_v1/metadata.json` again ships without `price_normalization`,
`feature_strategy`, `model_file` or `framework`. As-shipped the bundle trades **0 times** over the
60-day window (contained only by ETHUSDT's long-only guard, #1020). Injecting those four keys
restored a normal decision mix and the 7 trades in the patched column above. The committed artifacts
are the **unmodified** pipeline output; the patch existed only in the evaluation worktree and was
deleted afterwards. Until #1049 lands, no cloud bundle can be promoted without hand-editing
metadata, which is not an acceptable promotion path.

### New defect found this run: `--model-type lstm` silently builds a CNN-LSTM

Previous entries carried a caveat that the incumbent is a pure LSTM while the pipeline's price-only
path builds `cnn_lstm`, confounding architecture with data. This run passed `--model-type lstm` to
remove that confound. It did not work, and the reason is a mislabelled alias:

`create_model(model_type="lstm")` (`src/ml/training_pipeline/models.py:314`) calls
`build_price_only_model()`, which at `models.py:405-415` returns
`create_model("cnn_lstm", ...)`. The `lstm` branch is a pass-through to `cnn_lstm`.

Verified against the emitted graphs:

| | ONNX op types |
|---|---|
| Incumbent `2026-07-04_22h_v1` | `Conv=0  Loop=2  GRU=0`, 27 nodes — a real LSTM |
| Challenger `2026-09-06_07h21m47s_v1` | `Conv=2  Loop=0  GRU=2`, 75 nodes — a CNN-GRU |

Meanwhile the challenger's `metadata.json` records `"architecture": "lstm"`. The provenance field
therefore states something the artifact contradicts. Consequences:

- there is **no CLI path to train a pure LSTM**, so the architecture confound in this and the two
  prior weekly comparisons cannot currently be removed;
- any tournament or study that treated `lstm` and `cnn_lstm` as two distinct entrants was comparing
  a model against itself;
- `metadata["architecture"]` cannot be trusted for provenance until this is fixed.

### Caveats on this comparison

1. **The backtest is not out-of-sample for the challenger.** It trains through 2026-09-06, so all 60
   evaluation days are in-sample; the window is genuinely held out only for the incumbent (trained
   through 2026-07-04). The comparison is biased **toward** the challenger — which makes its RMSE
   loss robust and its 0.0052pp return "win" worthless. A clean read needs a second
   `--end-date`-shifted job (~$0.07 more).
2. **RMSE is not like-for-like** — each model's holdout is its own chronological split, and the
   challenger's split ends ~2 months later than the incumbent's.
3. **n=7 trades**, 100% win rate on both sides — far too small to separate two models.
4. **The cloud pipeline computes no `directional_accuracy`**, so the incumbent's 0.5312 has no
   counterpart and that metric could not be compared at all.
- **Refs**: weekly-model-retrain scheduled task; branch `chore/weekly-retrain-20260906`; #1049, #1041
