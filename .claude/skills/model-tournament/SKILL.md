---
name: model-tournament
description: Run a rigorous, reusable model tournament — train N candidate models (different architectures, data windows, targets, or features) under identical conditions and rank them honestly on out-of-sample trading performance. Use when comparing models, evaluating a new architecture/window/target, or answering "which model is best". Encodes the frozen-exam protocol and every leak/validity trap found in the July 2026 tournaments.
---

# Model Tournament

Compare N candidate models honestly. The referee is the **frozen shared exam** — never
holdout RMSE. Full system design: `docs/architecture/model_evaluation_system.md`.
Prior art (read one before your first tournament): `docs/research/experiments/
2026-07-05_window-tournament.md`, `2026-07-06_architecture-tournament.md`.

## Iron rules (each one paid for in blood — violations invalidate the tournament)

1. **No overlap between any candidate's training data and the exam window.** All
   candidates train with `--end-date <cutoff>`; the exam is strictly after the cutoff.
   A leaked eval window inflated results by 6pp once (−1.31% leaked vs −7.43% clean).
2. **One identical exam for all candidates**: same window, same strategy, same
   prod-matched flags (`--initial-balance 85 --risk-per-trade 0.02
   --max-risk-per-trade 0.03 --max-position-size 0.20` — update if prod config changes),
   same fees. Per-candidate holdouts are health checks, NOT rankings (holdout RMSE
   improved monotonically with shorter windows once while OOS P&L got WORSE).
3. **Rank on money metrics** (OOS return, profit factor, MTM MaxDD, win rate) plus
   directional accuracy — never on RMSE/MAE alone.
4. **Determinism guard**: before trusting any exam result, run the FIRST exam twice
   back-to-back — results must be identical. (PredictionEngine's inference timeout
   silently substituted HOLD under CPU load until fixed; guard against regressions.
   If needed, set the `MAX_PREDICTION_LATENCY` env override high for exam runs.)
5. **Identical treatment**: same feature pipeline, same target column, same
   epochs/batch/seq-length policy, same seed policy for every candidate. Verify the
   TARGET is `close_normalized` (a raw-`close` target once regressed dollar prices and
   produced RMSE ~2985 vs ~0.065 — incomparable garbage). Verify features via the
   training log's `feature_names`. No per-candidate hyperparameter tuning.
6. **Baselines in every table**: the current incumbent's numbers on the same exam, and
   naive persistence directional accuracy (~50%) computed on the exam window. If no
   candidate beats naive persistence, the answer is "no model", not "best of the bad".
7. **Statistics before verdicts**: n≈4,400 hourly exam bars → directional-accuracy SE
   ≈0.75pp — differences under ~1.5pp are NOISE. For decile/gradient claims use Wilson
   CIs + a trend test, and check the gradient survives OOS (a training-period gradient
   that vanishes OOS = overfitting, seen in the calibration study).
8. **Training data comes from the local Binance parquet cache** (prefill first:
   `atb data prefill-cache --symbols <S> --timeframes 1h --start <...> --end <...>
   --cache_dir cache/market_data`). Never allow silent third-party fallback into a
   training corpus. Verify loaded row count matches expectations in the log.
9. **1-epoch smoke test on candidate 1 before the full run** — check the metric is in
   the expected normalized range (~0.05–0.25 at 1 epoch, NOT thousands) and
   feature_names are correct. Smoke artifacts must be deleted (never left as `latest`).
10. **Multiple-comparisons discipline**: record how many candidates have faced each
    exam window; retire/rotate the window after ~10.

## Execution

**Cloud-first (standard, post-#918)**: parallel SageMaker jobs via
`atb train cloud <SYMBOL> --start-date <...> --end-date <cutoff> --model-type <arch>
--model-variant <v> --force-price-only ...` (~$0.10/candidate, all candidates in
parallel, <1h). Preconditions: ECR image NEWER than the latest
`src/ml/training_pipeline/` change (rebuild via `./src/ml/cloud/build-and-push.sh`,
linux/amd64), and the fixes from #918 merged. Sync bundles back, verify each has
`model.onnx`. Exams always run LOCALLY (backtests).

**Local fallback** (protocol-experimental runs needing unpushed pipeline patches):
strictly sequential — ONE training or backtest at a time machine-wide (thermal +
determinism); expect 1.5–4x nominal durations under load; run in a disposable worktree
from `origin/develop` (`git worktree add .claude/worktrees/<name> origin/develop
--detach`), document any worktree-local pipeline patches explicitly in the write-up.

**Process discipline** (agents running this): long steps via a single background
process, then END TURN — never poll. Wake-ups are lossy: keep a crash-safe incremental
state JSON in the scratchpad updated after EVERY stage, and leave a continuation recap
each turn so a resume (or the PM's backstop) can pick up statelessly. Verify claims
relayed by coordinators against the filesystem/logs before acting on them.

## Deliverables

1. `docs/research/experiments/YYYY-MM-DD_<name>-tournament.md`: per-candidate table
   (train window/config, holdout RMSE, OOS return/PF/MaxDD/win-rate/trades, confidence
   distribution median/IQR, directional accuracy + CI), baselines, statistical verdict,
   explicit answer to the tournament's question, staging-slot recommendation (winner →
   staging paper ≥48h before any promotion talk; a backtest win buys a trial, never a
   deployment).
2. GitHub issue (`type:experiment`, `owned-by:quant-researcher`) + `.claude/state/log.md`
   entry + a promotion row in `docs/research/model-promotions.md` if the tournament moves a
   `latest` symlink (the append-only promotion log; there is no `model-scoreboard.md`).
3. Remove the disposable worktree; delete smoke/stray model artifacts; never commit
   experiment-local pipeline patches without the PM's explicit upstreaming decision.

## Red flags — stop and re-check

- Any metric wildly out of scale (RMSE in dollars, MaxDD ~0 with big returns, 0% win
  rate with positive P&L) → accounting/units artifact, not a result.
- A candidate's exam beats everything by a margin that survives no re-run.
- Training log shows a network fetch or unexpected data source instead of cache.
- `latest` symlink in a SHARED registry moved by tournament activity (tournaments run
  in disposable worktrees precisely so this can't touch anything real).
