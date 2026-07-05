# Model Evaluation & Continuous Improvement System

*Status: DESIGN (2026-07-05, Board-directed). North-star question: **how do we make the models
as accurate as possible at predicting future prices, in any market condition?***

This codifies the verification methodology proven during the 2026-07-04/05 model campaign into a
reusable system. Every future model — new architecture, new window, new features, new symbol —
flows through the same pipeline and is judged by the same rules.

## Principles (empirically earned, not aspirational)

1. **Prediction error does not rank models.** W_3y beat W_full on holdout RMSE and lost the
   out-of-sample trading exam by 4 points of return. Selection keys on OOS *trading* metrics
   (profit factor, return, drawdown) with the real strategy, real fees, prod-matched limits.
2. **Leakage is the default, honesty is engineered.** The same model scored −1.31% leaked vs
   −7.43% clean. Every exam window must postdate every candidate's training cutoff; purge/embargo
   ≈ 1% of bars at boundaries; no shuffling anywhere.
3. **One shared exam, or no comparison.** Candidates are only comparable on identical eval
   windows. Per-model holdouts are health checks, not rankings.
4. **A backtest buys a staging ticket, never a live deployment.** Only observed forward
   performance (paper, then small live) buys risk.
5. **Regime robustness beats single-window wins** (charter). A candidate must not catastrophically
   fail any regime slice even if its aggregate wins.
6. **Always include dumb baselines.** A linear/naive-persistence baseline runs in every
   tournament (literature: LSTMs can lose to linear on noisy financial data). If the fancy model
   can't beat naive persistence OOS, the answer is "no model," not "worse model."
7. **Retrain on triggers and cadence, not reflexively.** Drift is real (~2.5-3.5%/mo decay in
   literature) but the only head-to-head showed naive daily retraining LOSING to drift-triggered
   adaptation. Cadence: weekly candidate generation; adoption only through the gates.

## The pipeline

```
 ┌────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌───────────────┐   ┌─────────────┐
 │ TRAIN      │ → │ L1 HOLDOUT   │ → │ L2 SHARED EXAM   │ → │ L3a STAGING   │ → │ L3b LIVE    │
 │ (local or  │   │ temporal     │   │ frozen OOS       │   │ paper ≥48h    │   │ small+      │
 │ SageMaker) │   │ 80/20        │   │ window, real     │   │ real future   │   │ monitored   │
 │ ~$0.37     │   │ sanity gate  │   │ strategy+fees    │   │ fake money    │   │ tripwires   │
 └────────────┘   └──────────────┘   └──────────────────┘   └───────────────┘   └─────────────┘
      ▲                                      │                      │                  │
      │                                      ▼                      ▼                  ▼
      │            ┌────────────────────────────────────────────────────────────────────┐
      └────────────│ REGISTRY + SCOREBOARD: every candidate's config, data window, git   │
   retrain trigger │ SHA, image digest, L1/L2/L3 results — append-only, auditable        │
   (cadence/drift) └────────────────────────────────────────────────────────────────────┘
```

### L1 — Temporal holdout (per candidate, cheap, automatic)
Chronological 80/20 split (already built into `atb train price` / the training pipeline).
**Gate**: model converged (early stopping), test RMSE within sane range of train RMSE (no gross
overfit), metadata + feature schema complete. Failing L1 = discard, no exam seat.

### L2 — Shared frozen exam (the referee)
- **Exam definition** = frozen artifact: `{cutoff_date, eval_start, eval_end, symbol, timeframe,
  strategy, engine flags}` committed to `docs/research/exams/`. Candidates must have
  `training_end ≤ cutoff`. Rotating quarterly exams prevent overfitting-to-one-exam.
- Runs the corrected backtest engine (post-#838 only) with prod-matched limits.
- **Metrics contract** (recorded for every candidate): OOS return, profit factor, MaxDD (MTM),
  win rate, trade count, Sharpe, confidence distribution (median/IQR of decision confidences),
  per-regime-slice breakdown (bull/bear/chop months within the window), vs-hold delta, and the
  naive-baseline delta.
- **Gate to L3a**: beats incumbent on ≥2 of {PF, return, MaxDD} on the shared exam AND does not
  fail any regime slice by more than X (default 5pp) where the incumbent passed.
- Multiple-comparison discipline: the more candidates tried against one exam, the higher the
  luck risk — record candidate count per exam; refresh the exam window when it exceeds ~10.

### L3a — Staging paper (real future, fake money)
Winner deploys to staging (`develop` sync + `latest` symlink in the staging build only).
≥48h and ≥N decisions observed. **Gate**: behavior consistent with L2 expectations (confidence
distribution, trade frequency, sizing); no guard/parity anomalies. Recorded to the scoreboard.

### L3b — Live (small, monitored, revocable)
Prod promotion per the model-promotion process (#887 pattern): explicit PR, rollback note,
risk-officer pass. Live monitoring treats performance as a hypothesis under test: standup
tripwires + (planned) drift detection on realized prediction error (ADWIN/Page-Hinkley family;
two-layer confirmation to cut false alarms). **Drift alarm ⇒ retraining trigger + optional
exposure reduction — never silent.**

## Retraining loops

| Loop | Trigger | What it does |
|---|---|---|
| **Weekly candidate** | `weekly-model-retrain` scheduled task (Sun 08:08) | Cloud-trains full-history-to-now candidate (~$0.37); L1+L2 vs incumbent; PR only if gates pass. Self-blocks if #890 unmerged or ECR image older than `training_pipeline/` code. |
| **Drift-triggered** (planned) | Live prediction-error drift detector | Same as weekly but immediate, plus pages PM; optionally FEATURE_ENTRY_PAUSE while degraded. |
| **Research tournament** | PM-dispatched (new windows/features/architectures) | N candidates through L1+L2; results to `docs/research/experiments/`; winner earns L3a. |

## What exists vs what to build

**Exists**: training pipeline w/ temporal split (L1); corrected backtest engine (#838); cloud
training (#890 hardening in flight — date ranges, fresh image, promotion command); model registry
w/ versioned bundles + `latest` symlinks; staging paper environment + observer tasks; standup
tripwires; walk-forward CLI (`atb walk-forward`); experiment framework (`atb experiment`).

**To build** (in order):
1. **`atb models exam` command**: run a frozen exam definition against candidate(s) and emit the
   metrics contract as JSON + markdown — replaces this weekend's hand-rolled scripts. Reuse the
   experiment framework/walk-forward internals; do not fork a new engine path.
2. **Scoreboard**: `docs/research/model-scoreboard.md` (append-only table) + per-exam JSON under
   `docs/research/exams/` — the institutional memory of every candidate ever tried.
3. **Regime slicing** in the exam runner (bull/bear/chop labels from monthly returns, simple and
   deterministic — not the live regime detector, to keep the exam stable).
4. **Naive baselines** in every exam (persistence + linear).
5. **Drift detector** on live prediction error feeding the drift-triggered loop (needs #871's
   sibling-generator cleanup for uniform prediction logging first).
6. **Confidence calibration study** (the current top profit lever: 53% directional accuracy
   gated at noise-level confidences — the raw-output→confidence mapping and the 0.05 gate were
   never calibrated to realized hourly move distributions).

## Open questions the system must answer over time
- Does any window/weighting scheme actually dominate across regimes, or is the honest answer
  "full history + drift-triggered refresh"? (Tournament #1 in flight: W_full ahead so far.)
- Is next-bar price the right target at all, vs direction-classification or vol-normalized
  returns (ReVol-style normalization showed large IC gains in literature)?
- Where does the 53%-direction / 0.03-confidence disconnect come from (magnitude compression),
  and does fixing calibration unlock the existing edge before any new model does?
