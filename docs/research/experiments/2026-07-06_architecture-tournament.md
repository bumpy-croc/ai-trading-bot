# Architecture Tournament — ETHUSDT 1h Price Prediction

Date: 2026-07-06 to 2026-07-08 (multi-day, interrupted by a local machine reboot and two
cloud-training infra bugs — see incident log below)
Author: ml-engineer
Status: COMPLETE
Protocol: `docs/architecture/model_evaluation_system.md` (L1 holdout + L2 shared frozen exam),
executing Phase 1 of the Nov 2025 architecture research (`docs/ml_architecture_research.md`,
commit 27e3e341) that was never run.
Issue: filed at close (see bottom).

## North star and scope

Prediction accuracy (directional accuracy primary, money exam secondary), per the task brief.
Entrants: `cnn_lstm/default` (control), `attention_lstm/default`, `attention_lstm/lightweight`,
`tcn/default`, `tcn/lightweight`. Excluded: TFT (too slow for this pass), ensemble (Phase 2,
gated on a +10% winner — see verdict).

## Headline result (read this first)

**No architecture is a clear winner, and the L2 money exam cannot currently distinguish them.**
All 5 entrants cluster within a **1.29 percentage-point band of directional accuracy** (53.16%
to 54.45%), which is at the edge of statistical noise for the sample size involved (see
Statistical Read). All 5 entrants also produced a **bit-identical realized trade blotter**
through the `hyper_growth` L2 exam — not because the models agree, but because of a real,
independently-verified defect in how `hyper_growth` sizes positions (see Harness-Validity
Finding). Architecture selection is not the binding constraint on this system's profitability;
this result is consistent with, and adds direct evidence for, the standing hypothesis that the
constraint lives elsewhere (target definition, confidence calibration, or — as newly
discovered here — the position-sizing harness itself).

Three infrastructure bugs were found and fixed as a byproduct of running this tournament,
independent of the model verdict — listed as a standalone contribution below.

## Protocol

- **Symbol/timeframe**: ETHUSDT, 1h.
- **Training window**: 2017-08-17 (ETHUSDT's actual Binance listing date) → 2025-12-31, hard
  cutoff. Full history, no recency weighting (per the 2026-07-05 window-tournament's finding
  that full history ties-or-beats shorter windows on risk-adjusted terms).
- **Eval window**: 2026-01-01 → 2026-07-04 (185 days), strictly non-overlapping with training.
  Same frozen exam window used by the window tournament and the currently-deployed
  `2026-07-04_22h_v1` model's own validation lineage.
- **Feature contract**: `PriceOnlyFeatureExtractor` — 5 features (`close_normalized`,
  `volume_normalized`, `high_normalized`, `low_normalized`, `open_normalized`), rolling min-max
  normalization, window = `sequence_length` = 120. Target = `close_normalized`.
- **Hyperparameters held constant across all 5 entrants**: `epochs=50` (requested; actual
  epochs varied by architecture via EarlyStopping — see per-entrant table), `batch_size=256`,
  `sequence_length=120`. No per-entrant hyperparameter tuning (that is a later study, per the
  anti-fooling discipline in the task brief).
- **L2 exam**: `hyper_growth` strategy, prod-matched risk flags
  (`--initial-balance 85 --risk-per-trade 0.02 --max-risk-per-trade 0.03 --max-position-size 0.20`),
  fees/slippage on by default (not a debug fee-free run).

### Honest protocol trail — training location differs by entrant

This is disclosed prominently because it is the kind of thing that undermines a tournament's
credibility if buried:

- **`cnn_lstm/default`** trained **locally**, in a frozen worktree
  (`.claude/worktrees/arch-tournament`, detached at pre-existing commit `09a2830f`, i.e. before
  PR #925 merged to `develop`), using three worktree-local patches (see Infra Bugs, below) that
  were necessary to get a fair, feature-contract-matched comparison out of `atb train model`.
- **`attention_lstm/default`, `attention_lstm/lightweight`, `tcn/default`, `tcn/lightweight`**
  trained in the **cloud** (AWS SageMaker), on a Docker image built from **fresh `origin/develop`
  post-#925** — a mid-tournament pivot to cloud-first training, directed by the Board (documented
  in PR #925's own merged body: "Board decision (2026-07-06): tournaments run cloud-first from
  now on — parallel SageMaker jobs instead of sequential local training"), triggered by a local
  machine reboot after entrant 2's original local training run starved the box (~15% CPU after
  4h under load averages of 39-77 from concurrent work in other worktrees).
- PR #925 upstreamed the tournament's worktree-local feature-pipeline and cache-ingestion
  patches **near-verbatim** (confirmed via the PR's own diff description), so entrants 2-5 use
  the same 5-feature `PriceOnlyFeatureExtractor` contract, the same `close_normalized` target,
  and the same cache-first, no-silent-fallback data ingestion as entrant 1 — with one cosmetic
  difference: the upstreamed version keeps `force_price_only` bundles under the `price/` registry
  namespace (not `basic/`, deliberately, to avoid a training run silently repointing the live
  `basic/latest` symlink). This is a folder-naming difference only, not a feature/data/
  architecture difference. All 5 model artifacts were normalized into
  `{tournament-worktree}/src/ml/models/ETHUSDT/basic/{version}_{arch}_{variant}` before
  evaluation so L1/L2 scoring code and paths are identical across entrants.
- **L1 directional accuracy for all 5 entrants was computed via ONNX Runtime** (not
  `tf.keras.models.load_model`), because the cloud-trained `.keras` files hit a genuine
  Keras-version deserialization mismatch between the SageMaker training container and the local
  evaluation venv (`Could not deserialize class 'Functional'... keras.src.engine.functional
  cannot be imported`). Entrant 1's DA was independently re-verified via the same ONNX path
  (0.534899, identical to the original Keras-based figure) before trusting any cross-entrant
  comparison, so all 5 are scored by one consistent method, not "4 via ONNX + 1 via a
  Keras path that happened to also work."
- **All L2 exams ran in the frozen `arch-tournament` worktree** regardless of where training
  happened, so the backtest/strategy code exercised is identical for every entrant. Entrant 1's
  L2 exam was run twice back-to-back to validate determinism (`MAX_PREDICTION_LATENCY=30` env
  override, applied uniformly to every entrant's L2 run — the tournament worktree is
  deliberately frozen at a commit predating PR #923's proper deterministic-inference fix, so the
  env override substitutes for it to keep every entrant's exam conditions identical).

## L1 — Temporal holdout results

Chronological 80/20 split within the training window (2017-08-17→2025-12-31), no shuffling.
Directional accuracy computed by the tournament (the training pipeline itself does not emit
this metric — confirmed gap, same one the 2026-07-05 window-tournament experiment found):
`sign(predicted_close_normalized[t] − actual_close_normalized[t-1])` vs
`sign(actual_close_normalized[t] − actual_close_normalized[t-1])`, evaluated on the model's own
validation split (n = 14,633 samples, 14,141 with a nonzero actual move — identical n for all 5
entrants since they share the same training window and split ratio).

| Entrant | Version | Trained | Epochs (early-stop) | Test RMSE | Train RMSE | MAPE | **Directional Accuracy** |
|---|---|---|---|---|---|---|---|
| **cnn_lstm/default** (control) | `2026-07-06_17h_v1` | local | 50/50 | 0.06760 | 0.06525 | 54.17% | **53.49%** |
| attention_lstm/default | `2026-07-07_12h32m05s_v1` | cloud | — | 0.08361 | 0.08209 | 60.02% | **53.96%** |
| attention_lstm/lightweight | `2026-07-07_12h01m28s_v1` | cloud | — | 0.08623 | 0.08515 | 64.84% | **54.45%** |
| tcn/default | `2026-07-07_13h15m02s_v1` | cloud | — | 0.07563 | 0.07360 | — | **53.16%** |
| tcn/lightweight | `2026-07-07_12h57m51s_v1` | cloud | — | 0.09619 | 0.08958 | — | **54.18%** |
| **Naive persistence** (continuation, computed on eval window not holdout — see caveat) | — | — | — | — | — | — | **48.04%** |

Wall-clock training time: cnn_lstm/default 15,728s (4.4h, local CPU); attention_lstm/default
4,770s; attention_lstm/lightweight 2,893s; tcn/default 2,323s; tcn/lightweight 1,284s (all cloud,
`ml.g4dn.xlarge`, GPU — the ~5-6x speedup vs local CPU training is expected and not itself a
finding).

**Cross-check finding (reconfirms the window-tournament's 2026-07-05 result on a new axis):**
holdout test RMSE does **not** rank models the same way directional accuracy does.
`attention_lstm/lightweight` has the *worst* RMSE (0.0862) of the five but the *best* directional
accuracy (54.45%); `tcn/default` has better RMSE than both attention_lstm variants (0.0756) but
the *worst* directional accuracy (53.16%). This is now demonstrated across two independent axes
(training window, and now architecture) — training-time regression error is not a reliable proxy
for directional or trading quality in this system, a caution that should generalize beyond this
one tournament.

**Naive-baseline caveat**: the naive-persistence figure (48.04%, "next move continues the
previous move's sign") is computed on the **2026-01-01→2026-07-04 eval window** (n=4,436 nonzero
moves), not on the training-time holdout the 5 models are scored on. This is an honest limitation
of this tournament, not a fabricated apples-to-apples comparison — the training pipeline has no
path to produce genuine per-bar OOS directional accuracy for `hyper_growth`'s execution path
(same missing-instrumentation gap flagged by the window-tournament experiment). All 5 entrants
clear 48% by 5-6pp, which is suggestive that all 5 have *some* real directional edge over pure
chance/persistence, but this is not a same-window, apples-to-apples statistical claim — flagged
explicitly rather than silently presented as one.

## Statistical read

All 5 entrants share the same validation set (n=14,141 nonzero-move bars), so their DA figures
are directly comparable to each other (not to the naive baseline — different window, see above).

- Single-entrant SE ≈ sqrt(p(1-p)/14141) ≈ **0.42 percentage points** for p≈0.535.
- Two-entrant (independent-samples approximation) pairwise-difference SE ≈ **0.59pp**. This is
  conservative: since all 5 models are scored on the *same* underlying bars, a proper paired test
  (e.g. McNemar's) would likely show a *tighter* effective SE where models agree on most bars —
  not computed here, flagged as a follow-up refinement.
- **Observed range**: 53.16% (tcn/default) to 54.45% (attention_lstm/lightweight) = **1.29pp**,
  or **~2.2 pairwise SEs** — at the edge of distinguishable, not comfortably clear of it.
  Adjacent-entrant comparisons (e.g. attention_lstm/default 53.96% vs attention_lstm/lightweight
  54.45%, a 0.49pp gap) are well within 1 SE and not distinguishable at all.
- **Multiple-comparison caution**: 5 entrants → 10 pairwise comparisons. Even the extreme-pair
  gap (tcn/default vs attention_lstm/lightweight) should not be read as a confident, corrected-for
  winner given how many comparisons were implicitly run. Per the evaluation-system protocol's own
  rule ("the more candidates tried against one exam, the higher the luck risk"), this tournament
  counts as 5 candidates against the shared exam — worth tracking toward the "refresh the exam
  window at ~10 candidates" threshold for future tournaments on this same window.

**Reading of the ranking**: attention_lstm variants (53.96%, 54.45%) trend slightly above
cnn_lstm (53.49%) and tcn/default trends slightly below (53.16%); tcn/lightweight (54.18%) sits
in the upper half. But no pairwise gap clears a confident significance bar once multiple
comparisons are accounted for. The honest conclusion is a **tight cluster with a very weak,
statistically fragile lean toward attention mechanisms** — not a decisive win for any
architecture.

## L2 — Frozen exam (money backtest) results

**Read the Harness-Validity Finding section before drawing any conclusion from this table.**

| Entrant | Return | Profit Factor | Max DD | Win Rate | Trades | Sharpe |
|---|---|---|---|---|---|---|
| cnn_lstm/default | -7.47% | 0.693097 | 10.59% | 77.78% | 54 | 0.073 |
| attention_lstm/default | -7.47% | 0.693097 | 10.59% | 77.78% | 54 | 0.073 |
| attention_lstm/lightweight | -7.47% | 0.693097 | 10.59% | 77.78% | 54 | 0.073 |
| tcn/default | -7.47% | 0.693097 | 10.59% | 77.78% | 54 | 0.073 |
| tcn/lightweight | -7.47% | 0.693097 | 10.59% | 77.78% | 54 | 0.073 |
| Hold (no-trade) | -40.98% | — | — | — | 0 | — |
| **W_full baseline** (2026-07-05 window tournament, `atb train price` legacy path) | -7.43% | 0.673 | 10.55% | 76.9% | 52 | 0.073 |

Every one of the 5 entrants' `profit_factor` and `final_balance` values are **bit-identical to
16 significant figures** (`0.6930967547208055`, `$79.08544675970002`) across all runs, including
entrant 1's own two-run determinism check. `sharpe_ratio`/`sortino_ratio`/`var_95` differ at the
6th-7th decimal only (equity-curve path noise between otherwise-identical trades, not a
meaningfully different result).

**Do not read this table as "all 5 architectures are equivalent."** This is the entire point of
the next section.

## Harness-validity finding (the tournament's second major contribution)

The bit-identical L2 result across 5 architecturally different models — independently confirmed
to produce genuinely different raw predictions (max abs diff 0.31 on identical synthetic input,
ruling out a registry/loading bug) — was investigated and root-caused. **Filed as GitHub issue
#938**, full technical write-up referenced there. Verified directly from source, not taken on
report alone:

- `FlatRiskManager.calculate_position_size` (`src/strategies/hyper_growth.py:97-118`) uses
  `signal.confidence` **only as a boolean gate** (`< 0.05 → 0`), then returns a flat
  `balance * risk_fraction` constant — no confidence or strength scaling term at all.
- `signal.strength` is never read anywhere in `hyper_growth.py` (confirmed via full-file grep:
  zero matches).
- `FixedFractionSizer` is instantiated with `adjust_for_confidence=False,
  adjust_for_strength=False` (confirmed at the exact call site).
- `LeveragedPositionSizer`/`LeverageManager` key only on `(regime.trend, regime.volatility,
  duration)` — confirmed via the code's own inline comment ("Keys are (TrendLabel, VolLabel)
  tuples") — never on the signal itself.
- Entry price, stop-loss, and take-profit are all fixed percentages off entry, independent of
  model output.
- An empirical sweep (predicted_return swept 0.06%→8%, an 8x spread far larger than any plausible
  real divergence between two trained models) produces exactly **one** non-zero position size for
  the entire range above the confidence gate — there is no intermediate regime where prediction
  magnitude affects sizing at all.

**Conclusion: `hyper_growth`'s L2 money exam cannot discriminate model quality beyond coarse
directional-sign agreement above its ~0.05 confidence gate.** Two models that agree on direction
at (almost) every bar will produce an identical trade blotter through this strategy regardless of
how different their actual calibration, magnitude, or confidence is. The L2 P&L table above is
reported in full for completeness and honesty about what was actually measured — but it is
**not the valid model-quality comparator this tournament was designed around**, and should not be
used to declare any architecture the winner or to declare the architectures equivalent.

**Recommendation for future architecture/target comparisons**: route the L2 exam through a
strategy that actually uses model confidence/magnitude in sizing — this codebase's
`ml_basic`/`ml_adaptive` pattern already uses `ConfidenceWeightedSizer`, which scales directly by
`signal.confidence` and would have produced differently-sized (and likely differently-timed)
trades per model. Alternatively, score architecture candidates directly on directional accuracy,
calibration (predicted-probability-vs-realized-frequency), and Brier score rather than treating
`hyper_growth` P&L as the arbiter. This applies immediately to the upcoming target-redesign
tournament (#933) — any preregistration for that work should state explicitly which
RiskManager/PositionSizer the exam uses and avoid `FlatRiskManager` + disabled-adjustment
`FixedFractionSizer` if the candidate's expected edge lives in confidence/magnitude (e.g.
meta-labeling, quantile targets).

## Verdict vs the Nov-2025 architecture research's +10% ensemble gate

The Nov 2025 research (`docs/ml_architecture_research.md`) proposed Attention-LSTM as "highest
priority" (12-15% expected improvement) and TCN as fast/competitive, with ensemble stacking gated
on a genuine improvement being found first. **Neither architecture clears any improvement bar
here**:

- No architecture beats the W_full baseline's L2 P&L in a way that means anything — the L2 exam
  cannot currently measure that (harness-validity finding above).
- On the metric that *does* carry information (L1 directional accuracy), the spread across all 5
  architectures is 1.29pp, at the edge of statistical noise for this sample size, with no
  pairwise comparison confidently clearing significance after accounting for 10 implicit
  comparisons.
- The Nov-2025 research's expected "12-15% improvement" for Attention-LSTM does not appear here
  in any metric that was actually measurable this cycle.

**Ensemble Phase 2 is NOT justified by this result.** The gate was "a winner clears +10%" — there
is no winner, clear or otherwise. Building an ensemble from 5 models that are statistically
indistinguishable from each other on the one valid metric would not be expected to produce a
material improvement, and would add real operational complexity (5x inference cost, 5x model
maintenance) for a benefit this tournament found no evidence for.

**This result strengthens the target-redesign case.** Architecture selection has now been tested
and found not to be the binding constraint, joining training-window selection (2026-07-05,
also found not to be the constraint) and echoing the standing hypothesis from the 2026-07-05
00:15 log entry and the window-tournament's own conclusion: the ~53% directional accuracy /
near-zero confidence disconnect is the more promising lever, and — as of this tournament — so is
the discovery that the live position-sizing strategy cannot express whatever signal quality *does*
exist. Recommended next steps, in priority order:
1. Confidence-calibration study (already flagged twice before this tournament; still unresolved).
2. Re-run any future architecture or target comparison through a confidence-weighted sizer
   (`ml_basic`/`ml_adaptive`) so L2 money-metrics are actually informative.
3. Target redesign (direction-classification or vol-normalized returns instead of next-bar price)
   — the original open question from the evaluation-system doc, now with two independent lines of
   evidence (window choice, architecture choice) pointing away from "pick a better model on the
   current setup" as the fix.

## Infrastructure bugs found (standalone contribution, independent of the model verdict)

Three real, previously-undiscovered defects were found and fixed while running this tournament —
listed here because they generalize beyond this one experiment and were costly enough (one
destroyed ~75 minutes of billable cloud GPU time) to warrant tracking independently of whether
any reader cares about the architecture verdict above.

1. **Model-factory `has_sentiment` kwarg contract drift** (upstream issue #928, fixed via PR
   #925/#928): `create_model()` unconditionally forwarded `has_sentiment` to every architecture
   factory, but only `create_adaptive_model` (cnn_lstm) accepted it — `attention_lstm/default`
   and `tcn/default` raised `TypeError` at model-construction time (before any training compute
   was spent). Found via a local training crash; fixed by popping the kwarg before dispatch.
2. **SageMaker input-channel day-boundary validation bug** (filed as issue #931; fixed
   worktree-locally first, then upstreamed to develop via PR #932 — merged, issue closed):
   `_validate_data_coverage` (the
   SageMaker S3-input-channel validator — a different function from the one PR #925 actually
   fixed) compared exact timestamps against a midnight `start_date`, rejecting correctly-loaded
   data with the self-contradictory error "Data starts at 2017-08-17 but training expects data
   from 2017-08-17" whenever a symbol's real listing time isn't exactly midnight (the common
   case — ETHUSDT's first-ever candle opens at 04:00). Broke all 4 initial cloud job launches at
   boot; fixed via a calendar-day comparison instead of exact-timestamp comparison.
3. **`evaluate_model_performance` metric-count contract drift** (filed as issue #936, fixed
   worktree-locally and — per the Board's separate dispatch — upstreamed as PR #937):
   `evaluate_model_performance` (`artifacts.py:209`) hard-unpacked exactly 2 return values from
   `model.evaluate()`, but `attention_lstm`, `tcn`, and `tcn_attention` all compile with
   `metrics=[rmse, mae]` (3 return values), crashing **after a full 22-epoch training run
   completed** — and because this evaluation step runs before artifact save, the crash destroyed
   the trained model entirely. Cost: ~75 minutes of combined billable SageMaker GPU time across
   two jobs produced zero artifacts before this was caught. Fixed via
   `model.evaluate(..., return_dict=True)` + lookup by metric name (robust to any architecture's
   metric set), plus a defensive try/except around the evaluation/diagnostics step so a *future*
   diagnostics failure degrades to a metadata gap rather than destroying training output.

All three are examples of the same underlying pattern: **per-architecture contract drift** — code
written and tested against one architecture (`cnn_lstm`) silently assumed properties (a specific
kwarg signature, a specific metric count, or a specific timestamp precision) that don't hold once
a second architecture is added. A regression test constructing every CLI-selectable
`(model_type, variant)` pair with the trainer's exact kwargs (added as part of the PR #925/#928
upstream fix) is the generalizable guard against this class of bug recurring for future
architectures (e.g. TFT, when it's eventually added to a tournament).

## What I'd want risk-officer / future-tournament designer to stress-test

- The paired-test refinement to the DA statistical read (McNemar's or similar) — the
  independent-samples SE used here is conservative and may be overstating how close this result
  is to distinguishable.
- Whether the 1.29pp DA spread reproduces on a different eval window (chop or bull regime) —
  this tournament, like the window tournament before it, is one draw, not a distribution.
- The `ConfidenceWeightedSizer`-routed re-run recommended above, to get a first *valid* L2 money
  comparison across these same 5 trained models before concluding anything about which
  architecture is best for live trading.

## Rollback / cleanup

Both worktrees used for this tournament (`.claude/worktrees/arch-tournament`,
`.claude/worktrees/arch-tournament-cloud`) are being removed at the close of this experiment; no
worktree-local patches were merged upstream by this tournament directly (issues #928/#931/#936 were
all upstreamed by separate dispatches per Board process — #931 via PR #932, merged and closed). No
`latest` symlink in the live registry was touched — all
model-artifact/registry work happened inside the two now-removed worktrees. No promotion
proposal accompanies this experiment; the verdict is "no winner, do not promote," which needs no
risk-officer gate under the model-promotion process.
