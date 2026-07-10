# TARGET-REDESIGN Tournament — Pre-registration

Date: 2026-07-10
Author: quant-researcher
Status: PLANNED — Phase 1 (pre-registration) complete, awaiting PM review before Phase 2 (build) begins
Issue: GH #933 (research survey + Board directives), follows #912 (confidence-calibration, closed,
H0 supported), #939/#938 (architecture tournament + harness-validity finding)
North star: `docs/architecture/model_evaluation_system.md` open question #2 ("Is next-bar price the
right target at all?")

**No backtest, training run, or code change accompanies this document.** This is Phase 1 only —
hypotheses, exam design, metrics, thresholds, and outcome-triggered decisions committed to paper
before any run, per `.claude/skills/experiment-preregister/SKILL.md`. Phase 2 (engineering build +
execution) begins only after PM sign-off on this document.

---

## 0. Constraints this preregistration is bound by

Restated from the dispatch brief (accumulated Board/PM decisions on GH #933 and its comments) so
this document is self-contained and auditable against them:

1. **Four entrants**, no more, no fewer, in Round 1: (a) meta-labeling secondary classifier with a
   genuinely richer feature set than the primary signal's own magnitude; (b) binary fixed-horizon
   direction classification (mandatory cheap baseline); (c) triple-barrier ternary classification
   with barriers matched to real stop/target geometry; (d) smoothed forward return over a
   configurable horizon (Board directive, FreqAI-style).
2. **Harness-wide rule**: every entrant's raw output → signal strength/confidence via statistics of
   its OWN training-set target distribution (percentile/z-score), never a hardcoded constant. The
   `confidence = |return| × 12` class of formula is prohibited everywhere in this tournament.
3. **Exam harness is NOT HyperGrowth's flat-sizer wiring.** Money-metric exam runs through a
   confidence-weighted sizer; DA/calibration/Brier are primary model-quality metrics, money exam is
   secondary/confirmatory. This document states explicitly which metric ranks and which merely
   gates (§4).
4. **Fresh exam data.** The 2026-01-01→2026-07-04 window must not be reused as the primary decision
   window (already spent ~7-12 candidate-facings). Options weighed explicitly (§3).
5. **Determinism precondition** before any exam number is trusted (§6).
6. **Cloud-first training** via `atb train cloud`, SageMaker, sequenced around the 2-concurrent
   `ml.g4dn.xlarge` quota and the Sunday ~08:08 UK weekly-retrain job; ECR image rebuilt from
   current develop before Round 1.
7. **Multiple-comparison discipline**: candidate count, correction method, per-metric success
   thresholds, and a full per-outcome decision table pre-committed before any run (§4, §5, §7).
8. **Honest Phase-2 engineering-work inventory** — what code must be built, in which files — because
   that work happens after this doc is locked, not before (§8).

---

## 1. Hypothesis

**H1**: Next-bar price regression (the incumbent target, `predicted_return = (price[t+1] -
price[t]) / price[t]`) is the wrong training target for this system — not because the model is
poorly trained, but because the target itself does not encode the information the strategy actually
needs (calibrated direction confidence, or awareness of the strategy's own exit mechanics). At least
one of the four reformulated targets below will produce a model whose **predicted-direction
probability is genuinely informative out-of-sample** (calibration/Brier and directional-accuracy
metrics beat both naive persistence and the incumbent price-regression control by a
pre-committed, statistically defensible margin), and this quality improvement will also be visible,
even if smaller, in money-exam metrics once routed through a sizer that can express confidence
(§4).

**H0 (falsifier, per entrant)**: For each entrant, the reformulated target's predicted output
carries **no more OOS directional/calibration information than the incumbent's already-falsified
confidence channel** (per #912: Cochran-Armitage Z=+0.43, p=0.669 on the frozen exam — magnitude
carries zero OOS signal). If H0 holds for all four entrants, the honest conclusion is that **target
reformulation alone does not create OOS edge in this system** — the constraint lives elsewhere
(features, exit geometry, or the underlying ~53% raw directional accuracy ceiling itself), not in
how the label is shaped.

**Per-entrant mechanism and falsification condition** (so this isn't one monolithic H1/H0):

| Entrant | Mechanism claimed | Falsified if |
|---|---|---|
| (a) meta-labeling | A richer feature set (vol regime, rolling hit-rate, session, regime label — NOT just `\|predicted_return\|`) can separate profitable from unprofitable instances of the primary signal firing, even though the primary signal's own magnitude cannot (per #912). | Meta-model's OOS Brier score / P(profitable) calibration is statistically indistinguishable from a coin flip, OR its OOS accuracy does not exceed the base rate of primary-signal profitability. |
| (b) binary direction classification | A model trained directly on `sign(close[t+H] - close[t])` with a proper classification loss produces a genuinely calibrated `P(up)`, unlike a magnitude-regression proxy. | OOS DA does not beat naive persistence by a margin exceeding ~2 pairwise SEs (§4), OR Brier score is no better than a constant-0.5 forecaster. |
| (c) triple-barrier ternary | Encoding actual stop/target/time mechanics into the label (not just sign of next-bar delta) produces a classifier whose confidence reflects "will this trade, as executed, actually work" rather than a generic direction guess. | OOS 3-class accuracy does not beat a class-frequency-weighted dummy classifier, OR the class distribution is degenerate (>90% one class) making the comparison meaningless. |
| (d) smoothed forward return | Averaging the target over a forward window denoises the single-bar-noise problem (FreqAI's own answer to this exact failure mode) without fundamentally changing "what" is predicted. | OOS magnitude-vs-hit-rate relationship (the same decile/Cochran-Armitage test #912 ran) remains flat, i.e. smoothing the target does not manufacture the magnitude-accuracy relationship #912 showed doesn't exist for the raw target. |

---

## 2. Entrant specifications

### (a) Meta-labeling secondary classifier

- **Primary signal**: the currently-deployed incumbent (next-bar price-regression `MLBasicSignalGenerator`
  path, `predicted_return`-derived direction). Meta-labeling is tested against the CURRENT primary
  signal, not against one of entrants (b)/(c)/(d) — this is a deliberate simplification of the
  research doc's original two-round design (see §9, Deviation 1).
- **Label**: for every bar where the primary signal fires non-HOLD, label = 1 if simulating that
  trade through **this tournament's own exam-harness exit geometry** (§3 — NOT HyperGrowth's
  10%/30%; see §9, Deviation 2) closes net-profitable after fees, else 0.
- **Feature set (required, richer than #912's falsified single-scalar version)**: 48-bar trailing
  realized volatility of bar-over-bar returns; rolling hit-rate of the primary signal over its
  trailing 20 fired signals; session/time-of-day bucket (cyclical encoding); the existing
  `EnhancedRegimeDetector`'s regime label (trend × volatility, already computed elsewhere — reuse,
  do not reimplement); the primary model's own `predicted_return` magnitude as ONE feature among
  the above, never the sole feature.
- **Model output**: `P(trade profitable | signal fired)`, binary classifier. Start with logistic
  regression as the dumb-baseline variant of this entrant itself (per north-star rule 6 applied
  within-entrant), gradient-boosted trees (LightGBM — already has a model factory,
  `models_lightgbm.py`) as the primary variant.
- **Confidence mechanism (harness-wide rule)**: raw output is already a probability in [0,1] by
  construction — no percentile/z-score transform needed for the *value* itself, but it MUST pass a
  calibration-correction step (Platt scaling or isotonic regression fit on a held-out slice of the
  training-period data, never the exam) before being trusted as `signal.confidence`, since
  tree/logistic probabilities are not guaranteed calibrated out-of-the-box.
- **Named risk (pre-registered, not discovered later)**: if the primary signal has ~zero real
  directional edge (open question — #912 falsified the *confidence* channel, not the base ~51.85%
  hit rate itself), meta-labeling has nothing real to filter and will, at best, filter training-set
  noise and fail OOS. This is Angle 4's caveat from the research doc, restated here as a formal
  falsifier, not a footnote.

### (b) Binary fixed-horizon direction classification

- **Label**: `y = 1 if close[t+H] > close[t] else 0`, `H = 1` bar (same horizon as the incumbent
  regression target — isolates "does the loss function/output type alone help" from "does horizon
  help," per the research doc's own reasoning).
- **Model output**: `P(up)` via sigmoid/softmax.
- **Confidence mechanism**: `confidence = |P(up) - 0.5| × 2` — bounded [0,1] by construction, no
  free multiplier. Still subject to a calibration-correction step (reliability check on
  training-period-adjacent data) before use, per the harness-wide rule's spirit (a classifier
  probability that LOOKS bounded can still be poorly calibrated).
- **Architecture path**: `create_model('tft', ...)` already exists as a binary-classification
  architecture (sigmoid output, `binary_crossentropy` loss, `models_tft.py`) — but see §8, this is
  currently unwired past model construction and cannot train end-to-end without pipeline work.

### (c) Triple-barrier ternary classification

- **Label**: for each bar, simulate forward with upper barrier `+take_profit_pct`, lower barrier
  `-stop_loss_pct` — set to **this tournament's exam-harness defaults, `take_profit_pct=0.04`,
  `stop_loss_pct=0.05`** (matching `src/config/constants.py`'s `DEFAULT_TAKE_PROFIT_PCT`/
  `DEFAULT_STOP_LOSS_PCT`, which also match `risk-limits.json`'s `stops.default_take_profit_pct`/
  `default_stop_loss_pct` — i.e. the Board-ratified risk-limits defaults; the prod HyperGrowth strategy overrides these to 10%/30%, see §9, Deviation 2), and a
  vertical time barrier at `max_holding_hours=336` (`risk-limits.json`
  `operational.max_holding_hours`). Label = {+1, -1, 0} for whichever barrier is hit first, using
  intrabar high/low via `src/engines/shared/` fill logic (reused, not hand-rolled — the research
  doc's explicit warning).
- **Model output**: 3-class softmax probability distribution.
- **Confidence mechanism**: `argmax` class → direction, `P(argmax class)` → confidence — same
  bounded-by-construction + calibration-correction treatment as (b).
- **Named risk**: barrier width is fixed a priori from the Board-ratified risk-limits defaults (not
  tuned on training or exam data) — this removes the barrier-width p-hacking risk the research doc
  flags, at the cost of not exploring whether a different (e.g. vol-scaled) barrier would do
  better; that is explicitly out of scope for Round 1.

### (d) Smoothed forward return (Board directive, entrant #4)

- **Label**: `y = mean(close[t+1..t+N] returns from close[t])` — mean of close-to-close returns
  over the next N candles, FreqAI's own convention (`docs/research/2026-07-07_ml-target-design-research.md`
  Angle 1).
- **Horizon N**: pre-committed **N = 6 hours** before any run (roughly a quarter-day swing horizon;
  distinctly different from the incumbent's implicit N=1; short enough to remain tradeable within
  `max_holding_hours=336`). A sensitivity check at **N=3 and N=12** is pre-registered as a
  diagnostic follow-up on this same entrant (not a new entrant — does not add to the 4-candidate
  comparison count), run only if N=6 clears the primary quality bar (§4), reported as sensitivity
  per the standard workflow's robustness step, never as a basis to retroactively pick a different N
  after seeing exam results.
- **Model output**: continuous predicted smoothed return.
- **Confidence mechanism (the direct fix for the `×12` bug — the harness-wide rule's clearest
  application)**: `confidence = percentile_rank(|predicted_smoothed_return|` within the frozen,
  training-set-only distribution of `|predicted_smoothed_return|` **computed once on the training
  split and never updated using eval-window data**). This is the literal FreqAI `&*_std`/`&*_mean`
  pattern the Board directive names.
- **Named risk**: this changes units, not necessarily information content — #912 already showed
  the underlying regression target has no OOS magnitude-accuracy relationship for N=1; if the same
  null holds for smoothed N=6, expressing it in percentile-rank units doesn't manufacture
  information that isn't there. Pre-registered as the null hypothesis, not discovered after the
  fact.

### Baselines (not counted against the 4-entrant budget, always run alongside)

- **Naive persistence**: `sign(close[t] - close[t-1])` continues, computed directly on each fold's
  eval window, no training required.
- **Linear baseline**: a simple linear/logistic regression on the same feature set as the incumbent
  — cheap, local (no SageMaker), per north-star rule 6 ("always include dumb baselines").
- **Incumbent control**: the currently-deployed next-bar price-regression model
  (`2026-07-04_22h_v1` lineage / equivalent retrain). Because this tournament's primary fold set
  (§3) is entirely pre-2026, and the incumbent's actual training cutoff is 2025-12-31, the
  incumbent's own already-trained artifact can be reused directly as the fold-3 control without
  retraining (its training window predates every fold's eval start by construction) — for folds 1
  and 2 (earlier cutoffs), a fresh incumbent-config retrain at each fold's cutoff is needed to keep
  the comparison fair (same target, same architecture, different-but-fold-matched cutoff).

---

## 3. Exam design (headline decision)

**Three options were weighed, per the dispatch brief's explicit instruction:**

1. **A single later training cutoff + new out-of-time window.** Given "today" is 2026-07-10, H2
   2026 barely exists (~9 days of data past 2026-07-01). Any window long enough to be statistically
   meaningful (the prior 185-day exam produced ~50 trades) would have to reach back into H1 2026 —
   which is exactly the already-judged 2026-01-01→2026-07-04 window. A later-cutoff single window
   is therefore either too short to be meaningful (if confined to genuinely-fresh H2 2026 data) or
   not actually fresh (if it reaches back into H1 2026).
2. **Purged/embargoed walk-forward folds (López de Prado style) across pre-2026 history.** All
   prior tournaments (#898, #912, #939) used 2018–2025 data only as TRAINING input for a single
   2026 eval window — none of them ever scored a candidate's OOS performance on a held-out slice
   *within* 2018–2025. Slicing that history into purged, embargoed expanding-window folds gives
   genuinely never-touched-as-eval windows, spans multiple real market regimes (bear, bull, chop),
   and directly serves the charter's stated preference ("robustness across multiple regimes over
   single-regime overfits") — at zero additional data cost, since the data already exists in the
   local cache.
3. **A combination.**

**Decision: option 3, weighted toward (2).** Primary ranking evidence comes from three purged
walk-forward folds drawn from pre-2026 history — genuinely fresh, multi-regime, zero overlap with
any previously-judged window. A short confirmatory-only check on the most recent available slice is
run alongside it, explicitly labeled non-deciding, to be honest about what recent-regime evidence
does and doesn't exist.

**Exact dates, pre-committed:**

| Fold | Role | Train (cutoff) | Embargo | Eval window | Regime (approx) |
|---|---|---|---|---|---|
| F1 | Primary | 2017-08-17 → 2022-12-31 | 48h | 2023-01-03 → 2023-06-30 | Post-crypto-winter chop/recovery |
| F2 | Primary | 2017-08-17 → 2023-12-31 | 48h | 2024-01-03 → 2024-06-30 | 2024 bull run (ETH ETF period) |
| F3 | Primary | 2017-08-17 → 2024-12-31 | 48h | 2025-01-03 → 2025-06-30 | 2025 H1 |
| F4 | Confirmatory only, non-deciding | 2017-08-17 → 2026-04-30 | 48h | 2026-05-03 → 2026-07-09 | Most recent available; **partially overlaps** the already-judged 2026-01-01→2026-07-04 window for 2026-05-03→2026-07-04, genuinely fresh only for 2026-07-05→2026-07-09 (~5 days) |

Embargo = 48 hours (2 days) around each fold's train/eval boundary, consistent with the 48-bar
realized-volatility lookback already used elsewhere in this research line (#912's vol_zscore_gate).
Expanding-window training (not rolling) — matches the #898 window tournament's finding that full
history ties-or-beats shorter windows.

**F4's role and limitation, stated honestly**: F4 is reported for every entrant but is explicitly
**not used to rank or gate** (§4) — its eval window is short (~68 days, likely <20-30 trades at
this system's typical trade frequency) and mostly re-treads ground already covered by 7-12 prior
candidate-facings on the old exam window. It exists only so that "does the verdict hold on the most
recent data we have" gets an honest, if statistically weak, answer, not a silent gap.

**Per-fold candidate count for multiple-comparison tracking**: this is a BRAND NEW fold set — zero
prior candidates have faced F1/F2/F3. This tournament's own facings: 4 entrants + linear baseline +
naive persistence (free) + incumbent control = 6 trained candidates × 3 primary folds = 18
fold-facings, well under the ~10-before-rotate guidance per exam window (each fold counts
separately since folds are genuinely different draws), leaving headroom for a determinism
re-run and one round of sensitivity checks (§2d) before any fold needs retiring.

---

## 4. Metrics — hierarchy and gating

Per constraint 3, this section states explicitly which metric ranks and which merely gates.

### Primary (ranks entrants)

- **Directional/classification accuracy** on each fold's eval window (sign-hit-rate for (a)/(d),
  classification accuracy for (b), 3-class accuracy for (c)), vs. naive persistence computed on the
  same fold.
- **Calibration quality**: Brier score (binary/ternary entrants) or the decile/Cochran-Armitage
  magnitude-vs-hit-rate test from #912 (entrant (d), continuous). Reliability diagrams reported for
  every classifier entrant.
- **Aggregation across folds**: per-entrant metric averaged across F1–F3 (F4 excluded from ranking,
  per §3), with a paired significance test per fold (McNemar's, since all entrants score the same
  bars within a fold) and Wilson 95% CIs. An entrant's win must survive **Bonferroni-corrected
  pairwise significance** — 4 entrants → 6 pairwise comparisons → α = 0.05/6 ≈ 0.0083 — to be
  called a confident winner, not just "highest observed number." This is a stricter bar than the
  architecture tournament used (which flagged the caution but didn't formally correct); adopted
  here because target-redesign is the higher-stakes comparison.

### Secondary / confirmatory (gates, does not rank)

- **Money exam**: OOS return, profit factor, MTM MaxDD, win rate, trade count, Sharpe — run through
  the exam harness (§5), fees/slippage on (`CostCalculator` defaults), prod-matched flags
  (`--initial-balance 85 --risk-per-trade 0.02 --max-risk-per-trade 0.03 --max-position-size 0.20`).
- **Gate condition** (must clear to be eligible for L3a staging, regardless of primary-metric rank):
  beats naive persistence on both OOS return and profit factor on the aggregate of F1–F3, trade
  count ≥30 aggregate (not per-fold — folds are short), and does not fail any single fold's MaxDD
  by more than the incumbent's own worst-fold MaxDD by >5pp (regime-robustness gate, per
  `model_evaluation_system.md` principle 5).
- An entrant can rank #1 on primary metrics and still fail this gate — that is a valid, reportable
  outcome ("quality win, not yet exam-actionable"), not grounds to lower the gate post hoc.

### Reported, never used to rank or gate

- **Accuracy-vs-coverage curves** for every probability-output entrant ((a),(b),(c), and (d) via
  its percentile-rank confidence): OOS metrics at 4 abstention thresholds (trade top 10%/25%/50%/
  100% of bars by confidence), per the research doc's Angle 5 finding — reported as a curve, not
  collapsed to one number.
- **L1 holdout RMSE/loss** and feature-schema sanity — health check only, per the north star's own
  principle 1 (prediction error does not rank models) and the window/architecture tournaments'
  repeated empirical confirmation that RMSE ranks backwards relative to what matters.

---

## 5. Exam harness (constraint 3 — explicit component wiring)

**Strategy config**: a new exam-only strategy factory, mirroring `src/strategies/ml_basic.py`'s
existing wiring pattern exactly (`CoreRiskAdapter(EngineRiskManager(RiskParameters(...)))` +
`ConfidenceWeightedSizer(base_fraction=0.2, min_confidence=0.3, min_confidence_floor=0.0)` —
`src/strategies/components/position_sizer.py:253`, already used by `ml_basic`/`ml_adaptive`, no new
sizer class needed). Each entrant's reformed `SignalGenerator` output plugs into this same
harness — **not** `FlatRiskManager` + `FixedFractionSizer(adjust_for_confidence=False)`
(HyperGrowth's wiring), which #938 proved cannot express confidence/magnitude differences at all.

**Why this matters concretely**: `ConfidenceWeightedSizer.calculate_size`
(`position_sizer.py:253-330`) scales `base_size = balance * base_fraction * confidence_factor`
directly by `signal.confidence` — two entrants that agree on direction but differ in confidence
WILL produce differently-sized trades under this harness, unlike HyperGrowth. This is the load-bearing
fix for the #938 finding, applied here explicitly rather than assumed.

**Stop/target geometry for the exam itself**: `stop_loss_pct=0.05`, `take_profit_pct=0.04` (the
system-default constants — `DEFAULT_STOP_LOSS_PCT`/`DEFAULT_TAKE_PROFIT_PCT`, `src/config/constants.py:126,129` —
i.e. the ml_basic-pattern harness defaults; NOT prod-HyperGrowth's 10%/30%. Same values used for
entrant (c)'s triple-barrier labels — self-consistent by construction, see §9 Deviation 2),
`max_holding_hours=336`. *(Provenance corrected at PM review: an earlier draft mislabeled these
values "prod-matched"; the prod strategy's geometry is 10%/30%.)*

---

## 6. Determinism guard (constraint 5)

Before any exam number from this tournament is trusted: train entrant (b) (the cheapest, simplest
architecture — binary classification is the mandated baseline arm) on fold F1 only, run its L2 exam
through the new exam harness (§5) **twice back-to-back**, identical config, identical model
artifact. Results (trade count, return, PF, MaxDD to at least 4 significant figures) must match
exactly.

This is a genuinely new re-verification, not an assumption that #923's fix "just works" here: #923
fixed the general inference-timeout non-determinism, but it was verified against HyperGrowth's
execution path — this tournament exercises a different `SignalGenerator`/sizer combination
end-to-end (new classification-native signal path, `ConfidenceWeightedSizer` instead of
`FlatRiskManager`/`FixedFractionSizer`), and #913's own lesson ("silent per-config non-determinism
until proven otherwise") applies to any new harness config, not just the one it was found in.

**If the repeat run diverges**: halt, do not interpret any other entrant's numbers, file as an infra
bug (same class as #913), fix, then re-run the determinism check before resuming.

---

## 7. Multiple-comparison discipline and per-outcome decision table (constraint 7)

**Comparison count**: 4 entrants (candidate-facing comparisons), 6 pairwise comparisons among them,
Bonferroni α=0.0083 for any pairwise "winner" claim (§4). Naive persistence and the incumbent
control are always-present reference points, not counted in the pairwise-comparison inflation
(consistent with north-star rule 6's treatment of dumb baselines).

**Pre-committed decision table:**

| Outcome | Decision |
|---|---|
| **One entrant clears BOTH the primary quality bar (Bonferroni-significant win vs. naive persistence AND vs. the incumbent control, aggregated F1–F3) AND the money-exam confirmatory gate (§4)** | Proceed to L3a staging paper trial (≥48h, per `model_evaluation_system.md`). PR opened describing the winning target design; ml-engineer dispatched for production wiring. **No promotion to live from this result alone** — L3a is mandatory regardless of exam strength (never-auto-promote rule). |
| **Multiple entrants cluster within Bonferroni-corrected noise (no confident pairwise winner)** | Report as "cluster, no winner." This corroborates — does not merely fail to refute — the standing hypothesis that model/target selection is not (yet) the binding constraint at this fidelity, joining the window tournament (#898) and architecture tournament (#939) as a third independent line of evidence. Close this round without a staging trial. Explicitly name the next lever implied by which entrants clustered (e.g., if (a)/(c) cluster near the incumbent but (b)/(d) don't move at all, that implicates exit-geometry-aware labels specifically as the more promising remaining direction). |
| **No entrant beats naive persistence on the primary quality bar** | Clean negative result, full write-up per the anti-p-hacking rule ("negative results get FULL write-ups"). Do NOT retry with hyperparameter tuning on the same fold set — that would spend comparison budget on noise-chasing. Redirect research priority explicitly (candidate next steps: feature-set expansion, the quantile/distributional entrant deferred from the research doc's Round 2, or accepting the ~53%/noise-confidence ceiling as structural pending a genuinely new data source, e.g. sentiment/orderbook features). |
| **An entrant fails to train (crash, contract drift, infra bug)** | Treat as the per-architecture-contract-drift pattern already seen 3× this cycle (#928/#931/#936). File a GH issue. One fix-and-retry attempt is in-scope IF the failure is a clear infra bug (kwarg mismatch, metric-unpack crash, etc.) unrelated to the target design itself; if the fix is unclear or non-trivial, the entrant is recorded as "did not train" and EXCLUDED from ranking — but the exclusion itself, and the reduced comparison count, must be logged (a silently-dropped entrant changes the multiple-comparison arithmetic). |
| **Entrant (a) specifically fails because the primary signal has no real edge to filter (§2a named risk)** | Distinguish explicitly from "meta-labeling as a technique failed." Report as "primary signal edge insufficient for meta-labeling to add value" — this is itself informative (further corroborates #912) and should not be read as evidence against meta-labeling in general, e.g. if entrant (b) or (c) later becomes the primary signal. |

---

## 8. Engineering-work inventory for Phase 2 (constraint 8, honest audit — no code written here)

Grounded in the current `develop` tree (checked directly, not assumed):

- **`create_model()` (`src/ml/training_pipeline/models.py:155`) already dispatches to a `tft`
  architecture** — `models_tft.py:373` (`create_tft_model`) compiles a genuine binary
  direction-classification head (`Dense(1, activation="sigmoid")`, `loss="binary_crossentropy"`,
  confirmed via #937's own PR description: "`tft` is excluded [from the eval-metrics fix] — it's a
  binary direction-classification architecture... a different evaluation contract entirely"). **This
  is architecturally the closest existing building block to entrant (b)** — but it is NOT usable
  end-to-end today:
  - `src/ml/training_pipeline/pipeline.py:178-179` builds `target_array =
    feature_data["close"].to_numpy(dtype=np.float32)` **unconditionally**, regardless of
    `model_type` — training `tft` through the existing pipeline today would fit a sigmoid/BCE head
    against a continuous close-price float target, a silent, real contract mismatch, not a
    theoretical one.
  - No CLI wiring found for `--model-type tft` beyond `create_model`'s own dispatch branch (no hits
    in `src/cli/`).
  - Unit-tested only at model-construction level (`tests/unit/ml/training_pipeline/test_models_tft.py`)
    — never exercised through a real training run end-to-end in this codebase's history, as far as
    this survey found.
  - **Phase 2 work**: build classification-target generation (see next bullet) and thread it
    through `pipeline.py` conditional on `model_type`, so `tft` (or another architecture reused
    with a classification head) trains against a real label.
- **No classification-label generation code exists anywhere in `src/ml/training_pipeline/`**
  (confirmed via grep — the only `classif` hit in the entire `src/ml/` tree is a docstring comment
  in `models_lightgbm.py`, not label logic). Entrants (b), (c), and (a)'s label all need new code,
  most naturally as a new `src/ml/training_pipeline/labels.py` module:
  - Binary direction label (b): trivial, `sign(close.shift(-H) - close)`.
  - Triple-barrier label (c): the substantial piece — a forward-simulation loop using intrabar
    high/low, which **must reuse `src/engines/shared/` exit/fill logic** (the same code
    `exit_handler.py` uses for live/backtest parity), not a hand-rolled reimplementation. This is
    the largest single piece of new Phase-2 code.
  - Meta-label (a): depends on (i) running the CURRENT incumbent signal generator forward over the
    training corpus to find its fire points, then (ii) simulating each fired trade through the exam
    harness's exit geometry (reusing the same triple-barrier simulation machinery built for (c)) to
    get the binary profitable/not-profitable label, plus (iii) computing the richer feature set
    (§2a) at each fire point.
  - Smoothed forward return (d): trivial, a rolling-mean transform of the existing return
    computation.
- **`evaluate_model_performance` (`artifacts.py`) now reads metrics by key** (post-#937, confirmed
  merged to develop) — this generalizes safely to classification metrics (`accuracy`, `auc`) without
  repeating the #936 positional-unpack crash. **Not yet built**: the reporting layer that actually
  computes and persists Brier score / reliability-diagram data for a classification head — #937's
  fix makes this *possible* without crashing, it does not itself add the calibration metrics this
  tournament needs.
- **Strategy-consumption layer is regression-shaped end to end.**
  `src/prediction/engine.py`'s `PredictionResult` (`:52-57`) has fields `price: float` and
  `direction: int` — **no native probability field**. `MLBasicSignalGenerator._get_ml_prediction`/
  `generate_signal` (`ml_signal_generator.py`) derives `predicted_return` from `price`, then
  confidence via the (prohibited, per constraint 2) fixed `×12` formula. A classification-native
  path for entrants (a)/(b)/(c) needs either (i) a new `SignalGenerator` variant that reads a
  probability output directly, applying only the calibration-correction step (§2), or (ii)
  extending `PredictionResult`/`PredictionEngine` to carry a probability field through the whole
  pipeline. **This touches money-path code** (`src/prediction/engine.py`,
  `src/strategies/components/ml_signal_generator.py`) and must go through the standard
  architecture-reviewer + code-reviewer gauntlet before any exam run trusts its output — a
  silent bug here would mis-score every classifier entrant identically and invisibly, the same
  class of failure #938 found in a different layer.
- **`ConfidenceWeightedSizer` and the `CoreRiskAdapter`/`EngineRiskManager` wiring pattern already
  exist and require no new code** (`position_sizer.py:253`, `ml_basic.py:96-176`) — the exam
  harness (§5) is a config/wiring exercise, not new component code, which meaningfully shrinks the
  Phase-2 scope relative to the signal-generation and label-generation work above.
- **`docs/research/exams/` and `docs/research/model-scoreboard.md` do not exist yet** — confirmed
  (directory absent, file absent). Both are named "to build" in `model_evaluation_system.md` and
  have not been built by any prior tournament (all have used hand-rolled scripts +
  `docs/research/experiments/*.md` write-ups instead). Recommend writing this tournament's frozen
  fold definitions (§3) as the first artifact under `docs/research/exams/` since the harness-wide
  rule and multi-fold design make a machine-readable exam definition more valuable than usual — but
  building the full `atb models exam` CLI command is **not** required for Phase 2 to proceed; flagged
  as optional/recommended, deferred to PM's call.

**Summary for PM**: the single largest, highest-risk piece of new code is the triple-barrier
forward-simulation label generator (shared by entrants (a) and (c)) plus the classification-native
signal/prediction path (shared by entrants (a)/(b)/(c)). Entrant (d) is comparatively cheap (a
rolling-mean transform + percentile-rank confidence, no new label-simulation machinery, no new
signal-generator type if the existing regression path is reused with a smoothed target). This is
worth knowing when sequencing Phase 2 — (d) could plausibly ship and exam-run before (a)/(b)/(c)'s
shared infrastructure is ready, if the PM wants an early partial read rather than waiting for all
four to be simultaneously ready.

---

## 9. Deviations from prior guidance, with reasoning

1. **Single-round, four co-equal entrants instead of the research doc's two-round design (meta-labeling
   gated behind a Round-1 winner).** The dispatch's binding constraint 1 names all four as Round-1
   entrants; GH #933's Board comment names smoothed-forward-return as "entrant #4" without
   qualification. Read together, these supersede the research doc's original sequencing
   recommendation (which predates the Board's decision to fix the entrant list at four). Reasoning
   for treating this as correct rather than a gap: meta-labeling entrant (a) is specified here
   against the CURRENT incumbent primary signal, which already exists and needs no Round-1 winner
   to be selected first — it is a self-contained target-design candidate in its own right, matching
   how the dispatch's constraint 1(a) frames it ("would be a wasted entrant" language implies it's
   being run now, not queued).
2. **Meta-labeling (a) and triple-barrier (c) labels use this tournament's exam-harness exit
   geometry (`stop_loss_pct=0.05`, `take_profit_pct=0.04`, the system-default/ml_basic-pattern
   values) rather than HyperGrowth's
   (10%/30%), which is what the research doc's original candidate-1 write-up specified.** The
   research doc predates #938's harness-validity finding. Using HyperGrowth's geometry for the
   LABEL while executing trades through a different harness (§5, mandated by constraint 3) would be
   internally inconsistent — the label would encode "would this trade work under HyperGrowth's
   stops" while the exam actually executes it under different stops. Matching label geometry to
   the harness that actually trades it is the more correct choice given evidence that postdates the
   original research doc.
3. **Only 3 of the research doc's original 6-candidate shortlist are run in Round 1** (vol-normalized
   regression and trend-scanning deferred, per the research doc's own recommendation that they wait
   for round-1 evidence) — this is not a deviation from the dispatch brief (which only ever named 4
   entrants), just a note that it also matches the research doc's own risk-tiering for the two
   candidates it explicitly flagged as thinnest-evidence/highest-implementation-cost.
4. **A three-fold purged walk-forward (F1-F3) is used as the PRIMARY exam instead of a single frozen
   window**, which is a more substantial protocol change than a simple "pick a new cutoff date" per
   the research doc's own suggestion. Reasoning is in §3 — a single window using 2026 data cannot
   simultaneously be long enough to be statistically meaningful and genuinely non-overlapping with
   the already-judged 2026-01-01→2026-07-04 window, given how little H2 2026 data exists. This is
   flagged explicitly for PM review since it is the single biggest design choice in this document
   and increases both engineering/compute cost (§8) and the number of cloud training jobs (§3, 18
   fold-facings) relative to a simpler single-window design.

---

## 10. Risks of false positive

- **Fold-regime luck.** Three folds is more robust than one window but still a small sample of
  "regimes" (3 specific slices of 2023-2025). A design that happens to fit these three particular
  half-years is not proven to generalize to 2026 conditions — F4's confirmatory (non-deciding) role
  exists specifically to catch a fold-set that doesn't transfer, but F4 itself is underpowered
  (§3), so this risk is only partially mitigated, not closed.
- **Label-construction bugs are the dominant new risk this tournament introduces.** Unlike prior
  tournaments (which reused an existing, already-debugged regression target), three of four
  entrants here need genuinely new label-generation code (§8). A subtle look-ahead bug in the
  triple-barrier or meta-label simulation (e.g., using a bar's own close instead of strictly-future
  bars, or leaking the label window into the feature set) would silently inflate every downstream
  metric for that entrant — this must be an explicit, named review focus (architecture-reviewer)
  before any Phase-2 exam number is trusted, over and above the standard money-path review.
- **Multiple-comparison risk is real even with the Bonferroni correction**: 6 pairwise comparisons ×
  3 folds × multiple metrics (accuracy, Brier, money-exam) is a large number of "looks" at the same
  underlying data, and Bonferroni only corrects the one comparison it's explicitly applied to (§4's
  pairwise accuracy claim) — the money-exam gate and coverage curves are additional, uncorrected
  looks, reported as confirmatory/diagnostic specifically so they are not read as independently
  significant findings.
- **Calibration-correction step (Platt/isotonic scaling, §2) is itself fit on training-period data**
  — if it overfits that slice, the "calibrated" probability could still be miscalibrated OOS. This
  is the same overfitting mechanism #912 diagnosed for the original `×12` formula, one level
  removed; the primary-metric Brier score on the eval fold is the check that would catch this, not
  an assumption that calibration-correction is sufficient by construction.
- **Meta-labeling's dependency on the primary signal's edge (§2a, §7)** means a negative result for
  entrant (a) is ambiguous between "meta-labeling doesn't work" and "there was nothing to filter" —
  the decision table (§7) pre-commits how to read this, but the ambiguity itself is a real
  limitation of testing meta-labeling this way rather than after a Round-1 winner is known.

---

## Next steps

This document is the complete Phase 1 deliverable. **Do not begin training, data prep, or any code
change beyond this file until PM reviews and signs off.** On approval, Phase 2 scope (per §8) should
be sequenced explicitly — recommend starting with the shared triple-barrier label-generation module
and the classification-native signal path (the highest-risk, most-reused pieces, needed by 3 of 4
entrants) before the per-entrant model-factory work.
