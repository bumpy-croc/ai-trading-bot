# INPUT SCREENING (Nonlinear) — Pre-registration

Date: 2026-07-12
Author: quant-researcher
Status: **COMPLETE — no arm graduates; btc_cross flagged as a regime-specific near-miss (§ Results)**
Issue: follow-up to GH #967 (linear screen, PR #969, merging on green), same Lane A input-tournament
line as GH #959 (Phase 0 audit)
Related: `docs/research/experiments/2026-07-12_input-screening-linear.md` (the linear screen this
doc re-screens with a different model family — read it first; folds, arms, data sources, and
alignment rules are reused verbatim from it, not redefined here), `docs/research/2026-07-12_input-candidates-audit.md`
(PR #958, merged `9e7ea5e8`)

**This is a NEW experiment, not an amendment.** The linear screen found zero arms graduate; the
named, unresolved risk in that report was that a linear detector can only find linearly-separable
signal, and several candidates' own literature support (funding-rate crowding, HAR-RV volatility)
describes regime/structure, not necessarily a linear function of the raw features. This doc
re-runs the identical screening design with a nonlinear model family to close that gap, per
PM authorization following the linear screen's recommendation.

---

## 1. Hypothesis

**H1 (per arm)**: Adding input class *k* to the price-only feature contract produces a
gradient-boosted-tree classifier whose next-bar directional accuracy (DA) is higher than the
price-only control's DA, by a margin that is both statistically significant (paired McNemar per
fold) and practically non-trivial (≥0.5pp averaged across F1–F3), on at least 2 of the 3 primary
folds — identical bar to the linear screen, so the two screens are directly comparable.

**H0 (per arm, the falsifier)**: Input class *k* adds no more OOS directional information than the
price-only control has, under a nonlinear detector either — the linear screen's null was not an
artifact of the linear-model-family choice.

**Why this specific falsification test, not a jump to deep models**: gradient-boosted trees can
capture interaction effects and non-monotonic relationships a linear/logistic model structurally
cannot (e.g. "funding rate matters only when realized vol is also elevated" — an interaction, not
a main effect). If trees also find nothing, that is much stronger evidence the six input classes
themselves carry no exploitable signal at this feature contract and horizon, as opposed to the
linear screen having used the wrong detector. Trees remain cheap (CPU, seconds-to-low-minutes per
fit) — still a screening gate, not the deep-model tournament itself.

**Named risk of false positive** (per arm, generic, restated from the linear screen since it still
applies): more model capacity (trees) can fit noise more easily than a linear model, especially
with the larger effective hypothesis space of learned splits/interactions. The same Bonferroni
correction and ≥2-of-3-folds requirement are the primary defenses. Feature-importance/gain
inspection (§4) is an additional, non-gating diagnostic for exactly this risk — an arm with a
significant DA gain but importance concentrated on a single implausible split is a candidate for
skepticism even if it technically clears the bar (named here, before any result exists).

**Named risk of false negative** (unique to this doc): a fixed, unsearched hyperparameter
configuration (§2) may simply be a poor fit for a given arm's feature scale/cardinality even if a
tuned configuration would find something. This is an accepted, pre-registered limitation of a
screening gate — tuning is explicitly out of scope (a hyperparameter search would spend the
comparison/multiple-testing budget this experiment is trying to conserve, and would reopen the
same p-hacking risk the linear screen's method section rejected for the same reason).

---

## 2. Model — gradient-boosted trees, ONE fixed configuration, no search

**Method**: `lightgbm.LGBMClassifier`, binary objective, predicting the identical next-bar
direction label as the linear screen (`y = 1 if close[t] > close[t-1] else 0`), on the identical
feature contract per arm (price-only 120-bar flattened sequence, 600 dims, plus each arm's extra
features exactly as defined in the linear screen's §4 — same columns, same alignment rules, same
`t-1`/`t` sampling convention). No feature scaling is applied (tree splits are scale-invariant;
`StandardScaler` was a linear-model-only concern in the prior screen).

**Fixed hyperparameters (pre-committed, not tuned per arm or per fold)**:

```
LGBMClassifier(
    n_estimators=300,
    max_depth=5,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=50,
    random_state=42,
    n_jobs=4,               # matches the BLAS thread cap used throughout this session
)
```

Chosen as a "modest, sensible default" — shallow-to-moderate depth and leaf count relative to the
600+-dimensional input, subsampling for variance reduction, a conservative learning rate offset by
early stopping (below) rather than a large tree count. These values are locked before any arm is
scored; if they turn out to be a poor choice for every arm, that is a limitation to disclose, not
a threshold to move after seeing results.

**Early stopping (in place of a fixed `n_estimators`, still no per-arm tuning)**: the LAST 10% of
each fold's training rows by timestamp (a "train-tail" validation split, itself still strictly
before the fold's train cutoff and thus never touching the embargo or eval window) is held out
from fitting and used only for early-stopping evaluation (`eval_set`, `early_stopping_rounds=20`
on binary log-loss). This is a single, uniform rule applied identically to every arm/fold — not a
per-arm choice — so it does not reopen the "no hyperparameter search" commitment above.

**Validity check (pre-committed, arm 0 only, mirrors the linear screen's own validity check but
against a different reference)**: report the price-only LightGBM control's DA alongside the
price-only **logistic** control's DA from the linear screen, per fold, as context — **not a
pass/fail gate**, since there is no external prior claiming trees must match or beat linear DA on
this contract (unlike the linear screen, which had the tournament's own reported number to
reproduce). Whatever the comparison shows, the LightGBM price-only control is the internal
reference every arm in THIS experiment is measured against, exactly as in the linear screen.

---

## 3. Folds — identical to the linear screen and the target-redesign tournament

Reused verbatim, no changes:

| Fold | Train (cutoff) | Embargo | Eval window |
|---|---|---|---|
| F1 | 2017-08-17 → 2022-12-31 | 48h | 2023-01-03 → 2023-06-30 |
| F2 | 2017-08-17 → 2023-12-31 | 48h | 2024-01-03 → 2024-06-30 |
| F3 | 2017-08-17 → 2024-12-31 | 48h | 2025-01-03 → 2025-06-30 |

The train-tail early-stopping validation split (above) is carved from the END of each fold's own
training window (still ≤ train cutoff) — it does not touch the embargo gap or the eval window, and
does not change the fold boundaries themselves.

---

## 4. Arms — identical to the linear screen, same 7, no additions or removals

| # | Arm | Extra features | Alignment rule |
|---|---|---|---|
| 0 | Price-only control | none | — |
| 1 | Multi-scale realized vol + range dynamics | rv_6h/24h/168h, Parkinson 24h, HL-range%, HL-range% MA24 | sampled at `t-1`, rolling windows causal |
| 2 | Calendar/session | hour/dow cyclical, hours-to-funding | uses bar `t`'s own timestamp |
| 3 | BTC→ETH cross-asset | BTC ret 1h/6h/24h, BTC rv 24h | BTC joined at ETH's `t-1` |
| 4 | Funding rate (ETHUSDT perp) | level, RoC, 30d z (frozen, train-only), extreme flag | last-settled-as-of-`t-1` |
| 5 | Basis/premium proxy | premium close, 24h vol | closed-bar close only, sampled at `t-1` |
| 6 | Fear & Greed | value, 7d momentum, extreme flag | 1-day-lagged, sampled at `t-1` |
| 7 | All-combined | arms 1+2+3+4+5+6 together | per-arm rules, applied jointly |

Same explicit exclusions as the linear screen (open interest/long-short ratio, on-chain, DXY/SPX/
NDX macro; `OnChainFeatureExtractor`/`MacroFeatureExtractor`/simulated sentiment components never
used).

---

## 5. Metrics, thresholds, graduation rule — identical bar to the linear screen

**Primary metric**: per-fold DA, plain classification accuracy vs. `1[close[t] > close[t-1]]`.

**Significance test**: per-fold McNemar's test, paired, arm vs. the price-only LightGBM control
(not the linear screen's control — this experiment's own internal reference, per §2).

**Multiple-comparison correction**: Bonferroni, α = 0.05/7 ≈ 0.0071 — same correction, same arm
count, as the linear screen.

**Graduation rule (pre-committed, IDENTICAL numbers to the linear screen, so the two screens are
directly comparable)**: an input class graduates to the deep-model input tournament if and only
if:

1. McNemar p < 0.0071 (Bonferroni-corrected) on **≥ 2 of the 3 folds**, AND
2. Average DA improvement across F1–F3 is **≥ +0.5 percentage points**.

Both conditions must hold, exactly as before.

**Reported, never used to rank or gate**:

- **Brier score**, every arm, every fold (LightGBM's `predict_proba` output).
- **DA on the naive-persistence-disagreement subset**, identical definition to the linear screen.
- **Feature importance/gain per arm** (LightGBM's built-in `gain`-based importance, summed per
  logical feature group — e.g. all 600 price-only dims as one group, each extra feature as its
  own). Pre-committed use: if an arm fails the graduation bar but one of its extra features shows
  gain far above the price-only baseline's per-feature average, that is flagged as "worth a closer
  look next round," not as a retroactive graduation — the bar in this section is the only thing
  that decides graduation.

---

## 6. Interpretation rule for this experiment's overall verdict (pre-committed, PM-authorized)

- **If this screen ALSO produces zero graduating arms**: the "new information sources" lever is
  formally retired for ETHUSDT-1h within the six audited input classes and this feature contract.
  The recorded standing conclusion becomes: four independent tournaments (window #898, architecture
  #939, target-redesign) plus two screens (linear, this nonlinear re-screen) all find the same
  ~51–53% DA ceiling, across every lever tried (training window, model architecture, target/label
  design, and now both a linear and a nonlinear view of six alternative input classes). Future
  research levers shift to trade geometry, frequency/symbol diversification, and the live-parity
  gap — not further feature-set expansion within this candidate list.
- **If anything graduates**: that arm (or arms) defines the entrant list for the deep-model input
  tournament — no further screening round before that tournament is preregistered separately.
- Either outcome is decision-grade and will be reported as such, per the anti-p-hacking discipline
  that negative results get full write-ups, not silent gaps.

---

## 7. Data, leakage discipline, compute plan

Identical to the linear screen (§6/§7/§8 there): same disk-cached raw sources
(`scripts/research/.cache/`, no re-fetch), same causal/embargo discipline (the 48h embargo gates
every rolling/z-score statistic, not just the label; no same-bar high/low lookahead). BLAS thread
caps (`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/`NUMEXPR_NUM_THREADS=4`) carried
over from the linear screen's compute-contention lesson. `lightgbm` is a new, research-only
dependency (`pip install lightgbm`, not added to `requirements.txt` — not used by any production/
serving code path, matching the existing pattern for research-only scripts under
`scripts/research/`).

Total fits: 7 arms × 3 folds = 21 LightGBM fits (plus the free naive-persistence baseline, already
computed and unchanged from the linear screen). Each fit includes early stopping, so wall-clock per
fit may exceed the linear screen's logistic-regression fits — still CPU-only, no GPU, no cloud.

---

## 8. Decision this experiment feeds

Screening gate, not a strategy-change proposal — identical scope statement to the linear screen.
No arm's outcome here authorizes any change to a live-affecting strategy, model, or
`risk-limits.json`. Per §6, this experiment's outcome (either branch) is itself the decision-grade
answer to "should feature-set expansion continue as a research lever" — that answer will be
reported plainly, not treated as an intermediate step needing a further round to become
actionable.

---

*Pre-registration locked at the above wording. Results appended below after the run, never by
editing the sections above.*

---

## Results

**Environment note**: `lightgbm` required installing the OpenMP runtime (`brew install libomp`)
on this macOS host — not previously needed by anything else in this repo. Noted for reproducibility,
not a threshold/method change.

**Run**: single complete pass, all 7 candidate arms × 3 folds, `scripts/research/run_input_screening_nonlinear.py`,
raw output committed at `scripts/research/input_screening_nonlinear_results.json`. 21 fits, each
with early stopping against the train-tail validation split (best iteration ranged 7–81 against
the `n_estimators=300` ceiling — early stopping engaged on every single fit, never hit the cap).
BLAS threads capped at 4 throughout (carried over from the linear screen's compute-contention
fix). Wall-clock: ~1–3s per fit, ~50s total — faster than the linear screen's logistic-regression
fits, since early stopping keeps most trees small.

### Validity check (context only, per §2 — not a pass/fail gate)

| Fold | Price-only LightGBM control | Price-only logistic control (linear screen) | Diff |
|---|---|---|---|
| F1 | 51.15% | 51.94% | −0.79pp |
| F2 | 52.96% | 53.10% | −0.14pp |
| F3 | 51.58% | 50.79% | +0.79pp |

Trees on the bare price-only contract track the linear control closely (within ~0.8pp every
fold, no systematic direction) — no evidence trees find extra structure in price-only alone, an
internally consistent finding. The LightGBM price-only control (not the linear one) is this
experiment's own reference for every arm below.

### Per-fold, per-arm results

**F1** (eval 2023-01-03→2023-06-30, n_eval=4,295, naive persistence DA=46.17%):

| Arm | n_train (fit/val) | DA | Δ vs LightGBM control (pp) | McNemar p | Brier | Best iter |
|---|---|---|---|---|---|---|
| price_only_control | 42,175 / 4,686 | 51.15% | — | — | 0.2495 | 20 |
| realized_vol_range | 42,131 / 4,681 | 50.83% | −0.33 | 0.652 | 0.2499 | 22 |
| calendar | 42,175 / 4,686 | 51.06% | −0.09 | 0.924 | 0.2497 | 24 |
| btc_cross | 42,175 / 4,686 | **54.99%** | **+3.84** | **6.9e-05** | 0.2478 | 37 |
| funding_rate | 24,387 / 2,709 | 51.64% | +0.49 | 0.579 | 0.2496 | 11 |
| basis_premium | 23,795 / 2,643 | 51.53% | +0.37 | 0.681 | 0.2496 | 16 |
| fear_greed | 38,491 / 4,276 | 51.46% | +0.30 | 0.736 | 0.2503 | 81 |
| all_combined | 23,795 / 2,643 | 53.88% | +2.72 | 0.0056 | 0.2477 | 45 |

**F2** (eval 2024-01-03→2024-06-30, n_eval=4,320, naive persistence DA=46.57%):

| Arm | n_train (fit/val) | DA | Δ vs control (pp) | McNemar p | Brier | Best iter |
|---|---|---|---|---|---|---|
| price_only_control | 50,058 / 5,562 | 52.96% | — | — | 0.2492 | 23 |
| realized_vol_range | 50,014 / 5,557 | 52.66% | −0.30 | 0.712 | 0.2491 | 50 |
| calendar | 50,058 / 5,562 | 51.81% | −1.16 | 0.097 | 0.2492 | 48 |
| btc_cross | 50,058 / 5,562 | 54.12% | +1.16 | 0.226 | 0.2478 | 28 |
| funding_rate | 32,270 / 3,585 | 51.78% | −1.18 | 0.173 | 0.2496 | 31 |
| basis_premium | 31,678 / 3,519 | 52.11% | −0.86 | 0.322 | 0.2496 | 17 |
| fear_greed | 46,374 / 5,152 | 52.22% | −0.74 | 0.360 | 0.2490 | 78 |
| all_combined | 31,678 / 3,519 | 53.54% | +0.58 | 0.555 | 0.2482 | 32 |

**F3** (eval 2025-01-03→2025-06-30, n_eval=4,296, naive persistence DA=48.21%):

| Arm | n_train (fit/val) | DA | Δ vs control (pp) | McNemar p | Brier | Best iter |
|---|---|---|---|---|---|---|
| price_only_control | 57,964 / 6,440 | 51.58% | — | — | 0.2498 | 38 |
| realized_vol_range | 57,920 / 6,435 | 51.98% | +0.40 | 0.588 | 0.2496 | 34 |
| calendar | 57,964 / 6,440 | 51.12% | −0.47 | 0.468 | 0.2498 | 39 |
| btc_cross | 57,964 / 6,440 | 51.26% | −0.33 | 0.741 | 0.2504 | 49 |
| funding_rate | 40,176 / 4,463 | 51.63% | +0.05 | 0.979 | 0.2497 | 7 |
| basis_premium | 39,583 / 4,398 | 52.12% | +0.54 | 0.539 | 0.2495 | 12 |
| fear_greed | 54,279 / 6,031 | 51.98% | +0.40 | 0.609 | 0.2496 | 32 |
| all_combined | 39,583 / 4,398 | 53.12% | +1.54 | 0.100 | 0.2494 | 30 |

### Graduation verdicts (per §5's pre-committed rule, applied literally)

| Arm | Avg Δ vs control (pp, F1–F3) | Folds with p < 0.0071 | Verdict |
|---|---|---|---|
| realized_vol_range | −0.08 | 0/3 | **does not graduate** |
| calendar | −0.57 | 0/3 | **does not graduate** |
| btc_cross | **+1.56** | **1/3** | **does not graduate** |
| funding_rate | −0.22 | 0/3 | **does not graduate** |
| basis_premium | +0.02 | 0/3 | **does not graduate** |
| fear_greed | −0.01 | 0/3 | **does not graduate** |
| all_combined | **+1.61** | **1/3** | **does not graduate** |

**No arm graduates** — the rule requires ≥2/3 significant folds AND avg Δ≥+0.5pp; `btc_cross` and
`all_combined` clear the magnitude bar but only 1 of 3 folds each clears Bonferroni significance,
short of the required 2. Applying §5 literally, exactly as pre-committed, the graduation column is
uniformly "does not graduate."

### The btc_cross near-miss, named honestly (not folded into "clean null")

**This screen is NOT a repeat of the linear screen's clean, uniform null — `btc_cross` behaves
genuinely differently under trees than under logistic regression**, and this deserves being said
plainly rather than smoothed into "still nothing graduates":

- F1: DA 54.99% vs control 51.15%, **Δ=+3.84pp, p=6.9e-05** — this clears Bonferroni significance
  by four orders of magnitude, not a borderline call.
- F2: Δ=+1.16pp, p=0.226 — same direction, much weaker, not significant.
- F3: Δ=−0.33pp, p=0.741 — reverses sign, not significant.

**Feature-importance/gain (reported, never gating, per §5)** shows `btc_ret_1h`/`btc_ret_6h`
carrying real, consistent gain in ALL THREE folds (16–22% and 6–8% of total gain respectively,
`btc_cross` arm) — not just F1. This is the same pattern the target-redesign tournament flagged
for its own confidence signal: **gain/importance does not straightforwardly track OOS DA
improvement** — the tree consistently finds BTC-return splits worth making, but that only
translates into a significant, generalizing accuracy edge in one of three regimes (F1, "post-
crypto-winter chop/recovery" per the fold table). `all_combined`'s pattern is driven by the same
BTC features (still the two largest non-price-only gain contributors in every fold of that arm).

**Read plainly**: this is exactly the kind of result the linear screen's own named false-negative
risk predicted — a nonlinear detector found something a linear one did not. It does not graduate
under the pre-committed rule (regime-specific, not persistent across ≥2/3 folds), and the rule is
applied as written, not loosened after seeing this number. But "found nothing at all" would be the
wrong headline for this arm specifically; "found a real, regime-dependent signal that does not
generalize across the three tested regimes" is the accurate one, and is exactly the kind of
disclosure §5's feature-importance reporting exists to surface.

### Reading this result honestly, against the pre-committed interpretation rule (§6)

Applying §6 literally: since **zero arms graduate** (the rule's condition, checked exactly as
written), the "new information sources" lever is formally retired for ETHUSDT-1h within these six
input classes and this feature contract, per the PM-authorized interpretation rule. This is now
six converged results — four tournaments (window #898, architecture #939, target-redesign) plus
two screens (linear, this nonlinear re-screen) — all finding the same ~51–53% DA ceiling under
every lever tried: training window, model architecture, target/label design, and now both a
linear and a nonlinear view of six alternative input classes.

**The one qualification to that headline, stated as directly as the evidence supports**:
`btc_cross` is not "no signal" the way the other five candidates are — it is "a real,
highly-significant-in-one-regime signal that a linear model cannot see and a nonlinear model can,
but which does not persist across the two other tested regimes." This is not strong enough to
graduate under the pre-committed bar (correctly not graduating — one significant fold out of three
is exactly the kind of single-lucky-draw result the ≥2/3 requirement exists to filter), but it is
also not nothing, and closing the "new information sources" lever entirely would be slightly
overclaiming if it silently dropped this distinction. Recommended framing: the lever closes for
five of six input classes with high confidence; `btc_cross` closes for THIS screening design
specifically (i.e., "detect a fold-persistent linear-or-tree-detectable edge") while leaving open
a narrower, separately-scoped question — whether BTC-ETH lead-lag is a regime-conditional
relationship (stronger in some vol/trend regimes than others, consistent with the audit's own
literature read: "plausible, weak, almost certainly time-varying") worth testing with an explicit
regime-interaction design, not a blanket four-feature addition. That is a distinct, narrower
research question, not a re-run of this screen, and is not decided or committed to here.

### Recommendation

**Confirms the pattern; formally retires the "new information sources" lever for five of six input
classes; flags `btc_cross` as a narrower, separately-scoped open question, not a graduating
candidate.** Per §6's pre-committed interpretation, this experiment's outcome (zero graduating
arms) is itself decision-grade: future research levers for this system shift to trade geometry,
frequency/symbol diversification, and the live-parity gap, rather than further feature-set
expansion within the six audited classes. A regime-conditional BTC-ETH lead-lag question is named
as a candidate for a future, separately-preregistered, narrowly-scoped experiment — not
authorized or scheduled here.

**For risk-officer / pm**: nothing here proposes a live-affecting change; there is nothing to
stress-test.

**Status**: COMPLETE.
