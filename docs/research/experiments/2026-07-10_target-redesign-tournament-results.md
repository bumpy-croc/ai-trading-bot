# TARGET-REDESIGN Tournament — Results

Date: 2026-07-10 to 2026-07-11
Author: quant-researcher
Status: **COMPLETE — fact-check pass done, F4 confirmatory numbers inserted, ready for PR**
Issue: GH #933 (research survey), locked prereg PR #946, Amendment 1 (PR #951, merged),
Amendment 2 (PR #956, merged to develop as of `25e0a202`)
Preregistration: `docs/research/experiments/2026-07-10_target-redesign-tournament-prereg.md`
(read in full, including both amendments — this report answers to it section by section)
Protocol: `docs/architecture/model_evaluation_system.md`, `.claude/skills/model-tournament/SKILL.md`
North star: "Is next-bar price the right target at all?" (open question #2)

**Source of every number in this report**: `results.json` and `methods.md`, produced in
`.claude/worktrees/target-redesign-tournament` (detached at `6fc224c0`, PR #950 on #948). No
number below was recomputed or derived by this writer — where a number appeared missing or
inconsistent, it is flagged in the TODO list at the end rather than filled in.

---

## 1. Executive verdict (read this first)

**No entrant proceeds to L3a staging. The round closes without a staging trial.**

Applying prereg §7's decision table literally, row by row, against the CORRECTED aggregation
(`results.json` → `aggregate_stats_CORRECTED`, per-fold-averaged accuracy, Bonferroni α=0.0083,
significance tested per fold not pooled):

| §7 row | Literal condition | Applies? |
|---|---|---|
| 1. One entrant clears BOTH the primary quality bar (Bonferroni-significant vs naive AND incumbent, aggregated F1–F3) AND the money-exam gate | Entrant (c) `triple_barrier` clears the **first half** — significant wins vs naive persistence and vs incumbent control on **every one of F1/F2/F3 individually and averaged** (`aggregate_stats_CORRECTED.per_fold_significance_bonferroni_alpha_0.0083`: `c_vs_naive` and `c_vs_incumbent` both `SIGNIFICANT (c wins)` on F1, F2, F3). It does **not** clear the second half — its exam profit factor is 0.4953 (F1), 0.4085 (F2), 0.3827 (F3), all deeply below 1.0 (`L2_results_full.entrant_c_triple_barrier`). **The AND fails. Row 1 does not fire, for any entrant.** | **No** |
| 2. Multiple entrants cluster within Bonferroni-corrected noise, no confident pairwise winner | Does not describe what happened. Entrant (c) is not clustered with anything — it is a statistically **confident** pairwise winner over naive and incumbent on the L1 letter, every fold. Entrant (b) is a narrower, partial winner (beats naive on F1/F2, not F3; loses to incumbent on F1/F2, ties F3). This is not "no winner" in the literal sense — it's "a winner exists on the L1 letter, but the mechanism behind that win is the concern" (see §2 below). | **No, not the literal fit — see caveat below** |
| 3. No entrant beats naive persistence on the primary quality bar | False — (b) and (c) both beat naive on at least one fold each; (c) on every fold. | **No** |
| 4. An entrant fails to train (infra bug) | Applies narrowly to entrant (d)/F3: two collapsed attempts, one retry per the pre-committed one-retry policy, final state "trained but degenerate," excluded from ranking (F3 has zero usable (d) model). This is a **partial**, already-executed instance of row 4's protocol, not itself the tournament's overall verdict. | **Partially, for (d)/F3 only** |
| 5. Entrant (a) fails specifically because the primary signal has no edge | **Resolved during fact-check** (gap closed, see §3): entrant (a)'s OOS accuracy is 0.5063 (F1), 0.5243 (F2), 0.5307 (F3) — an **exact tie with its own majority-class dummy baseline on all three folds** (`L1_results_full_entrant_a_OOS_GAP_CLOSED`). This is the SAME base-rate-collapse mechanism confirmed for entrant (b), in a different model family (sklearn LogisticRegression on 8 tabular features, not a tft classifier). Distinguishable from row 5's literal wording ("primary signal has no edge") — the primary signal (§2a's own named risk) is a separate question from the meta-classifier itself collapsing; both may be true, but what's directly confirmed here is the classifier-level collapse, matching (b) and (partially) (c). | **No, not row 5's literal fit — but a confirmed collapse nonetheless, now folded into the mechanism finding (§2)** |

**The honest resolution**: the table's rows don't cleanly nest this specific outcome ("one entrant
statistically clears the L1 letter on every fold, but the L1 win is itself of questionable
character, and the money gate fails universally"). §4 of the prereg pre-committed language for
exactly this shape of outcome: *"An entrant can rank #1 on primary metrics and still fail this
gate — that is a valid, reportable outcome ('quality win, not yet exam-actionable'), not grounds
to lower the gate post hoc."* That is the correct characterization of entrant (c), and by
extension of the tournament as a whole: **a "quality win, not yet exam-actionable" outcome, for
entrant (c) only, combined with a universal money-gate failure that closes the round for every
entrant.** No staging trial follows. If this reading is wrong — if the PM's intent was for this
to trigger row 1 or row 2 as literally worded — that disagreement should be resolved before this
goes to PR; flagged explicitly rather than silently forced into either box.

### The statistical letter vs. the mechanism, side by side

| Entrant | L1 statistical letter (Bonferroni, per fold) | Mechanism (from `probability_distribution_diagnostic`) | Money exam (PF, every fold) |
|---|---|---|---|
| (b) binary_direction | Beats naive F1/F2 (not F3); **loses to incumbent** F1/F2, ties F3 | **CONFIRMED**, clean 3-for-3: `P(up)` mean tracks each fold's own class-1 base rate within 0.4–0.8pp, std ≈0 (F3 literally bit-identical constant). Classic BCE-with-no-signal degenerate optimum. | 0.501 / 0.449 / 0.381 — all net-lossy |
| (c) triple_barrier | Beats naive **and** incumbent, every fold, Bonferroni-significant | **Fold-dependent, NOT unified** — F1 and F3 are exact ties to the majority-class dummy classifier (accuracy == dummy_accuracy to 4+ sig figs, 100% single-class predictions); F2 is genuinely non-degenerate (predicted-class std 30–40x larger than F1/F3, real +2.16pp edge over dummy). The fold-averaged "win" is driven almost entirely by triple_barrier's own ~59–60%-majority-class label skew, not learned discriminative skill, except in F2. | 0.495 / 0.409 / 0.383 — all net-lossy |
| (d) smoothed_return | Only F1 usable (F2/F3 collapsed); F1 not significant vs naive, loses to incumbent | **Regression analogue of (b)'s collapse**: prediction std 127–137x below label std at F1; 3 of 5 total training attempts across F2/F3 produced literal constant output (std = 0.000000) | F1 0.312, F2 0.405 (F3 N/A) — net-lossy |
| (a) meta_label | **Exact tie with dummy, every fold** (0.5063/0.5243/0.5307, all == dummy_accuracy) — gap closed during fact-check | **CONFIRMED base-rate collapse**, same family as (b): `P(profitable)` mean sits near 0.55–0.56 with std 0.017–0.023 (looser than (b)'s 0.000–0.004 but still never crosses 0.5 the wrong way on any fold) — a THIRD entrant, in a different model family (sklearn LogisticRegression vs (b)'s tft), converging to the same degenerate optimum | 0.384 / 0.392 / 0.408 — all net-lossy, worst-clustered of the four |
| Incumbent control | Reference point, not ranked | Not diagnosed (regression target, different mechanism family) | (deployed-semantics) 0.369 / 0.342 / 0.364 — also net-lossy |
| Linear baseline | 53.2–53.6% accuracy, closely tracking incumbent's 53.0–54.8% — corroborating evidence, not ranked | — | Not exam-run (L1 health check only, per north-star rule 6) |

**The money-exam gate itself has a disclosed measurement gap**: no naive-persistence variant of
the money exam was built this session, so the literal gate text ("beats naive on both OOS return
and profit factor") cannot be checked against a naive money-exam number directly
(`L2_results_full.naive_persistence_money_exam`). What **is** directly measured: every entrant's
exam profit factor across every fold sits in **0.31–0.50**, i.e. every entrant loses money net of
fees on every fold, with no exception. Results.json's own language: this makes it "very unlikely
ANY entrant robustly clears the gate even if naive's own exam PF were computed" — an inference,
not a verified fact, and reported as such.

---

## 2. The mechanism finding (likely the tournament's most valuable output)

Three independent tournaments have now asked "what's the binding constraint on this system's
~53% directional-accuracy ceiling?" and answered it three different ways — window choice (#898),
architecture choice (#939), and now, for the first time, *why the models behave the way they do*
rather than just *that they cluster*:

**Entrant (b) — clean, textbook confirmation of BCE degeneracy.** Across all three primary folds,
`P(up)`'s mean tracks that fold's own actual class-1 base rate to within 0.4–0.8 percentage points,
while its standard deviation collapses toward zero (F1 std=0.0024, F2 std=0.0037, F3 std=0.0000 —
literally bit-identical across all 4,273 eval bars,
`probability_distribution_diagnostic.entrant_b.F3`). This is the textbook degenerate optimum for a
binary-cross-entropy classifier facing a target with no real discriminative signal in its inputs:
absent a learnable relationship, the loss-minimizing strategy is to predict the training-set base
rate as a near-constant, which is exactly what happened, and the degeneracy *worsens* fold over
fold. A synthetic-input control (random noise fed to the same ONNX session) does produce varying
sigmoid output (0.494–0.601), confirming the network retains some raw input-sensitivity — it just
never crosses 0.5 on real ETHUSDT feature sequences specifically. Entrant (b) also **loses to the
incumbent control on 2 of 3 folds** despite "beating naive" on 2 of 3 — a classifier converged to
the base rate can still edge out naive persistence (which is itself a poor baseline here, 46–48%)
without being informative in any absolute sense.

**Entrant (a) — a THIRD entrant, a different model family, the same collapse (found during
fact-check, closing a gap the writer's draft had flagged).** Meta-labeling's sklearn
`LogisticRegression` classifier, scored OOS by generating fresh fires over each fold's own eval
window (not the training window) and comparing its `P(profitable)` against those genuinely-OOS
trade resolutions: accuracy of 0.5063 (F1), 0.5243 (F2), 0.5307 (F3) — an **exact tie with its own
majority-class dummy baseline on all three folds**, no exception. `P(profitable)`'s std (0.017–0.023)
is looser than entrant (b)'s (0.000–0.004), but the effect is identical: the classifier never
crosses the 0.5 threshold in the minority direction on any real eval bar, in any fold. This is now
**three** entrants ((a), (b), and 2 of (c)'s 3 folds), across **two different model families**
(binary classification networks and a linear sklearn classifier), converging to the same failure
mode. The coverage-curve check below adds a further wrinkle even for the one place real signal
does appear (entrant (c)/F2): the variation that exists there is not well-calibrated as usable
confidence.

**Entrant (d) — the regression analogue of the same failure.** Under MSE loss with a
near-zero-variance smoothed-return target, the degenerate optimum is "predict close to the
training-set mean," and the model converges there wholesale rather than partially: prediction
std ratios of 137x (F1), ~895x (F2 retry), and full constant collapse (std = 0.000000) on **3 of
5 total training attempts** across F2-attempt1, F3-attempt1, and F3-retry
(`probability_distribution_diagnostic.entrant_d_regression_to_mean`). F3 has no usable model at
all after a retry — final state, per the pre-committed one-retry policy.

**Entrant (c) — genuinely fold-dependent, and the tournament's own diagnostic explicitly refuses
to force a unifying story here** (per the PM's instruction not to overclaim, honored in
`probability_distribution_diagnostic.entrant_c.verdict`). F1 and F3 collapse to an exact tie with
the majority-class dummy classifier — 100% single-class predictions, accuracy identical to
dummy_accuracy to the reported precision. F2 is real: predicted-class-probability std 30–40x
larger than F1/F3, a full 0.14–0.86 range, and a genuine +2.16pp edge over the fold's own dummy
baseline. Read plainly: triple_barrier's aggregate "win" over naive and incumbent is substantially
a label-imbalance artifact (the barrier geometry produces a skewed ~59–60%-majority-class target
in most folds, which a degenerate always-predict-majority classifier exploits automatically) —
except in F2, where something real appears to be happening. This is not "meta-labeling doesn't
work" or "triple-barrier doesn't work" in general; it's a specific, disclosed caveat about what
this tournament's three folds actually showed.

**The linear baseline corroborates from a different angle.** A plain `sklearn.LinearRegression` on
the *exact same* feature contract as the incumbent (`PriceOnlyFeatureExtractor`, 120-bar
flattened, `close_normalized` target) scores 53.2–53.6% accuracy across the three folds
(`training_matrix.linear_baseline`), closely tracking the incumbent neural model's own 53.0–54.8%
(`L1_results_full.incumbent_control`) — a cheap, local, no-training-required linear model gets
essentially the same OOS accuracy as the deployed architecture. This is consistent with, and adds
a third independent line of evidence to, the standing hypothesis first raised by the window
tournament (#898, training-window selection is not the binding constraint) and the architecture
tournament (#939, model architecture is not the binding constraint): **the price-only 1h feature
set itself appears to be the ceiling.** No amount of target reshaping (this tournament), model
architecture (#939), or training-window choice (#898) has moved the needle past ~53–54%
directional accuracy or produced a money-exam profit factor anywhere near 1.0. All four
mechanisms found this session — (a)'s and (b)'s clean base-rate collapses (two different model
families), (d)'s MSE analogue, (c)'s partial (F1/F3) analogue with a genuine exception (F2, itself
undermined by a mis-calibrated confidence signal) — are different symptoms of the same root
cause: with no real discriminative signal in the inputs, every loss function converges to its own
flavor of "predict the unconditional distribution."

---

## 3. Full results tables

### L1 — directional/classification accuracy, vs naive AND vs dummy, Brier

| Entrant | Fold | n | Accuracy | Dummy accuracy | Brier | Note |
|---|---|---|---|---|---|---|
| (b) binary_direction | F1 | 4,272 | 0.5068 | **0.5068 (exact tie, retro-confirmed: 100% class-1 predictions)** | 0.2499 | predicted 100% class 1 |
| (b) binary_direction | F2 | 4,297 | 0.5148 | **0.5148 (exact tie, retro-confirmed)** | 0.2498 | predicted 100% class 1 |
| (b) binary_direction | F3 | 4,273 | 0.5099 | **0.5099 (exact tie)** | 0.2499 | predicted 100% class 1 |
| (c) triple_barrier | F1 | 4,272 | 0.5941 | **0.5941 (exact tie)** | 0.5055 | total collapse, 100% single-class |
| (c) triple_barrier | F2 | 4,297 | 0.5958 | 0.5741 (+2.16pp real edge) | 0.5047 | genuinely non-degenerate |
| (c) triple_barrier | F3 | 4,273 | 0.6000 | **0.6000 (exact tie)** | 0.4911 | total collapse again |
| (d) smoothed_return | F1 | 4,272 | 0.4846 | — | — | sign-hit-rate, post metadata-bridge-patch |
| (d) smoothed_return | F2 | 4,297 | 0.5125 | — | — | retry model, gap closed |
| (d) smoothed_return | F3 | — | **N/A** | — | — | both attempts collapsed, no usable model |
| (a) meta_label | F1 | 4,152 | 0.5063 | **0.5063 (exact tie)** | 0.2549 | fires + resolutions generated fresh over the EVAL window (not training window), scored OOS |
| (a) meta_label | F2 | 4,297 | 0.5243 | **0.5243 (exact tie)** | 0.2510 | OOS, exact tie again |
| (a) meta_label | F3 | 4,273 | 0.5308 | **0.5308 (exact tie)** | 0.2496 | OOS, exact tie again — 3 of 3 folds |
| Incumbent control | F1 | 4,152 | 0.5373 | — | — | fold-matched retrain |
| Incumbent control | F2 | 4,138 | 0.5478 | — | — | fold-matched retrain |
| Incumbent control | F3 | 4,157 | 0.5297 | — | — | **Amendment-2-corrected** retrain (`price/2026-07-11_11h17m33s_v1`), NOT the excluded live artifact |
| Naive persistence | F1 | 4,268 | 0.4616 | — | — | free, no training |
| Naive persistence | F2 | 4,296 | 0.4662 | — | — | |
| Naive persistence | F3 | 4,271 | 0.4821 | — | — | |
| Linear baseline | F1 | 4,151 | 0.5324 | — | — | health check, not ranked |
| Linear baseline | F2 | 4,137 | 0.5361 | — | — | |
| Linear baseline | F3 | 4,156 | 0.5318 | — | — | |

**Averaged accuracy (F1–F3, prereg §4 method)**: naive 0.4700, (b) 0.5105, (c) 0.5966, (d) 0.4846
(F1 only — F2/F3 excluded, trained-but-degenerate), incumbent 0.5383, linear 0.5334
(`aggregate_stats_CORRECTED.averaged_accuracy`). **Entrant (a)'s OOS accuracy (0.5063/0.5243/0.5307,
averaged 0.5205) is deliberately NOT folded into this same pairwise table** — its metric measures
"is this fired trade profitable," a different semantic than (b)/(c)/(d)'s "is next-bar/barrier
direction correct," so a naive vs./incumbent pairwise z-test on the same footing would compare
different questions. What IS directly comparable and reported: entrant (a)'s OOS accuracy exactly
ties its own majority-class dummy on all three folds, exactly like (b) and 2 of (c)'s 3 folds — see
§2.

**Per-fold Bonferroni significance (α=0.0083)** — full table, `aggregate_stats_CORRECTED.per_fold_significance_bonferroni_alpha_0.0083`:

| Fold | b vs c | b vs d | c vs d | b vs naive | c vs naive | d vs naive | b vs incumbent | c vs incumbent | d vs incumbent |
|---|---|---|---|---|---|---|---|---|---|
| F1 | SIG, c wins (z=-8.11) | not sig (z=2.06) | SIG, c wins (z=10.16) | SIG, b wins (z=4.18) | SIG, c wins (z=12.27) | not sig (z=2.13) | SIG, incumbent wins (z=-2.81) | SIG, c wins (z=5.26) | SIG, incumbent wins (z=-4.84) |
| F2 | SIG, c wins (z=-7.55) | N/A (d excluded) | N/A | SIG, b wins (z=4.50) | SIG, c wins (z=12.03) | N/A | SIG, incumbent wins (z=-3.04) | SIG, c wins (z=4.45) | N/A |
| F3 | SIG, c wins (z=-8.38) | N/A | N/A | not sig (z=2.58) | SIG, c wins (z=10.94) | N/A | not sig (z=-1.82) | SIG, c wins (z=6.51) | N/A |

Method note: this is a two-proportion z-test approximation per fold (not pooled), the same
disclosed limitation the 2026-07-06 architecture tournament used — true per-bar-paired McNemar's
was not computed this session (would need per-bar correctness vectors, not captured).

### L2 — money exam (trades, return, profit factor, MaxDD), fees/slippage on, per fold

| Entrant | Fold | Trades | Return % | Profit factor | Max DD % | Final balance |
|---|---|---|---|---|---|---|
| (a) meta_label | F1 | 660 | -6.73 | 0.3840 | 6.75 | 79.29 |
| (a) meta_label | F2 | 757 | -8.54 | 0.3917 | 8.54 | 77.74 |
| (a) meta_label | F3 | 811 | -9.77 | 0.4077 | 10.09 | 76.71 |
| (b) binary_direction | F1 | 659 | -4.98 | 0.5011 | 5.02 | 80.78 |
| (b) binary_direction | F2 | 753 | -7.03 | 0.4489 | 7.11 | 79.03 |
| (b) binary_direction | F3 | 833 | -8.78 | 0.3812 | 8.79 | 77.53 |
| (c) triple_barrier | F1 | 659 | -5.34 | 0.4953 | 5.39 | 80.47 |
| (c) triple_barrier | F2 | 1,020 | -9.40 | 0.4085 | 9.45 | 77.01 |
| (c) triple_barrier | F3 | 833 | -10.62 | 0.3827 | 10.62 | 75.97 |
| (d) smoothed_return | F1 | 428 | -5.94 | 0.3120 | 5.98 | 79.96 |
| (d) smoothed_return | F2 | 543 | -7.09 | 0.4047 | 7.29 | 78.97 |
| (d) smoothed_return | F3 | — | N/A | N/A | N/A | — |
| Incumbent (deployed semantics) | F1 | 127 | -3.07 | 0.3687 | 3.10 | 82.41 |
| Incumbent (deployed semantics) | F2 | 198 | -4.72 | 0.3423 | 4.80 | 80.99 |
| Incumbent (deployed semantics) | F3 | 277 | -6.41 | 0.3638 | 6.53 | 79.56 |

Note: incumbent's L2 exam runs `MLBasicSignalGenerator`'s own deployed production confidence
mechanism (a hardcoded-constant-derived formula) — deliberately, by PM ruling, because the
control's job is to represent the status quo *as deployed*; the harness-wide no-hardcoded-constant
rule governs entrants only. All other components (sizer, risk wiring, exam-harness geometry) are
identical across every row.

**Determinism guard (§6, prereg constraint 5)**: entrant (b)/F1 run twice back-to-back,
byte-for-byte identical full results JSON both times (trades=659, return=-4.980356186688639%,
PF=0.5011210759184279) — **PASS**. This is the number that appears in the L2 table above; every
other number in this report is trusted on the strength of this guard passing.

### Accuracy-vs-coverage curves (prereg §4: reported, never ranked/gated)

Gap closed during fact-check (`coverage_curve.py`, actual numbers, not qualitative estimates).
Accuracy at the top 10%/25%/50%/100% most-confident bars, ranked within each fold's eval window:

| Entrant | Fold | 10% | 25% | 50% | 100% | Confidence range (full) |
|---|---|---|---|---|---|---|
| (b) | F1 | 0.5012 | 0.5243 | 0.5112 | 0.5068 | 0.0026–0.0185 |
| (b) | F2 | 0.5221 | 0.5289 | 0.5209 | 0.5148 | 0.0012–0.0231 |
| (b) | F3 | 0.4965 | 0.4963 | 0.5014 | 0.5099 | 0.0000009 (bit-identical) |
| (c) | F1 | 0.5714 | 0.5805 | 0.5510 | 0.5941 | 0.5231–0.5456 |
| (c) | F2 | 0.5455 | 0.5754 | 0.5475 | 0.5958 | 0.4982–0.8573 |
| (c) | F3 | 0.5878 | 0.5880 | 0.6213 | 0.6000 | 0.5141–0.8603 |

**Read plainly**: every single curve above is **non-monotonic** — higher stated confidence does
NOT reliably track higher accuracy, on any entrant, any fold, including F2 (entrant (c)'s one
fold with genuine, non-degenerate probability variation). Entrant (c)/F2's top-10%-most-confident
bars score **54.6%**, actually *worse* than its unconditional 100%-coverage accuracy of 59.6% —
the opposite of what a well-calibrated confidence signal would produce. This is a second, separate
finding from the base-rate-collapse story: even where real predictive variation exists (F2), it is
**not well-calibrated as a confidence ranking**. For entrant (b), the curves are trivially
uninformative for the mechanical reason already established — confidence barely varies at all
(F1's full range spans 1.6 percentage points across all 4,272 bars), so ranking by it is close to
a random subset draw. Entrant (a)'s literal coverage curve was not separately computed (time-boxed
after the OOS-accuracy gap closure above) — given its own confirmed exact-tie-to-dummy collapse,
the same qualitative conclusion (little to no abstention-useful signal) is the reasonable
expectation, disclosed as inferred rather than directly measured for this one entrant.

### F4 — confirmatory only, NEVER RANKED per prereg §3 [COMPLETE]

F4 (train cutoff 2026-04-30, eval 2026-05-03→2026-07-09) is explicitly non-deciding per the
prereg — it exists only to check whether the F1–F3 verdict holds on the most recent available
data, not to change the verdict itself. All four training slots (b, c, d, incumbent-control) are
now complete, including one uniform-policy retry for entrant (d) (collapsed both times — see
below).

| Entrant | Fold | n | Accuracy | Dummy accuracy | Trades | Return % | PF | Max DD % |
|---|---|---|---|---|---|---|---|---|
| Naive persistence | F4 | 1,609 | 0.4767 | — | — | — | — | — |
| (b) binary_direction | F4 | 1,609 | 0.4891 | 0.5109 | 263 | -2.74 | 0.3493 | 2.79 |
| (c) triple_barrier | F4 | 1,536 | 0.3685 | 0.6315 | 263 | -3.37 | 0.3440 | 3.43 |
| (d) smoothed_return | F4 | — | **N/A, no usable model** | — | — | — | — | — |
| Incumbent control | F4 | 1,551 | 0.5377 | — | 58 | -0.79 | 0.5768 | 1.01 |

**F4 delivers the cleanest confirmation of the base-rate-collapse mechanism in the entire
tournament.** Both (b) and (c) are STILL predicting 100% of bars as class 1 — the identical frozen
behavior confirmed on F1–F3 — but F4's actual regime flipped to a class-(-1) majority (51.1% for
(b)'s binary labels, 63.2% for (c)'s triple-barrier labels). The frozen collapse doesn't adapt: it
now actively *underperforms* its own fold's dummy baseline on both entrants. Entrant (c)'s F4
accuracy (0.3685) is almost exactly `1 - dummy_accuracy` (0.6315) — i.e., getting it wrong on
almost exactly the fraction of bars its frozen "always class 1" behavior now collides with the
flipped regime. This is strong, independent evidence that the collapse mechanism (§2) is anchored
to the *training-period* base rate, not something that happens to luckily track each window's own
majority class by coincidental design — it fails exactly where a genuine regime shift exposes it.

**Entrant (d)/F4: collapsed on both the initial run and the one uniform-policy retry** (constant
outputs `6.19e-05` and `7.31e-05` respectively, both confirmed via direct input probing). Per the
same uniform one-retry-then-accept policy applied to (d)/F2 and (d)/F3, this is final —
trained-but-degenerate, no usable model, excluded. Combined with F2/F3's own history, this brings
the total tally to **4 outright collapses out of 6 entrant-(d) training attempts across all four
folds** (only F1, near-collapse but usable, and F2's retry, genuinely usable, produced a
non-degenerate model) — overwhelming, not just suggestive, confirmation that the near-zero-variance
smoothed-return target is fundamentally difficult for this architecture/loss combination to fit,
independent of which fold or how much data.

**Conclusion**: F4 does not change the headline verdict (§1) — it was never going to, per its own
non-deciding design — but it substantially *strengthens* the tournament's central mechanism
finding rather than merely repeating it. The regime-flip is the single most direct piece of
evidence in this report that entrants (b) and (c)'s apparent "wins" over naive/incumbent on
F1–F3 are an artifact of a frozen, non-adaptive collapse meeting a training-period base rate that
happened to align with those three folds' own regimes — not a sign of genuine, generalizable
skill.

---

## 4. Honest trail

### Amendments to the locked prereg (both pre-data, both validity-strengthening)

- **Amendment 1 (PR #951, merged)**: the original §2a text specified entrant (a)'s primary signal
  as "the currently-deployed incumbent," which — taken literally — would have used a model trained
  through 2025-12-31 to generate fires/features on F1 (eval 2023-01-03→2023-06-30) and F2 (eval
  2024-01-03→2024-06-30), both of which fall entirely inside that training window. This is
  lookahead contamination of exactly the kind the tournament's whole purge/embargo fold design
  exists to eliminate. Caught and fixed before any entrant-(a) job was submitted or any entrant-(a)
  number existed. Fix: each fold's entrant (a) now uses **that fold's own incumbent-control
  retrain** as its primary signal, at zero additional training cost (the fold-matched retrains
  were already required for the incumbent-control baseline row).
- **Amendment 2 (PR #956, merged to develop)**: the prereg's original baselines section claimed
  the live incumbent artifact (`basic/2026-07-04_22h_v1`) could stand in as the F3 control without
  retraining, because its "training cutoff is 2025-12-31." Direct inspection of that artifact's
  own `metadata.json` (`training_params.end_date`) showed the **actual** cutoff is 2026-07-04 —
  covering F3's entire eval window (2025 H1) plus a year beyond it. Same contamination class
  Amendment 1 fixed, this time for the F3 baseline itself and for entrant (a)/F3's intended
  primary signal, both of which were about to consume the contaminated artifact under an earlier,
  same-day PM ruling that inherited the same wrong premise. Fix: the fold-matched F3 retrain
  (`price/2026-07-11_11h17m33s_v1`, cutoff 2024-12-31, originally logged as an "unnecessary"
  accidental job) is now authoritative for F3's L1 row, L2 exam, and entrant (a)'s primary signal;
  the live artifact is excluded from F3 entirely, not even as a cross-check.

### Entrant (d) metadata bug — found, bridge-patched, upstreamed

This tournament's own scaffolding (`pipeline.py`, PR #948/#950, pre-#954) seeded a bundle's
`target_distribution` metadata from **training labels** (`np.abs(y_train)`) instead of the
model's own predictions over the training split. Since prediction variance is far smaller than
label variance under MSE regression-to-the-mean, this starved `percentile_rank_confidence` to
near-zero on every window — a false "0 trades" reading for entrant (d)/F1 before the bug was
found (confirmed directly: F1 label_std=0.017309 vs prediction_std=0.000127, ratio ~137x). A
per-bundle bridge patch (`patch_smoothed_return_distribution.py`) rebuilt `target_distribution`
from actual model predictions, replicating `pipeline.py`'s exact train-split construction; the
real fix landed upstream as **PR #954** (merged to develop 2026-07-11) — this worktree
deliberately never rebased onto it, staying pinned at `6fc224c0` for training consistency across
the whole tournament, so **every entrant-(d) number in this report reflects the bridge-patched
metadata, not PR #954's fix directly.**

### Retry policy and outcomes

A uniform one-retry-per-fold policy for entrant (d) weight collapse was stated as policy *before*
any fold's outcome was known. F2's retry succeeded (`price/2026-07-11_11h50m31s_v1`, usable). F3's
retry collapsed again, identically to the first attempt (both literal constant output,
prediction_std=0.000000) — final, recorded as trained-but-degenerate, excluded from ranking per
the pre-committed policy, not re-retried further.

### Meta-label local-only fallback and platform constraints

- **Cloud training has no artifact-shipping channel for entrant (a)'s primary model** — filed as
  **GH #953**, open, `state:proposed`. `MLBasicSignalGenerator`/`PredictionModelRegistry.select_bundle()`
  reads the local model registry only; SageMaker has no mechanism to ship a specific primary-model
  bundle into the training container, so entrant (a) training fell back to `--provider local`
  entirely (chunked/checkpointed via `meta_label_chunk_fires.py` + `meta_label_finish_fast.py`,
  not the monolithic CLI path).
- **Hard ~60-minute background-task lifetime cap** confirmed on this platform: two independent
  long-running local meta-label training attempts were killed by the harness at the **exact same
  elapsed second** (3540s/59min). No error, no partial-state save unless the caller builds its own
  checkpointing. Worked around via the chunked/checkpointed driver once the true per-bar rate was
  measured (~200 bars/sec, not the ~7.5 bars/sec a naive one-time-startup-cost read had
  suggested). Flagged by the PM as "a platform constraint every agent must know," going on the
  weekly retro agenda.
- **Two algorithmic performance defects in `meta_labels.py`** — filed as **GH #955**, open,
  `state:proposed`: (1) `build_meta_label_features()`'s O(n² log n) "eligible_prior" reconstruction
  made feature-building effectively unbounded at this tournament's fire volume (46k–64k fires per
  fold, ~98%+ fire rate); fixed locally via a `bisect.insort`-based incrementally-maintained sorted
  list. (2) `EnhancedRegimeDetector.detect_regime()` re-annotates the entire 46k–64k-row DataFrame
  from scratch on *every* call (a local variable discarded after each call); annotating once
  upfront was the larger of the two fixes by far. Combined: **>500x speedup** (the same
  46,033-fire feature-build step went from >40min/never-finished to 38.4s). Both fixed
  bridge-locally (not upstreamed to develop in this worktree, by design); GH #955 tracks the
  upstream fix.

### Registry fragility (structural, not a bug requiring a fix)

`select_bundle()`'s reliance on the shared, mutable `price/latest` symlink (used by entrant (a)
and by the incumbent-control's deployed-semantics baseline) created two confirmed near-miss
incidents where the wrong fold's artifact would have been used as a primary signal, caught within
~40 seconds each via a mandatory re-sync-then-`readlink`-immediately-before-launch discipline
(adopted after the first near-miss, applied unconditionally thereafter). No wasted SageMaker
budget either time.

### Other incidents (all remediated, zero real data loss)

- The tournament worktree was deleted entirely by the repo's own eod-worktree-prune scheduled task
  (ran before `.agent-active` sentinel protection existed). Recreated off `origin/develop`, no
  local-only commits lost; the prune task was patched to hard-skip any worktree with
  `.agent-active` (PR #952).
- A cwd-relative model-registry path bug, combined with the worktree deletion, caused one
  misdirected artifact sync into the PM's own worktree and one false "0 trades, all-HOLD" exam
  reading — both caught via direct verification (readlink, `ps aux`, log tails) before being
  reported as real results; the false reading was re-run correctly.
- At least 4 separate points this session where a relayed status ("job finished," "this cutoff is
  X") was independently verified against a primary source before being acted on — at least 2 of
  those checks caught a materially wrong premise (a "finished" job still running/actually killed;
  the training-cutoff claim that led to Amendment 2).

### Budget actuals vs. the pre-approved estimate (FINAL)

Pre-approved budget: ~18–20 training jobs, ~$8–10. **Final actual: 19 SageMaker training jobs, all
complete** — (b) ×4 folds, (c) ×4 folds, (d) ×7 attempts (F1, F2×2, F3×2, F4×2 — one uniform-policy
retry each on F2/F3/F4, all pre-authorized), incumbent-control ×4 folds. Modestly at the upper end
of the pre-approved range, entirely explained by the entrant-(d) retry policy (3 retries, each
pre-committed as in-scope before its outcome was known) plus F4 — not scope creep. Entrant (a)
trained locally (`--provider local`, 3 folds), not SageMaker-billable. Total billable compute:
**51,089 seconds (~14.2 hours of `ml.g4dn.xlarge` on-demand time)**, a sum of every job's logged
duration, **not independently re-verified against AWS Cost Explorer** — reported as logged, not
converted to a dollar figure, since actual-cost verification was not run this session.

### Known gaps (disclosed in `results.json`, restated here in full)

1. **No naive-persistence money exam was built.** The §4 gate's "beats naive on both OOS return
   and profit factor" can only be inferred (every entrant's PF sits deeply below 1.0 across every
   fold) rather than directly verified against a naive money-exam number.
2. **True McNemar's paired significance test was not computed** — would need per-bar correctness
   vectors per entrant per fold, not captured this session. The two-proportion z-test
   approximation used throughout is disclosed as conservative/imprecise, the same limitation the
   2026-07-06 architecture tournament had.
3. **Entrant (a)'s literal accuracy-vs-coverage curve** was not computed (only inferred from its
   confirmed base-rate collapse) — a residual, disclosed gap, not fabricated.

### Additional gaps found by this writer while assembling the report — CLOSED during the PM's fact-check pass

4. **CLOSED. Entrant (a)'s OOS L1 result** was generated fresh during fact-check
   (`entrant_a_oos_score.py`): fires regenerated over each fold's own EVAL window (not training
   window) using that fold's fold-matched primary signal, resolved under the same exam-harness exit
   geometry, and the already-trained meta-classifier scored against those genuinely-OOS
   resolutions. Result: exact tie with dummy on all 3 folds (§2, §3). §7 row 5 still cannot be
   cleanly applied in its literal wording (that row is specifically about the *primary signal*
   having no edge, a different question from the meta-classifier's own collapse) — restated
   precisely in §1's row 5.
5. **CLOSED. Accuracy-vs-coverage curves** computed for (b) and (c) across all 3 folds
   (`coverage_curve.py`, real numbers, see §3) — every curve is non-monotonic, including entrant
   (c)/F2's one genuinely non-degenerate fold, itself a separate, disclosed finding (real
   predictive variation is not well-calibrated confidence). Entrant (a)'s literal curve was not
   computed (time-boxed after closing gap 4) — disclosed as inferred, not measured, given time
   constraints.

---

## 5. Recommendations

Per §7's own text, and per §4's pre-committed "quality win, not yet exam-actionable" language, the
prescribed next action is **not** a staging trial and **not** a hyperparameter-tuning retry on this
same fold set (which the decision table explicitly warns against as noise-chasing). What the table
does prescribe, read plainly:

- **Close this round.** No entrant is promoted to L3a staging. This is a clean, reportable
  negative-to-mixed result, not a silent gap — per the anti-p-hacking rule, it gets this full
  write-up rather than a one-line "didn't work."
- **Name the implicated next lever, as §7's row 2 language directs even though this isn't a literal
  cluster**: entrants (b) and (d) — the two `SignalGenerator`-native reformulations that changed
  loss function/output type without changing the feature set — did not move the needle at all
  (both collapse to variants of "predict the unconditional distribution"). Entrant (c) is the one
  entrant showing a real, non-degenerate signal, but only in one of three folds (F2), and its
  aggregate "win" is mostly a label-imbalance artifact in the other two. This pattern — reshaping
  the target moves the needle only when the label itself encodes new information (triple-barrier's
  exit-mechanics-aware label, sometimes), not when it's a bare relabeling of the same 1h
  price-only signal — is the most concrete, mechanism-level evidence yet for **where the ceiling
  actually lives**.
- **The input-ceiling implication, stated as directly as the evidence supports**: three
  independent tournaments (window choice #898, architecture choice #939, and now target design,
  this report) have each tested a different lever on the *same* price-only 1h feature set and each
  found that lever is not the binding constraint. The linear baseline matching the incumbent's
  accuracy almost bar-for-bar (§2) is corroborating, not just suggestive, evidence that the
  feature set itself — not the model, not the window, not the target shape — is the ceiling. The
  next research lever this implies is **new information sources** (order-book/microstructure
  features, sentiment, cross-asset signals, or a genuinely different data modality) rather than
  further reshaping or re-architecting what can be extracted from price-only OHLCV at 1h
  resolution.
- **Frame this as input to the next preregistration, not as a decision already made.** This report
  recommends; it does not commit resources. Any next step (a features-expansion tournament, a
  formal acceptance of the ~53% ceiling as structural pending new data, or a differently-scoped
  meta-labeling round once entrant (a)'s L1 gap is resolved) should go through
  `experiment-preregister` in the normal way, with its own hypothesis, thresholds, and decision
  table, informed by but not pre-decided by this write-up.

**What risk-officer should stress-test, if any of this were ever proposed for live capital**
(it is not, per the verdict above, but naming it for completeness): none of these entrants are
being proposed for staging, so there is nothing to stress-test yet. If a future round produces an
entrant that clears both halves of §7 row 1, the standing asks from the architecture tournament
still apply — regime-shift behavior across bull/bear/chop, and a paired (not independent-samples)
significance re-check before treating any pairwise "win" as confident.

---

## Fact-check pass (ml-engineer, PM's dispatched fact-checker)

Verdict: **the draft's numbers and framing check out.** Every number I spot-checked against
`results.json`/`methods.md` traces 1:1 (L1/L2 tables, the corrected aggregate stats, the
determinism guard, the amendment/incident/budget narrative). GH #953, #954, #955 independently
re-verified against live GitHub state (not just the draft's claims) — all exist, all correctly
described, PR #956's merge commit (`25e0a202`) confirmed.

Resolutions to the writer's own TODOs:

1. **Entrant (a)'s missing OOS L1 result — CLOSED.** Genuinely never scored this session before
   the writer flagged it; not something that existed elsewhere and was omitted. Freshly generated
   during this fact-check pass (§2, §3, §1 row 5) — exact tie with dummy, all 3 folds, closing the
   loop on a real gap the writer was right to surface rather than paper over.
2. **Accuracy-vs-coverage curves — CLOSED.** Confirmed never built before this pass; now computed
   with real numbers for (b)/(c) (§3). Non-monotonic on every single curve, including (c)/F2 — an
   independently interesting finding in its own right, folded into §2.
3. **F4 numbers — CLOSED, inserted in full (§3).** All four F4 slots trained and scored, including
   one collapsed-then-retried-then-collapsed-again entrant-(d) attempt (final, per the uniform
   one-retry policy — no usable (d)/F4 model). F4 does not change the headline verdict (never
   going to, per its own non-deciding design) but substantially strengthens §2's mechanism finding:
   both (b) and (c) are still frozen predicting their F1–F3 training-period majority class, and
   F4's regime flip exposes this as a real, non-adaptive failure rather than a coincidental fit.
4. **§1 verdict-table reading — CONFIRMED by the PM directly** (relayed here for the record): (c)
   clears the L1 letter per-fold but fails the money gate universally, so §4's pre-committed
   "quality win, not yet exam-actionable" language applies, combined with a round-closing
   money-gate failure for all entrants — nothing proceeds to L3a. The "doesn't nest into a single
   §7 row" honesty in §1 stays as written; this is the PM's own confirmed reading, not a writer
   guess.
5. GH issue confirms (#953/#954/#955) — independently re-verified, see above, no corrections
   needed.

**Status**: complete. F4 numbers are inserted (§3), all training is finished (19 SageMaker jobs +
3 local entrant-(a) folds), the fact-check pass is done, and this document is ready to open as a
PR against `develop`.
