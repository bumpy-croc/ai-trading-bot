# Confidence-Calibration Study — ETHUSDT

Date: 2026-07-05
Author: quant-researcher
Status: COMPLETE — hypothesis REJECTED (H0 supported); redirect to target redesign
Issue: https://github.com/bumpy-croc/ai-trading-bot/issues/912 — follows from #898 (window tournament)
North star: `docs/architecture/model_evaluation_system.md` — item 6 of "To build",
open question #3.

## Hypothesis

**H1**: The ETHUSDT basic model carries real directional edge (~53% accuracy per
#898/#887, holding across all training windows tried) but the raw-output→confidence
mapping compresses that edge into noise-level scores (median ~0.03, barely above
HyperGrowth's `min_confidence=0.05` gate), so the edge exists but the current
gate/sizing cannot discriminate high-conviction from low-conviction bars. A
magnitude-aware recalibration (rank/z-score against the model's own realized
prediction distribution) will let the gate pass more true-edge bars and/or size them
better, improving OOS return/PF without materially raising drawdown.

**H0 (falsifier)**: Predicted-delta magnitude carries no information about
directional accuracy (conditional hit-rate is flat across magnitude quantiles). If
so, the confidence channel is information-free by construction — no recalibration of
the *mapping* can help, and the fix belongs upstream in the training target
(vol-normalized returns / direction classification), not in the strategy layer.

## Metric

- Phase 2: conditional directional accuracy by |predicted delta| quantile (the key
  table — this is what decides H1 vs H0).
- Phase 3: OOS return, profit factor, MaxDD, win rate, trade count on the frozen exam
  window (2026-01-01 → 2026-07-04, hyper_growth, prod-matched flags), vs the #898
  W_full baseline (return −7.43%, PF 0.673, MaxDD 10.55%, 52 trades).

## Success threshold

A calibration variant is "ready for risk review" only if, on a single one-shot exam
run (parameters chosen from the training-period distribution, not tuned on the exam):
OOS return improves by ≥3pp vs W_full baseline AND MaxDD does not worsen by more than
2pp AND trade count stays large enough to be statistically meaningful (>30). Anything
weaker is "promising but not ready." If Phase 2 shows H0, no variant is tested against
the promotion bar — the recommendation redirects to target redesign.

## Risks of false positive

- **Multiple-comparison / p-hacking risk**: testing 2-3 calibration variants on the
  same frozen exam window inflates luck risk (this is exactly what
  `model_evaluation_system.md` warns about — "refresh the exam window when candidate
  count exceeds ~10"). Mitigated by picking each variant's parameters from the
  *training*-period prediction distribution and running each variant on the exam
  exactly once (no iteration on the exam itself).
- **Small trade count**: HyperGrowth's `ignore_signal_reversal` + partial-exit
  mechanics mean the 185-day exam window produces ~40-55 trades. A single-variant
  return delta of a few points on ~50 trades is not strong evidence; report it as
  such, not as a settled result.
- **Regime specificity**: the exam window is a single continuous bear market. A
  calibration that "helps" here may just be re-fitting to this bear's magnitude
  distribution. Flag for risk-officer regime-shift stress test explicitly.
- **Confusing the gate with the sizer**: see Phase 1 finding below — for HyperGrowth
  specifically, confidence has NO effect on position size once the signal clears the
  gate (`FixedFractionSizer(adjust_for_confidence=False)`). A calibration that only
  reshapes the *gate* threshold cannot "size up" high-conviction trades under the
  current strategy config — that would require a second, separate change (turning on
  confidence-weighted sizing), which is out of scope here and would need its own
  sensitivity/robustness pass if proposed later.

---

## Phase 1 — Code trace: raw prediction → confidence → gating → sizing

### 1. Raw model output → predicted return

`src/strategies/components/ml_signal_generator.py`, `MLBasicSignalGenerator._get_ml_prediction`
(lines 802-856) returns the model's raw predicted **price** for bar `t+1` (ONNX
regression output, `result.price`). `generate_signal` (line 731) converts this to a
return:

```python
predicted_return = (prediction - current_price) / current_price
```

This is a raw hourly return prediction, not a probability, not vol-normalized, and not
compared against any distribution — a single scalar in the range of ETHUSDT's typical
hourly move (empirically ~0.1-0.5% per bar, see Phase 2).

### 2. Predicted return → confidence

`MLBasicSignalGenerator._calculate_confidence` (`ml_signal_generator.py:858-869`):

```python
CONFIDENCE_MULTIPLIER = 12.0  # class default, line 539

def _calculate_confidence(self, predicted_return: float) -> float:
    confidence = min(1.0, abs(predicted_return) * self.confidence_multiplier)
    return max(0.0, confidence)
```

So `confidence = clip(|predicted_return| * 12.0, 0, 1)`. This is a **fixed linear
scale with no reference to the model's own output distribution or to realized
volatility**. For `confidence` to reach the gate value 0.05, `|predicted_return|`
must exceed `0.05 / 12 = 0.4167%` — a 0.42% hourly move prediction. For confidence to
reach 1.0 (saturate), the model would need to predict a ~8.3% hourly move, which never
happens for ETHUSDT at 1h resolution. The practical range of `confidence` is therefore
compressed into roughly [0, 0.15] for the vast majority of bars — consistent with the
observed median ~0.03 (implies typical `|predicted_return|` ≈ 0.25%).

`strength` uses a different, more permissive multiplier (`abs(predicted_return) * 10`,
`ml_signal_generator.py:736/739`) but is **not used by HyperGrowth's sizer** at all
(see below) — it only matters for strategies with `adjust_for_strength=True`.

This is the compression the task description hypothesized: **the confidence formula's
implicit normalization constant (12.0, chosen presumably against BTCUSDT's historical
move distribution or by intuition, not calibrated) is far too aggressive relative to
the model's actual predicted-delta magnitudes**, so nearly every bar's confidence sits
near the floor of the [0,1] range regardless of how much real directional information
the magnitude carries.

### 3. Gating — `min_confidence=0.05`

`src/strategies/hyper_growth.py`: `create_hyper_growth_strategy(min_confidence=0.05)`
(line 173) wires this straight into `FlatRiskManager(min_confidence=min_confidence)`
(line 232). The actual gate:

`FlatRiskManager.calculate_position_size` (`hyper_growth.py:94-115`):

```python
if signal.confidence < self.min_confidence:
    return 0.0
# Flat risk — no further confidence/strength scaling
return balance * self.risk_fraction
```

Given step 2, `confidence >= 0.05` requires `|predicted_return| >= 0.4167%`. Anything
below that is a hard `HOLD`-equivalent veto (zero position), no matter how strong the
directional edge is at smaller magnitudes. If the true edge lives mostly in the
0.1-0.4% predicted-move band (very plausible for an hourly ETH model — see Phase 2),
**the gate is discarding most of the tradeable signal by construction**, independent
of whether the model is any good.

### 4. Gate passed → sizing — confidence has ZERO further effect for HyperGrowth

This is the key structural finding that scopes Phase 3. HyperGrowth's position sizer
chain (`hyper_growth.py:235-254`):

```python
base_sizer = FixedFractionSizer(
    fraction=base_fraction,
    adjust_for_confidence=False,   # <-- confidence ignored past the gate
    adjust_for_strength=False,     # <-- strength ignored too
)
position_sizer = LeveragedPositionSizer(
    base_sizer=base_sizer,
    leverage_manager=leverage_manager,   # regime-based multiplier, NOT confidence-based
    max_leveraged_fraction=0.50,
)
```

`FixedFractionSizer.calculate_size` (`position_sizer.py:171-217`) only applies a
confidence multiplier `if self.adjust_for_confidence` (False here) — so with
`adjust_for_confidence=False`, `multiplier` stays 1.0 from that term entirely. Sizing
is a flat `balance * risk_fraction` (0.20 by default), further scaled only by
regime-based leverage (disabled at `max_leverage=1.0` per the module's own
docstring/comment, lines 11-15) — confidence and strength are **fully inert** for
sizing under the current live/prod config. This matches the module docstring's stated
design intent (lines 17-22): *"Standard risk managers multiply by confidence, crushing
positions to $10 ... This strategy uses a FlatRiskManager ... the ML direction filter
IS the edge, not the per-bar confidence score."* — i.e. the strategy was deliberately
built to route around the compressed-confidence problem for sizing, but the **binary
gate** (step 3) is still fully exposed to the same compression, and that gate is a
much blunter instrument (all-or-nothing veto) than a continuous size multiplier would
be.

**Consequence for scope**: a calibration fix for HyperGrowth can only ever act through
the **gate** (which bars pass vs. get vetoed), not through position sizing (which is
architecturally decoupled from confidence here). This is fine — the gate is where the
edge is currently being thrown away — but it means Phase 3 candidate (c) (recalibrated
gate) is the one most likely to move the needle for HyperGrowth specifically; (a) and
(b) (rank/z-score confidence *values*) only matter if a future strategy change also
flips `adjust_for_confidence=True`, which is explicitly out of scope for this study.

### Where this leaves the diagnosis before Phase 2

The formula is a **fixed linear scale of predicted return with an apparently
uncalibrated constant (12.0)**, feeding a **binary threshold gate**, with **sizing
fully decoupled from confidence** in the strategy actually deployed. The open
empirical question (Phase 2) is whether |predicted_return| magnitude — independent of
the arbitrary ×12 scaling — actually predicts hit rate. If yes, recalibrating the gate
threshold to the model's own realized magnitude distribution (rather than an
arbitrary 0.4167% cutoff implied by ×12) should recover trades that carry edge but
currently get vetoed. If no, the entire confidence channel is decorative and the
fix has to happen at the model/target level.

---

## Phase 2 — Empirical distributions

Retrained the #898 tournament's winning config (W_full, full history) for a clean
ETHUSDT instrument: `atb train price ETHUSDT --start-date 2017-08-17 --end-date
2025-12-31 --timeframe 1h --epochs 50 --batch-size 256 --sequence-length 120`, in
the isolated `.claude/worktrees/calibration-study` worktree (fresh from
`origin/develop`, detached; never touched the main checkout, staging, or prod;
worktree-local `src/ml/models/` filesystem is entirely separate from the real
registry). Trained 41 epochs (early-stopped), test RMSE 0.06566, train/test loss
gap tight — no gross overfit, consistent with the #898 W_full result (0.06586,
different random init/stopping epoch — see Phase 3 caveat on baseline variance).
Zero training/eval-window overlap: `training_params.end_date = 2025-12-31`, exam
window starts 2026-01-01, both verified directly against `metadata.json`.

Scripted the walk with `experiments/confidence_calibration_diagnostic.py`, which
reuses `MLBasicSignalGenerator.generate_signal()` directly (the exact
`PredictionEngine`/ONNX/rolling-minmax-denormalization pipeline the live strategy
uses — no hand-rolled inference, per backtest-live parity). It records
`predicted_return` (from `signal.metadata`), the realized next-bar return, and the
strategy's own `confidence` value, per bar.

Ran it over two windows:
- **Exam window**: 2026-01-01 → 2026-07-04 (the frozen OOS window itself; n=4,415
  bars scored).
- **Training-period slice**: 2025-07-01 → 2025-12-31 (the last 6 months of the
  training data — in-sample-adjacent, used only to pick Phase 3 thresholds
  *before* touching the exam, per the no-p-hacking rule; n=4,391 bars scored).

### Distributions (exam window)

| | p01 | p05 | p25 | p50 | p75 | p95 | p99 | mean |
|---|---|---|---|---|---|---|---|---|
| \|predicted_delta\| | 0.000044 | 0.000251 | 0.001188 | 0.002639 | 0.005389 | 0.014314 | 0.026603 | 0.005211 |
| realized \|hourly move\| | 0.000055 | 0.000241 | 0.001153 | 0.002628 | 0.005274 | 0.013635 | 0.024992 | 0.004210 |
| confidence (=\|pred\|×12, clipped) | 0.0005 | — | 0.0143 | 0.0317 | 0.0647 | 0.1718 | 0.3192 | — |

Median confidence 0.0317 (fraction ≥ 0.05 gate: 34.18%) — confirms the compression
described in Phase 1 with the freshly trained model, not just the previously
deployed `2026-07-04_22h_v1`. The predicted-delta distribution tracks the realized
move distribution reasonably closely in scale (both medians ≈0.0026), so the
model isn't wildly miscalibrated in magnitude — the compression is specifically
that `confidence_multiplier=12.0` maps this ≈0.26%-median signal to a
≈3%-median confidence, an order of magnitude below where the 0.05 gate sits.

### THE KEY TABLE — directional hit rate by |predicted_delta| decile, with statistics

Formal statistics (binomial 95% Wilson confidence intervals per decile; a
Cochran-Armitage trend test on the underlying binomial counts; cross-checked with
a logistic-regression slope test and a Spearman rank correlation on the
decile-level hit rates — all four give consistent verdicts):

**Exam window (2026-01-01 → 2026-07-04), n=4,415, overall hit rate 51.85%:**

| Decile | n | hit rate | 95% CI (Wilson) | mean \|pred\| |
|---|---|---|---|---|
| 0 | 442 | 49.77% | [45.13%, 54.42%] | 0.000240 |
| 1 | 441 | 49.89% | [45.24%, 54.53%] | 0.000704 |
| 2 | 442 | 52.49% | [47.83%, 57.10%] | 0.001189 |
| 3 | 441 | 53.29% | [48.62%, 57.90%] | 0.001727 |
| 4 | 442 | 54.98% | [50.32%, 59.55%] | 0.002327 |
| 5 | 441 | 50.79% | [46.14%, 55.43%] | 0.003054 |
| 6 | 441 | 52.83% | [48.17%, 57.45%] | 0.004065 |
| 7 | 442 | 49.77% | [45.13%, 54.42%] | 0.005452 |
| 8 | 441 | 53.51% | [48.85%, 58.12%] | 0.007811 |
| 9 | 442 | 51.13% | [46.48%, 55.76%] | 0.025522 |

Every decile's 95% CI overlaps every other decile's — no decile is distinguishable
from any other at conventional significance, and there is no monotonic pattern
(decile 4 and decile 7, adjacent in magnitude rank, differ by 5pp with no
trend continuity). Formal trend tests confirm this is noise:

- **Cochran-Armitage trend test**: Z = +0.427, p = 0.669
- **Logistic regression slope** (hit ~ decile index): slope = +0.0045 (SE 0.0105),
  Z = +0.427, p = 0.669, odds ratio 1.0045/decile step (i.e., ~0.45% higher odds
  of a hit per decile — indistinguishable from zero)
- **Spearman rank correlation** (decile vs. hit rate): ρ = 0.255, p = 0.477

**Verdict on the exam window: no evidence that predicted-delta magnitude carries
directional information.** The decile-9 mean confidence (0.207, well above the
0.05 gate) has a hit rate (51.13%) statistically indistinguishable from decile-0
(mean confidence 0.003, 49.77%). The confidence channel, as currently gated, does
not separate high-conviction from low-conviction bars in a way that predicts
outcomes.

**Training-period slice (2025-07-01 → 2025-12-31), n=4,391, overall hit rate 51.58%:**

| Decile | n | hit rate | 95% CI (Wilson) | mean \|pred\| |
|---|---|---|---|---|
| 0 | 440 | 46.36% | [41.76%, 51.03%] | 0.000270 |
| 1 | 439 | 50.80% | [46.13%, 55.45%] | 0.000831 |
| 2 | 439 | 50.80% | [46.13%, 55.45%] | 0.001393 |
| 3 | 439 | 51.25% | [46.59%, 55.90%] | 0.002032 |
| 4 | 439 | 51.25% | [46.59%, 55.90%] | 0.002754 |
| 5 | 439 | 51.03% | [46.36%, 55.67%] | 0.003628 |
| 6 | 439 | 53.99% | [49.31%, 58.59%] | 0.004701 |
| 7 | 439 | 54.90% | [50.22%, 59.49%] | 0.006217 |
| 8 | 439 | 51.48% | [46.81%, 56.12%] | 0.008787 |
| 9 | 439 | 53.99% | [49.31%, 58.59%] | 0.028413 |

- **Cochran-Armitage trend test**: Z = +2.353, **p = 0.019** (significant at 0.05)
- **Logistic regression slope**: slope = +0.0248 (SE 0.0105), Z = +2.352,
  **p = 0.019**, odds ratio 1.025/decile step
- **Spearman rank correlation**: ρ = 0.881, **p = 0.0008** (significant, strong)

**Verdict on the training-period slice: there IS a statistically real, monotonic-ish
positive trend** — hit rate climbs from ~46-51% in the bottom deciles to ~54-55%
in the top deciles. Three independent tests agree (CA, logistic, Spearman), so this
is not a single-test artifact.

**The critical comparison — this is the central finding of Phase 2:** the gradient
that is statistically significant in the training-period slice (p=0.019, Spearman
p=0.0008) **completely vanishes** in the frozen, never-touched exam window
(p=0.669, Spearman p=0.477). This is the textbook signature of **overfitting of the
confidence channel to the training distribution** — the magnitude-vs-accuracy
relationship the model exhibits on data adjacent to what it was trained on does not
generalize to genuinely unseen future data. It is not merely "the effect is
smaller OOS" — the odds ratio drops from 1.025/decile (train-period) to
1.0045/decile (exam), a ~5x shrinkage, consistent with a real in-sample pattern
being mostly or entirely noise once evaluated honestly.

This directly answers the H1/H0 framing: **H0 is supported on the exam window**
(the confidence channel, evaluated where it matters — genuinely unseen data — is
information-free), while the apparent H1 support in the training-period slice is
best explained as overfitting rather than a real, exploitable signal. Per the
success-threshold section above, this result should be reported plainly rather
than papered over.

---

## Phase 3 — Candidate calibrations on the frozen exam

Per the coordinator-confirmed decision branch: because the training-period slice
*did* show a statistically real (if likely overfit) gradient, Phase 3 ran in full
(all three candidates), not just the single-control fallback — with the explicit
expectation, stated before running, that any improvement is more likely to reflect
noise than a transferable pattern, given the Phase 2 exam-window null result.

### Thresholds (chosen from the training-period slice, before touching the exam)

All three variants act ONLY on the gate (per the Phase 1 sizing-decoupling
finding) — `FixedFractionSizer`'s `adjust_for_confidence=False` means nothing
downstream of the gate changes size, so every variant here isolates "which bars
trade" as the only free variable:

- **rank_gate**: percentile-rank of `|predicted_return|` within a trailing
  500-bar window; gate passes at the 60th percentile — chosen as the
  decile-6 boundary in the training-period table (where hit rate first climbs
  decisively above ~51%).
- **vol_zscore_gate**: `z = predicted_return / trailing_realized_vol` (48-bar
  realized vol of bar-over-bar returns); gate at `|z| ≥ 0.5` — the
  training-period median `|z|`.
- **recalibrated_fixed_gate**: same linear confidence formula, gate threshold
  moved to `|predicted_return| ≥ 0.0041` — the same decile-6 boundary in raw
  magnitude terms (vs. the baseline's implied 0.004167 cutoff from
  `0.05 / confidence_multiplier(12.0)` — nearly the same value; see caveat below).

Implemented as `FlatRiskManager` subclasses in
`experiments/confidence_calibration_variants.py`
(`RankGateRiskManager`, `VolZScoreGateRiskManager`,
`RecalibratedFixedGateRiskManager`), constructed via `create_hyper_growth_strategy`
with every other parameter (risk_fraction=0.20, base_fraction=0.20,
stop_loss_pct=0.10, take_profit_pct, leverage config, partial-exit/trailing-stop
config) byte-identical to the prod-matched defaults — the gate logic is the only
variable. Ran through the real `Backtester` (same engine as `atb backtest`),
default `CostCalculator` fee/slippage settings (fee_rate 0.001, slippage_rate
0.0005 — not fee-free), `RiskParameters(base_risk_per_trade=0.02,
max_risk_per_trade=0.03, max_position_size=0.20)`, `initial_balance=85` — matching
the #898 tournament protocol exactly. Each variant ran on the frozen exam
(2026-01-01 → 2026-07-04) exactly once; no iteration on the exam.

**Bugs found and fixed during script development, before trusting any numbers**:
(1) the model-registry key format needs colons (`SYMBOL:timeframe:type:version`),
not slashes — didn't actually matter here since only one ETHUSDT/basic version
existed at smoke-test time, but would silently break pinning once a second version
exists; (2) `create_hyper_growth_strategy()` requires an explicit `symbol=`
argument when called directly (bypassing `call_strategy_factory`) — omitting it
defaults to `MLBasicSignalGenerator`'s `DEFAULT_SYMBOL="BTCUSDT"`, which would have
silently reproduced the exact cross-symbol bug #867 fixed (scoring ETH candles
with the BTCUSDT model). Caught via a smoke-test numbers change after the fix
(baseline return moved from -1.47%/PF 0.558 to -2.30%/PF 0.404 on the same tiny
window pre- vs. post-fix) — confirming the bug was real and silent, not cosmetic.

### Results

| Variant | Trades | Win rate | Return | MaxDD | PF | Final $ |
|---|---|---|---|---|---|---|
| **baseline** (min_confidence=0.05 gate) | 46 | 65.2% | **-11.36%** | 14.92% | 0.393 | $74.80 |
| rank_gate (p60 trailing rank) | 47 | 70.2% | -12.98% | 14.19% | 0.376 | $74.41 |
| vol_zscore_gate (\|z\|≥0.5) | 49 | 69.4% | **-9.91%** | **11.53%** | **0.482** | $76.87 |
| recalibrated_fixed_gate (0.0041) | 46 | 65.2% | -11.36% | 14.92% | 0.393 | $74.80 |

Raw results: `experiments/confidence_calibration_variants.py --out-json` output,
cross-checked directly against this table (not taken from any summary).

**Important caveat, root-caused rather than merely noted**: this run's baseline
(-11.36% return, PF 0.393, 46 trades, MaxDD 14.92%) does not match the #898
tournament's W_full baseline (-7.43%, PF 0.673, MaxDD 10.55%, 52 trades), and —
more concerning — **it also does not reproduce against itself**. Re-running the
identical baseline variant (same trained model file, same code, same exam window,
no code changes) produced a third result: 55 trades, -10.33% return, PF 0.512,
MaxDD 12.81%. Same model, same window, three different answers.

Root cause, traced rather than assumed: `PredictionEngine._get_timeout_seconds()`
(`src/prediction/engine.py:880-882`) returns `self.config.max_prediction_latency`,
which defaults to **0.1 seconds**
(`DEFAULT_MAX_PREDICTION_LATENCY = 0.1`, `src/config/constants.py:10` — named and
documented as a latency-alerting budget, not an inference-abort deadline) and this
value is what actually gates `run_with_timeout(model.predict, ...)` at
`engine.py:189-196`. Both the original Phase 3 sweep and the rerun logged
`ERROR src.infrastructure.timeout: ML model inference exceeded timeout of 0.1s`
a handful of times (3 of 17,188 decisions in the original 4-variant sweep, 1 in
the single-variant rerun — 0.006-0.02% of bars). When this fires,
`PredictionEngine.predict()` raises `ModelInferenceError`, which
`MLBasicSignalGenerator._get_ml_prediction()` catches with a bare
`except Exception` and converts to `None` → that bar's `generate_signal()` falls
back to `HOLD, confidence=0.0` (the `"prediction_failed"` metadata path).

Because `run_with_timeout` is thread-based wall-clock timing (not CPU-time), this
is **directly sensitive to system load at the moment of inference** — this
machine had several other concurrent worktree sessions running during this study
(confirmed via `git worktree list` / `ps`), and the same ONNX inference call took
different wall-clock time on different attempts purely from OS scheduling
contention, not from any change to inputs or code. A single missed bar can matter
disproportionately given `ignore_signal_reversal=True` (HyperGrowth holds through
signal flips) — silently forcing one entry bar to HOLD can mean missing an entire
multi-day position, which plausibly explains why a ~0.01% per-bar event rate
produced a ~20% swing in trade count between runs.

**This is a real, pre-existing backtest-live-parity and backtest-reproducibility
defect in the shared prediction pipeline** (`src/prediction/engine.py`), not
something introduced by this study's scripts, and not limited to this experiment
— any backtest run on a loaded machine is subject to it, and worse, a backtest run
under different load than the eventual live/staging deployment is not a clean
comparison. It fully explains this experiment's inability to reproduce #898's
baseline (that comparison was always confounded by this, not just training-run
variance — training-run variance is a real, separate, ADDITIONAL factor, since the
model weights genuinely differ from #898's W_full run, but the load-dependent
timeout is the dominant, more alarming mechanism because it means even a single
fixed model's backtest result is not deterministic).

**Consequence for how to read the Phase 3 table above**: the numbers are still
the actual output of real runs and are reported honestly, but given confirmed
run-to-run non-determinism, none of the small deltas between variants (a few
percentage points of return, a few points of PF) should be read as precise —
they sit within a noise band whose width is now known to be nontrivial (order:
the ~1-2pp differences separating variants from baseline are comparable in
magnitude to the ~1pp-of-return swing observed from re-running the SAME variant).
This makes the Phase 3 verdict (no variant clears the pre-registered bar) more
robust, not less — the one variant that looked directionally favorable
(vol_zscore_gate) was already sub-threshold before this was discovered, and this
finding is an additional reason not to trust it further, not a reason to
suspect a real result was masked.

This is filed as a follow-up (see below) rather than fixed in this session,
since fixing shared inference-timeout infrastructure is out of scope for a
research experiment and deserves its own dedicated review (the fix likely
involves raising `DEFAULT_MAX_PREDICTION_LATENCY` to something realistic for
actual inference cost, or decoupling "latency alerting" from "abort inference,"
plus making backtests deterministic regardless of system load — e.g., a
much larger timeout for backtest/research contexts specifically, since live
trading may have a legitimate reason to bound latency tightly but a backtest
should never sacrifice correctness for a wall-clock budget tuned for production
alerting).

**recalibrated_fixed_gate ≈ baseline is not a bug, it's a coincidence of
thresholds**: 0.0041 (chosen from the training-period decile boundary) and
0.0041667 (baseline's implied cutoff from `0.05/12.0`) are almost the same value,
so the two variants gate nearly the same set of bars and produce nearly identical
results (46 vs 46 trades, -11.356% vs -11.356%, agreeing to 3 decimal places) —
this is expected given how close the two thresholds are, and is a useful sanity
check that the harness is behaving deterministically given near-identical inputs,
not evidence of a broken variant.

**vol_zscore_gate is the only variant that moved metrics in a favorable direction**
(return -9.91% vs -11.36%, PF 0.482 vs 0.393, MaxDD 11.53% vs 14.92%, on 49 vs 46
trades — comparable trade count, not a small-sample artifact from filtering down
to a handful of trades). This is the one candidate that normalizes against
*realized* market volatility rather than the model's own raw output distribution,
which may make it more robust to the exact overfitting failure mode Phase 2
diagnosed (a gate keyed to the model's own in-sample magnitude distribution, like
the fixed and rank variants, inherits that distribution's overfit shape; a gate
keyed to independently observable market vol does not).

**However, none of this clears the pre-registered success threshold** (≥3pp
return improvement AND MaxDD not worse by >2pp AND >30 trades, from a single
one-shot exam run). vol_zscore_gate's return improved by 1.45pp (needed ≥3pp) and
its MaxDD improved rather than worsened, so it clears the MaxDD/trade-count bars
but misses the return bar by more than half. Given Phase 2's exam-window null
result on the same underlying mechanism (predicted-delta magnitude carries no
OOS directional information), a plausible read is that vol_zscore_gate's
improvement here is itself largely noise on a 46-49-trade sample — consistent
with, not contradicting, the Phase 2 verdict. It is not strong enough evidence to
override the Phase 2 finding.

### How this could lose money (adversarial self-review, applies to any of these
variants if considered for further work)

1. **The one variant that "worked" is also the one most likely to be noise.**
   vol_zscore_gate's improvement is within the range plausible from ~50-trade
   sampling variance; Phase 2 already showed the underlying mechanism
   (magnitude-conditioned accuracy) has zero OOS signal. Promoting this variant
   on this evidence would be fitting the exam window, exactly the
   multiple-comparison risk flagged in the Risks section.
2. **Single bear-market window.** All four variants were tested on one 185-day
   continuous bear regime. A gate that happens to filter well in a bear market
   (e.g., by coincidentally suppressing whipsaw noise during high-vol chop) may
   behave completely differently in a trending bull, where the realized-vol
   denominator in vol_zscore_gate shrinks and the gate becomes systematically
   easier to pass.
3. **Training-run non-determinism undermines any single-model conclusion.** Given
   the baseline-vs-#898 discrepancy documented above, a different retrain of the
   "same" W_full config could shift every one of these numbers again, in either
   direction, before any strategy-layer change is even considered.
4. **The gate change touches nothing about the underlying signal quality.**
   Every variant here can only decide which existing (noisy) predicted-direction
   bets to take, not improve the predictions themselves. Best case, a working
   gate filters some noise trades; it cannot manufacture edge that Phase 2 shows
   isn't there.

### What risk-officer should stress-test, IF any variant were escalated (it is not)

Not applicable — no variant is being recommended for promotion (see verdict
below). If a future iteration on target redesign (see recommendation) produces a
model that clears Phase 2's magnitude-vs-accuracy bar, the eventual gate/sizing
change built on top of it should be stress-tested for: regime-shift behavior
(bull vs. bear vs. chop, not just the single bear window tested here), sample-size
robustness (require a larger multi-window/multi-symbol trade count before trusting
a MaxDD/PF claim), and training-run reproducibility (multiple retrains of the same
config, not a single run, before treating any one model's behavior as
representative).

---

## Verdict and recommendation

**H0 is supported: the confidence channel, as currently constructed
(`confidence = clip(|predicted_return| × 12.0, 0, 1)`), is information-free on
genuinely out-of-sample data.** The apparent magnitude-vs-accuracy gradient
visible in training-period-adjacent data (Cochran-Armitage p=0.019, Spearman
p=0.0008) does not survive contact with the frozen exam window (p=0.669, p=0.477)
— a textbook overfitting signature, not a real, exploitable pattern. Three of four
Phase 3 gate-recalibration variants (rank-based, and the fixed-threshold
recalibration which nearly degenerates to the existing gate) produced no
improvement or a worse outcome than baseline; the fourth (vol-normalized z-score
gate) showed a directionally favorable but sub-threshold result (+1.45pp return
vs. the required ≥3pp) on a sample too small to distinguish from noise, especially
given Phase 2's null finding on the same underlying mechanism.

**This is not a "the calibration formula needs tuning" result — no recalibration
of the mapping from raw prediction to confidence score can manufacture
information that isn't in the underlying `predicted_return` signal's magnitude in
the first place.** Per Phase 1, HyperGrowth's sizer is architecturally decoupled
from confidence anyway (`adjust_for_confidence=False`), so even a perfect
magnitude-based gate would only ever change *which* bars trade, never *how much*
— and Phase 2 shows magnitude doesn't reliably tell you which bars are worth
taking, OOS.

**Recommendation: redirect to target redesign, not further calibration-layer
work.** Per the north-star doc's own open question #2 ("Is next-bar price the
right target at all, vs direction-classification or vol-normalized returns"), the
next tournament should test:

1. **Direction classification** as the training target (binary/ternary up-down-flat
   classifier with a calibrated probability output) instead of next-bar price
   regression — a probability output is a much more natural confidence signal
   than a magnitude-based proxy, and directly optimizes for the thing HyperGrowth
   actually gates on (direction correctness), rather than optimizing RMSE on a
   price target and hoping the residual magnitude happens to correlate with
   accuracy (Phase 2 shows, cleanly, that it does not OOS).
2. **Vol-normalized returns** as an alternative regression target (ReVol-style
   normalization, referenced in the north-star doc's prior research synthesis) —
   if magnitude is going to be used as a confidence proxy at all, it should be
   computed in units of "how large relative to normal volatility" from the start,
   not retrofitted post-hoc as this study's vol_zscore_gate attempted (and which
   showed the most promise of the three variants, weakly).

**This experiment is CLOSED, not "ready for risk review."** No variant is
recommended for promotion, staging trial, or further parameter sweeps on the
current price-regression target. Follow-up work belongs in a new research
tournament (target redesign), not in this experiment.

**Recommendation to pm: rejected as calibration-layer fix; redirect to
target-redesign tournament (direction-classification and/or vol-normalized
returns) as the next research priority for closing the 53%-accuracy /
noise-confidence gap.**

## Follow-ups filed

- GitHub issue tracking a target-redesign tournament (direction-classification
  vs. vol-normalized-return vs. current price-regression target, same L1/L2 exam
  discipline as #898) — recommended next step, not started here.
- **Load-dependent backtest non-determinism (spawned as a background task during
  this session)**: root-caused during Phase 3 — `PredictionEngine`'s inference
  timeout defaults to 0.1s (a latency-alerting budget, not an inference-abort
  deadline; `src/prediction/engine.py:880-882`,
  `DEFAULT_MAX_PREDICTION_LATENCY` in `src/config/constants.py:10`), and firing it
  silently substitutes `HOLD` for that bar via a bare `except Exception` in
  `MLBasicSignalGenerator._get_ml_prediction`. Re-running the IDENTICAL baseline
  backtest (same model file, same window, same code) twice produced 46 trades/
  -11.36%/PF 0.393 and then 55 trades/-10.33%/PF 0.512 — from as few as 1-3
  timeout events per run (0.006-0.02% of bars), amplified by
  `ignore_signal_reversal`. This is a general backtest-reproducibility defect,
  not specific to this study, and directly threatens the model-evaluation-
  system's frozen-exam comparability premise. Filed for dedicated investigation
  (separate from training-run variance, which is real but secondary).
- Training-run reproducibility gap (a distinct, additional factor from the
  timeout issue above): a fresh `atb train price` run on the identical W_full
  config as #898 also differs due to genuinely different learned weights
  (different random init / early-stopping epoch), independent of the timeout
  issue. The model-evaluation-system's scoreboard design
  (`docs/architecture/model_evaluation_system.md`) should account for both
  effects — consider requiring N≥2-3 retrains per candidate before trusting any
  one model's OOS behavior as representative.
- `min_confidence` override gap in `src/experiments/runner.py`'s
  `ExperimentRunner._apply_strategy_attribute` (component_targets maps
  `min_confidence` to `[position_sizer]` only, but HyperGrowth's gate lives on
  `risk_manager`/`FlatRiskManager` instead) — spawned as a background task during
  this session (not re-filed here to avoid duplication).
