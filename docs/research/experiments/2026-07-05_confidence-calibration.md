# Confidence-Calibration Study — ETHUSDT

Date: 2026-07-05
Author: quant-researcher
Status: IN PROGRESS (Phase 1 complete, Phase 2/3 pending)
Issue: TBD (opened at write-up time) — follows from #898 (window tournament)
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

## Phase 2 — Empirical distributions (pending)

Plan: retrain the #898 tournament's winning config (W_full, full 2017-08-17→2025-12-31
history) for a clean ETHUSDT instrument via
`atb train price ETHUSDT --start-date 2017-08-17 --end-date 2025-12-31 --timeframe 1h
--epochs 50 --batch-size 256 --sequence-length 120` in an isolated worktree (never
touching the main checkout / prod). Then script ONNX inference over the 2026 exam
window's candles and characterize:

1. Distribution of `|predicted_delta|` vs. distribution of realized `|hourly move|`.
2. **The key table**: conditional directional accuracy by predicted-delta magnitude
   quantile — does a larger predicted move correlate with a higher hit rate?

(To be appended once compute completes.)

## Phase 3 — Candidate calibrations on the frozen exam (pending)

(To be appended once Phase 2 determines whether this phase is warranted, and if so,
which candidates make sense given the Phase 1 sizing-decoupling finding.)
