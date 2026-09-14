# Phase 1 (gate): Does model confidence carry exploitable signal to size on? — GH #1067

Author: quant-researcher
Status: COMPLETE — H0 SUPPORTED (null). Phase 2 (sizer evaluation) NOT started — gate did not clear.
Issue: https://github.com/bumpy-croc/ai-trading-bot/issues/1067 (Board-directed, Alex 2026-08-13)

## Why this experiment, and why it is not a re-run of #912

GH #912 (2026-07-05, CLOSED, REJECTED) already asked and answered almost this exact
question — "does `|predicted_return|` magnitude/confidence predict directional hit
rate, out of sample?" — for the same strategy component
(`MLBasicSignalGenerator._calculate_confidence`, unchanged formula
`confidence = clip(|predicted_return| * 12.0, 0, 1)`) and found H0 supported: flat,
non-monotonic hit-rate-by-decile, Cochran-Armitage p=0.669, Spearman p=0.477 on a
frozen exam (2026-01-01 → 2026-07-04), against a model #912 trained itself
(end_date=2025-12-31).

Per role instructions, re-running an already-answered experiment needs explicit
justification. The justification here: **#912's exam window is now stale, and the
model has changed** — the currently deployed `ETHUSDT/basic/latest`
(`2026-07-04_22h_v1`, trained through 2026-07-04) is a *different* trained model than
#912's, and there are ~7 weeks of genuinely fresh data (2026-07-06 → 2026-08-24) that
neither #912's exam nor any prior study has touched. This experiment is a **light,
single-script confirmatory replication** on that fresh window, using the exact
production model (`model_version` pin, not `latest` resolution, per GH #988's
point-in-time pinning) and the exact production signal-generation code path — not a
new backtest, not a parameter sweep, not a new confidence-proxy invention.

## Hypothesis

**H1**: The currently deployed ETHUSDT model's confidence/predicted-return magnitude
predicts realized next-bar direction and/or magnitude on genuinely unseen data,
meaning `FlatRiskManager`'s binary gate is discarding real, sizeable information at
the sizing layer.

**H0 (falsifier)**: Confidence/magnitude carries no exploitable OOS signal — hit rate
is flat across confidence/magnitude buckets (no monotonic trend beyond what a
bootstrap/analytic null would produce), and even if bucket means differ, the pattern
is not significant/robust across multiple bucket definitions.

## Metric

Conditional directional hit rate by `|predicted_return|` quintile/decile and by
`confidence` quintile (`sign(predicted_return) == sign(realized_return)`), with
Wilson 95% CIs per bucket; Cochran-Armitage trend test and Spearman rank correlation
(bucket index vs hit rate) as the formal significance tests, replicating #912's
exact statistical methodology so results are directly comparable. Secondary: Spearman
correlation of `|predicted_return|` vs `|realized_return|` (does magnitude predict
magnitude, independent of direction).

## Success threshold (the gate)

Per the issue's explicit gate: **if high-confidence predictions are not measurably
better than low-confidence ones (both direction and magnitude), STOP — conviction
sizing cannot help.** Concretely: gate clears (proceed to Phase 2) only if at least
one bucket definition shows a Cochran-Armitage/Spearman trend test p < 0.05 (no
multiple-comparison correction needed to justify *stopping* — correction would only
matter for justifying *proceeding*, and BH-FDR across the 3 bucket definitions tried
here is even more conservative, so the uncorrected p is the more permissive bar and
still must be cleared). Given #912 and #1105's convergent priors, the base rate for
this gate clearing is treated as low going in — stated explicitly to avoid moving the
goalposts if the result is null, per the anti-p-hacking rule.

## Risks of false positive

- **Small window** (n≈1,177 usable bars, ~7 weeks, single continuous regime segment)
  — any positive result here would need replication across a longer/different window
  before being trusted, same caveat #912 already logged.
- **Multiple bucket definitions tried** (quintile-by-prediction, decile-by-prediction,
  quintile-by-confidence) inflate false-positive risk if only the best-looking one
  were reported. All three are reported below, not just the cleanest.
- **#1106 caveat**: backtest/paper scoring does not predict live decisions exactly
  (43.2% of live decisions at minute-5 disagree with the closed-bar decision) — this
  script scores closed bars only (matches backtest, not live-forming-bar behavior),
  so even a positive finding here would need a forming-bar-aware follow-up before any
  live sizing change.

## Method

Script: `docs/research/experiments/scripts/2026-08-25_confidence_bucket_oos_check.py`.
Reuses `MLBasicSignalGenerator.generate_signal()` directly (the exact
`PredictionEngine`/ONNX/rolling-minmax pipeline the live strategy uses — no
hand-rolled inference), pinned to `model_version="2026-07-04_22h_v1"` (currently
deployed `ETHUSDT/basic/latest`). No backtest engine, no `RiskManager`, no
`CostCalculator` involved — pure scoring + statistics, so fees/slippage are N/A here
(this is a signal-quality question, not a P&L question; Phase 2, if reached, would be
the P&L question and would use full costs).

Data: ETHUSDT 1h OHLCV, prefetched via `atb data prefill-cache --symbols ETHUSDT
--timeframes 1h --years 1` (fresh pull this session, cache verified non-stale).
OOS window: 2026-07-06 → 2026-08-24 (1 day of buffer after the model's training
`end_date=2026-07-04`, through the latest cached data at run time). No look-ahead:
`current_price = df["close"].iloc[i]` uses only bar `i`'s already-closed candle;
realized return compares `close[i+1]` vs `close[i]`, matching the model's own
next-bar-price training target exactly (verified by reading
`ml_signal_generator.py:295-370` directly, not assumed).

Worktree verified: `python -P -c "import src; print(src.__file__)"` resolves under
this session's own worktree (`.claude/worktrees/agent-a450d07f9a8e9b09b`), shim
installed and checked before any number was trusted.

## Results

n=1,177 scored bars (all with nonzero predicted sign — no HOLDs/failed predictions
excluded). Overall directional hit rate 54.29%.

**By `|predicted_return|` quintile:**

| bucket | n | hit rate | 95% CI | mean\|pred\| |
|---|---|---|---|---|
| 0 (lowest) | 236 | 50.42% | [44.09%, 56.74%] | 0.000323 |
| 1 | 235 | 54.89% | [48.50%, 61.13%] | 0.001035 |
| 2 | 235 | 62.55% | [56.21%, 68.49%] | 0.001932 |
| 3 | 235 | 51.06% | [44.71%, 57.39%] | 0.003485 |
| 4 (highest) | 236 | 52.54% | [46.18%, 58.82%] | 0.008755 |

Cochran-Armitage Z=0.041, **p=0.967**; Spearman ρ=0.200, **p=0.747**.

**By `|predicted_return|` decile** (n≈118/bucket): same non-monotonic, no-trend
pattern; Cochran-Armitage Z=-0.005, **p=0.996**; Spearman ρ=-0.036, **p=0.920**.

**By `confidence` quintile** (identical bucketing to the prediction-magnitude
quintile, since `confidence` is a monotone transform of `|predicted_return|` under
the current formula — reported separately per the brief's instruction not to collapse
distinct proxy definitions, but the numbers are literally identical to the
prediction-magnitude quintile table above by construction): same result, Cochran-
Armitage p=0.967, Spearman p=0.747. Highest-confidence quintile (mean confidence
0.105, well clear of the 0.05 gate) hit rate 52.54% — statistically indistinguishable
from the lowest quintile (mean confidence 0.004, hit rate 50.42%; CIs overlap by a
wide margin).

**Confidence distribution**: median 0.0228, only 23.45% of bars clear the live
`min_confidence=0.05` gate — reproduces #912's "compression" finding almost exactly
(#912's exam: median 0.0317, 34.18% clearing gate; close enough given a different
model/window that the compression mechanism is confirmed structural, not
model-specific).

**Secondary — magnitude-vs-magnitude correlation**: Spearman ρ=0.204, **p=1.5e-12**,
n=1,177. This *is* significant. `|predicted_return|` weakly but genuinely correlates
with `|realized_return|` — the model's confidence signal appears to track something
like a volatility/regime read (it predicts bigger moves are coming when bigger moves
actually happen), independent of whether it gets the *direction* of that move right.
This is a real, if modest, effect (ρ=0.20 explains ~4% of variance) and is reported
in full rather than discarded, per the no-p-hacking rule — but it does **not** answer
the question conviction-weighted sizing needs answered. A directional strategy sizing
up on "confidence" that predicts move *size* but not move *direction* would size up
into larger, still-coin-flip bets — this is exactly the "amplifies noise" failure
mode the issue names as the live null hypothesis, not a path to a better sizer. It is
flagged here as a candidate for a *separate* future question (does move-size
predictability have value for, e.g., a vol-targeting stop/TP adjustment rather than
directional position sizing) — explicitly out of scope for #1067's conviction-sizing
question, not silently smuggled into a Phase 2 justification.

## Reconciliation with prior work

- **#912** (previous model, exam Jan–Jul 2026): null (CA p=0.669, Spearman p=0.477).
  **This experiment** (current model, fresh window Jul–Aug 2026): null (CA p=0.967,
  Spearman p=0.747). Same mechanism, different model instance, different window,
  same verdict — this is now a **replicated** null, not a single-study result.
- **#1105** (model-free predictability scan, 2026-08-13): 0/20 symbol×timeframe cells
  clear round-trip costs on directional momentum; most cells sit at or below 50% DA.
  Consistent — if the *model itself* carries only a ~51-54% DA edge with no magnitude-
  conditional structure, its confidence channel has no extra information to size on.
- **PR #1010 slippage/EV-conditioning study** (2026-07-12): found no entry observable
  predicts per-trade EV, run against `FlatRiskManager`'s binary sizing — the issue
  brief correctly flagged this as *not* a clean refutation of the conviction-sizing
  question, since flat sizing could only let through what the binary gate passed.
  This experiment closes that gap directly: it scores *every* bar regardless of
  whether it would clear the live gate, and still finds no directional signal in the
  magnitude/confidence channel. So #1010's null and this experiment's null are now
  **independently confirming**, not just consistent-by-construction.

## Verdict

**H0 supported. Gate does NOT clear.** High-confidence predictions are not
measurably better than low-confidence ones on direction (the question that matters
for a directional strategy's sizing), across three independent bucket definitions
(quintile-by-magnitude, decile-by-magnitude, quintile-by-confidence — all report
literally the same null because confidence is a monotone transform of magnitude
under the current formula), on a genuinely fresh out-of-sample window scored with the
actual production model and inference path. The one significant finding (magnitude-
vs-magnitude, ρ=0.20) does not answer the directional-sizing question and does not
justify overriding the gate.

**Per the pre-registered gate and the issue's own framing: STOP HERE. Phase 2
(conviction-weighted sizer evaluation, `ConfidenceWeightedSizer`,
`kelly_momentum`-vs-`FlatRiskManager` comparison) is NOT started.** Building or
backtesting a sizer against a signal that three independent studies now show carries
no directional information would be, in the issue's own words, "a more elaborate way
to size noise."

## Recommendation to pm

1. **Close #1067 as a completed, negative research question** — not a failure, a
   real answer. The Board's underlying concern ("is FlatRiskManager throwing away
   real information") is now answered: no, there is no directional information in the
   confidence channel to throw away, given the current price-regression target.
2. **Do not build or evaluate `ConfidenceWeightedSizer`/`kelly_momentum` against
   this confidence channel** — same root cause as #912's redirect recommendation:
   the fix, if there is one, is upstream at the model-target level (direction
   classification / vol-normalized returns), not at the sizing-layer.
3. **#938 (HyperGrowth structurally blind to model quality) should be updated/closed
   with a cross-reference to this finding** — the "blind to confidence" architecture
   is no longer a risk given the confidence channel carries no information for it to
   discard.
4. If the Board wants this reopened later, the trigger condition should be: a
   genuinely redesigned training target (direction-classification or vol-normalized
   regression, per #912's own recommendation) that clears a magnitude-vs-hit-rate
   check on its own frozen exam — not a re-test of the current price-regression
   target's confidence formula.

No code change proposed. No `.claude/state/proposals/` file — nothing live-affecting
recommended (the recommendation is "do not build," which is the status quo action).
