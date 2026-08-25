# GH #1067 — FlatRiskManager conviction-blind sizing: findings

Author: quant-researcher
Date: 2026-08-25
Status: Phase 1 gate FAILED (null). Phase 2 NOT started. Recommend closing #1067.

## TL;DR

Ran the Phase 1 gate the issue prescribes: does the model's own confidence signal
predict realized outcome (direction or magnitude), out of sample, once you stop
letting `FlatRiskManager`'s binary gate hide most of the distribution from you? No.
Confirmed on a genuinely fresh out-of-sample window (2026-07-06 → 2026-08-24,
n=1,177 bars) using the currently deployed production model
(`ETHUSDT/basic/2026-07-04_22h_v1`) and the exact production inference path — no
proxy confidence signal was invented. Three bucket definitions tried
(prediction-magnitude quintile, decile, confidence quintile), all null
(Cochran-Armitage p=0.967–0.996, Spearman p=0.747–0.920). This replicates GH #912's
2026-07-05 finding on a different model and a different, previously-untouched window
— the null is no longer a single study, it is a replicated result, further
corroborated by #1105's independent model-free scan (0/20 symbol×timeframe cells
clear costs) and PR #1010's EV-conditioning null.

**Conclusion: there is no directional signal in the confidence channel to size on.
Conviction-weighted sizing (`ConfidenceWeightedSizer`, `kelly_momentum` vs
`FlatRiskManager`) is not evaluated, because doing so against this confidence signal
would just be a more elaborate way to size noise, per the issue's own stated
concern.** No code change. No sizer backtest. No proposal.

## What I found before running anything new

Before writing any new script, I searched prior experiments (per the "don't re-run an
already-answered experiment" rule) and found the Phase 1 question had already been
asked, almost verbatim, in **GH #912** (2026-07-05, CLOSED):

> H0 (falsifier): Predicted-delta magnitude carries no information about directional
> accuracy (conditional hit-rate is flat across magnitude quantiles).

#912's verdict: **H0 supported.** A freshly retrained ETHUSDT model, scored on a
frozen exam (2026-01-01 → 2026-07-04, n=4,415), showed a flat, non-monotonic hit-rate
table by `|predicted_return|` decile (Cochran-Armitage p=0.669, Spearman ρ=0.255,
p=0.477). A superficially significant gradient existed on training-period-adjacent
data (p=0.019) but vanished on the frozen exam — textbook overfitting of the
confidence channel to the training distribution, not real signal.

This is not a different question from #1067's Phase 1 — it is the same mechanism
(`MLBasicSignalGenerator._calculate_confidence`, `confidence = clip(|predicted_return|
* 12.0, 0, 1)` — unchanged since #912, verified by reading the current source), same
strategy component, same statistical framing (bucket by magnitude, test hit-rate
trend). Per the role's own rule, re-running it needed explicit justification, not a
blind repeat.

**Justification for a light follow-up rather than treating #912 as fully closing the
question**: #912's exam window ended 2026-07-04, and the model it scored was a
worktree-local retrain, not the model actually deployed to `latest` today. The
currently deployed model (`ETHUSDT/basic/2026-07-04_22h_v1`) was trained through the
same date but is a genuinely different trained instance, and there are ~7 weeks of
data since (2026-07-06 → 2026-08-24) that no prior study has scored. That is a real,
previously-untouched OOS window using the actual live model — worth a cheap
confirmatory check, not a full new tournament.

## What I ran

A single scoring script (no backtest engine, no P&L, no fees/costs involved — this is
purely "does the signal carry information," not "does trading on it make money"):
`docs/research/experiments/scripts/2026-08-25_confidence_bucket_oos_check.py`.

- Loads ETHUSDT 1h OHLCV (freshly prefetched this session via `atb data
  prefill-cache`, verified non-stale).
- Instantiates `MLBasicSignalGenerator(symbol="ETHUSDT",
  model_version="2026-07-04_22h_v1")` — pins to the exact production model version
  (GH #988's point-in-time pin), not `latest` resolution, so the result is
  reproducible regardless of any future promotion.
- Calls `generate_signal()` directly — the real `PredictionEngine`/ONNX/
  rolling-minmax pipeline the live strategy uses. No hand-rolled inference.
- For every bar in 2026-07-06 → 2026-08-24 (1 day of buffer past the model's
  2026-07-04 training cutoff), records `predicted_return`, `confidence`, and the
  realized next-bar return (`close[i+1]` vs `close[i]` — matches the model's own
  next-bar-price training target, verified by reading the signal generator's
  `generate_signal()` source, not assumed).
- No look-ahead: `current_price` is bar `i`'s already-closed candle; the realized
  return uses only `i+1`, one bar forward of the decision point.

Verified I was scoring my own worktree's code (`python -P -c "import src; print(src.
__file__)"` resolves under `.claude/worktrees/agent-a450d07f9a8e9b09b`), and installed
the worktree import shim first (it was not active — confirmed and fixed before
running anything, per the #1070 lesson).

## Results (full — every bucket definition tried, not just the cleanest)

n=1,177 usable bars, overall directional hit rate 54.29%. Confidence distribution:
median 0.023, only 23.45% of bars clear the live `min_confidence=0.05` gate — this
reproduces #912's "compression" finding (median 0.032, 34% clearing gate on a
different model/window) closely enough to confirm the compression is a structural
property of the formula, not model-specific noise.

**|predicted_return| quintile** (n≈235/bucket): hit rates 50.4% / 54.9% / 62.6% /
51.1% / 52.5% — non-monotonic, no trend. Cochran-Armitage Z=0.041, **p=0.967**.
Spearman ρ=0.200, **p=0.747**. Every bucket's 95% Wilson CI overlaps every other's.

**|predicted_return| decile** (n≈118/bucket): same non-monotonic pattern.
Cochran-Armitage Z=-0.005, **p=0.996**. Spearman ρ=-0.036, **p=0.920**.

**confidence quintile**: identical numbers to the magnitude quintile (confidence is a
monotone transform of magnitude under the current formula, so this isn't independent
evidence, but is reported per the brief's instruction to show every proxy tried, not
just one). Highest-confidence quintile (mean confidence 0.105, well clear of the 0.05
gate): 52.5% hit rate. Lowest-confidence quintile (mean confidence 0.004): 50.4% hit
rate. Statistically indistinguishable.

**One real finding, reported honestly even though it doesn't help**: `|predicted_
return|` DOES correlate with `|realized_return|` — Spearman ρ=0.204, **p=1.5e-12**,
n=1,177. The model appears to weakly track something like realized-volatility regime
(it predicts bigger moves are coming when bigger moves actually happen) — but this is
orthogonal to *direction*, which is what a directional strategy's position sizing
needs. Sizing up on a signal that predicts move size but not move direction is
exactly the "amplifies noise" failure mode the issue itself names as the plausible
null — scaling into a bigger, still-coin-flip bet. I flagged this as a possibly
interesting question for a *different* purpose (vol-adaptive stop/TP width, not
directional position sizing) but did not fold it into a Phase 2 justification —
that would be exactly the kind of goalpost-moving the pre-registered gate exists to
prevent.

## Reconciliation with the two most relevant prior findings the brief called out

- **`docs/research/notes/2026-07-12_slippage-and-ev-conditioning.md` (PR #1010)**:
  found no entry observable predicts per-trade EV, but ran against
  `FlatRiskManager`'s binary gate, so it could only see what the gate let through —
  the issue correctly flagged this as not a clean refutation on its own. This
  experiment closes that gap: it scores every bar unconditionally (gate or no gate)
  and still finds no directional signal. The two nulls are now independently
  confirming rather than one being a weaker echo of the other.
- **`docs/research/experiments/2026-08-13_model-free-predictability-scan.md` (#1105)**:
  0 of 20 symbol×timeframe cells show tradeable directional structure after
  correcting for multiple comparisons and costs. Consistent top-down: if the market
  itself shows no exploitable directional structure at this resolution, there's no
  reason a single model's confidence *channel* specifically would encode more
  information than the model's raw directional call already carries (~51-54% DA,
  itself barely above chance).

## Caveats logged (per standing instructions)

- **#1106 (backtest/live parity gap, in flight)**: this script scores closed bars
  only, matching backtest behavior, not live's forming-bar decisions (43.2% of live
  decisions at minute 5 disagree with the closed-bar call). Even if this had come
  back positive, it would need a forming-bar-aware follow-up before informing any
  live change. Since the result is negative, this caveat doesn't change the
  conclusion, but is stated because the brief requires it on every backtest-derived
  claim, and this is backtest/scoring-adjacent even though it never runs the
  `Backtester` engine.
- **No fees/costs involved** — this is a signal-quality question (does the number
  carry information), not a P&L question. No `drawdown_cap_mode`/`early_stopped`
  concern applies because no backtest was run; noted explicitly so this isn't
  mistaken for a P&L claim.
- **Small window, single regime segment** (~7 weeks, one continuous stretch of
  2026-07/08 market conditions) — same caveat #912 logged about its own exam. A null
  here is not proof no window could ever show signal, but three separate
  studies (different models, different windows, different statistical framings —
  #912's decile study, #1105's model-free scan, and this bucket study) now agree,
  which is the standard the role instructions ask for before treating a null as
  solid ground to stop on.

## Recommendation to pm

1. **Close #1067** — treat this as a complete, valuable negative result per the
   issue's own framing ("Honest null is a success outcome"), not an incomplete task.
2. **Do not build or backtest `ConfidenceWeightedSizer` or evaluate `kelly_momentum`
   against this confidence channel.** The root-cause fix, if the Board wants to
   pursue it, is upstream at the model-training-target level (direction
   classification or vol-normalized-return targets — #912's own redirect
   recommendation, never executed as far as I can tell from the open-issue list;
   worth checking with pm/ml-engineer whether a target-redesign tournament should be
   scheduled).
3. **Cross-reference GH #938** ("HyperGrowth's flat position sizing makes it
   structurally blind to ML model quality") — the risk that framing describes no
   longer applies with force: there is no confidence information for flat sizing to
   be "blind" to. Recommend a short update comment there rather than a new issue.
4. If the Board later wants this reopened, the trigger should be a redesigned
   training target that clears its own magnitude-vs-hit-rate check on a frozen
   exam — not a retest of today's price-regression target's confidence formula.

## Files

- `docs/research/experiments/2026-08-25_conviction-sizing-phase1.md` — full
  preregistration-style write-up (hypothesis, metric, threshold, risks, method,
  results, verdict).
- `docs/research/experiments/scripts/2026-08-25_confidence_bucket_oos_check.py` —
  the analysis script.
- `docs/research/experiments/scripts/2026-08-25_confidence_bucket_oos_results.json`,
  `2026-08-25_confidence_bucket_oos_scored.csv` — raw output.
