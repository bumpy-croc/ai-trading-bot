# Why cnn_lstm/default and attention_lstm/default produce a bit-identical HyperGrowth blotter

**Date:** 2026-07-08
**Author:** quant-researcher (investigation, no code changes)
**Trigger:** architecture tournament (`.claude/worktrees/arch-tournament`) found two models with
genuinely different raw ONNX outputs (max abs diff 0.31 on identical synthetic input) produce
identical realized trades (54/54, same entries/exits/sizes, PF and final_balance match to 16 sig
figs) when scored through HyperGrowth on the frozen exam (ETHUSDT 1h, 2026-01-01→2026-07-04).

## Verdict

**Config issue specific to HyperGrowth's component wiring, not structural to
`ml_signal_generator.py`.** The signal generator itself preserves continuous variation in
`predicted_return` almost all the way through (into `strength` and `confidence`). The
information is destroyed one step later, inside HyperGrowth's `FlatRiskManager` +
`FixedFractionSizer(adjust_for_confidence=False, adjust_for_strength=False)` +
`LeveragedPositionSizer` (leverage keyed to regime, never to the signal). This is a **deliberate,
documented** design choice (see the module docstring), not an oversight — but it has a
side-effect nobody had traced through before: it makes HyperGrowth structurally blind to model
quality above a low bar, which quietly invalidates using HyperGrowth as the tournament's judge.

## The transformation chain, file:line by file:line

1. **Raw model → price prediction.** `src/strategies/components/ml_signal_generator.py:1061`
   (`MLBasicSignalGenerator._get_ml_prediction`) — `pred = float(result.price)`. Whatever the
   ONNX head outputs, `PredictionEngine` inverse-transforms it to a real price. This step is
   lossless (continuous in, continuous out) — the 0.31 raw-output divergence should still show up
   here unless the two models happen to converge on similar prices for real market windows.

2. **Price → predicted_return.** `ml_signal_generator.py:885` —
   `predicted_return = (prediction - current_price) / current_price`. Lossless, continuous.

3. **predicted_return → direction.** `ml_signal_generator.py:888-896`
   (`MLBasicSignalGenerator.generate_signal`):
   ```
   if predicted_return > self.long_entry_threshold:      # 0.0
       direction = BUY
   elif predicted_return < self.short_entry_threshold:    # -0.0005
       direction = SELL
   else:
       direction = HOLD
   ```
   **First lossy step** — continuous → 3 buckets. Expected/necessary for any directional
   strategy; not itself the anomaly (a direction-only bet is the intended edge, per the
   docstring in `hyper_growth.py:17-22`). Two models only need to agree on *sign* here, not
   magnitude, to start converging.

4. **predicted_return → strength / confidence.** `ml_signal_generator.py:890,899` and `:1073-1084`:
   ```
   strength   = min(1.0, abs(predicted_return) * 10)     # saturates at |pr| > 0.10
   confidence = min(1.0, abs(predicted_return) * 12.0)   # saturates at |pr| > 0.0833
   ```
   Continuous below saturation. For hourly ETH, `|predicted_return|` saturating either of these
   (8-10%/bar) essentially never happens, so this step is *not* where variation dies — confidence
   still tracks the model's actual predicted_return faithfully up to this point.

5. **confidence → risk_amount.** `src/strategies/hyper_growth.py:97-118`
   (`FlatRiskManager.calculate_position_size`):
   ```python
   if signal.confidence < self.min_confidence:   # 0.05
       return 0.0
   return balance * self.risk_fraction           # FLAT — no confidence/strength term at all
   ```
   **This is the critical lossy step.** Confidence is consumed *only* as a boolean gate
   (`>= 0.05`, i.e. `|predicted_return| >= 0.05/12 ≈ 0.42%`). Once a signal clears that gate,
   `risk_amount` is a constant (`balance * 0.25` by default) irrespective of whether
   `predicted_return` was 0.5% or 8%. `signal.strength` is never read anywhere in
   `hyper_growth.py` or `leverage_manager.py` — it's dead for this strategy.

6. **risk_amount → final position size.** `position_sizer.py:148-225`
   (`FixedFractionSizer.calculate_size`), instantiated at `hyper_growth.py:256-260` with
   `adjust_for_confidence=False, adjust_for_strength=False`:
   ```python
   multiplier = 1.0
   if self.adjust_for_confidence: ...   # skipped — flag is False
   if self.adjust_for_strength: ...     # skipped — flag is False
   if regime is not None:
       multiplier *= self._get_regime_multiplier(regime)   # regime-only, model-independent
   final_size = base_size * multiplier
   final_size = min(final_size, risk_amount)                # capped by the flat risk_amount
   ```
   Then `LeveragedPositionSizer.calculate_size` (`position_sizer.py:1119-1156`) multiplies by
   `leverage_manager.get_leverage_multiplier(regime)` (`leverage_manager.py:109-146`), which is a
   pure function of `(regime.trend, regime.volatility, regime duration)` — the `signal` argument
   is never touched. With HyperGrowth's default `max_leverage=1.0` this multiplier is capped
   ≤1.0 and is identical for both models anyway since regime is computed from price data, not
   from the ML model.

   **Net result: `final_position_size = f(balance, regime)` only**, once direction and the
   confidence gate are satisfied. Entry price (current close), stop-loss and take-profit are also
   fixed percentages off entry (`stop_loss_pct=0.10`, `take_profit_pct=0.30`,
   `hyper_growth.py:184,209`) — again model-independent. So **every field of the order
   (direction, size, entry, stop, target) is invariant to the model's output magnitude** once (a)
   sign of `predicted_return` agrees with the threshold and (b) `|predicted_return|` clears the
   0.42% confidence gate. Only the intra-trade equity curve differs, because `predicted_return`'s
   exact value still gets logged into per-bar metadata used for Sharpe/Sortino/VaR bookkeeping —
   consistent with the tournament's own observation that only the 6th-7th decimal of those
   diverges.

## Empirical verification

Ran the actual `FlatRiskManager` + `FixedFractionSizer` + `LeveragedPositionSizer` objects
(HyperGrowth's exact default config) against a fixed `BUY` signal with `predicted_return` swept
from 0.06% to 8% (spanning and exceeding the plausible range two different real models would
diverge by) at fixed `balance=10000`, `regime=(TREND_UP, LOW_VOL)`:

| predicted_return | confidence | strength | risk_amount | **final_size** |
|---|---|---|---|---|
| 0.0006 | 0.0072 | 0.0060 | 0.00 | 0.000000 |
| 0.0015 | 0.0180 | 0.0150 | 0.00 | 0.000000 |
| 0.0040 | 0.0480 | 0.0400 | 0.00 | 0.000000 |
| 0.0100 | 0.1200 | 0.1000 | 2500.00 | **2000.000000** |
| 0.0200 | 0.2400 | 0.2000 | 2500.00 | **2000.000000** |
| 0.0500 | 0.6000 | 0.5000 | 2500.00 | **2000.000000** |
| 0.0800 | 0.9600 | 0.8000 | 2500.00 | **2000.000000** |

An 8x spread in `predicted_return` (0.01 → 0.08) — larger than any plausible real divergence
between the two tournament models — collapses to the exact same `final_size` the instant both
clear the 0.42% gate. Below the gate, both are exactly zero regardless of magnitude. There is no
intermediate regime where magnitude matters at all. Script: see this note's companion analysis
(not committed — reproducible from `src/strategies/hyper_growth.py` + `position_sizer.py` +
`leverage_manager.py` directly).

## Config vs. structural

- **`ml_signal_generator.py` itself is not the bottleneck.** It computes `strength` and
  `confidence` as continuous (pre-saturation) functions of `predicted_return`. Any strategy that
  wires this generator to a sizer which *does* respect those fields will see model differences
  propagate into position size.
- **Confirmed by contrast:** `src/strategies/ml_basic.py:534-549` and `ml_adaptive.py:93`
  both use `ConfidenceWeightedSizer`, whose `calculate_size` scales `base_fraction` by
  `signal.confidence` directly (`position_sizer.py:253-...`) — no flat gate-then-constant
  behavior. `ml_basic`'s `CoreRiskAdapter` `confidence_weighted` sizer path
  (`ml_basic.py:534-549`) does the same. Two models scored through `ml_basic` or `ml_adaptive`
  **would** show differently-sized trades whenever `predicted_return` magnitude differs, even at
  identical direction.
- HyperGrowth's flat sizing is intentional and documented (`hyper_growth.py:17-22`): the
  strategy's own thesis is "the ML direction filter IS the edge, not the per-bar confidence
  score," because standard confidence-weighted sizers were crushing positions to ~$10 given the
  model's typically tiny raw confidence (0.01-0.05). That thesis may well be correct for P&L —
  but it has an unexamined corollary: **if direction-only is the edge, then any two models with
  correlated directional calls are indistinguishable to this strategy, no matter how different
  their calibration/magnitude/architecture.**

## Implication for both tournaments

**Architecture tournament:** HyperGrowth is not a valid judge of model quality beyond a coarse
directional-agreement test. A "better" model (better calibrated, better magnitude-ranked, better
Brier score) will never show up as better P&L under HyperGrowth unless it also flips the
*sign* of `predicted_return` on different bars than the current model — i.e., unless it makes
different binary direction calls. Any ranking of cnn_lstm vs attention_lstm (or any future
architecture) run through HyperGrowth is really measuring "how often do these two models agree on
sign, weighted by whether both clear a 0.42% gate," not P&L-relevant prediction quality. To
compare architectures meaningfully, the tournament should either (a) score models through a
sizer that respects confidence continuously (`ml_basic`'s `ConfidenceWeightedSizer` config, or
`ml_adaptive`), or (b) compare models directly on prediction-quality metrics (directional
accuracy, calibration, Brier/RMSE per regime) rather than through a P&L pipeline that structurally
cannot express the difference. If HyperGrowth-routed P&L is the promotion bar for hyper_growth
specifically, that's a defensible strategy-level choice — but it should not be read as evidence
of "no meaningful difference between architectures" in general, only "no difference HyperGrowth's
current config can act on."

**Target-redesign tournament (meta-labeling / direction-classification candidates):** This
partially generalizes and partially doesn't.
- Any candidate that ultimately gets consumed through HyperGrowth's exact component wiring (flat
  risk manager + confidence-blind sizer) hits the *same* wall: a meta-label classifier's
  confidence score, however well-calibrated, is thrown away the same way `predicted_return`-derived
  confidence is today. If the redesign's payoff thesis depends on position sizing responding to
  meta-label confidence (e.g., bigger size on higher-confidence direction calls), it must not be
  evaluated through HyperGrowth's current sizer, or the tournament will again only measure
  directional agreement.
- Where it does NOT generalize: a direction-classification target changes what's being predicted
  (P(up) vs. point-return), not the sizer. If the redesign's exam harness routes candidates
  through `ml_basic`/`ml_adaptive`-style confidence-weighted sizing (as it should, since
  HyperGrowth's control-arm role per charter.md is about the current live strategy, not a general
  evaluation harness), this bottleneck doesn't apply and confidence differences would propagate
  normally.
- Recommendation: the target-redesign tournament should explicitly state, per candidate, which
  RiskManager/PositionSizer combo the exam uses, and confirm it's one where `signal.confidence`
  and `signal.strength` are load-bearing (i.e. not `FlatRiskManager` + fully-disabled
  `FixedFractionSizer`) — otherwise a genuinely better meta-labeling model could silently fail to
  show up as better P&L for the same structural reason found here.

## What this does NOT show

- It does not show the two models are equivalent in quality — only that HyperGrowth can't tell
  them apart above a low directional-agreement bar. Whether cnn_lstm and attention_lstm actually
  *do* agree on sign at the 54 entry-transition bars in this exam window (plausible, since
  price-prediction models commonly converge on a momentum/continuation heuristic regardless of
  architecture) was not separately re-verified here — that's the natural follow-up to confirm the
  full causal story (need: dump `predicted_return` per bar for both models, or per the earlier
  confidence-calibration study `docs/research/experiments/2026-07-05_confidence-calibration.md`,
  compare the sign series). This note establishes the *mechanism* (where and how magnitude
  information is destroyed); a per-bar sign comparison would close the loop on *why 54/54 trades
  matched exactly* rather than just "matched whenever direction agreed."
- Not a proposal to change HyperGrowth. `risk_fraction`/flat-sizing is a live-affecting parameter;
  any change needs full backtest + sensitivity + risk-officer review per the standard proposal
  process before touching it.
