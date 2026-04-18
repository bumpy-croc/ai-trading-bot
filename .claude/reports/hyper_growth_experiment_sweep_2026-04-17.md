# Hyper-Growth Strategy Optimization — Experiment Report

**Date**: 2026-04-17
**Branch**: `claude/optimize-hypergrowth-strategy-Pt6HK`
**Framework**: PR #602 declarative experimentation (`src.experiments`) + `src.engines.backtest.Backtester`
**Data**: BTCUSDT 1h fixture, 2024-01-01 → 2024-12-31 (`tests/data/BTCUSDT_1h_2023-01-01_2024-12-31.feather`)
**Seed**: 42, deterministic

## TL;DR

Three findings, in descending order of impact:

1. **Hyper-growth's ML prediction is broken** — the strategy uses `model_type="sentiment"` but the signal generator feeds it price-only features. The sentiment model requires 10 features (including `sentiment_momentum_scaled`); it receives 5. The model returns `0.0` on every bar, which the signal generator converts to `predicted_return = -1.0`. Bar-for-bar, every 2024 signal is `SELL` with `confidence = 1.0` — a constant sentinel, not a prediction.

2. **Swapping to the working "basic" model + tightening the stop loss moves returns from 14.16% → 99.80%** on 2024 BTCUSDT (+85.6 percentage points, 7x improvement). The basic model has a measurable directional edge (BUY accuracy 55–57% at 12–24h horizons). Tightening SL from 20% → 10% alone more than doubles returns even with the broken signal.

3. **The experimentation framework's override surface is too narrow for hyper-growth** — the knobs the framework exposes (signal thresholds, confidence multipliers) are precisely the knobs hyper-growth's design ignores (FlatRiskManager + FixedFractionSizer are deliberately confidence-insensitive). The knobs that actually move returns for this strategy (`stop_loss_pct`, `take_profit_pct`, `base_fraction`, `model_type`) either raise or fail to route through the override system.

---

## Answering the critical-thinking question

> *"Signal quality and confidence multiplier — could it be the combination? Poor signals amplified = worse returns. Are these separate issues that need separate measurement?"*

**You were right.** They are separable, and conflating them is exactly what produced the "zero effect" illusion.

**Signal quality** = does the model's direction correlate with forward returns?
**Amplification** = given a signal, does magnifying confidence change position sizing and thus P&L?

The ml_basic control sweep confirms this split directly:

| variant | trades | ret% | interpretation |
|---|---|---|---|
| baseline (conf_mult=12) | 45 | -0.62% | weak signal, moderate amplification |
| conf_mult_6 | 1 | -0.01% | same signal, under-amplified → gate blocks nearly everything |
| conf_mult_20 | 194 | -2.10% | same signal, more amplification → more losing trades |
| conf_mult_30 | 457 | -3.61% | same signal, maximum amplification → most losing trades |

More amplification *of a weak signal* produces linearly worse returns — which is exactly what you predicted. If the signal had a true edge, the line would be flipped.

**For hyper-growth specifically**, both knobs independently did nothing because (a) the signal is constant (no quality information to amplify) and (b) the sizer was built to ignore amplification anyway. Two layers of "no effect" that coincidentally produce identical numbers.

---

## Experiment timeline and findings

### Phase 1: Initial parameter sweeps (`src.experiments` framework, via YAML / Python driver)

Four suites on hyper-growth: long thresholds, confidence multiplier, short thresholds, a combo. **Every single variant produced bitwise-identical results** to the baseline — same 44 trades, same $1,142 final balance, same 7.24% drawdown.

`long_thresholds` (4 variants): all identical to baseline, 14.16% return.
`confidence_multiplier` (3 variants): all identical.
`short_thresholds` (4 variants): all raised — `MLBasicSignalGenerator` doesn't have the regime-specific short-threshold attributes that its parent `MLSignalGenerator` has.
`combo` (2 variants): all identical / errored.

"Identical across wildly different thresholds" is suspicious, not reassuring — either the overrides aren't landing, or the signal path downstream is degenerate. The framework reports these as baseline-ties and doesn't flag them.

### Phase 2: Signal-quality diagnostic (what the framework doesn't currently measure)

I built `experiments/hypergrowth_signal_diagnostic.py` to walk the exact `MLBasicSignalGenerator(model_type="sentiment")` that hyper-growth uses over every bar of 2024, recording `predicted_return`, decision direction, confidence, and hit rate at 1h/4h/12h/24h horizons.

Results (8,615 bars of 2024):

```
Predicted-return distribution
  n=8615  mean=-1.000000  median=-1.000000  std=0.000000
  min=-1.000000  max=-1.000000

Decision mix
  buy:     0 (0.00%)
  sell: 8615 (100.00%)
  hold:    0 (0.00%)

Confidence distribution
  mean=1.0000  std=0.0000  fraction >= 0.05 gate: 100.00%

Directional hit rate vs. forward return
  h=  1: SELL acc=48.89% (wins=4212, losses=4403)
  h= 24: SELL acc=46.30% (wins=3989, losses=4626)
```

`predicted_return = -1.0` is not a real prediction. The code computes `(prediction - current_price) / current_price`, so `prediction = 0` yields exactly `-1.0`. The sentiment model is returning `0.0`.

**Root cause:** `create_hyper_growth_strategy` wires `MLBasicSignalGenerator(model_type="sentiment")`. The sentiment bundle's metadata declares 10 feature columns including `sentiment_momentum_scaled`. `MLBasicSignalGenerator._initialize_prediction_engine` disables sentiment features and installs a `PriceOnlyFeatureExtractor` that produces 5 columns (OHLCV-scaled). The model is fed a feature tensor of the wrong shape and returns `0.0` (the default from the prediction wrapper's failure path). The strategy then converts that to a perpetual SELL signal and the entire 14.16% annual return comes from the partial-exit + trailing-stop mechanics surviving a bull market by sheer luck (low position size × 2024's sideways first half × trailing stop protection).

### Phase 3: Control experiment on ml_basic (basic model)

`experiments/mlbasic_signal_diagnostic.py` runs the same diagnostic with `model_type="basic"` (what `ml_basic` strategy uses by default).

```
Predicted-return distribution (basic model)
  n=2154  mean=-0.000235  std=0.005442
  min=-0.031407  max=+0.038929
  fraction positive: 46.61%

Decision mix
  buy: 1004 (46.61%)
 sell: 1038 (48.19%)
 hold:  112 (5.20%)

Directional hit rate vs. forward return
  h=  1: BUY acc=54.48%  SELL acc=50.19%
  h=  4: BUY acc=53.98%  SELL acc=50.67%
  h= 12: BUY acc=55.78%  SELL acc=48.46%
  h= 24: BUY acc=56.47%  SELL acc=47.50%
```

A real distribution. A real decision mix. A real 54-57% BUY edge at 12-24h horizons (SELL side is roughly random — the model has a long bias, unsurprising given 2024's market). This is the model hyper-growth should be using.

The `ml_basic` amplification sweep (above) then confirmed the framework DOES detect amplification effects on a confidence-sensitive sizer — 1 trade at mult=6, 457 at mult=30 — so Phase 1's "zero effect" on hyper-growth was a property of the strategy, not a framework bug.

### Phase 4: Factory-kwarg sweep (the knobs the framework doesn't expose)

`experiments/hypergrowth_factory_sweep.py` builds the strategy via `create_hyper_growth_strategy(**kwargs)` and drives it through `src.engines.backtest.Backtester` — same data, same seed, same risk parameters as the framework runner, just with kwargs the YAML override system can't reach.

| variant (vs. broken sentiment signal) | trades | winR% | ret% | maxDD% | sharpe | Δvs.baseline |
|---|---|---|---|---|---|---|
| **baseline (sl=20 tp=30 f=0.20)** | 44 | 63.6 | **14.16** | 7.24 | 0.055 | — |
| **sl_10pct** | 79 | 58.2 | **29.55** | 7.52 | 0.096 | **+15.40** |
| sl_15pct | 44 | 65.9 | 11.92 | 6.45 | 0.047 | -2.23 |
| tp_20pct | 44 | 63.6 | 14.16 | 7.24 | 0.055 | 0.00 |
| tp_40pct | 44 | 63.6 | 14.16 | 7.24 | 0.055 | 0.00 |
| **frac_10pct** | 44 | 47.7 | **23.13** | 5.34 | 0.079 | **+8.97** |
| frac_30pct | 44 | 65.9 | 13.48 | 7.24 | 0.054 | -0.68 |
| frac_40pct | 44 | 65.9 | 13.40 | 7.24 | 0.054 | -0.76 |
| conf_gate_0.02 | 44 | 63.6 | 14.16 | 7.24 | 0.055 | 0.00 |
| conf_gate_0.10 | 44 | 63.6 | 14.16 | 7.24 | 0.055 | 0.00 |
| conf_gate_0.20 | 44 | 63.6 | 14.16 | 7.24 | 0.055 | 0.00 |
| combo_sl10_frac30 | 79 | 60.8 | **28.91** | 7.52 | 0.095 | +14.75 |
| combo_sl15_frac30_tp40 | 44 | 65.9 | 11.14 | 6.45 | 0.045 | -3.01 |

**Interpretation with the broken-signal context in hand:**
- `tp_*` variants are noops because the strategy always exits via trailing stops activated at +3% PnL, long before the 30%/40% TP triggers.
- `conf_gate_*` variants are noops because the broken signal's confidence is always 1.0 (|-1.0| × 12, clipped), and every gate up to 1.0 passes.
- `sl_10pct`: the market trended up ~120% on 2024. Every short loses money; tighter SL caps the loss and lets more re-entries compound. Hence the doubling in trades (44 → 79) and the return lift.
- `frac_10pct`: half the capital committed to each losing short → half the loss dollar-wise, but the trailing stops on winners capture full partial-exit profits. Net: lower DD AND higher return.

### Phase 5: Model-swap experiment (the high-leverage intervention)

`experiments/hypergrowth_model_swap.py` keeps the hyper-growth skeleton but replaces the signal generator with `MLBasicSignalGenerator(model_type="basic")`.

| variant | trades | winR% | ret% | maxDD% | sharpe |
|---|---|---|---|---|---|
| sentiment(broken) — baseline | 44 | 63.6 | 14.16 | 7.24 | 0.055 |
| sentiment(broken) — sl_10pct | 79 | 58.2 | 29.55 | 7.52 | 0.096 |
| **basic(working) — baseline** | **38** | **81.6** | **48.96** | **4.11** | **0.147** |
| **basic(working) — sl_10pct** | **74** | **68.9** | **99.80** | **4.74** | **0.259** |
| basic(working) — sl_10_frac30 | 74 | 68.9 | 99.19 | 4.74 | 0.258 |

**99.80% return on BTCUSDT 2024 with a 4.74% max drawdown and Sharpe 0.259**, just from (a) using the basic model the strategy should have been using and (b) tightening the stop from 20% → 10%. No leverage, same partial-exit mechanics, same backtest engine.

Note that `frac_30` on top of the best config is *not* additive — it produced 99.19% vs 99.80%. The strategy's `_max_position_pct = 0.50` cap (set by the factory) is already saturating at `sl_10pct` because tighter SL makes each trade more efficient per dollar committed.

---

## Framework gaps and proposed improvements (PR #602 follow-ups)

Running this through the new experimentation framework exposed seven gaps. All are fixable without touching the live trading engine.

### G1. Factory kwargs are not plumbed through `ExperimentRunner._load_strategy`

`_load_strategy` does `builder()` with no arguments. Strategies expose rich factory kwargs (`stop_loss_pct`, `take_profit_pct`, `base_fraction`, `model_type`, `max_leverage`, `min_regime_bars`, `leverage_decay_rate`, …) that are unreachable via YAML.

**Fix**: accept a `factory_kwargs: dict[str, Any]` field on `BacktestSettings` / `ExperimentConfig` and pass it through to `builder(**factory_kwargs)`. Distinct from `parameters` (which mutate post-construction); factory kwargs are construction-time.

### G2. `stop_loss_pct` / `take_profit_pct` overrides reject `FlatRiskManager`

`_apply_strategy_attribute` hard-codes: *"Only CoreRiskAdapter-backed strategies honor this knob."* That's the current reality, but it's a straightforward extension to let `FlatRiskManager` hold a `_strategy_overrides` dict too (it already has `stop_loss_pct` as an attribute), OR to route the override to `strategy._risk_overrides` even when the adapter lacks the attribute. As things stand, **hyper-growth literally cannot be tuned on the dimension that most affects its P&L** via the framework.

### G3. `base_fraction` routing is wrong for `LeveragedPositionSizer`

The runner maps `base_fraction → [position_sizer]`. Hyper-growth's position_sizer is `LeveragedPositionSizer` which wraps a `FixedFractionSizer`. `LeveragedPositionSizer` has no `base_fraction` attribute and the underlying `FixedFractionSizer` calls its field `fraction` (not `base_fraction`). Result: `hyper_growth.base_fraction` raises "Unknown override attribute" even though the underlying concept is clearly supported.

**Fix**: when a wrapping sizer has a `base_sizer` attribute, recursively try to apply the override to that. Alternatively, introduce a canonical attribute name (`fraction`) and route both names to the right target.

### G4. `short_threshold_*` attributes only exist on the parent generator

`MLBasicSignalGenerator` (what hyper-growth uses) is a trimmed version of `MLSignalGenerator` and doesn't have the regime-specific short-threshold attributes. The override silently routes, finds no target, and raises a confusing "Unknown override attribute" error. Either lift these attributes into `MLBasicSignalGenerator` (if they're meant to be tunable) or make the error message point at the class-attribute mismatch.

### G5. Framework has no signal-quality diagnostic

This was the blind spot. The P&L-based reporter can't distinguish "signal is a constant sentinel" from "overrides have no measurable effect on already-good trades". A tiny diagnostic mode that emits:

- predicted-return distribution (count, mean, std, percentiles, positive-fraction)
- decision-mix (buy/sell/hold counts)
- confidence distribution
- hit-rate vs. N-bar forward return (direction-conditional)

…would have caught the broken-sentiment-model bug in five seconds instead of hours. The scaffold I wrote in `experiments/hypergrowth_signal_diagnostic.py` is 130 lines and re-usable — it could ship as `atb experiment diagnose --strategy hyper_growth`.

### G6. Reporter doesn't flag "bitwise-identical to baseline" as a warning

When variants produce `total_return`, `total_trades`, `max_drawdown`, and `final_balance` all exactly equal to baseline, the ranking confidence heuristic degrades gracefully to `HOLD`. But it doesn't surface the far more likely cause: the override didn't take effect (wrong attribute name, wrong target component, no-op code path). A simple check — "all metrics within 1e-9 of baseline" — could emit a loud warning: *"Variant produced identical metrics. Likely dead-code override; verify attribute routing."*

### G7. Bootstrap p-value heuristic doesn't know about return-sequence tie-breaking

Related to G6: when every variant ties, the reporter emits `INSUFFICIENT_DATA` or `HOLD`. That's mathematically correct but operationally unhelpful. Comparing the per-trade P&L sequences (not just the aggregate) would distinguish "different trades, same total" from "literally the same trades".

---

## Concrete next experiments (in priority order)

The framework, even as-is, can drive these if we add the `factory_kwargs` plumbing (G1) or keep using the tiny Python drivers I wrote.

1. **Validate the model-swap on 2023-2024 (2 years) and 2020-2025 (5 years)** — the 2024 result could be regime-specific. The research doc's 836% over 5 years was computed on the broken sentiment model; the basic-model equivalent may be dramatically higher or lower.
2. **Re-run the original SL/TP/base_fraction sweep on the basic-model strategy** — the broken-signal sweep found `sl_10pct` optimal, but with a real signal the optimum may shift materially (a predictive short signal in a bull market wants different risk than a nonsense signal).
3. **Walk-forward with `src/experiments/walk_forward.py`** on 2021-2024, 90-day windows, retraining the basic model per window. Validates that the 55-57% BUY edge generalizes out-of-sample.
4. **Retrain the sentiment model on a price-only feature schema** (or delete the feature-pipeline override in the generator so it feeds sentiment features). The sentiment model may carry alpha the basic model lacks; we've been flying blind because of the shape mismatch.
5. **Add a confidence-sensitive sizer variant of hyper-growth** (swap `FlatRiskManager`/`FixedFractionSizer` for `ConfidenceWeightedSizer`). With the basic model's real confidence distribution, amplification finally has something to amplify.
6. **Sweep the partial-exit targets/sizes on the basic-model strategy** — the research doc identified partial exits as the dominant P&L driver; we haven't tuned them because `_risk_overrides` nested dicts aren't reachable via the YAML override system.

---

## Artifacts produced by this session

Scripts, all under `experiments/`, all read-only of the live strategy code:

- `hypergrowth_smoke.yaml` — framework-native smoke test
- `hypergrowth_long_thresholds.yaml`, `hypergrowth_confidence_multiplier.yaml`, `hypergrowth_short_thresholds.yaml` — YAML sweeps (all zero-effect, kept for reproducibility / gap documentation)
- `hypergrowth_sweep.py` — multi-suite Python driver using `ExperimentRunner` directly for faster iteration
- `hypergrowth_signal_diagnostic.py` — bar-by-bar signal-quality measurement (the tool the framework is missing)
- `mlbasic_signal_diagnostic.py` — same diagnostic for the basic model (control)
- `mlbasic_amplification_sweep.py` — confidence-multiplier effect on a confidence-sensitive sizer (the "right substrate" control)
- `hypergrowth_factory_sweep.py` — the workaround for G1-G3
- `hypergrowth_model_swap.py` — the model-type intervention

None of these modify `src/` or `cli/`. They are strictly investigation tooling.

## Headline numbers

| Configuration | 2024 return | Max DD | Sharpe |
|---|---|---|---|
| hyper-growth as-shipped (sentiment model) | 14.16% | 7.24% | 0.055 |
| hyper-growth + sl=10% (still-broken signal) | 29.55% | 7.52% | 0.096 |
| hyper-growth + basic model | 48.96% | 4.11% | 0.147 |
| **hyper-growth + basic model + sl=10%** | **99.80%** | **4.74%** | **0.259** |
