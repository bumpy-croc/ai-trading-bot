# Forming-Bar Decision Flip-Rate: Sizing the Blast Radius of the No-`is_closed`-Gate Kline Buffer

**Date**: 2026-07-06
**Researcher**: quant-researcher
**Status**: complete (measurement only — no fix implemented here)
**Engine**: `MLBasicSignalGenerator` + `EnhancedRegimeDetector`, resolving `ETHUSDT/basic/2026-07-04_22h_v1` (the `latest` symlink at study time) via the real prediction engine / model registry — no mocked components.
**Worktree**: disposable `.claude/worktrees/fliprate-study` (detached HEAD off `develop`), removed at end of session. Harness script and checkpoints preserved at `/private/tmp/claude-501/-Users-alex-Sites-ai-trading-bot--claude-worktrees-elated-vaughan-5625e7/d6be54e9-2423-48d6-bda7-7b5e9102452a/scratchpad/fliprate_study/` (see Reproducibility, §8).
**Related**: `src/engines/live/kline_buffer.py` (the component under discussion for a closed-bar gate fix — not modified in this study), `.claude/state/log.md` 2026-07-04/2026-07-05 entries on the cross-symbol-scoring fix (#867/#887/#905 — prod now scores ETHUSDT with the native ETHUSDT model as of 2026-07-05).

## Hypothesis

**H1 (falsifiable)**: The live engine's `KlineBuffer` rewrites the tail 1h candle's OHLCV on every websocket tick with no `is_closed` gate. If `MLBasicSignalGenerator` is evaluated against this constantly-mutating tail candle throughout the hour, its trading decision (direction, confidence) will disagree with the decision it would reach once the candle is fully closed often enough, and on economically meaningful (actionable) decisions often enough, that gating live signal evaluation to closed bars only is a materially different trading system, not a cosmetic fix.

**Falsified if**: forming-bar decisions agree with the closed-bar decision on the overwhelming majority of hours, agreement holds up even harder on the actionable (confidence ≥ 0.05) subset, and forming-bar decisions predict H+1's realized return no worse than closed-bar decisions.

**Supported if**: flip rates are large, actionable-decision flips are non-trivial, and/or forming-bar decisions predict worse than closed-bar decisions.

Result, up front: **H1 is supported.** Flip rates are large and mostly outright direction reversals, not gate-boundary noise; the effect decays smoothly through the hour as expected; and the *directional accuracy* of forming-bar decisions is statistically indistinguishable from closed-bar decisions (both close to a coin flip) — so the practical risk is churn and reactive whipsaw trading, not a hidden edge being thrown away.

## Mechanism correction (read this before the numbers)

The brief for this study started from a hypothesis that the model might be scoring **out-of-distribution features** on a malformed forming candle. Reading the actual code path shows that is **not** what happens, in either engine:

- `MLBasicSignalGenerator._get_ml_prediction` (`src/strategies/components/ml_signal_generator.py:975-977`):
  ```python
  window_df = df[["open", "high", "low", "close", "volume"]].iloc[
      index - self.sequence_length : index
  ]
  ```
  This Python slice is **exclusive of `index`**. Both engines point `index` at the "current" row — live via `current_index = len(df) - 1` (`src/engines/live/trading_engine.py:1536`), backtest via the identical convention over a static historical frame. So the model's 120-bar feature window is **always the 120 closed bars strictly before the tail row**. The tail/forming candle's OHLCV never enters the model's input tensor, in either engine. **The model's raw `prediction` (a predicted price level) is constant for the entire forming hour** — it only changes at the top of the next hour, once a full 120-closed-bar window has rolled forward.

- What *does* vary intra-bar is `current_price = df["close"].iloc[index]` (`ml_signal_generator.py:866`), read directly off the tail row. `predicted_return = (prediction - current_price) / current_price` (`ml_signal_generator.py:885`) is then compared against `long_entry_threshold` / `short_entry_threshold` to pick `direction`, and its magnitude drives `confidence` via `_calculate_confidence` (`ml_signal_generator.py:1073-1084`, `confidence = min(1.0, abs(predicted_return) * confidence_multiplier)`, `confidence_multiplier=12.0` default). Since `prediction` is pinned for the hour and `current_price` is a live-updating tick, **the entire intra-bar variation in direction and confidence comes from a fixed prediction being divided by a moving reference price** — not from the model re-evaluating anything.

- Regime detection is a *partial* exception. `EnhancedRegimeDetector.detect_regime(df, index)` (`src/regime/enhanced_detector.py:177-212`) does read `working_df.iloc[index]` directly for `trend_label`/`vol_label`/`regime_confidence`, and those columns come from `RegimeDetector.annotate()` (`src/regime/detector.py:237-310`), whose `trend_score` is a rolling OLS slope on `close` over a 50-bar window (`slope_window=50`) **ending at and including** `index`. So regime context *can* legitimately shift intra-bar as the forming candle's close moves — this is a second, smaller channel of forming-bar sensitivity, distinct from the ML prediction path, and it feeds `MLBasicSignalGenerator`'s output only indirectly for the "basic" generator (`MLBasicSignalGenerator.generate_signal` ignores `regime` for direction — only the regime-aware `MLSignalGenerator`, not used by `hyper_growth`, adjusts thresholds by regime). Confirmed neither engine pre-annotates regime columns before calling `process_candle`/`generate_signal` (`grep` for `regime_label`/`annotate(` in `src/engines/live/trading_engine.py` and `src/engines/backtest/engine.py` returns nothing) — both call `EnhancedRegimeDetector.detect_regime` fresh each time, exactly as this harness does.

**Bottom line**: the closed-bar gate fix under discussion will not change what features the model sees (they were already closed-bars-only). What it will change is **freezing `current_price` to the bar's final close** instead of letting it float tick-by-tick, which freezes `predicted_return`'s denominator and therefore freezes `direction`/`confidence` for the whole hour (recomputed once, at close, instead of on every tick). This study measures exactly that mechanism.

## 1. Sanity check: 1m→1h reconstruction fidelity (done first, as required)

Before any decision replay, the forming-bar reconstruction method (running max/min/sum of 1m bars from minute 0 through minute *m* inclusive, open fixed from the hour's first 1m bar) was validated against the **real** closed 1h klines, using all 60 minutes of 1m data per hour (i.e. `m=59`, the point at which the reconstruction should exactly equal the real bar).

| Field | Max abs diff | Max rel diff | Tolerance (rtol) | Mismatches |
|---|---|---|---|---|
| open | 0.0 | 0.0 | 1e-6 | 0 |
| high | 0.0 | 0.0 | 1e-6 | 0 |
| low | 0.0 | 0.0 | 1e-6 | 0 |
| close | 0.0 | 0.0 | 1e-6 | 0 |
| volume | 1.455e-11 | 4.19e-16 | 1e-6 | 0 |

840 hours checked (the full study window), 1 hour skipped for incomplete 1m coverage at the window's trailing edge. Zero fields exceeded tolerance; the only nonzero delta (volume) is floating-point summation-order noise, ~13 orders of magnitude below the tolerance. The reconstruction method is faithful — no systematic mismatch to report or reconcile.

## 2. Data and scope

- **Symbol**: ETHUSDT.
- **1h bars**: fetched via `CachedDataProvider` (`src/data_providers/cached_data_provider.py`), reused/extended the existing read-only parquet cache at `/Users/alex/Sites/ai-trading-bot/cache/market_data`. Warmup from 2026-05-01 00:00 UTC through study end, giving the ML 120-bar sequence window and the regime detector's 252-bar ATR-percentile lookback both real, non-degenerate history before the study window starts (not a NaN-driven cold-start regime).
- **1m bars**: fetched directly from the base Binance provider (`create_data_provider(provider_type="auto")`) in 3-day chunks — bypassing `CachedDataProvider`'s year-based caching, which forces a whole-calendar-year fetch per call (~525k 1m candles) and blew the `DATA_FETCH_TIMEOUT_SECONDS` budget on the first attempt. Chunking is still the repo's normal Binance pagination (`python-binance`'s `get_historical_klines`), just bounded to short windows instead of a full year. This is a **scope-relevant deviation from the brief's suggested `atb data prefill-cache`/`CachedDataProvider` path**, done because the year-forcing behavior is impractical for 1m granularity; documented here rather than silently worked around.
- **Study window**: 2026-05-25 00:00 UTC → 2026-06-29 00:00 UTC — **5 weeks**, 840 hourly bars, all 840 valid for the study (120 closed bars of prior history available, H+1's real closed bar available for the realized-return metric, full 60 minutes of 1m data present).
- **Offsets**: all 11 requested (`m ∈ {5,10,...,55}`) — no reduction was needed. Per-call `generate_signal` overhead measured at ~9ms/call (regime detection ~7ms) after model warmup, so 840 hours × 12 snapshots (11 offsets + 1 closed) ≈ 10,080 decisions ran in 144 seconds as a single background process. Batch inference (`ENGINE_BATCH_INFERENCE`/`use_engine_batch`) was not needed and not used.
- **Actionability gate**: `confidence >= 0.05`, matching `create_hyper_growth_strategy`'s literal `min_confidence: float = 0.05` default (`src/strategies/hyper_growth.py:179`), enforced solely in `FlatRiskManager.calculate_position_size` (`hyper_growth.py:115`, `if signal.confidence < self.min_confidence: return 0.0`). This is the entire actionability gate at the signal-generation level — RiskManager sizing curves, leverage, and other engine-level filters are out of scope for this signal-agreement study.
- **Caveat**: 3 of ~10,080 decision calls logged `ML model inference exceeded timeout of 0.1s` (the real `MAX_PREDICTION_LATENCY` production default, `src/config/constants.py:10`), almost certainly due to CPU contention from a concurrent training tournament running elsewhere on the machine during this run (the study was run as a single background process per instructions, but the *other* process was not under this study's control). Verified zero corrupted/NaN `predicted_return` values in the final dataset — these transient timeouts did not propagate into recorded decisions. Noted for completeness, not a finding about the studied mechanism.

## 3. Methodology recap

For each hour H in the study window: build a DataFrame of exactly 120 real closed 1h bars immediately preceding H, followed by one "current" row occupying position `len(df)-1` (matching live's `current_index = len(df) - 1` exactly). That current row is populated 12 different ways per hour:

- 11 forming-bar snapshots, one per minute offset `m`: `open`=H's real open, `high`/`low`/`volume` = running max/min/sum of 1m bars from minute 0 through `m` inclusive, `close` = the 1m close at minute `m`.
- 1 fully-closed snapshot: H's real closed 1h OHLCV (the normal closed-bar decision, identical to what a backtest would evaluate).

Both `EnhancedRegimeDetector.detect_regime(df, index)` (fresh, per-snapshot — matching what both engines actually do) and `MLBasicSignalGenerator.generate_signal(df, index, regime)` are called directly against the freshly-built DataFrame — the real production classes, the real resolved model, no mocks. `direction` and `confidence` are read off the returned `Signal`. H+1's real closed 1h close is used only for the metric-(e) realized-return comparison, never as a decision input — by construction no snapshot's DataFrame contains any row at or after H+1.

## 4. Results

### (a) Per-offset flip rate vs. the closed-bar decision (all 840 hours)

| Offset (min) | Flips | Total | Flip rate |
|---|---|---|---|
| 5  | 363 | 840 | 43.2% |
| 10 | 324 | 840 | 38.6% |
| 15 | 313 | 840 | 37.3% |
| 20 | 285 | 840 | 33.9% |
| 25 | 244 | 840 | 29.0% |
| 30 | 210 | 840 | 25.0% |
| 35 | 208 | 840 | 24.8% |
| 40 | 178 | 840 | 21.2% |
| 45 | 161 | 840 | 19.2% |
| 50 | 118 | 840 | 14.0% |
| 55 | 89  | 840 | 10.6% |

Flip rate decays smoothly and monotonically (with one small bump at m=35) as the hour progresses and the forming candle's close converges toward its final value — exactly the expected shape if the mechanism is "reference price hasn't settled yet," not noise.

### (b) Hours with ≥1 flip across any offset

**551 / 840 hours (65.6%)** show at least one forming-bar snapshot disagreeing with the closed-bar decision at some point during the hour. Nearly two-thirds of all hours would have shown the live engine a different directional call at some tick than the one a closed-bar-gated engine would ever have produced.

### (c) Flip rate restricted to actionable decisions (confidence ≥ 0.05 on either side)

| Offset (min) | Actionable pairs | Flips | Flip rate |
|---|---|---|---|
| 5  | 313 | 108 | 34.5% |
| 10 | 326 | 88  | 27.0% |
| 15 | 329 | 82  | 24.9% |
| 20 | 341 | 75  | 22.0% |
| 25 | 342 | 49  | 14.3% |
| 30 | 342 | 46  | 13.5% |
| 35 | 352 | 37  | 10.5% |
| 40 | 352 | 32  | 9.1%  |
| 45 | 343 | 13  | 3.8%  |
| 50 | 333 | 12  | 3.6%  |
| 55 | 320 | 4   | 1.3%  |
| **All offsets combined** | **3,693** | **546** | **14.8%** |

This is the subset that matters for P&L: pairs where *either* the forming or the closed decision cleared the 0.05 confidence gate hyper_growth actually trades on. Even restricted to economically actionable decisions, **1 in 7 such pairs flip direction** overall, and in the first 20 minutes of an hour, roughly 1 in 3 to 1 in 4 do. Of the 2,493 total flip pairs across all offsets (actionable and not), **62.4% (1,556) are outright BUY↔SELL reversals** — not soft crossings of the confidence gate to/from HOLD — confirming these are substantive directional disagreements, not boundary noise. Example, hour 2026-05-25 01:00 UTC: model `prediction` fixed at 2103.97 for the whole hour; decisions across the 11 offsets were hold→buy→sell→sell→sell→hold→sell→buy→buy→buy→buy, with the closed-bar decision landing on buy — six direction changes inside one hour, purely from `current_price` moving between 2100.57 and 2107.30 against the frozen prediction.

### (d) Confidence trajectory within the hour

- Mean |Δconfidence| between consecutive offsets: **0.0162**; median **0.0086**.
- Mean confidence range (max−min) within an hour: **0.0755**; median **0.0448**.
- Mean number of times an hour's confidence trajectory crosses the 0.05 actionability gate: **1.07** per hour.
- Gate-crossing distribution (840 hours): 0 crossings in 421 hours (50.1%), 1 in 165 (19.6%), 2 in 116 (13.8%), 3 in 78 (9.3%), 4 in 40 (4.8%), 5+ in 20 (2.4%, max observed 8).

Half of all hours never touch the actionability gate at all (confidence stays consistently above or below 0.05 the whole hour) — consistent with the low median confidence (~0.03-0.05) already flagged in `.claude/state/log.md` (2026-07-05 window-tournament entry) as a system-wide calibration bottleneck independent of this study. But the other half cross the gate at least once, and a non-trivial tail (16.5%) crosses it 3+ times in a single hour — i.e. the position-sizing gate itself is flickering on and off within the hour purely from reference-price movement.

### (e) Directional accuracy vs. realized H+1 close-to-close return (non-HOLD only)

| | Hit rate | n (non-HOLD) |
|---|---|---|
| Closed-bar decision | 50.38% | 798 |
| Forming-bar decisions, aggregate (all offsets) | 50.71% | 8,639 |

Per-offset forming-bar hit rate:

| Offset (min) | Hit rate | n (non-HOLD) |
|---|---|---|
| 5  | 47.5% | 764 |
| 10 | 49.6% | 766 |
| 15 | 51.6% | 781 |
| 20 | 50.4% | 772 |
| 25 | 50.1% | 785 |
| 30 | 50.6% | 799 |
| 35 | 51.5% | 789 |
| 40 | 52.3% | 791 |
| 45 | 52.7% | 793 |
| 50 | 51.3% | 798 |
| 55 | 50.2% | 801 |

Both closed-bar and forming-bar hit rates sit at essentially a coin flip (~50-53%), consistent with the already-documented ~53.1% directional-accuracy ceiling for this model class (`.claude/state/log.md`, 2026-07-05 00:15 ml-engineer entry: "directional_accuracy 0.5312 on temporal holdout"). There is no evidence in this data that deciding early is *worse* at predicting the next hour's direction than waiting for close — the two are statistically indistinguishable given the sample sizes here. This is an important qualifier on the flip-rate findings above: the flips are real and frequent, but neither the forming-bar nor the closed-bar decision has much standalone predictive edge to lose or gain from timing.

## 5. Robustness notes

- The offset-decay shape in (a)/(c) is monotonic modulo one minor bump (m=30→35 flip-rate ticks up slightly, 25.0%→24.8%, effectively flat) — not an artifact of a single outlier hour, since it holds at both the raw and actionable-only cuts.
- The BUY↔SELL-reversal share (62.4% of all flips) was checked as a robustness cross-cut beyond the five requested metrics, specifically to rule out "these are just soft HOLD-boundary flickers" as an alternative explanation — it is not supported by the data.
- No parameter sweep was performed in this study (it measures an existing mechanism, not a proposed change) — sensitivity analysis is not applicable here in the quant-researcher proposal-template sense. The one true "parameter" is the offset grid itself, and all 11 requested offsets were run (no p-hacked subset).

## 6. Verdict

**H1 is supported. Intra-bar decisioning is materially different from closed-bar decisioning, and the difference is a large, directionally-substantive flip rate — not a subtle edge case.**

Be blunt about what the numbers show:

- Two-thirds of all hours produce at least one forming-bar decision that disagrees with the eventual closed-bar call (metric b).
- Restricted to decisions that would actually have sized a trade (confidence ≥ 0.05 on either side), roughly 1 in 7 still flip overall, and roughly 1 in 3-4 flip in the first 20 minutes of the hour (metric c) — this is the number that should worry anyone reasoning about live P&L today.
- Nearly two-thirds of flips are outright direction reversals (BUY↔SELL), not gate-boundary noise.
- **But**: neither the forming-bar decisions nor the closed-bar decisions predict H+1's realized return meaningfully better than a coin flip (metric e). This means the practical cost of the current no-gate behavior is not "the live engine is throwing away a real edge by deciding early" — there is barely any edge to throw away either way, given this model's ~50-53% ceiling. The cost is **decision churn and reactive whipsaw**: a live engine can flip its stance on the same nominal hour multiple times, size into a position, size out, re-enter opposite, purely because the reference price wiggled — with no corresponding improvement (or, from this data, meaningful degradation) in whether that stance was ultimately right.

## 7. What the closed-bar gate fix will actually change in production

Given the corrected mechanism (§ Mechanism correction), the closed-bar gate is best described as: **freeze `current_price` (and, indirectly, `predicted_return`, `direction`, `confidence`, and the regime context) to the values computed once at bar close, instead of recomputing them on every websocket tick throughout the hour.** It is not "stop feeding the model malformed data" — the model was never fed forming-bar data in the first place.

Concretely, expect:

- **Fewer signal-generation calls that matter, not fewer inputs.** The live engine can still receive ticks and rewrite the tail candle's OHLCV as often as it does today (this study did not evaluate whether the gate also stops `KlineBuffer` from rewriting the buffer itself, only whether signal evaluation reads the mutating tail) — but whichever mechanism enforces the gate should cause `process_candle`/`generate_signal` to only be invoked, or only be actioned, once per closed hour rather than on every intermediate tick.
- **Lower trade/order churn**, roughly proportional to the ~14.8% actionable flip rate in metric (c) and the gate-crossing distribution in metric (d) — today, up to ~50% of hours cross the 0.05 confidence gate at least once mid-hour, meaning position sizing can toggle on/off within a single nominal hour. A closed-bar gate collapses each hour to exactly one decision, eliminating that intra-hour toggling by construction.
- **No expected change in raw directional accuracy**, per metric (e) — this fix should not be sold as an alpha improvement. Its value is operational (fewer redundant orders, less exchange fee churn, cleaner backtest-live parity for anyone reasoning about "what did the strategy decide this hour") rather than predictive.
- **Interaction with the cross-symbol-scoring fix (#867/#887/#905)**: that fix (native ETHUSDT model live as of 2026-07-05, replacing a foreign-symbol substitute) is a *feature-input* correction — it changed what the model was trained on relative to what it scores. This study's mechanism is a *reference-price* correction — it does not touch model inputs at all, only the denominator used to convert a fixed prediction into a directional call. The two fixes are orthogonal and additive: the cross-symbol fix makes `prediction` itself trustworthy; the closed-bar gate (if implemented) would make the `direction`/`confidence` derived from that prediction stable for the hour instead of tick-dependent. Neither fix compensates for the other's absence.
- **Backtest-live parity**: since the backtest engine already only ever evaluates closed historical bars (there is no "forming candle" concept in a static historical replay), a closed-bar gate in live moves live's *effective* decision cadence into alignment with what backtest has always modeled. Today, backtest results implicitly assume the closed-bar cadence this study shows live does not have — that gap is itself a backtest-live parity issue independent of P&L, worth flagging per `CODE.md`'s parity requirement.

## 8. Reproducibility

- **Model**: `ETHUSDT/basic/2026-07-04_22h_v1` (the `latest` symlink at the time of this study, `src/ml/models/ETHUSDT/basic/latest`).
- **Data window**: ETHUSDT 1h, warmup 2026-05-01 00:00 UTC, study window 2026-05-25 00:00 UTC → 2026-06-29 00:00 UTC (5 weeks, 840 hourly bars). ETHUSDT 1m, same study window.
- **Harness script**: `fliprate_harness.py`, plus `fetch_data.py`/`fetch_1m.py` (data fetch), `sanity_check.py` (§1), `analyze.py` (metrics a-e) — all preserved at `/private/tmp/claude-501/-Users-alex-Sites-ai-trading-bot--claude-worktrees-elated-vaughan-5625e7/d6be54e9-2423-48d6-bda7-7b5e9102452a/scratchpad/fliprate_study/` (the disposable fliprate-study worktree this was developed in has been removed; this scratchpad path is the durable copy). Checkpoint JSONs (`checkpoints/cp_*.json`, per-100-hour) and the combined `all_results.json` (840 hours × 12 decisions = 10,080 decision records) are in the same directory, along with `sanity_check_result.json` and `metrics_result.json`.
- **No look-ahead**: re-verified at both the reconstruction level (each forming-bar snapshot's DataFrame contains only 1m bars from minute 0 through minute *m* of hour H, plus 120 real closed bars strictly before H — nothing from H's remaining minutes or H+1 onward) and the metric level (H+1's realized return is read only for scoring in metric (e), never passed into any `generate_signal`/`detect_regime` call).
- **No fees/execution modeled.** This is a decision-agreement study (does the signal generator's direction/confidence output change), not a P&L backtest — no `CostCalculator`, no position sizing, no order execution was invoked. The "how this could lose money" framing from `CODE.md`'s live-affecting-change checklist does not directly apply here since no strategy change is proposed; the closest analogue is captured in §7's expected-impact discussion above.
- **Fees/slippage**: N/A (no trades simulated).
- **Reused components**: `MLBasicSignalGenerator` and `EnhancedRegimeDetector` are imported directly from `src/strategies/components/ml_signal_generator.py` and `src/regime/enhanced_detector.py` — no reimplementation, no logic duplicated outside `src/engines/shared/` (this study reads decisions from existing shared/strategy-layer code; it does not introduce new financial-math logic).

## 9. Recommendation to pm

**Ready for risk-officer review of the proposed closed-bar gate fix (being built in a separate engineering session), not this session's job to build.** This measurement supports prioritizing that fix: the blast radius is large (two-thirds of hours affected, ~15% actionable-decision flip rate, mostly outright direction reversals) even though the underlying signal has no meaningful directional edge to lose. The risk this creates in production is **operational** (order churn, fee bleed from toggling positions within an hour, noisier live logs for anyone debugging "why did the strategy do X") rather than a hidden alpha loss — risk-officer stress-testing should focus on trade-frequency/fee-churn impact and confirm the gate doesn't introduce a new lag (e.g. a full extra hour of delay before acting on a fresh signal) that trades off against the churn it removes.
