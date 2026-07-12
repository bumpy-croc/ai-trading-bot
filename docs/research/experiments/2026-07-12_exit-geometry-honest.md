# EXIT-GEOMETRY (honest-engine rerun): can trade-management design flip HyperGrowth's expectancy?

**Date**: 2026-07-12
**Researcher**: quant-researcher (Lane B)
**Status**: PREREGISTERED — locked before first result is read. Results appended below the line marked `## RESULTS`.
**Worktree**: `.claude/worktrees/exit-geometry-honest`, branch `claude/exit-geometry-honest`, off `origin/develop @ 79c70aaf` (`.agent-active` sentinel present, per #952's prune-guard).
**Related**:
- `docs/research/experiments/2026-07-04_hypergrowth-exit-geometry.md` — the prior exit-geometry sweep. Its NUMBERS are unreliable (pre-#838 partial-exit-units fix, pre-#867 symbol-wiring fix); its QUESTIONS (full-stop losers dominate net loss; winners give back most of MFE) motivate this rerun on corrected plumbing.
- `docs/research/notes/2026-07-08_hypergrowth-confidence-collapse.md` — HyperGrowth's `FlatRiskManager` + confidence-blind sizer makes it structurally blind to model quality above a low directional-agreement bar. This is why exit/trade-management design — not model quality — is the only lever left to test for this specific live strategy.
- `docs/research/experiments/2026-07-10_target-redesign-tournament-results.md` — three independent tournaments (window #898, architecture #939, target #933) each conclude the price-only 1h feature set is the ceiling on entry quality. Exit geometry is the other half of expectancy and has not been re-tested honestly since #838/#867.
- `docs/research/notes/2026-07-12_live-trade-review.md` (PR #960) — Lane D's live-fill autopsy, **received and incorporated into this prereg before it was locked/committed** (see Sec. 3.1). Key finding: on real prod fills, winners already capture ~72% of MFE (TP/trailing is not the main leak); losers ride ~91% of MAE to the stop (the stop side is the leak). Also flags a live-vs-backtest trade-frequency divergence (matched-config backtest: 6 trades/-0.78% vs. live: 12 trades/+9% over the same period) — see Sec. 7 (limitations).

## 1. Hypothesis

**H1 (mechanism, carried over from the 2026-07-04 doc, now to be re-verified on honest plumbing)**: HyperGrowth's net loss is dominated by a small number of "full-stop" losing trades that ride price to (near) the full `stop_loss_pct` distance, while winning trades exit well before their own peak favorable excursion (MFE). Live-fill evidence (Lane D, Sec. 3.1) now provides an independent, real-fills confirmation of the loser side of this mechanism (91% MAE-ride) and a correction to the winner side (72% MFE-capture live vs. ~47% in the pre-#838/#867 backtest — flagged as a backtest-vs-live divergence to characterize, not assume away).

**H2 (geometry fix — this experiment's actual test)**: Changing ONLY the exit/trade-management configuration of HyperGrowth (stop-loss width, take-profit width, or a bounded holding-time cutoff) — with identical entries, identical model, identical sizing — measurably improves profit factor and/or total return across the primary exam folds (F1–F3) without breaching the portfolio MaxDD cap, and without needing any change to `src/`.

**Falsifiable statement**: An arm is a "promotion candidate, ready for risk-officer review" only if it clears every one of:
1. Total return improves vs. control on **every one of F1, F2, F3** individually (not just averaged — multi-regime robustness, same bar the 2026-07-04 doc and the 2026-07-10 tournament both used).
2. Profit factor improves vs. control on every one of F1, F2, F3.
3. MaxDD stays inside the 20% portfolio hard cap (`risk-limits.json: portfolio.max_drawdown_pct`) on every fold.
4. The return improvement is statistically significant at a Bonferroni-corrected threshold (α = 0.05 / 6 arms = 0.0083, two-sided) on a bootstrap difference-in-means test over each fold's per-trade P&L sequence (10,000 resamples).
5. No fabrication signature (0%-win positive return; near-zero MaxDD with multi-% return; return/win-rate/trade-count inconsistency) — same checklist the tournament reruns use.

If no arm clears all five bars on all three folds, the honest conclusion is **"no geometry arm ships"** or **"promising but not ready"** (partial clearance) — not a forced recommendation. F4 (2026H1) is confirmatory-only per Sec. 4 and never used to grant or deny promotion-candidate status.

## 2. Strategy under test

`hyper_growth` (`src/strategies/hyper_growth.py::create_hyper_growth_strategy`), **live prod config as control**:
- `stop_loss_pct=0.10`, `take_profit_pct=0.30`
- `risk_fraction=0.25`, `base_fraction=0.25` (flat sizing, no confidence/strength scaling — `FlatRiskManager` + `FixedFractionSizer(adjust_for_confidence=False, adjust_for_strength=False)`)
- Partial-exit ladder `[0.08, 0.15, 0.30]` → `[20%, 30%, 50%]` exit sizes (hardcoded, unchanged across every arm)
- Trailing stop: activation 3%, distance 1.5% (hardcoded, unchanged across every arm)
- Breakeven: threshold 5%, buffer 0.8% (hardcoded, unchanged across every arm)
- `max_leverage=1.0` (leverage disabled, matches live)
- Symbol: ETHUSDT, timeframe 1h, signal source `ml`/`basic` (currently-deployed live model via `MLBasicSignalGenerator`, no retraining — see Sec. 7 caveat)
- `ignore_signal_reversal=True` (hold through signal flips, exit only on SL/TP/trailing/time — matches live)

Every arm below changes **only** the exit/trade-management knobs listed; entries, model, and position sizing are byte-identical to control.

## 3. Expressibility audit — what can and cannot be varied without touching `src/`

Per instruction, this experiment uses only the checked-in `src/experiments/runner.py` (`ExperimentRunner`) and `hyper_growth`'s own factory kwargs — no strategy or engine file is modified. Read `hyper_growth.py`, `src/experiments/runner.py`'s `_apply_strategy_attribute` component-target map, and `src/engines/shared/risk_configuration.py`'s `build_trailing_stop_policy`/`build_time_exit_policy` directly to determine what's real:

| Knob | Expressible without `src/` change? | Mechanism |
|---|---|---|
| `stop_loss_pct` | **Yes** | `FlatRiskManager._direct_runtime_overrides = {"stop_loss_pct", "min_confidence"}` — reads `self.stop_loss_pct` directly at trade time; runner setattrs it. Unit-tested (`tests/unit/experiments/test_overrides.py::...G2...`). |
| `take_profit_pct` | **Yes** | `FlatRiskManager` has no TP logic itself, but `create_hyper_growth_strategy` wires it into `strategy._risk_overrides["take_profit_pct"]` via `set_risk_overrides(...)`; `resolve_stop_loss_take_profit_pct` (`src/engines/shared/entry_utils.py`) reads strategy overrides as highest priority and sets `trade.take_profit`, which the exit handler's barrier check honors independently of the partial-exit ladder. Unit-tested. |
| `time_exits.max_holding_hours` | **Yes, via a different channel** | `hyper_growth`'s hardcoded `set_risk_overrides` dict has **no** `"time_exits"` key, so `_build_time_exit_policy`/`build_time_exit_policy` falls through to `risk_manager.params.time_exits` — i.e. the generic `RiskParameters(time_exits=...)` passed to `ExperimentConfig.risk_parameters`, NOT the `parameters`/setattr override path. |
| Trailing-stop `activation_threshold` / `trailing_distance_pct` | **No** | `build_trailing_stop_policy`: `activation = cfg.get("activation_threshold") or params.trailing_activation_threshold` — hyper_growth's `cfg` (from `get_risk_overrides()`) always supplies a truthy `activation_threshold=0.03` and `trailing_distance_pct=0.015`, so the `or`/`cfg.get(...)`-first logic never falls through to any `RiskParameters` value. Not a factory kwarg either (`create_hyper_growth_strategy`'s signature has no trailing-stop parameters). **Would need a `src/strategies/hyper_growth.py` signature change. SKIPPED.** |
| Breakeven `breakeven_threshold` / `breakeven_buffer` | **No** | Same mechanism as trailing stop — strategy-declared `cfg` always wins. **SKIPPED.** |
| Partial-exit `exit_targets` / `exit_sizes` | **No** | `engine.py:330` comment: "Strategy-declared partial_operations win; risk-parameter fallback used only when the strategy has none." hyper_growth always declares them. Not a factory kwarg. **SKIPPED.** |
| A true MFE-conditioned "early-cut if no favorable move within N hours" rule (Lane D's hypothesis 2, Sec. 3.1) | **No** | No existing `ExitHandler`/`RiskManager` policy conditions an exit on MFE-within-a-time-window; only unconditional time cutoffs (`max_holding_hours`) and price-level triggers (SL/TP/trailing) exist. Building this is new logic, not a parameter — a real `src/engines/shared` / `ExitHandler` change. **SKIPPED here; recommended as the top candidate for the next round's prereg** (see Sec. 8). |

**Consequence for arm design**: the mechanism most directly implicated by both the 2026-07-04 backtest finding (winners give back MFE via trailing) and Lane D's live-fill autopsy (losers ride MAE to the stop) is **not fully expressible** at the trailing-stop/breakeven layer without a code change. What IS expressible — stop width, take-profit width, and an unconditional time cutoff — is a narrower slice of "exit geometry" than the ideal design. This experiment tests that narrower, honestly-scoped slice; it does not (and cannot, without patching `src/`) directly test trailing-distance or MFE-conditioned early-cut variants.

### 3.1 Mid-flight evidence incorporated before locking (Lane D, `docs/research/notes/2026-07-12_live-trade-review.md`)

Received while this prereg was still open (not yet committed). Per instruction, arms were reweighted before the first backtest ran (not amended after seeing results): stop-side arms outnumber TP-side arms 4:2 (was 2:2 in the initial internal draft), and a short `max_holding_hours=18` arm was added as the closest **expressible** proxy for Lane D's "early-cut if MFE < ~1.5% within 12–18h" hypothesis — an unconditional time cutoff, NOT a true MFE-conditioned rule (that needs new `src/` logic, per Sec. 3's table, and is deferred to Sec. 8). Every result below also reports MAE-ride fraction (losers) alongside MFE-capture ratio (winners), per Lane D's request.

## 4. Exam windows

Primary (decision-bearing), matching the target-redesign tournament's fold definitions for comparability:
- **F1 = 2023-01-01 → 2023-06-30**
- **F2 = 2024-01-01 → 2024-06-30**
- **F3 = 2025-01-01 → 2025-06-30**

Confirmatory only, never gates a verdict, run **only for the control arm plus any arm that clears the F1–F3 bar** (protects the comparison budget per the dispatch brief — "that window's comparison budget is spent"):
- **F4 = 2026-01-01 → 2026-06-30**

## 5. Arms (control + 6 variants — within the 5–7 arm budget)

All entries/model/sizing identical across every row; only the listed exit knob(s) differ from control.

| Arm | `stop_loss_pct` | `take_profit_pct` | `max_holding_hours` | Rationale |
|---|---|---|---|---|
| `control` | 0.10 | 0.30 | 336 (default) | Live prod config |
| `sl_08` | 0.08 | 0.30 | 336 | Mild stop tightening — sensitivity point |
| `sl_06` | 0.06 | 0.30 | 336 | Moderate stop tightening — re-test of the 2026-07-04 sweep's direction on honest plumbing |
| `sl_04` | 0.04 | 0.30 | 336 | Aggressive stop tightening — directly targets Lane D's "91% MAE-ride to a wide stop" finding |
| `tp_06` | 0.10 | 0.06 | 336 | Tight full-close TP (below the first partial-exit tier of 0.08 — **note**: this bypasses the partial-exit ladder entirely rather than tightening its final tier; see Sec. 6 caveat) |
| `maxhold_18` | 0.10 | 0.30 | 18 | Closest expressible proxy for an early-cut rule (Sec. 3.1); unconditional, not MFE-gated |
| `combo_sl06_tp15` | 0.06 | 0.15 | 336 | Asymmetric combo — changes reward:risk from 3:1 to 2.5:1 while tightening the loss side |

6 variants × 3 primary folds = 18 runs, + control × 3 folds already counted in that 18 (control is one of the 7 rows) = **21 primary runs**, + confirmatory F4 for control and any qualifying arm(s), + 1 determinism recheck (control/F1 run twice). Matches the dispatch brief's ~21-run budget estimate.

## 6. Metrics

Per arm, per fold: total trades, win rate, total return %, annualized return %, max drawdown %, Sharpe, **profit factor** (read from the engine's own `perf_metrics.profit_factor` in the raw `backtester.run()` results dict — not recomputed, per CODE.md's "financial math lives in `src/engines/shared`, never duplicated"), final balance, turnover (trade count as a proxy, fold length fixed), and fee/slippage drag is implicit in `total_return` since `CostCalculator` defaults are **on** for every run (`fee_rate=0.001`, `slippage_rate=0.0005` — never disabled for this study).

**Mechanism metrics** (the "why," not just the "what," per dispatch brief and Lane D's request):
- **MFE-capture ratio** (winners): mean realized return ÷ mean MFE, matching the 2026-07-04 precedent method.
- **MAE-ride fraction** (losers): mean |realized loss| ÷ mean |MAE|. 1.0 ≈ exits essentially at the worst point reached (full-stop behavior); well below 1.0 ≈ the trade recovered/exited before its worst point.

**Caveat on capture-ratio/MAE-ride units**: a same-day smoke test (`control`/`tp_04`/`maxhold_48`, F1, dry run before locking the final arm list) produced capture ratios both below and **above** 1.0 (0.82, 1.55, 1.22). `Trade.pnl_percent` is documented as "sized percentage return" while `Trade.mfe`/`mae` are documented as "peak unrealized profit %" — plausibly the same units, but the 2026-07-04 precedent doc independently found `Trade.pnl` accounting has a disclosed completeness gap around partial exits (un-itemized realized P&L from partial closes, fees deducted from balance but not written back onto the `Trade` record). Capture ratio / MAE-ride fraction below are reported as directional, within-fold-comparable indicators, not as a strict bounded-[0,1] physical quantity — a ratio near or above 1.0 should be read as "captures close to all of, or more than, its own peak" rather than over-interpreted as a precise fraction.

**Caveat on `tp_06`/`combo_sl06_tp15`'s take-profit mechanism**: `take_profit_pct` values below the first partial-exit tier (0.08) do not "tighten the final target while leaving partials in place" — they fire the hard full-position TP check before any partial-exit tier is ever reached, so those two arms are better read as "replace the partial-exit ladder + trailing-stop upside management with a single fixed full-close target," not "the same ladder with a smaller cap." This is disclosed here rather than silently treated as a clean single-variable change.

## 7. Known limitations (disclosed before running, not discovered after)

1. **No fold-matched model retraining — this is a fixed-entries study, not an OOS model-quality claim.** Every fold uses the currently-deployed live ETHUSDT `basic` model (via the model registry `latest` symlink), which per the 2026-07-10 tournament's Amendment 2 finding has a training cutoff of 2026-07-04 — i.e. it has seen data from within and after every fold tested here, including the F4 "confirmatory" window. This makes the **absolute** P&L numbers non-conservative relative to true live performance. It does **not** invalidate the **relative** arm-vs-control comparison, which is this experiment's actual question ("given a fixed, identical set of entries — including whatever the model already knows — does changing only the exit config change expectancy?"), and it mirrors exactly what would happen if this exit config shipped to the live bot today (no retraining involved). Any promotion recommendation from this study inherits this caveat and requires a forward staging-paper validation period before being treated as forecast-grade — independent of, and in addition to, the trade-frequency divergence in point 3 below.
2. **Data-quality gap, non-differential across arms.** A smoke-test run of `control`/F1 logged ~120 `MLBasicSignalGenerator: prediction failed ... Input data contains non-positive values` warnings (~2.8% of bars in a 6-month window) around late March 2023, despite no non-positive raw OHLCV in the cached candles at that timestamp — the failure is in a derived feature, not raw price. Root cause not chased further (out of scope, pre-existing, not introduced by any override tested here). Because every arm shares the identical signal-generation pipeline and identical data, this affects every arm and control equally and does not bias the relative comparison; it may modestly understate absolute trade count/return for all arms in the affected window.
3. **Live-vs-backtest trade-frequency divergence (Lane D, Sec. 3.1).** A matched-config backtest over the same period as Lane D's live sample produced 6 trades/-0.78% vs. live's 12 trades/+9% — a known, disclosed gap consistent with prior forming-bar research (#912), not resolved by this study. Fold-based internal comparisons (arm vs. control, same backtest engine, same window) still stand, but **any arm that clears the promotion bar here still requires staging-paper validation before a live/prod change**, independent of this study's own verdict.
4. **Trailing-stop/breakeven/partial-exit-ladder mechanisms are NOT varied** (Sec. 3) — the mechanism most directly implicated by both the pre-existing backtest finding and Lane D's live evidence on the *winner* side (MFE capture) is held fixed in every arm, including control. This study can only speak to the stop/TP/time-cutoff slice of exit geometry.

## 8. Recommended follow-up (not built here — flagging per instruction, not patching `src/`)

If this round finds no arm clears the promotion bar, or clears it only partially, the most concrete next lever — implied by Sec. 3's expressibility gaps and Lane D's mechanism finding — is a genuine `src/engines/shared`/`ExitHandler` feature: an MFE-conditioned early-cut policy (e.g. "flatten if unrealized MFE has not reached X% within Y hours of entry"), plus making trailing-stop activation/distance and breakeven threshold real `RiskParameters`-driven knobs for `hyper_growth` specifically (currently locked to the strategy's hardcoded `set_risk_overrides` dict). Both would need their own prereg, their own risk-officer sign-off path (this is a `src/` change to money-path-adjacent code), and are explicitly out of scope for this experiment.

## 9. Determinism spot-check

`control`/F1 re-run once, back-to-back, same config. Must match on `total_trades`, `total_return`, `profit_factor` to reported precision (post-#923 deterministic inference — this should be exact, not approximate).

---

## RESULTS

(Appended below this line only after every primary-fold run completes. Nothing above this line is edited post-hoc.)
