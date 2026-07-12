# EXIT-GEOMETRY ROUND 2: testing the arms round 1 could not express

**Date**: 2026-07-12
**Researcher**: quant-researcher
**Status**: PREREGISTERED — locked before first result is read. Results appended below the line marked `## RESULTS`.
**Worktree**: `.claude/worktrees/exit-geometry-round2`, branch `claude/exit-geometry-round2`, off `origin/develop @ 3721a835` (`.agent-active` sentinel present).
**Related**:
- `docs/research/experiments/2026-07-12_exit-geometry-honest.md` (round 1, #970/#971, CLOSED) — NO-GO on all 6 arms (stop-tightening monotonically harmful; `tp_06` directionally positive on all 3 folds but never statistically significant at n=28-70 trades/fold). Round 1's own Sec. 3 expressibility audit found trailing-stop distance/activation, breakeven threshold/buffer, and any MFE-conditioned early-cut rule were **not expressible** without a `src/` change — those are exactly what this round tests.
- `docs/research/notes/2026-07-12_live-trade-review.md` (Lane D) — the motivating live evidence: live winners capture ~72% of MFE (range 50-99%), live losers ride ~91% of MAE to the wide stop. H1: a tighter trailing stop could lock in more of the winners' MFE. H2: a trade that has built <1.5% MFE within ~12-18h is unlikely to recover and could be cut early without truncating the win-streak trades.
- `docs/research/2026-07-12_returns-levers-synthesis.md` — Board rollup ranking exit/trade-management as the #1 return lever, with the explicit caveat that the effect size found in round 1 (`tp_06`) is small (~1pp return, ~0.1 PF) and this round's job is to (a) properly test the previously-inexpressible mechanisms and (b) give `tp_06` more statistical power via more folds.
- PR #976 (merged to develop today, closes #971's follow-up) — makes `early_cut_mfe_threshold_pct`/`early_cut_evaluation_window_hours` (new `EarlyCutPolicy`), and `enable_trailing_stop`/`trailing_activation_threshold`/`trailing_distance_pct`/`breakeven_threshold`/`breakeven_buffer` (now key-presence resolved, explicit `None` honored) real `create_hyper_growth_strategy` factory kwargs. Its own regression evidence reproduces round 1's control numbers bit-identically on F1-F3 (used below as this round's validity gate). GH #977 (open): live early-cut fires on wall-clock, backtest fires on bar close — same divergence class as existing time exits, decision-identical, execution-timing only; noted here per #977's own request, revisited in Sec. 7.

## 1. Hypothesis

**H1 (carried from Lane D)**: A trailing stop calibrated to activate earlier and/or trail tighter than HyperGrowth's current 3%-activation/1.5%-distance config would lock in more of a winner's peak favorable excursion, improving realized return without a matching increase in loss frequency.

**H2 (carried from Lane D)**: A trade that has not built at least a small favorable excursion within a bounded window after entry (the round-1 `maxhold_18` proxy suggested ~12-18h) is unlikely to recover; an MFE-conditioned early-cut rule (not an unconditional time cutoff — the actual mechanism, now buildable via PR #976) could cut such losses without truncating the winners that build their gains over multiple days.

**H3 (ablation, new to round 2)**: Round 1 could not test whether the winner-side capture behavior (0.73-0.82 across folds) is being driven by the trailing-distance mechanism, the breakeven-move mechanism, both, or neither (i.e. TP/SL/time exits alone would produce the same capture ratio). Decomposing control's combined trailing+breakeven config into two single-mechanism arms answers this directly.

**Falsifiable statement** (unchanged structure from round 1, thresholds identical where applicable): an arm is a **staging-trial candidate** only if it clears every one of:
1. Bonferroni-significant return improvement (two-sided bootstrap diff-in-means on per-trade P&L, 10,000 resamples, α = 0.05/6 = 0.0083) on **at least 2 of the 3 primary folds** (F1, F2, F3) — loosened from round 1's "all 3 folds" bar because this round adds 2 extension folds for power and a perfection-on-3-folds bar becomes needlessly strict once 5 folds are being read in aggregate.
2. Aggregate profit-factor AND aggregate total-return improve vs. control, where "aggregate" = computed over the pooled per-trade P&L sequence across all 5 folds tested (F1-F3 + the two extension folds), not a simple average of per-fold headline numbers — see Sec. 6 for the exact pooling definition, pre-committed here before any result is read.
3. No fold — any of the 5 — has MaxDD worse than control's MaxDD on that same fold by more than 2.0 percentage points.
4. No fabrication signature (0%-win positive return; near-zero MaxDD with multi-% return; return/win-rate/trade-count inconsistency) — same checklist round 1 and the tournament reruns use.

If no arm clears all four bars, the honest conclusion is **"no arm ships"** or **"promising but not ready"** (partial clearance) — not a forced recommendation. Anything that does clear the bar is a **staging-trial candidate only** — it goes to `risk-officer` + staging paper-trading, never straight to prod, per the standing rule.

## 2. Strategy under test

`hyper_growth` (`src/strategies/hyper_growth.py::create_hyper_growth_strategy`), **live prod config as control** — identical to round 1's control:
- `stop_loss_pct=0.10`, `take_profit_pct=0.30`
- `risk_fraction=0.25`, `base_fraction=0.25` (flat sizing, `FlatRiskManager` + `FixedFractionSizer(adjust_for_confidence=False, adjust_for_strength=False)`)
- Partial-exit ladder `[0.08, 0.15, 0.30]` → `[20%, 30%, 50%]` (hardcoded, unchanged across every arm — still not a factory kwarg after PR #976, out of scope for this round too)
- Trailing stop: activation 3%, distance 1.5%; breakeven: threshold 5%, buffer 0.8% (now real factory kwargs, still equal to control's defaults — only the arms below change them)
- No early cut (control default: OFF)
- `max_leverage=1.0` (leverage disabled, matches live)
- Symbol: ETHUSDT, timeframe 1h, signal source `ml`/`basic` (currently-deployed live model via `MLBasicSignalGenerator`, no retraining — same fixed-entries caveat as round 1, restated in Sec. 7)
- `ignore_signal_reversal=True`

Every arm below changes **only** the exit/trade-management knobs listed; entries, model, and position sizing are byte-identical to control.

## 3. Model-contamination is controlled — restated honestly (per round 1's own precedent)

Every arm uses the identical, currently-deployed ETHUSDT `basic` model (registry `latest`, no retraining). Per the `2026-07-10` target-redesign tournament's finding, this model's training cutoff (2026-07-04) is **after** every fold tested here, including the two new extension folds — this makes **absolute** P&L numbers non-conservative relative to true live/historical performance (the model has, in a leakage sense, "seen the future" relative to 2021-2025 in the sense that its architecture/weights reflect knowledge of the full historical relationship, not that it was trained on these exact rows out of order — no walk-forward retraining is being claimed here). This is disclosed, not hidden.

**What stays valid despite this**: because every arm shares the identical entries (same model, same signal generator, same sizing) on every fold, the **BETWEEN-ARM comparison** — "given a fixed, identical set of entries, does changing only the exit/trade-management config change expectancy?" — is unaffected by the absolute-return inflation. The inflation is a level shift common to every arm and every fold; it does not selectively favor one arm's exit-geometry variant over another's. This is the identical logical argument round 1's prereg made (Sec. 7 there) and it applies unchanged here. Any arm that clears Sec. 1's bar still requires a forward staging-paper validation period before being treated as forecast-grade, independent of and in addition to this study's own verdict.

## 4. Expressibility audit — what PR #976 makes real, and one gotcha found before locking

Per PR #976's body (config surface) and direct verification against `src/strategies/hyper_growth.py` and `src/engines/shared/risk_configuration.py` on this worktree's `origin/develop @ 3721a835`:

| Knob | Mechanism | Verified buildable? |
|---|---|---|
| `early_cut_mfe_threshold_pct` + `early_cut_evaluation_window_hours` (factory kwargs, paired) | `build_early_cut_policy` reads `strategy.get_risk_overrides()["early_cut"]`, which `create_hyper_growth_strategy` populates only when both kwargs are non-None | **Yes** — constructed and inspected directly for all 3 early-cut arms below; `EarlyCutPolicy.validate_for_timeframe("1h")` passes for all three windows (12h/18h/24h — all > 1 bar and whole multiples of the 1h bar interval, satisfying PR #976's review-added guard) |
| `trailing_only` (trailing distance kept, breakeven disabled) | Factory kwarg `breakeven_threshold=None` → `get_risk_overrides()["trailing_stop"]["breakeven_threshold"] = None` → `build_trailing_stop_policy`'s key-presence resolution honors the explicit `None` (`cfg_has_breakeven=False`), `trailing_distance_pct` stays at the default 0.015 | **Yes** — verified directly: `build_trailing_stop_policy` returns `TrailingStopPolicy(activation_threshold=0.03, trailing_distance_pct=0.015, breakeven_threshold=None, ...)` |
| `breakeven_only` (breakeven kept, trailing distance disabled) | Factory kwarg `trailing_distance_pct=None` | **Yes, but only after fixing a real gotcha** — see below |

**Gotcha found and fixed before locking**: naively passing only `trailing_distance_pct=None` to the factory does **not** produce a "no distance-based trailing" policy. `RiskManager`'s default `RiskParameters.trailing_atr_multiplier` is `1.5` (a pre-existing default, unrelated to this study), and `hyper_growth`'s `trailing_stop` overrides dict has no `trailing_distance_atr_mult` key — so `build_trailing_stop_policy`'s key-presence resolution falls through to that params default and builds `TrailingStopPolicy(trailing_distance_pct=None, atr_multiplier=1.5, ...)`. `src/engines/shared/trailing_stop_manager.py::_calculate_trailing_distance` tries **percentage-based distance first**; only when `trailing_distance_pct is None` does it fall through to the ATR branch (`atr_multiplier * ATR-from-df`) — meaning for control and every arm where `trailing_distance_pct` stays non-`None`, this `atr_multiplier=1.5` leak is genuinely inert (dead code, never reached), exactly as PR #976's own body claims ("deliberately kept bit-identical...it is inert at runtime: TrailingStopManager prefers pct distance"). But for `breakeven_only`, where `trailing_distance_pct` is deliberately set to `None`, that fallback branch **is** reached — so the naive construction would have silently built an ATR-based trailing policy (1.5× ATR) instead of the intended "breakeven move only, no distance-based trailing at all."

**Fix, verified directly** (`build_trailing_stop_policy` called with each candidate `RiskManager`): passing `risk_parameters={"trailing_atr_multiplier": None}` (the same `ExperimentConfig.risk_parameters` → `RiskParameters(**...)` channel round 1 used for `time_exits`) neutralizes the fallback — confirmed: `TrailingStopPolicy(trailing_distance_pct=None, atr_multiplier=None, breakeven_threshold=0.05, breakeven_buffer=0.008)`. Because this setting is provably inert for every other arm (pct branch always wins when `trailing_distance_pct` is non-`None`), it is applied **globally, to every arm including control**, for methodological cleanliness (one fewer moving part between arms) with zero effect on any arm except `breakeven_only`. This does not threaten the validity gate (Sec. 5): control's runtime behavior is unchanged because the pct branch short-circuits before `atr_multiplier` is ever read.

**Still not expressible / out of scope for this round** (unchanged from round 1's Sec. 3): the partial-exit ladder's targets/sizes and scale-in thresholds remain hardcoded — not a factory kwarg, not varied here.

## 5. Validity gate — reproduce round 1's control numbers bit-identically on F1-F3

Before reading any arm result, `control` is run on F1/F2/F3 and compared against round 1's published control numbers (`docs/research/experiments/2026-07-12_exit-geometry-honest.md` RESULTS and PR #976's own regression table):

| Fold | round-1 trades | round-1 return % | round-1 PF | round-1 MaxDD % |
|---|---|---|---|---|
| F1 2023H1 | 31 | -2.8817542... | 0.6619... (0.662) | 4.85 |
| F2 2024H1 | 46 | -6.643012775559887 | 0.528 | 7.65 |
| F3 2025H1 | 70 | -11.557713074441967 | 0.446 | 12.78 |

If this round's `control` run does not match on `total_trades` and `total_return` to full float precision on all three folds, **the run is invalid and must be re-diagnosed before any arm comparison is trusted** — this is a hard gate, not a sanity nicety, because it is the only guarantee that this fresh worktree's cache/model/engine state genuinely matches the state round 1 and PR #976 both verified against.

### 5.1 ADDENDUM — validity gate FAILED as originally stated; root cause found, gate revised before any arm was run

**This addendum is written after Sec. 1-11 above were locked and committed (`git log`: `a43e7a1a`), and before any arm-vs-control comparison was read** — it documents a methodology-blocking discovery made while executing Sec. 5's own gate, not a post-hoc adjustment to arm results. No arm has been run at the time this addendum is written; only `control` diagnostics.

**Finding**: a bare `create_hyper_growth_strategy()` call (zero kwargs — exactly what round 1's checked-in `experiments/exit_geometry_sweep.py` does for its `control` arm, since it never sets `factory_kwargs`) resolves `signal_generator.symbol = "BTCUSDT"` (verified by direct construction and inspection). `MLBasicSignalGenerator.DEFAULT_SYMBOL = "BTCUSDT"`, and a BTCUSDT `basic` model genuinely exists in the registry, so there is no fail-fast — the strategy silently scores **ETHUSDT price candles with the BTCUSDT model** instead of ETHUSDT's own deployed model. `src/experiments/runner.py::ExperimentRunner._load_strategy` never threads `config.symbol` into the strategy factory — unlike `src/engines/live/runner.py::load_strategy` and `cli/commands/backtest.py::_load_strategy`, which both explicitly do this "so model registry selection matches" (their own docstrings). This is a gap in the shared research-driver harness, not something round 1 introduced.

**Isolated and proven by direct test** (four combinations run against this worktree, `develop @ 3721a835`, `control`/F1):

| Engine code | Symbol threading | Result |
|---|---|---|
| Main checkout's stale `src/` (see below) | `symbol=None` (BTCUSDT model) | 31 trades, **+45.65%**, PF 0.146, MaxDD 4.01% |
| This worktree's `src/` (develop) | `symbol="ETHUSDT"` (correct) | 29 trades, **-1.69%**, PF 0.797, MaxDD 4.31% (deterministic: reproduced bit-identically twice) |
| **This worktree's `src/` (develop)** | **`symbol=None` (BTCUSDT model)** | **31 trades, -2.881754238240264%, PF 0.6623077792846185, MaxDD 4.846836585357486%** |

The third row is an **exact, full-float-precision match** to round 1's published control (`-2.881754238240264` to the last digit) and to PR #976's own regression table. **Conclusion: round 1's entire exit-geometry-honest study — and PR #976's "reproduces round 1 bit-identically" regression-evidence claim, which reused the identical script — ran the correct, honest, post-#838/#867 `develop` engine code, but scored every fold's ETHUSDT candles with the BTCUSDT `basic` model, not the currently-deployed ETHUSDT model as the document states throughout ("currently-deployed live ETHUSDT `basic` model", Sec. 2).** This is a genuine, previously-undetected validity bug, not a data/cache/non-determinism artifact (both the buggy and corrected configurations are independently deterministic).

**A second, separate bug was found and fixed along the way** (first row above): invoking a driver script as `python3 experiments/foo.py` (a script path) sets `sys.path[0]` to the script's own directory, not the caller's cwd — falling through to whatever editable "atb" install is on site-packages, which points at **the main checkout** (`/Users/alex/Sites/ai-trading-bot`, branch `main`, months behind `develop`), not the worktree actually being tested. This worktree's own `experiments/exit_geometry_round2_sweep.py` now guards against this with an explicit `sys.path.insert(0, ...)` at the top of the file (Sec. "script" below). This bug is orthogonal to the symbol-threading one; it happened to be present in how *I* first invoked the script and does not appear to explain round 1's specific numbers (the exact-match row above uses the correct worktree engine code, not the shadowed one) — flagged and fixed here so it cannot corrupt this round's own runs, and reported separately below since it could affect any other `experiments/*.py` script invoked the same way.

**What this means for round 1's verdict**: because every arm in round 1 shared the identical symbol-threading gap, the **relative, between-arm comparison** that round 1's own NO-GO verdict rests on is not automatically invalidated by this — the same "model contamination is controlled because entries are identical across arms" argument round 1's own Sec. 7 makes for the no-retraining caveat extends to this bug (every arm scored the same wrong model the same way). But the **absolute numbers, the "currently-deployed live ETHUSDT model" characterization, and the mechanism metrics (MFE-capture/MAE-ride, computed on BTCUSDT-model-driven trades)** are all suspect, and staging validation of any arm that would have cleared round 1's bar must be re-run against the correct model before being trusted. This is escalated as its own finding (Sec. 12 below), not silently absorbed into this round's numbers.

**Revised validity gate for this round**: since round 1's published numbers are now understood to reflect a bug rather than a ground truth, this round does **not** attempt to reproduce them. Instead, `control` is re-established as a **new, correct baseline** — worktree `develop` engine code (guarded by the `sys.path` fix) with `symbol="ETHUSDT"` explicitly threaded into every arm's `factory_kwargs` (all arms in Sec. 7 below carry this correction) — and the gate is: **the corrected `control` arm must be internally deterministic (re-run twice, bit-identical)**, which is verified in Sec. 9. All arm-vs-control comparisons in this round are against this corrected baseline, not round 1's published numbers.

## 6. Exam windows

**Primary** (decision-bearing for the "≥2 of 3" significance bar):
- **F1 = 2023-01-01 → 2023-06-30**
- **F2 = 2024-01-01 → 2024-06-30**
- **F3 = 2025-01-01 → 2025-06-30**

**Power-extension** (new to round 2 — reported in the same aggregate, per dispatch brief; NOT separately gating the "≥2 of 3 primary" significance bar, but every fold — primary or extension — is checked individually against the MaxDD-worsening cap in Sec. 1 bar 3, and both extension folds are included in the pooled "aggregate" calculation in bar 2):
- **F0a = 2021-01-03 → 2021-06-30**
- **F0b = 2022-01-03 → 2022-06-30**

No F4/2026H1 confirmatory fold this round — round 1's F4 budget was already spent (per the returns-levers synthesis's own note: "that window's comparison budget is spent") and this round's 2021/2022 extension folds serve the same statistical-power purpose without re-touching it.

**Aggregate pooling definition (bar 2 of Sec. 1), locked here before any run**:
- **Aggregate return improvement**: arithmetic mean, across all 5 folds tested, of (arm's `total_return` on that fold − control's `total_return` on that same fold). Each fold is an independently-initialized $85 backtest; this is a mean of 5 independent deltas, not a compounded quantity.
- **Aggregate profit factor improvement**: pool every trade's `pnl_percent` from all 5 folds into one combined list (per arm, and separately for control), then compute `PF = sum(positive pnl_percents) / abs(sum(negative pnl_percents))` on each pooled list. Compare arm's pooled PF vs. control's pooled PF.

## 7. Arms (control + 6 variants — 7 rows × 5 folds = 35 primary runs)

All entries/model/sizing identical across every row; `trailing_atr_multiplier=None` applied globally per Sec. 4. Only the listed knob(s) differ from control.

| Arm | Factory kwargs (delta from control) | Rationale |
|---|---|---|
| `control` | none | Live prod config |
| `early_cut_1p5_12h` | `early_cut_mfe_threshold_pct=0.015, early_cut_evaluation_window_hours=12` | H2 at the tightest live-autopsy-suggested window |
| `early_cut_1p5_18h` | `early_cut_mfe_threshold_pct=0.015, early_cut_evaluation_window_hours=18` | H2 at the widest live-autopsy-suggested window; also the window closest to round 1's `maxhold_18` proxy, for a direct "unconditional cutoff vs. MFE-conditioned cutoff" contrast on the same clock |
| `early_cut_1p0_24h` | `early_cut_mfe_threshold_pct=0.01, early_cut_evaluation_window_hours=24` | Looser threshold, longer window — tests whether a more forgiving early-cut rule still helps without over-truncating slow-building winners (trade 9/11/12 in the live autopsy took days to build MFE). **Pre-committed trim target**: if pace requires dropping an arm, this one drops first (it is the third point on the same H2 axis, the least novel of the three). |
| `trailing_only` | `breakeven_threshold=None` (trailing stays at defaults 3%/1.5%) | H1/H3 ablation: does the trailing-distance mechanism alone reproduce control's winner-capture behavior, without the breakeven move? |
| `breakeven_only` | `trailing_distance_pct=None` (breakeven stays at defaults 5%/0.8%; `trailing_atr_multiplier=None` globally per Sec. 4 fix) | H1/H3 ablation: does the breakeven move alone reproduce control's winner-capture behavior, without active distance-trailing? |
| `tp_06_rerun` | `take_profit_pct=0.06` | Round 1's only directionally-positive arm (all 3 primary folds, never significant at 28-70 trades/fold) — rerun with 2 additional folds for power, per the returns-levers synthesis's explicit recommendation |

**Deliberately not run this round**: the combo arm (`early_cut_1.5%/18h` + `tp_06`) mentioned as optional in the dispatch brief. The 6-variant set above already reaches the ~35-run budget (7 rows × 5 folds); adding a 7th variant would push to 40 runs, and a combo arm is only interpretable once the two individual mechanisms it combines have their own independent verdicts. Deferred to a round 3 if either `early_cut_*` or `tp_06_rerun` clears the Sec. 1 bar independently.

## 8. Metrics

Per arm, per fold: total trades, win rate, total return %, max drawdown %, Sharpe, **profit factor** (read from `backtester.run()`'s own `perf_metrics.profit_factor` / results dict — never recomputed, per CODE.md), final balance, per-trade P&L sequence (`trade_pnl_pcts`, needed for the bootstrap test and the aggregate PF pooling in Sec. 6).

**Mechanism metrics** (unchanged method from round 1):
- **MFE-capture ratio** (winners): mean realized return ÷ mean MFE.
- **MAE-ride fraction** (losers): mean |realized loss| ÷ mean |MAE|.

**New this round — early-cut mechanism metrics, from PR #976's exam metadata (no recomputation, read directly from `Trade.metadata["early_cut_window_mfe_pct"]` and `exit_reason`)**:
- **Cut rate**: fraction of an early-cut arm's total trades whose `exit_reason` starts with `"Early cut"`.
- **Window MFE distribution of cut trades**: descriptive (mean/median) of `metadata["early_cut_window_mfe_pct"]` for cut trades — sanity check that cuts are firing near, not far below, the configured threshold.
- **Cut-precision (matched-entry proxy) — the pre-committed interpretation rule for "fraction of cut trades that would have ended losers"**: A true counterfactual ("what would this exact trade have done had it not been cut?") is not directly available — cutting a trade early can change the timing of the *next* entry (the strategy holds one position at a time), so an early-cut arm's trade sequence can diverge from control's after the first cut. The honest, pre-committed proxy used here: for each cut trade in an early-cut arm, look up the control arm's trade in the **same fold** with an identical `entry_time` (bit-for-bit match — before any divergence, entries are driven by the same deterministic signal generator and are identical across arms). If a match exists, record whether that control-arm trade's `pnl_percent < 0` (i.e., the same entry, left to run under control's exit rules, would have closed as a loser). **Cut-precision = (# matched cut trades whose control counterpart was a loser) / (# matched cut trades)**. Trades with no control match (entries that only exist in the early-cut arm's sequence because an earlier cut freed up capital sooner) are excluded from the ratio and reported separately as a **match rate** (matched / total cut trades) so the metric's coverage is disclosed, not silently assumed complete. This is descriptive evidence for interpreting the mechanism, not a promotion-gating metric — Sec. 1's bars are the only gate.

## 9. Determinism spot-check

`control`/F1 re-run once, back-to-back, same config. Must match on `total_trades`, `total_return`, `profit_factor` to reported precision (post-#923 deterministic inference — exact, not approximate).

## 10. Known limitations (disclosed before running)

1. **Fixed-entries study, not an OOS model-quality claim** — restated in Sec. 3.
2. **Live-vs-backtest trade-frequency divergence** (Lane D, round 1 Sec. 7 point 3) is a known, disclosed, unresolved gap — this study's internal arm-vs-control comparisons still stand (same engine, same window, same divergence affecting every arm equally), but any arm clearing the Sec. 1 bar still requires staging-paper validation before any live/prod change.
3. **Live early-cut fires on wall-clock, backtest fires on bar close (GH #977)** — same divergence class as the pre-existing time-exit check; affects only *when within a window* a cut executes and at what price, never *whether* the frozen-window MFE decision itself differs (the decision is computed from the same completed-bar history in both engines). Not expected to matter for this round's arm-vs-control backtest comparison (both run in the same engine); flagged here because any early-cut arm that clears the bar and proceeds to staging validation must have this divergence checked empirically, not assumed negligible, before being read as forecast-grade.
4. **Earlier folds have smaller effective training context for the model** and the round-1 doc found a non-differential ~2.8%-of-bars signal-generator failure rate in one tested window (Q1 2023) — worth checking the new 2021/2022 folds don't show a materially worse rate, reported descriptively in the results below, not gating (affects every arm equally, per round 1's own precedent).
5. **Data-quality**: cache prefilled fresh in this worktree (`atb data prefill-cache --symbols ETHUSDT --timeframes 1h --start 2020-11-01 --end 2025-07-05`) rather than reused from round 1's (now-pruned) worktree cache. The validity gate in Sec. 5 is the check that this fresh cache/environment reproduces the same numbers.

## 12. Escalation — round 1's validity is now in question (filed, not silently absorbed)

Per Sec. 5.1's finding, two GitHub issues are filed alongside this preregistration:
1. **`ExperimentRunner`/research-driver symbol-threading gap** (`src/experiments/runner.py::_load_strategy`) — any past or future `experiments/*.py` script that constructs an ML strategy via the bare `ExperimentRunner` machinery without manually adding `symbol` to `factory_kwargs` silently scores the configured trading symbol's candles with whatever model `MLBasicSignalGenerator.DEFAULT_SYMBOL` ("BTCUSDT") resolves to. This is a systemic gap, not specific to hyper_growth or this study — it affects any prior experiment using this pattern for a non-BTCUSDT symbol.
2. **Round 1 (`docs/research/experiments/2026-07-12_exit-geometry-honest.md`, #970/#971, PR #976's regression-evidence table) requires re-verification** — its published control/arm numbers are now known to reflect BTCUSDT's model scoring ETHUSDT candles, not "the currently-deployed live ETHUSDT model" as stated throughout. The relative arm-vs-control NO-GO verdict likely still holds (same contamination-is-controlled argument as the no-retraining caveat — every arm shared the identical bug), but this needs an explicit re-run to confirm before anyone treats round 1's specific numbers, or its mechanism metrics (MFE-capture/MAE-ride), as ground truth. **Not re-run in this document** — round 2 is scoped to the arms round 1 could not express, not a full re-verification of round 1's own 6 arms; that re-run is filed as separate follow-up work.

A third, narrower issue is filed for the `sys.path` shadowing bug found in Sec. 5.1 (invoking any `experiments/*.py` script as a path, rather than fixing the caller's `PYTHONPATH`/cwd handling, can silently import the main checkout's stale `src/`) — orthogonal to the symbol bug, fixed locally in this round's own script, flagged for anyone else invoking these scripts the same way.

## 13. Promotion rule (pre-committed, restated from the dispatch brief)

An arm is a **staging-trial candidate** only if it clears all four bars in Sec. 1. Any such arm goes to `risk-officer` for stress-testing (drawdown scenarios, correlation, regime-shift behavior) and then staging paper-trading — **never straight to prod**, regardless of how clean the backtest result looks, per the standing autonomy-envelope rule for anything the daemon itself has not independently validated live.

---

## RESULTS

_(to be appended after all runs complete)_
