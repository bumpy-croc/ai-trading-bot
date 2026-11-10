# Experiment: Volatility-Targeted Position Sizing for HyperGrowth (Phase 1, Vol-Regime Program #1)

**Author**: quant-researcher
**Status**: PREREGISTERED — locked before any statistic is computed. Do not edit thresholds after seeing results; corrections are new dated sections.
**Program**: Vol/regime program (this doc's sibling `agents/research/vol-regime-program.md`), successor to the 2026-08-13 model-free scan (#1105).
**Blocked on**: `#1106` (closed-candle-gating parity fix) for any *promotion* decision. See §10 for exactly what may and may not run before it lands.

---

## 0. Why this experiment, and what it is not

The 2026-08-13 model-free predictability scan (#1105, `docs/research/experiments/2026-08-13_model-free-predictability-scan.md`) found **zero of 20 (symbol, timeframe) cells** graduate on directional prediction — no cell's momentum or mean-reversion baseline clears the measured ~11bps round-trip cost after BH-FDR correction. But the same scan found **ARCH volatility clustering significant in 20/20 cells** (`docs/research/experiments/scripts/output/2026-08-13_fdr_grid.csv`), the single most universal, robust finding in the whole grid. Volatility itself is highly autocorrelated in every symbol/timeframe tested; direction is not.

This experiment does **not** attempt to predict direction. It tests whether **sizing positions inversely to realized volatility** — a purely defensive, direction-agnostic response to a finding that is not in dispute — improves risk-adjusted return and reduces drawdown for the existing HyperGrowth entries, without changing what HyperGrowth decides to buy or sell or when.

**Why this is tractable now, cheaply**: `src/strategies/components/position_sizer.py::VolatilityTargetSizer` already exists (built under GH #805, closed), is already wired to consume `regime.metadata["atr_percentile"]` from the `EnhancedRegimeDetector`, already lives in the parity-safe component layer (used identically by both engines per `docs/architecture.md`'s `SignalGenerator`/`RiskManager`/`PositionSizer` model), and HyperGrowth's `create_hyper_growth_strategy` (`src/strategies/hyper_growth.py`) already instantiates an `EnhancedRegimeDetector` and threads regime context through `LeveragedPositionSizer`. Wrapping the strategy's current `FixedFractionSizer` base in `VolatilityTargetSizer` is a **sizing-config change, not a new `src/` feature** — this experiment can run without shipping new code, only a new strategy-construction call using an already-reviewed class.

**Relationship to `2026-08-25_model-magnitude-vol-signal-prereg.md`** (sibling document, program item #0): GH #1067's Phase 1 gate (PR #1116) found the deployed ETHUSDT model's `|predicted_return|` carries no directional signal (replicated null) but does correlate with `|realized_return|` at Spearman rho=0.204, p=1.5e-12 — a third independent line pointing at second-moment, not first-moment, structure. That sibling document tests whether this generalizes beyond the one model/window it was found on. **This experiment does not wait on that result** — `VolatilityTargetSizer` already consumes `atr_percentile`, a realized-volatility measure, which is a different (and already-built) signal source from the model's predicted-magnitude. If the sibling document's generalization check succeeds, a natural follow-up (explicitly deferred, not run here) would test blending or swapping in the model-magnitude signal alongside or instead of ATR-percentile — this experiment's ATR-percentile-based result stands as a baseline for that comparison either way.

**Relationship to GH #1067** (open, "FlatRiskManager is conviction-blind... evaluate replacement"): #1067 asks a related but distinct question — whether HyperGrowth's sizer should scale with the *model's own confidence output* (`ConfidenceWeightedSizer`, Kelly variants), which is a directional-information-preserving claim, contingent on the model's confidence actually being calibrated. This experiment deliberately does not test that; it tests scaling by *realized market volatility*, which requires no assumption about model quality or calibration at all — the model-free scan's ARCH finding holds regardless of what any model outputs. #1067's evaluation item 4 ("does variable sizing change the drawdown profile favourably or simply add variance — a sizer that scales into a 51%-accurate signal amplifies noise") is the same concern Arm C (§4) is built to isolate, applied here to volatility rather than confidence. This experiment answers a piece of #1067's question space without closing #1067 itself.

## 1. Hypotheses

**H0 (null)**: Wrapping HyperGrowth's `FixedFractionSizer` in `VolatilityTargetSizer` (inverse-ATR-percentile scaling) does not improve risk-adjusted return (Sortino) or reduce MaxDD relative to the current flat-fraction sizer, holding entries identical. Any apparent improvement is not distinguishable from the effect of simply running at a lower average exposure (a strawman: uniformly de-risking would also lower MaxDD).

**H1 (vol-targeting hypothesis)**: Because realized volatility clusters in time (ARCH significant in 20/20 cells, this is not new evidence being fished for — it is the input to this design), scaling position size inversely to the current ATR-percentile reduces the size of trades entered into volatility regimes that are more likely to produce large adverse moves, at zero cost to the count or timing of entries. This should show up as: (a) lower MaxDD than the flat-fraction control at comparable or better total return, and/or (b) higher Sortino/Sharpe at materially lower average dollar-volatility per trade — the "constant risk per trade" property the sizer is designed to produce.

- *Mechanism if true*: ARCH-clustering means "this bar's volatility is informative about the next several bars' volatility" even though it says nothing about direction. A sizer that reads that signal and shrinks size accordingly removes exposure precisely when adverse moves (which scale with volatility) are more likely to be large, without needing to know which direction those moves go.
- *Falsified if*: vol-targeted sizing does not beat a **volatility-matched flat-fraction control** (see §4, control B) on both MaxDD and Sortino across the pre-registered exam windows, or the improvement is not distinguishable from H0's uniform-de-risking strawman.

## 2. Metric

**Primary**: Sortino ratio and MaxDD (`drawdown_cap_mode=measure`, see §7) on the frozen out-of-sample exam window (§5), HyperGrowth/ETHUSDT/1h, prod-matched risk params, fees/slippage on (`CostCalculator` defaults, never disabled).

**Secondary** (reported, not decisive): total return, Sharpe, win rate, trade count, average dollar-notional per trade (the sizer's own diagnostic — should show materially lower variance across trades than the flat-fraction control if the mechanism is working as designed), `vol_target_applied` fraction (from `VolatilityTargetSizer.get_last_sizing_metrics()` — confirms the sizer actually activated rather than passing through due to missing regime metadata).

Trade-count floor: **≥15 trades** in the exam window per arm, per the skill's anti-p-hacking rule. HyperGrowth's honest 365d backtest produced 104 trades (`docs/research/experiments/2026-07-04_hypergrowth-365d-drawdown-stress-review.md`), so this is expected to clear comfortably, but is checked and reported, not assumed.

## 3. Success threshold (pre-committed, numeric)

The vol-targeted arm (Arm B, §4) **beats the control (Arm A)** if, on the frozen OOS exam window:
- MaxDD is lower by **≥2 percentage points**, AND
- Sortino is higher or within **0.05** of the control (i.e., the drawdown reduction is not purely bought by giving up all upside), AND
- `early_stopped == False` for both arms (or, if `True` for the control under `enforce` mode, the comparison is re-run under `measure` mode for both arms and reported as such per §7).

If MaxDD improves but Sortino drops by more than 0.05, this is reported as "risk reduction at a Sharpe/Sortino cost" — a legitimate but different finding from a clean win, and is NOT reported as "supported" without that qualifier.

Statistical bar: given ~104 trades/365d historically, a full exam window (see §5) is expected to produce on the order of 150-250 trades. Two-arm return-distribution comparison uses a paired bootstrap (same entries, different sizing → paired by construction) at **Bonferroni-corrected** significance (this is a confirmatory 2-arm test, not a screen, so Bonferroni applies per the skill's guidance, not BH-FDR) across the 2 primary metrics (MaxDD, Sortino) → alpha = 0.025 per metric.

## 4. Arms (pre-committed, no post-hoc arm added)

- **Arm A (control)**: current live sizing — `FixedFractionSizer(fraction=0.25)` → `LeveragedPositionSizer` → engine cap 0.20, exactly as shipped (`src/strategies/hyper_growth.py::create_hyper_growth_strategy` defaults).
- **Arm B (vol-target)**: `VolatilityTargetSizer(base_sizer=FixedFractionSizer(fraction=0.25), target_atr_percentile=0.5)` wrapped identically into `LeveragedPositionSizer`, all other params (regime detector, leverage manager, engine cap) unchanged. `target_atr_percentile=0.5` is chosen as the class default/median — no tuning pass on this parameter happens before the frozen exam (see §8 sensitivity, which is a *post-hoc* robustness check on the already-decided primary threshold, not a search for a better one).
- **Arm C (uniform de-risk strawman, for H0)**: `FixedFractionSizer(fraction=0.20)` (a uniform 20% cut vs Arm A's 0.25, chosen to roughly match Arm B's expected average size reduction — computed and reported post-hoc from Arm B's realized average multiplier, not tuned to it). Exists solely to test whether any MaxDD improvement Arm B shows is just "smaller positions are safer" rather than "vol-conditional sizing specifically helps." If Arm C matches Arm B's MaxDD improvement, H1 is **not** supported even if Arm B beats Arm A.

## 5. Data window / protocol

- **Training/tuning window** (never used for the decision metric): 2023-01-01 → 2024-12-31, used only to compute Arm C's fraction from Arm B's realized average multiplier, and for the sensitivity pass in §8.
- **Frozen exam window**: `--start 2025-01-01 --end` (most recent complete UTC day before this experiment's run, recorded as an explicit calendar date in the results appendix — never `--days N` relative to run-date; see the `--start`/`--end` requirement below). Not touched during arm construction. This is a fresh window, deliberately not reusing the F1/F2/F3 = 2023H1/2024H1/2025H1 folds from `2026-07-12_exit-geometry-honest.md`, because that document's provenance relative to the #1081 stale-import defect (GH #1070) is not on the "confirmed safe" list and this experiment does not want to inherit an unconfirmed baseline.
- **Symbol/timeframe**: ETHUSDT/1h (the live-traded pair), matching `charter.md`'s active symbol.
- **`--start`/`--end`, never `--days N`.** `2026-08-13_hypergrowth-tier-restore-reproduction.md` demonstrated a `--days 365` run relative to "today" landed in an unrelated market regime and produced +103% return vs. the intended window's -28%/-20% — a purely date-arithmetic artifact, not a real result. Every run in this experiment uses explicit `--start`/`--end` calendar dates, recorded exactly in the results appendix.
- **Worktree import-path guard (GH #1070, P0).** The same document found that `atb backtest` run naively from a worktree can silently execute the primary checkout's stale code via an editable-install path finder, with no error. Every command in this experiment is run as `PYTHONPATH="$(pwd)" atb backtest ...` from the worktree root, and the worktree's own code is confirmed active before trusting any number (e.g. checking for a worktree-only code seam known to differ from `main`, per that document's verification method) — logged in the results appendix, not assumed.
- Command shape: `PYTHONPATH="$(pwd)" atb backtest hyper_growth --symbol ETHUSDT --timeframe 1h --start 2025-01-01 --end <most recent complete UTC day> --risk-per-trade 0.02 --max-risk-per-trade 0.03 --max-position-size 0.20 --initial-balance 85 --drawdown-cap-mode measure --log-to-db` for each arm, run **strictly sequentially** (Mac thermal constraint), never in parallel. Long-only enforced (matches current prod per GH #1020 — no `allow_shorts` override).
- Worktree: new, disposable, branched from `develop` at the commit this preregistration is committed against; recorded in the results section when run.
- Engine version: `develop` HEAD at worktree creation, recorded exactly (commit hash) in the results appendix.
- Determinism guard: each arm's exam run is executed twice; results must be bit-identical (nothing in this design is stochastic beyond what the engine itself already seeds) before the write-up is finalized, per the skill's determinism-guard rule.

## 6. Decision each outcome triggers (pre-committed, all branches)

- **Supported** (Arm B clears §3 against Arm A **and** is not matched by Arm C): recommend to pm as "promising, not ready" — a staging paper-trial candidate for `hyper_growth`, contingent on #1106 landing first (see §10). Write proposal file per quant-researcher's standard workflow once #1106 lands and a live-representative re-run confirms the effect on live-timed entries.
- **Matched by Arm C** (Arm B ~= Arm C on MaxDD/Sortino): H1 rejected — the improvement is generic de-risking, not vol-conditional value-add. Recommend against building further vol-targeting machinery for HyperGrowth specifically; note that `VolatilityTargetSizer` may still have value for a strategy with a wider dynamic sizing range than HyperGrowth's binary gate-then-flat-fraction design (see `2026-08-13_capital-sizing-knee.md` — HyperGrowth has no continuous sizing middle today).
- **Rejected** (Arm B does not clear §3 against Arm A): close as a NO-GO, full write-up, same as `2026-07-12_exit-geometry-honest.md`'s standard for negative results. Note whether the mechanism failed (sizer didn't activate — check `vol_target_applied`) or the mechanism activated but didn't help (real negative finding) — these have different implications for the rest of the program.
- **Inconclusive** (e.g., `early_stopped` fires differently across arms and `measure` mode re-run doesn't resolve it, or `atr_percentile` is missing/degenerate for a material fraction of bars): flag per-arm, do not force a verdict; next step is a targeted data-quality check before re-running.

## 7. Drawdown-truncation discipline (#1102)

Every run in this experiment uses `--drawdown-cap-mode measure` for the characterization exam (this is explicitly a research characterization, not a promotion decision — measuring the true drawdown profile of each arm is the point). The `early_stopped` boolean is checked and reported for every single run; if any run under `measure` mode still needed to be compared against a prior `enforce`-mode number, that comparison is invalid and is not made. If and when this experiment reaches a promotion recommendation, a final confirmatory run under `--drawdown-cap-mode enforce` (live-representative) is required before that recommendation goes to risk-officer, per §10.

## 8. Sensitivity analysis (pre-committed to run, not pre-committed to a specific outcome)

Two parameters, tested only on the training window (§5), never touching the frozen exam:
1. `target_atr_percentile` in {0.3, 0.5, 0.7} — does the qualitative MaxDD/Sortino ordering (B better than A) survive a ±20 percentage-point wiggle around the default, or does it collapse (a "collapses under a 10% wiggle" result must be stated plainly per the standard robustness bar).
2. `min_multiplier`/`max_multiplier` bounds (class defaults vs a tighter [0.5, 1.5] band) — does clipping the sizer's dynamic range change the qualitative result.

Both reported alongside the primary result, not used to pick a different "best" configuration to then re-test on the frozen exam (that would be exactly the post-hoc threshold move the skill prohibits).

## 9. Risks of false positive

- **Single-regime draw**: the 2025-01-01→present exam window is one continuous stretch, not multiple independent market cycles. A result here is "worth a closer look," not "regime-robust," consistent with how the model-free scan treats its own findings.
- **ATR-percentile availability**: `VolatilityTargetSizer` is a pass-through (falls back to the control behavior) if `regime.metadata["atr_percentile"]` is missing. If this happens for a large fraction of bars, Arm B silently degenerates toward Arm A and any "no difference" result would be a data-plumbing failure, not evidence against H1 — checked explicitly via `vol_target_applied`, not assumed.
- **Arm C is a strawman with its own construction risk**: computing Arm C's fraction from Arm B's realized average multiplier could itself embed the same information vol-targeting uses (indirectly, via how much size Arm B ends up using on average) — if Arm C's fraction is close to Arm A's fraction and still matches Arm B, that is weaker evidence against H1 than if Arm C is a genuinely different sizing regime. This asymmetry is disclosed and the realized Arm C fraction is reported before interpreting the C-vs-B comparison.
- **Fee/slippage sensitivity is not this experiment's target but is not turned off**: `CostCalculator` defaults stay on throughout; this experiment tests a sizing overlay, not a costs-off sanity check, and is never reported as fee-free.
- **This experiment holds entries fixed at backtest-generated (closed-bar) entries.** See §10 — this is the load-bearing caveat for the whole document.

## 10. Parity status — what this experiment can and cannot conclude before #1106

The closed-candle-gating parity fix (`fix/closed-candle-gating`, tracked in GH #1106) is **not yet merged**. The forming-bar flip-rate study (`docs/research/experiments/2026-07-06_forming-bar-fliprate.md`) measured 43.2% of live decisions at minute 5 disagreeing with the closed-bar backtest decision. This experiment compares Arm A vs Arm B vs Arm C **on the identical, fixed set of backtest-generated entries** — the sizing overlay is applied symmetrically across all three arms on the same trade sequence. That symmetry means:

- **What is parity-insensitive and may run now**: the *relative* ranking of Arm A vs B vs C — "does vol-conditional sizing help more than uniform de-risking, holding a given entry sequence fixed" — is a question about the sizing layer in isolation, and the entry-sequence gap between backtest and live affects all three arms identically. This characterization run is authorized to execute now, labeled clearly as **pre-parity characterization**.
- **What is parity-sensitive and must wait for #1106**: any absolute claim about what this would do to *live* P&L, and any staging/production promotion recommendation. Live entries at minute-5 decisions disagree with closed-bar entries 43.2% of the time — the actual ATR-percentile regime each *live* entry lands in could differ systematically from what a closed-bar backtest shows (e.g., if the flip-prone decisions cluster in specific volatility regimes, which is plausible and untested). A "supported" verdict from §6 is downgraded to "promising but not ready — pending parity" until a live-representative re-run (post-#1106, ideally on staging with the gate flag on) reproduces the qualitative result.

## 11. What this experiment explicitly does not do

- Does not predict direction. Entries are HyperGrowth's existing signal generator, completely unchanged.
- Does not touch `RiskParameters`, `ExitHandler`, or any exit/trade-management logic (that is the separate `2026-08-25_exit-geometry-tp06-power-prereg.md` track).
- Does not propose a live-affecting change by itself. Any graduating result is a staging-trial candidate pending #1106, not an immediate proposal.
- Does not modify `src/` production code — `VolatilityTargetSizer` already exists and is already reviewed; this experiment is a strategy-construction/config change only, run inside a disposable research worktree.

---

*Locked 2026-08-25, before any statistic is computed. Results appended below in a dated section per the anti-p-hacking rule — this header and everything above it is not edited after results are known; corrections are new dated sections.*
