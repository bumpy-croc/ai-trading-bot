# Experiment: Regime-Conditional Exposure for HyperGrowth (Phase 1, Vol-Regime Program #2)

**Author**: quant-researcher
**Status**: PREREGISTERED — locked before any statistic is computed. Do not edit thresholds after seeing results; corrections are new dated sections.
**Program**: Vol/regime program (this doc's sibling `agents/research/vol-regime-program.md`), successor to the 2026-08-13 model-free scan (#1105).
**Blocked on**: `#1106` for any promotion decision — see §10. Also depends on resolving whether `2026-07-04_hypergrowth-365d-drawdown-stress-review.md`'s counterfactual numbers are safe to cite as prior evidence — see §0.1.

---

## 0. Why this experiment, and what it is not

The model-free scan found **regime persistence significant in 17/20 cells** (`docs/research/experiments/scripts/output/2026-08-13_fdr_grid.csv`) — once a symbol/timeframe is in a high- or low-volatility regime, it tends to stay there (Markov transition probability well above 0.5). This is, like ARCH clustering, a direction-agnostic finding: it says nothing about which way price moves, only that the *regime* itself clusters in time and is therefore forecastable in a way direction is not.

This experiment does not predict direction. It tests whether **conditioning total exposure on the persistent regime state** — reducing size or pausing new entries specifically in bear/high-vol regimes, independent of what the entry signal says — improves risk-adjusted return, using machinery that is largely **already built and currently dormant or under-configured** for the live strategy.

### 0.1 This is NOT a from-scratch question for Arm B — a reproduction already exists, and this experiment extends it rather than re-asking it

`docs/research/experiments/2026-07-04_hypergrowth-365d-drawdown-stress-review.md` (independent quant-researcher + risk-officer review) originally ran a counterfactual (CF-A) on the honest 365d HyperGrowth backtest: applying the **graduated drawdown throttle exactly as ratified in `risk-limits.json`** (`dynamic_drawdown_thresholds_pct: [0.05, 0.10, 0.15]`, `dynamic_risk_reduction_factors: [0.8, 0.6, 0.4]`) — rather than HyperGrowth's own looser, strategy-specific override — cut MaxDD from 21.84% to 17.01% and improved return by 4.1pp. That document is explicitly named on GH #1081's "flagged for re-validation" list.

**It has since been independently re-derived**, correctly, on current `develop`: `docs/research/experiments/2026-08-13_hypergrowth-tier-restore-reproduction.md` (GH #1071, open) reproduced this exact question from scratch, using `PYTHONPATH`-forced worktree execution (discovering and filing GH #1070 in the process — a naive worktree `atb backtest` invocation silently ran the primary checkout's 131-commits-stale code and produced +103% return instead of the intended window's honest number) and explicit `--start`/`--end` calendar dates (a naive `--days 365` relative to run-date landed in an unrelated regime). Its verdict, on current `develop @ 9d900992`, long-only enforced (matching prod's actual live config per GH #1020), window `--start 2025-07-04 --end 2026-07-04`:

| Config | Return | MaxDD | Trades | Win Rate |
|---|---|---|---|---|
| Current live override `[0.15,0.30,0.45]`/`[0.8,0.5,0.2]` | -28.29% | 31.27% | 89 | 62.92% |
| Ratified tiers `[0.05,0.10,0.15]`/`[0.8,0.6,0.4]` | -18.95% | 22.23% | 89 | 62.92% |

**Directional claim holds** (ratified tiers reduce both drawdown and losses, identical trades — throttle-only difference, as expected). **Magnitude claim from the original July CF-A does not hold**: the reproduced baseline is materially worse (31.27% vs. the cited 21.84%) and the ratified-tier result **still breaches the 20% cap** (22.23%, not "fits inside the cap for free" as originally claimed) — the divergence traced to the long-only deployment lock (#1020, merged after the July review) removing a hedge that was doing real work in the original long/short backtest. GH #1071's own Step 3 ("fold generalization... **not run**... recommend re-scoping as a follow-up") explicitly left multi-window robustness and the secondary metrics (Sortino, regime time-in-state) as open. This experiment **is** that scoped follow-up — it does not re-ask "does tier restoration help" (answered: yes, directionally, on one window), it asks "does the effect generalize across regimes, and how does it compare on the same footing to the two other dormant mechanisms (Arms C, D) this program is testing." Also unresolved by #1071: even the ratified tiers still breach the 20% cap on that window, meaning Arm B alone is not expected to be sufficient by itself — this experiment reports that expectation explicitly rather than treating a clean Arm B win as the default outcome.

## 1. Hypotheses

**H0 (null)**: Regime-conditional exposure control (graduated drawdown throttle reinstatement, and/or `ExposureGovernor` regime caps, and/or `RegimeAdaptiveSizer`) does not improve risk-adjusted return or MaxDD relative to HyperGrowth's current configuration, once re-measured on a clean, provenance-confirmed backtest run.

**H1 (regime-persistence hypothesis)**: Because volatility/drawdown regimes persist (17/20 significant), a rule that reduces exposure once a drawdown or high-vol regime is *already underway* — not predicting when one will start, only recognizing it has started and is likely to continue — captures real, already-measured structure and should reduce MaxDD without a proportionate cost to return, because it only cuts size during periods that are (on average, given persistence) more likely to keep going the way they're already going.

- *Mechanism if true*: regime persistence means "high-vol/drawdown now" is informative about "high-vol/drawdown in the near future" even without knowing direction. Reducing size during confirmed-adverse regimes removes exposure disproportionately from the periods most likely to still be adverse, which is a different (and cheaper, already-built) claim than "predict when a bad regime will start."
- *Falsified if*: none of the three tested mechanisms (§4) improves MaxDD without a Sortino cost larger than the pre-committed tolerance, on the frozen exam window.

## 2. Metric

**Primary**: MaxDD (measured, `drawdown_cap_mode=measure`) and Sortino, frozen OOS exam window, HyperGrowth/ETHUSDT/1h, prod-matched risk params, fees/slippage on.

**Secondary**: total return, Sharpe, trade count, fraction of bars spent in each regime bucket (bull/bear x low/high-vol, from `RegimeHelper`), realized average exposure by regime (does the mechanism actually cut size where it's supposed to — a mechanism-activation check, not a decision metric).

Trade-count floor: ≥15 trades per arm.

## 3. Success threshold (pre-committed, numeric)

Each arm (B, C, D in §4) is evaluated independently against Arm A (control) using the same bar: MaxDD lower by **≥2 percentage points** AND Sortino within **0.05** of control (same shape as the vol-sizing experiment's threshold, deliberately matched so the two experiments' results are comparable on the same scale). Bonferroni-corrected paired bootstrap across 3 arms x 2 primary metrics -> alpha = 0.05/6 ≈ 0.0083 per test (confirmatory multi-arm test, Bonferroni per the skill's guidance).

If more than one arm clears the bar, they are ranked by MaxDD improvement per unit of Sortino given up (a simple efficiency ratio), not just by raw MaxDD reduction — a mechanism that halves MaxDD but also halves Sortino is not automatically better than one that trims MaxDD by 3pp for free.

## 4. Arms (pre-committed)

- **Arm A (control)**: current live HyperGrowth configuration, unmodified — including its currently loosened, strategy-specific drawdown thresholds (`hyper_growth.py`'s own risk-override call, per the exit-geometry-honest expressibility audit referenced in the returns-levers synthesis).
- **Arm B (graduated throttle reinstatement)**: HyperGrowth wired to the **ratified** `risk-limits.json` graduated thresholds (`drawdown_thresholds=[0.05, 0.10, 0.15]`, `risk_reduction_factors=[0.8, 0.6, 0.4]`, `recovery_thresholds=[0.02, 0.05]`) via the existing `src/engines/shared/dynamic_risk_handler.py` Layer-3 mechanism, **replacing** HyperGrowth's current live override (`drawdown_thresholds=[0.15, 0.30, 0.45]`, `risk_reduction_factors=[0.8, 0.5, 0.2]`, `recovery_thresholds=[0.08, 0.15]` — exact values per GH #1071's reproduction). Parity-safe by construction (Layer 3 is explicitly documented as ensuring backtest/live parity in its own module docstring). Per §0.1, GH #1071 already confirmed the directional effect on one window (2025-07-04→2026-07-04) — this arm's job here is the multi-window/secondary-metric generalization that #1071 explicitly deferred, not a first-pass answer.
- **Arm C (`ExposureGovernor` activation)**: `enable_exposure_governor` feature flag turned on with `DEFAULT_EXPOSURE_CAPS` (`src/config/constants.py`), which caps gross exposure per regime bucket independent of the drawdown throttle — tests the exposure-cap mechanism specifically (regime-state-based, not drawdown-trajectory-based, a different signal from Arm B even though both reduce exposure in adverse conditions). Composed with Arm A's existing config (not combined with Arm B, to isolate each mechanism — a combined arm is out of scope for this preregistration; see §12).
- **Arm D (`RegimeAdaptiveSizer` swap)**: HyperGrowth's `FixedFractionSizer` base replaced with `RegimeAdaptiveSizer(base_fraction=0.03, volatility_adjustment=True)` at class defaults (bull_low_vol 1.8x ... bear_high_vol 0.2x), wrapped in the same `LeveragedPositionSizer` as Arm A. This is the most aggressive of the three mechanisms (multiplier range 0.2x-1.8x vs. the throttle's 0.4x-1.0x) and is expected, if anything, to show the largest MaxDD reduction and the largest Sortino cost — reported as such, not as an automatic winner just because effect size is larger.

Each arm changes exactly one mechanism relative to Arm A. No arm combines mechanisms; that is an explicit follow-up, not this preregistration (§12).

## 5. Data window / protocol

Identical window, symbol, timeframe, and command shape to `2026-08-25_volatility-target-sizing-prereg.md` §5 (frozen exam `--start 2025-01-01 --end` most recent complete UTC day; training/tuning window 2023-01-01 -> 2024-12-31 for sensitivity only). Using the same window across both sibling preregistrations means their results are directly comparable and, if both graduate, a combined-mechanism follow-up (§12) has a clean shared baseline. This is a deliberately different window from GH #1071's `--start 2025-07-04 --end 2026-07-04` — reusing it verbatim would make this experiment a pure duplicate of an already-answered question; a different window is exactly what "fold generalization," the thing #1071 said was missing, requires.

- Command shape: `PYTHONPATH="$(pwd)" atb backtest hyper_growth --symbol ETHUSDT --timeframe 1h --start 2025-01-01 --end <most recent complete UTC day> --risk-per-trade 0.02 --max-risk-per-trade 0.03 --max-position-size 0.20 --initial-balance 85 --drawdown-cap-mode measure --log-to-db`, long-only enforced (no `allow_shorts`, matching GH #1020/current prod), run strictly sequentially, one worktree, one at a time. Same `PYTHONPATH`-forcing and worktree-identity verification as the sibling document, per GH #1070.
- Determinism guard: each arm run twice, results must match, before write-up.
- Engine version and worktree commit hash recorded in the results appendix.

## 6. Decision each outcome triggers

- **One or more arms supported**: rank by the efficiency ratio in §3; recommend the best-ranked arm to pm as a staging-trial candidate, contingent on #1106 (see §10). If Arm B (the ratified-limits reinstatement) is the winner, additionally flag to risk-officer that HyperGrowth's current live config is running looser thresholds than the Board-ratified default **on live capital today** — that is a live-configuration divergence worth an explicit call-out regardless of this experiment's outcome, since `risk-limits.json`'s own header states any `constants.py` divergence is a P0, and a strategy-level override loosening the ratified default deserves the same scrutiny even though it isn't a `constants.py` mismatch. (This is descriptive, not a request to change anything — the change itself would go through the standard proposal path.) **Pre-committed expectation, stated before running**: per GH #1071, Arm B alone still breached the 20% cap on its one tested window (22.23% MaxDD) — this experiment does not expect a clean Arm B pass on §3's bar by itself, and a "supported" verdict for Arm B specifically should be read as "helps, on this window, but the cap-breach question raised by #1071 is not thereby closed" rather than a full resolution.
- **No arm supported**: full NO-GO write-up. Explicitly reconciles with `2026-07-04_hypergrowth-365d-drawdown-stress-review.md`'s CF-A finding — either the effect fails to replicate cleanly (informative about that document's #1081 exposure) or it replicates in direction but not magnitude (also informative, and reported precisely).
- **Arm B specifically fails to replicate CF-A's qualitative direction** (even partially): this is treated as a signal that the July hypergrowth-365d document's provenance concern (§0.1) may be substantive, and is escalated as a note on GH #1081, not silently absorbed.
- **Inconclusive**: same per-arm flagging discipline as the sibling document.

## 7. Drawdown-truncation discipline (#1102)

Same as the sibling vol-sizing document: `--drawdown-cap-mode measure` throughout for characterization; `early_stopped` checked and reported per arm; a final `enforce`-mode confirmatory run required before any promotion recommendation goes to risk-officer.

**Note of caution specific to this experiment**: Arms B, C, and D all explicitly *reduce* exposure in adverse regimes by design, so `early_stopped` under `enforce` mode is, if anything, expected to fire *less* for the treatment arms than for the control — that asymmetry is itself part of what this experiment is testing, and is reported explicitly rather than treated as a nuisance parameter.

## 8. Sensitivity analysis

Training-window only (2023-01-01 -> 2024-12-31), never touching the frozen exam:
1. Arm B: does the qualitative result survive using the *next* threshold set up or down (dynamic thresholds shifted by +-2pp at each tier)?
2. Arm D: does the qualitative result survive a milder multiplier band (e.g., halving the distance of each multiplier from 1.0) — i.e., is the effect driven by the mechanism or by how aggressive the specific multipliers happen to be?

## 9. Risks of false positive

- **CF-A provenance risk** (§0.1) — explicitly why this experiment re-derives rather than cites.
- **Single continuous exam window** — same caveat as the sibling document; a persistence-based mechanism is *especially* exposed to this risk, since its entire value proposition depends on the window containing at least one real adverse-regime episode to reduce exposure into. If the frozen exam window happens to contain no material drawdown episode, all three treatment arms will show no effect for a reason unrelated to whether the mechanism works — checked and disclosed via the regime-bucket time-in-state secondary metric (§2), not silently absorbed into a "no effect" verdict.
- **Regime-detector look-ahead risk**: `EnhancedRegimeDetector`'s regime classification and `atr_percentile` computation must be trailing-only (no centered windows using future bars) — this is inherited, not new, machinery, but is spot-checked in the results write-up rather than assumed correct because it is "already reviewed code."
- **Mechanism double-counting**: Arms B and C both reduce exposure in adverse states via different code paths (`dynamic_risk_handler.py` vs `exposure_governor.py`); `exposure_governor.py`'s own docstring states it "never double-counts the graduated drawdown throttle" by taking the most-restrictive of the two — this experiment tests them **separately** (Arm B alone, Arm C alone) specifically so a combined-mechanism interaction isn't accidentally attributed to just one of them.

## 10. Parity status

Same framing as the sibling vol-sizing document (§10 there): these are sizing/exposure-layer changes applied to an unchanged, closed-bar-generated entry sequence. Relative ranking of Arm A vs B vs C vs D is parity-insensitive and may run now, labeled pre-parity characterization. Any absolute live-P&L claim or promotion recommendation is parity-sensitive and waits for #1106 plus a live-representative (or staging, flag-on) re-run — regime persistence is a property of the underlying price series, but *which* bar HyperGrowth's live engine is actually positioned on at any given moment depends on entries that diverge from the backtest 43.2% of the time at minute 5, so the exact regime each live position is exposed to at entry may differ from what this backtest characterizes.

## 11. What this experiment explicitly does not do

- Does not predict direction; does not modify the signal generator.
- Does not touch exit/trade-management logic (separate track, §12/program overview).
- Does not combine mechanisms (Arms B+C+D together) — that is an explicit, cheap follow-up once the individual mechanisms are characterized, not bundled into this preregistration to avoid an implicit "kitchen sink" arm with no clean attribution.
- Does not propose a live-affecting change by itself.

## 12. Deferred, not forgotten

If Arm B and Arm C both graduate independently, a combined-mechanism arm (both active simultaneously, per `exposure_governor.py`'s stated non-double-counting composition rule) is the natural next preregistration — deliberately deferred rather than run speculatively now, per the skill's "no post-hoc arm added" discipline.

---

*Locked 2026-08-25, before any statistic is computed. Results appended below in a dated section per the anti-p-hacking rule.*
