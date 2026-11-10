# HyperGrowth Dynamic-Risk Tier Restore — Reproduction Attempt

**Date**: 2026-08-13
**Researcher**: quant-researcher
**Status**: **reproduction FAILED for the load-bearing magnitude claim; directional claim holds; implementation PAUSED pending PM/Board review**
**Engine**: `develop @ 9d900992` (post-#838/#843, post-#1020 long-only)
**Worktree**: `.claude/worktrees/hg-tier-restore-1e227d` (disposable)
**Related**: GH #986, #1065 (merged risk review), #1070 (new — environment defect found during this session), `docs/research/experiments/2026-07-04_hypergrowth-365d-drawdown-stress-review.md`, `docs/research/risk-snapshots/2026-08-13_1800_max-drawdown-cap-30pct-review.md`

## Hypothesis (as framed by PM, 2026-08-13)

Restoring HyperGrowth's dynamic-risk throttle tiers to the Board-ratified values (`drawdown_thresholds=[0.05,0.10,0.15]`, `risk_reduction_factors=[0.8,0.6,0.4]`, in place of the live override `[0.15,0.30,0.45]`/`[0.8,0.5,0.2]`) contains the strategy's honest 365d MaxDD **inside** the existing 20% cap, at a **better** return, superseding a proposal to raise the cap to 30%.

## Metric / success threshold

Same 365-day window, same params, fees+slippage on (default `CostCalculator`), current `develop` engine:
- **Success**: ratified tiers produce MaxDD < 20% (no breach) AND return ≥ current-override return (matching CF-A: 17.01% MaxDD / -16.08% return vs baseline 21.84% / -20.15%).
- **Failure**: either axis doesn't hold, or the baseline itself doesn't reproduce (per explicit pre-registration in the dispatch: "if the numbers do not reproduce, STOP and report rather than implementing").

## Step 1 — reproduction

### Methodology bug found first (see GH #1070)

The first two reproduction attempts (naive `atb backtest ...` from this worktree) silently ran the **primary checkout's** code (`/Users/alex/Sites/ai-trading-bot`, branch `main`, verified **131 commits behind `origin/main`**, last touched 2026-07-04), not this worktree's `develop`-based code. The shared venv's editable install hardcodes `src`/`cli` imports to that path via a generated finder (`__editable___ai_trading_bot_0_1_0_finder.py`). Filed as GH #1070 (P0) — this is a systemic risk for every agent that runs `atb` from a worktree without forcing `PYTHONPATH`. All results below use `PYTHONPATH="$(pwd)" atb backtest ...` from the worktree root, confirmed to pick up the worktree's own `src/` (verified via presence of the `resolve_strategy_max_position_size` seam and the `allow_shorts` kwarg, both worktree-only at time of writing).

### Window

`--start 2025-07-04 --end 2026-07-04` (matching the 2026-07-04 review's `--days 365` run exactly, not `--days 365` relative to today — the two differ by ~40 days and land in unrelated market regimes; an initial attempt using `--days 365` from today produced +103% return, underscoring how much this matters).

### Command (both arms identical except the code edit under test)

```
PYTHONPATH="$(pwd)" atb backtest hyper_growth --symbol ETHUSDT --timeframe 1h \
  --start 2025-07-04 --end 2026-07-04 --initial-balance 85 \
  --risk-per-trade 0.02 --max-risk-per-trade 0.03 --max-position-size 0.20
```

**Effective sizing**: `--max-position-size 0.20` passed explicitly on every run (matches prod's pinned `--max-position 0.20`, and matches the original 2026-07-04 review's command). This sidesteps GH #1021's `ExperimentRunner` default-clamp bug entirely — that bug affects unspecified `RiskParameters()` defaults in the harness path, not an explicit CLI flag. Model pinned to `2026-07-04_22h_v1` implicitly (the only version present in the registry checked into this worktree; `latest` resolves to it).

### Results — current override vs ratified tiers, current develop, long-only enforced (as prod actually runs today)

| Run | Config | Return | MaxDD | Trades | Win Rate | Sharpe |
|---|---|---|---|---|---|---|
| Baseline (current live override) | `[0.15,0.30,0.45]`/`[0.8,0.5,0.2]` | **-28.29%** | **31.27%** | 89 | 62.92% | 0.16 |
| Ratified tiers restored | `[0.05,0.10,0.15]`/`[0.8,0.6,0.4]` | **-18.95%** | **22.23%** | 89 | 62.92% | 0.10 |

**Cited baseline (2026-07-04 review, never re-verified since — see risk-officer's own 2026-08-13 log entry: "no new backtests run (compute discipline)")**: -20.15% / 21.84% MaxDD.
**Cited CF-A (ratified tiers)**: -16.08% / 17.01% MaxDD.

### Verdict on reproduction

**Directional claim reproduces**: restoring the ratified tiers measurably reduces both drawdown (-9.04pp) and losses (+9.34pp better return) on this window, with identical trade count/win-rate (same entries, same exits — only sizing under drawdown differs, as expected from a throttle-only change).

**Magnitude claim does NOT reproduce, and the specific conclusion the Board decision rests on is false under current conditions**:
- My reproduced baseline (31.27% MaxDD) is **9.4pp worse** than the cited 21.84%.
- My reproduced ratified-tiers result (22.23% MaxDD) **still breaches the 20% cap** — it does not "fit inside the existing cap" as claimed. It is worse than even the cited *baseline* number.
- The central selling point of [D-2026-08-13 risk review, §1 finding 1 / condition C3] — "restoring tiers contains the strategy inside the cap for free" — is not true on current `develop`.

### Root cause of the divergence (isolated, not merely asserted)

The 2026-07-04 review's baseline predates GH #1020 (ETHUSDT/hyper_growth long-only deployment lock, merged 2026-07-12). To confirm this is the actual mechanism rather than unrelated engine drift, I re-ran the baseline (current override tiers) with shorts explicitly re-enabled (bypassing the deployment lock via `allow_shorts=True`, same window/model/sizing, everything else identical):

| Run | Return | MaxDD | Trades | Win Rate |
|---|---|---|---|---|
| Long-only (today's actual prod config) | -28.29% | 31.27% | 89 | 62.92% |
| Shorts re-enabled (pre-#1020 historical config) | **-17.75%** | **18.99%** | 116 | 74.14% |
| Cited 2026-07-04 original | -20.15% | 21.84% | 104 | 71.15% |

The shorts-enabled run lands close to (not exact — residual engine drift since #838/#843/#1020-era commits, consistent with the risk review's own §10 caveat that this was never re-verified) the originally-cited baseline. **The long-only lock removes a hedge that was doing real work in the original analysis; today's actual live configuration (long-only) bleeds meaningfully more than the number everyone has been citing.** This also means the 20% cap breach the Board is trying to fix is currently *worse* than believed, not better.

## Step 2 — implementation

**Paused.** Per the pre-registered stop condition ("if the numbers do not reproduce, STOP and report rather than implementing"), the code edit (removing HyperGrowth's `dynamic_risk` override so it inherits the ratified defaults) was tested locally, confirmed to produce the directional improvement shown above, then **reverted** — not committed, no PR opened. The change itself is low-risk and strictly improves both axes with zero apparent downside (identical trades/win-rate, only sizing-under-drawdown differs), and remains available to ship on request. But shipping it under the framing "this resolves the cap breach" would be publishing a claim I disproved in the same session.

The concrete diff, for reference (matches the ratified JSON exactly; `recovery_thresholds` falls back to the engine default `[0.02, 0.05]` when the override key is dropped, per `DEFAULT_RECOVERY_THRESHOLDS` in `src/config/constants.py:277` — not itself part of the ratified JSON schema, which has no `recovery_thresholds` field):

```diff
- "dynamic_risk": {
-     "enabled": True,
-     # Wider drawdown tolerance for hyper-growth target
-     "drawdown_thresholds": [0.15, 0.30, 0.45],
-     "risk_reduction_factors": [0.8, 0.5, 0.2],
-     "recovery_thresholds": [0.08, 0.15],
- },
+ "dynamic_risk": {
+     "enabled": True,
+     "drawdown_thresholds": [0.05, 0.10, 0.15],
+     "risk_reduction_factors": [0.8, 0.6, 0.4],
+     "recovery_thresholds": [0.02, 0.05],
+ },
```

This is consistent with [D-2026-07-14-04] item 2's PRUNE-ONLY ruling on the dead `≥0.20` tiers (this supersedes rather than contradicts it, per the original task framing) — the ratified factor set `[0.8,0.6,0.4]` is unchanged and this review does not recommend re-spacing it (per the July risk review's §4.1 reasoning, which is unaffected by this reproduction finding: front-loaded de-risking under whatever cap is in force remains the correct ramp shape).

## Step 3 — fold generalization

**Not run.** Given the foundational reproduction gap found in Step 1, spending further sequential-backtest compute (each of the F1/F2/F3 + #898 bear-window folds takes 5-10 minutes; 8 runs ≈ 60-80 minutes) on top of an unresolved discrepancy would be premature — the Board needs the corrected baseline numbers before deciding whether continued investigation is the right next step, or whether this line of research is now moot pending a decision on the cap itself. Recommend re-scoping as a follow-up once PM has reviewed this finding.

For what it's worth, #898's bear window is `2026-01-01` to `2026-07-04` (185 days, the confirmed OOS window from the ETHUSDT basic-model training-window tournament, GH #898) — flagging this since the task description's "#898 bear window" reference doesn't correspond to an existing HyperGrowth backtest of that name; #898 is a model-tournament experiment that established a bear-market OOS window, which this task borrows by reference for a HyperGrowth-specific test that has not yet been run.

## Step 4 — the honest caveat (per instructions, stated regardless of outcome)

Even in the best case shown here, this is a losing-strategy-loses-less finding, not an edge finding. HyperGrowth's honest 365d expectancy is negative under every configuration tested in this session and in every prior study referenced (Sharpe well below the charter's 0.5 minimum, PF < 1.0 in every fold of every prior tournament). Whether HyperGrowth is superseded by the Phase 0 research programme (GH #1059) is a live, unresolved question — if Phase 0 finds something, this entire line of tier-tuning work on the incumbent becomes moot. Until then, if the Board wants risk reduction on the incumbent specifically, this reproduction shows the ratified tiers are a real but **partial** mitigation (not the "fits inside the cap for free" result that was cited) — the underlying choice between raising the cap and accepting a still-present (smaller) breach has not been resolved by this study.

## Recommendation to PM

**Not ready to implement as originally framed.** Three things need PM/Board attention before any code ships:
1. The specific numbers driving the "prefer tier-restore over cap-raise" decision do not hold on current `develop` — bring the corrected numbers (this doc) back to the sitting before finalizing that preference.
2. Even after restoring ratified tiers, current honest MaxDD (22.23%) still breaches the ratified 20% cap. The tier-restore alone does not make the cap-raise question moot; it may still be needed, or some other measure is, alongside the tier restore.
3. GH #1070 (environment defect) should be triaged with real urgency — it silently invalidates any backtest run naively from a worktree, and it is plausible some other artifacts already in `docs/research/` were affected the same way (not audited in this session; out of scope, flagged only).

If the Board still wants a "measured risk improvement, taken now" despite an incomplete fix: the tier-restore diff above is safe to ship on its own honest merits (net risk reduction, zero downside observed, matches the Board's own ratified values) — just not under the claim that it resolves the breach.
