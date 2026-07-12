# Session handover — 2026-07-12 (Sunday research+audit day)

Written ~14:20 UTC as the ~9h autonomous window closed. Everything below is either merged, PR-open, or a filed issue — nothing lives only in memory.

## Headline outcomes

**Research program (morning):** four lanes → a strategic conclusion.
- **"New input data" lever RETIRED** for ETHUSDT-1h across 6 audited input classes (derivatives/cross-asset/sentiment/microstructure/calendar). Linear AND nonlinear screens both graduated ZERO of 7 arms (PRs #969, #973). Sixth independent null at the ~51-53% DA ceiling. One asterisk: BTC→ETH lead-lag had a real regime-specific edge in F1 only (+3.8pp, sign-flips elsewhere) — narrow future question, not scheduled.
- **Signal-path audit DISCONFIRMED the "broken pipeline" hypothesis**: feature normalization clean (raw features hit the same ceiling), sequence alignment clean, ONNX faithful. The ceiling is REAL, not an artifact. BUT two downstream findings explain why a real 53% edge yields flat P&L: (a) exam slippage ~5x overstated (#984), (b) flat sizing discretizes the edge. Both fixable.
- **Levers synthesis** (PR #974, merged): ranked roadmap. #1 exit-design round 2; #2 elevated to the live-vs-backtest parity gap; then symbols, then BTC-lead-lag, then longer timeframes.

**Parity investigation (PR #987, merged) — the session's biggest single finding:** the "live beat backtest" mystery was TWO mechanisms, not forming-bars: (1) the matched backtest used TODAY's model but live trades predate the Jul-5 promotion (model-version confound → fixed by point-in-time pinning, PR #1006 open); (2) **live has effectively been trading SHORT-DISABLED** — an inventory guard rejects shorts whenever free ETH > $1 dust, only ever blocking shorts (9 long / 3 short real vs 50/50 signals). Filed #990 with forensics→counterfactual→risk-review sequence. This is a genuine returns lever in EITHER direction.

**Codebase deep audit (5 subsystems):** money path is SOUND (all historical bug classes verified remediated). Real finds fixed/filed:
- P1 drawdown-gate fired one iteration late → **PR #1001** (risk-officer SAFE-WITH-CONDITIONS, gating peak-check CLEARED at $84.42; merging).
- P1 market-close SELL could round up past holdings → **#994 MERGED**.
- Reconciliation edge paths (double-count, partial-exit reset, silent divergence) → **#996 MERGED**.
- mfe/mae columns corrupted (sized-vs-unsized, 10-23x) → **#992** (merging).
- Filed: #979 (partial-ops landmine, gated on #734), #982/#983 (data-path), #986 (RISK-RATIFICATION BUNDLE — circuit breakers OFF in prod + 3 config-drifts, needs Alex+risk-officer), #989 (emergency-close -2010 sites), #1005 (reconciliation nits), #1003 (weights-blind checkpoint), #993/#985/#991/#995 (mfe restart + lightgbm-env dup).

## EXIT-LANE CORRECTION (told Alex, important)
Exit-geometry ROUND 1 (#970, merged) was computed with the WRONG model (BTCUSDT on ETHUSDT — the #997 ExperimentRunner bug). Its NO-GO verdict SURVIVES as a between-arm statement (all arms shared entries) but its ABSOLUTE numbers don't represent live. **Round 2 (in progress, PID 73178) verified to be on the CORRECT model — no re-run needed; its verdict WILL be fully valid.** Round-1 driver fixed on branch claude/exit-geometry-round2 (commit cd9ce6ef). #997/#1004/#1008 fix the harness; #998 tracks the round-1 re-verify.

## IN FLIGHT at handover (all backstopped/watched)
- **exit-round-2 sweep** (PID 73178, agent afad3a86b0b94cc78): running arms on CORRECT model; agent writes prereg-locked verdict + PR when done. THE session's live returns experiment.
- **slippage/EV analysis** (agent adb692bc122d8bafb): measures real slippage from prod fills + tests whether per-trade EV varies with any entry observable (gates whether a sizing experiment is worth preregistering). Respects the calibration-study null.
- **staging soak** (PR #1009, agent a3178d0ed3833452d): syncing develop→staging with 5 change-specific boot checks (esp. drawdown-gate must NOT false-trip). Gates the prod promote.
- Open PRs needing review/merge: #1000 (inference-context #926/#927), #1004+#1008 (symbol threading #997/#1002), #1006 (model pinning #988), #992/#1001 (merging).


## LEVERS RANKING — REVISED after slippage/EV verdicts (PR #1010, 14:30 UTC)
Two of the signal-path audit's downstream suspicions REFUTED by measurement:
- Slippage NOT ~5x overstated (my earlier relay of the audit ESTIMATE was wrong): measured 5.1-5.8 bps/side ≈ exam's 5bps default. Cost model honest. No re-exam needed. #984 recommends REJECT the cut.
- No EV-conditioning signal: no entry observable (magnitude/confidence/vol/regime/session) predicts per-trade EV → flat sizing is optimal, NO sizing experiment worth running. Extends the calibration null to trade level.
CONSOLIDATED conclusion: ceiling real + cost model honest + no signal to size by. Remaining genuine levers, RE-RANKED:
1. **#990 SHORT-SUPPRESSION** (promoted to #1) — strategy runs half-crippled (inventory guard blocks ~all shorts); fixing lets the EXISTING strategy express trades it's designed for. Highest-value, forensics→counterfactual→risk-review sequence filed.
2. **Exit design** (round 2 running, correct model) — trade-management lever.
3. Symbols/frequency diversification.
4. Parity gap — now mostly EXPLAINED (model-version confound #1006 + short-suppression #990), less of an open mystery.
DEAD-ENDED this session (measured, not assumed): new inputs, target reformulation, architecture, windows, stop-tightening, slippage-recalibration, conditional-sizing.

## PROD PROMOTE — DEFERRED DELIBERATELY (recommendation for next session)
NOT promoted tonight. Reasoning is risk-based not caution: prod at 0.5% DD; the drawdown-gate bug only bites at 20% (nowhere near), close-cap only on a fee-haircut close (infrequent, 1 open position). Fixes are risk-REDUCING but NOT urgent → staging soak first is strictly better than same-day dozen-PR prod push at session tail. **Runbook for the promote (do after staging soaks clean):** parity promote develop→main; the 3 risk-officer conditions on #1001 (peak-check DONE; changelog-conflict-resolved+CI-on-merged-tree DONE at merge; 24-48h post-deploy watch for spurious close-only). Charter-clean window: before CPI pause arms Mon 13:30 UK (12:30 UTC).

## PROD STATE (read-only verified 14:00 UTC)
Session 20, peak $84.4159, current $84.4025 (~0.5% DD), 899 account_history rows since 2026-06-05, one open ETHUSDT short (#22). Healthy. Running OLD code (safety fixes NOT yet promoted — the bugs are latent, not currently firing).

## SCHEDULED (autonomous)
- CPI pause-on Mon 13:30 UK / pause-off Tue; weekly-retrain Sun 08:08 (already ran); alert-monitor 6-hourly; weekly-retro Mon 10:27 (agenda has ~18 items incl. today's: changelog-per-PR collision = top fix, wake-loss recurrences, sub-delegation chain depth, merge-before-3rd-reviewer, PM-premise-verification).
- Retro AGENDA.md (in .claude/skills/weekly-retro/) — append items there, not memory.

## NEXT-SESSION FIRST MOVES (suggested order)
1. Read exit-round-2 verdict + slippage/EV verdict + staging soak result (all should be in by then).
2. Clear the review/merge queue (#1000/#1004/#1008/#1006, drawdown-gate if not merged).
3. If staging clean → prod promote per runbook above (before 12:30 UTC Mon).
4. Decide the #990 short-suppression investigation (biggest latent returns lever) and #984 slippage recalibration + the exit-round-2 outcome — these three define the next experiment.
5. #986 risk-ratification bundle needs Alex (circuit breakers off in prod is the notable one).
