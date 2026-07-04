# HyperGrowth 365d Drawdown Breach: Independent Review & Stress Test

**Date**: 2026-07-04
**Researcher**: quant-researcher + risk-officer (dual role, dispatched by pm as an independent review — reproduction and stress analysis performed from scratch in this session, not copied from the originating session)
**Status**: complete
**Engine**: `develop @ e1d24239` (post-#838 corrected drawdown accounting, post-#843)
**Worktree**: disposable `.claude/worktrees/hg-365d-repro` (detached at `e1d24239`), removed at end of session. Production DB accessed strictly read-only (`SET default_transaction_read_only = on`).
**Related**: issue #844 / `docs/research/experiments/2026-07-04_kelly-active-evaluation.md` (where the finding surfaced as a benchmark side-effect), issue #749 (live max-drawdown enforcement gap, open since 2026-06-10), issue #807 (account-level circuit breakers), `docs/observability_audit_2026-06-08.md`.

## The question

The first-ever full-year (365d, multi-regime) backtest of HyperGrowth/ETHUSDT/1h — the incumbent **live** strategy — showed **-20.15% return with 21.84% MaxDD**, breaching the 20% portfolio hard cap in `.claude/state/risk-limits.json`. Is this a backtest-window artifact (backtest lacks live's protections), or genuine tail risk in the live configuration?

**Verdict up front: genuine tail risk of the live configuration** — established by the exact reproduction, the drawdown's multi-regime slow-bleed anatomy, and four stacked control-layer failures verified in code. The strategy's honest full-year profile breaches the 20% cap and no automated layer would halt or even notice.

> **Correction (2026-07-04, same day, ledger-verified — supersedes this review's initial live-breach claim).** The first version of this review reported that production had *already* realized a 20.33% peak-to-trough drawdown (peak $103.82 on 2026-04-22 → trough $82.71 on 2026-06-06) and was 19.18% below peak. pm challenged the peak's provenance and the ledger confirms the challenge: `account_history.balance` is **software-pinned in the pre-sync era** (Mar: 2 distinct values; Apr: 4; May: literally **one** distinct value, 99.9789, across 451 hourly rows). The April "$103.82 equity peak" was a frozen ~$100 book base plus unrealized wiggle — an optimistic `session_start` reset value, **not a true exchange read**. True margin-equity reads begin only with the #655 sync (2026-06-03, $84.14). Therefore **no true-equity 20% breach can be established**, and the "one stop-out from re-breach" urgency claim is withdrawn. Adopted baseline policy (pm, 2026-07-04): drawdown peak = peak *true* equity since the last reconciled reset (2026-06-05 / session 20, ≈$84.40) → current live DD ≈ **0.6%**, standup tripwires ($80.18 soft / $75.96 reduce / $67.52 hard) stand. Everything else in this review — the reproduction, the four control failures, and the counterfactuals — is unaffected by this correction and stands as written.

## 1. Reproduction (independent, fresh worktree)

Same params as the original run: `atb backtest hyper_growth --symbol ETHUSDT --timeframe 1h --days 365 --initial-balance 85 --risk-per-trade 0.02 --max-risk-per-trade 0.03 --max-position-size 0.20`, fees/slippage ON (default CostCalculator), cache symlinked from main checkout.

| Metric | Original session | This reproduction |
|---|---|---|
| Total Return | -20.15% | **-20.15%** |
| Max Drawdown | 21.84% | **21.84%** |
| Trades | 104 | **104** |
| Win Rate | 71.15% | **71.15%** |
| Sharpe | 0.119 | **0.12** |
| Final balance | $67.38 | **$67.38** |
| 2025 / 2026 split | -16.72% / -4.81% | **-16.72% / -4.81%** |

Exact match. The number is real, not a one-off anomaly of the prior session. Context: buy-and-hold ETHUSDT over the same window was **-31.03%** (the strategy beat holding by ~11pp — this was a brutal year for ETH, but "less bad than holding" is not the KPI; Sharpe 0.12 is far below the charter's 0.5 minimum).

### Drawdown anatomy (trade-level, from `--log-to-db` rerun, session 27 local DB)

- **Slow bleed, not a crash**: worst single trade -$2.19 (2.6% of capital). Monthly realized P&L: Jul 2025 -$6.10, Aug -$0.89, Sep -$1.96, Oct -$4.78, Nov +$0.43, Dec +$0.13, Jan 2026 -$2.05, then roughly flat into the 2026-06-04 realized trough ($68.17).
- Realized balance **never rose above the $85 start** — peak = initial balance, decline began in week one (-$6.10 across the first 7 trades, all at or near the 20% position cap).
- 71% win rate with profit factor 0.47: frequent small wins, larger stop-outs (death by a thousand stop losses). All 5 largest losses were `Stop loss` exits at 0.18–0.20 size.
- This multi-month, multi-regime bleed profile means **no plausible event-window entry-pause pattern would have avoided it** — there was no single event to pause around.

## 2. Live production drawdown history (read-only, prod `account_history`) — CORRECTED, see note in header

Coverage 2026-03-29 → 2026-07-04, 1,949 hourly equity rows. No deposits/withdrawals distort the curve (verified via `account_balances.update_reason` ledger).

| Metric | Initial reading | Corrected reading |
|---|---|---|
| Book-equity peak | $103.82 (2026-04-22) | **phantom-era book value** — balance base software-pinned at ~$100 Mar–May (May: one distinct balance value across 451 rows); not a true exchange read |
| Trough | $82.71 (2026-06-06) → "20.33% breach" | book-value drawdown only; **no true-equity 20% breach can be established** (no true reads exist before 2026-06-03) |
| Current equity | $83.92, "19.18% below peak" | **≈0.6% below the post-reconciled-reset true peak (~$84.40)** per the adopted baseline policy |
| Monthly equity path | Apr 99.25–103.82 → May min 93.60 → Jun min 82.71 → Jul ~83.9 | same numbers, but Mar–May values are book, not truth |

What live history *does* still establish, post-correction:

- Real capital went from a true ~$100 (initial deposit, March) to a true ~$84 (first honest sync reads, June) — a real ~16% capital erosion, previously post-mortemed (SL-fail cascade + phantom accounting, #648/#653/#655). The system's books hid it while it happened, which is the same observability failure mode this review documents in code.
- Since honest accounting began (2026-06-03), equity has been flat ~$83–84.4 — live under the current config is ~2 days old (#835, 2026-07-03) and has essentially no drawdown history yet. **The live corroboration for the 21.84% backtest tail is therefore weaker than this review initially claimed** — live hasn't realized this tail under honest books; it simply has no honest history long enough to test it. The backtest evidence and the code-level control-failure evidence carry the finding on their own.

## 3. Why no circuit breaker caught it (four stacked control failures)

Layer-by-layer, verified at `develop@e1d24239`:

1. **The 20% hard cap is dead code in live.** `PortfolioRiskManager.check_drawdown` (`src/risk/risk_manager.py:846`) has zero call sites. The live engine has no drawdown halt of any kind (`rg early_stop|max_drawdown|halt src/engines/live/trading_engine.py` → nothing). Known since 2026-06-10 as issue #749 (still open, still P2 — this review argues that priority is wrong).
2. **The backtest's own halt defaults to 50%, not 20%.** `--max-drawdown` defaults to `0.5` (`cli/commands/backtest.py:326-330`) while `DEFAULT_MAX_DRAWDOWN = 0.20` in `src/config/constants.py:101` and `risk-limits.json` says 0.20. So the standard backtest invocation everyone runs does not enforce the risk-limits cap either — the 21.84% run sailed through. (`risk-limits.json` header: "Must match src/config/constants.py. Any divergence is a P0.")
3. **HyperGrowth loosens the graduated breakers ~3x, in both engines.** `create_hyper_growth_strategy` overrides `dynamic_risk` to thresholds `[0.15, 0.30, 0.45]` / factors `[0.8, 0.5, 0.2]` (`src/strategies/hyper_growth.py:294-300`, "Wider drawdown tolerance for hyper-growth target") versus risk-limits.json's `[0.05, 0.10, 0.15]` / `[0.8, 0.6, 0.4]`. Both engines honor the strategy override via the same `merge_dynamic_risk_config` (backtest `engine.py:296`, live `trading_engine.py:826`). Consequence: the first size cut (×0.8) arrives at 15% DD — three quarters of the way to the kill line — and the second (×0.5) at **30% DD, past the cap**. In the reproduction this is visible directly: entries run at the 0.20 cap until ~Nov 2025, then step to exactly 0.16 (0.8×cap) for the rest of the run. The breaker was *active and correctly implemented* — its thresholds are simply calibrated to permit a cap breach.
4. **Live's breaker input has peak amnesia.** Live sources drawdown from `PerformanceTracker`/`BalanceTracker`, whose `peak_balance` initializes to the balance at engine start (`src/performance/tracker.py:205`, `src/engines/live/pnl.py:19-25`) with no rehydration from `account_history` (nothing in `recovery.py` restores a peak). Prod restarts on every deploy → after the 2026-07-03 deploy, live's dynamic risk sees peak ≈ $84.44 and current DD ≈ **0.6%**, while true DD from the April peak is **19.18%**. Even perfectly calibrated thresholds would currently do nothing. (Same defect class as the operational standup tripwires, which were computed from the post-reset $84.40 "peak.")

Plus the standing observability finding: `alert_webhook_url` unset in prod → even the events that do fire page nobody (`docs/observability_audit_2026-06-08.md`).

## 4. Counterfactual stress runs (same window, same params otherwise)

| Run | Config | Return | MaxDD | Trades | Final $85 | Notes |
|---|---|---|---|---|---|---|
| Baseline (repro) | live config as-is | -20.15% | 21.84% | 104 | $67.38 | breach; no halt (50% CLI default) |
| CF-B: hard cap enforced | `--max-drawdown 0.20` | -20.41% | 20.50% | 81 | $67.65 | engine halted ("Maximum drawdown exceeded") — but only after a 0.5pp overshoot, and late in the window |
| CF-A: risk-limits.json breaker thresholds | strategy `dynamic_risk` override removed → defaults `[0.05,0.10,0.15]`/`[0.8,0.6,0.4]` | -16.08% | **17.01%** | 104 | $71.08 | **no breach** — same trades, same 71.15% win rate, sizes throttled earlier and harder |

Interpretation:

- **CF-A is the load-bearing result**: on the identical bad year, the graduated breakers *as the Board configured them in risk-limits.json* keep MaxDD at 17.01% — inside the 20% cap and near the charter's <15% target — purely by cutting size earlier (0.8x from 5% DD, 0.6x from 10%, 0.4x from 15%). HyperGrowth's deliberate loosening of those thresholds ("wider drawdown tolerance for hyper-growth target") is the difference between a contained drawdown and a hard-cap breach. Cost of the protection on this path: none — it *saved* 4.1pp of return ($3.70). (Sharpe dips 0.12→0.09 since the same negative-expectancy trades happen at smaller size; on a positive-expectancy year the throttle would cost some upside — that is the intended trade of a drawdown breaker.)
- **CF-B shows the hard halt is a backstop, not prevention**: it fires per-candle *after* the line is crossed (realized MaxDD 20.50%, an ~0.5pp overshoot) and on this path only near the end of the window. Its value is stopping the *next* leg down and forcing the human decision the charter requires — it does not retroactively save the 20% already lost. Both layers are needed; neither substitutes for the other.

## 5. Artifact-or-genuine verdict

**Genuine tail risk of the live configuration.** Every artifact hypothesis examined and rejected:

- *"Backtest lacks live's entry pauses / human interventions"* — rejected. `FEATURE_ENTRY_PAUSE` only shipped 2026-07-03, is manual, and defaults off; the historical human interventions (June flatten+reset) happened *after* losses were realized, they did not prevent drawdown. And the drawdown anatomy (12-month bleed, no single event) offers nothing for an event-window pause to bite on.
- *"Backtest measures drawdown more harshly post-#838"* — rejected as an artifact claim: marked-to-market drawdown is exactly what live equity (`account_history.equity`, exchange-synced) experiences. Pre-#838 backtests *understated* it; the correction created honesty, not pessimism.
- *"Live regime timing has been luckier"* — **open, not resolvable from live data** (corrected): live has only ~1 month of honestly-booked equity (post-#655 sync) and ~2 days under the current config, so it can neither corroborate nor refute the backtest tail. The initial version of this review claimed live had already realized a 20.33% drawdown; that rested on a phantom-era book peak and is withdrawn (see Correction). The verdict does not need the live leg: the backtest reproduction and the code-level control failures stand on their own.
- *"Fees/slippage assumptions overstate losses"* — implausible direction/magnitude: verified live fee reality is ~$0.17 total over two weeks of comparable trading; the backtest's PF 0.47 with 71% win rate is a payoff-structure problem, not a cost-model problem.

One honest caveat: the 365d window is a single, historically bad year for ETH (hold -31%), and HyperGrowth's original promotion story (#567, "737% over 5 years, Sharpe 2.19") was built on a different era and a pre-#838 engine whose partial-exit returns are now known to have been fabricated (#839). The tail this year is real; whether the strategy has positive expectancy in friendlier regimes is a separate question this review does not answer.

## 6. Recommendation to pm

**Structural tightening is warranted — but not an emergency halt.** (Recommendation revised with the Correction: the "one stop-out from re-breach" imminence claim is withdrawn; under the adopted baseline policy current live DD is ≈0.6%.) The strategy's honest full-year profile shows the breach is structural, and all four defense layers that should contain a repeat are broken or miscalibrated. Concretely, in priority order:

1. **P1, ops (no code)**: keep the standup tripwires from the post-reconciled-reset peak ($84.40: soft $80.18 / reduce $75.96 / hard $67.52 → `FEATURE_ENTRY_PAUSE` + page human) as the interim manual control, and treat them as *binding*, not advisory — they are currently the only functioning drawdown control in production. No immediate entry-pause is warranted at ≈0.6% DD.
2. **P1 (code)**: land the live max-drawdown halt (#749; branch `fix/live-max-drawdown-halt` exists, no commits yet at review time). Two requirements this review adds evidence for: (a) the peak must survive process restarts — rehydrated from a persistent store, **scoped per pm's adopted baseline policy** (peak true equity since the last reconciled reset / session), since an unqualified all-time peak would resurrect phantom-era book values; the residual gap (re-baselining on a future clean restart creating a new session — "20% per session") needs a durable cross-session anchor as a follow-up; (b) halt = block new entries + alert, not liquidate.
3. **P2 (code, one block)**: delete or drastically tighten HyperGrowth's `dynamic_risk` loosening (`hyper_growth.py:294-300`) so live inherits risk-limits.json's `[0.05,0.10,0.15]`/`[0.8,0.6,0.4]`. CF-A quantifies the benefit on the bad year. Sizing config was already halved once (#835) — this closes the *drawdown-reactive* gap that sizing changes don't touch.
4. **P2 (config)**: change the backtest CLI `--max-drawdown` default from 0.5 to `DEFAULT_MAX_DRAWDOWN` (0.20) so every future backtest enforces the same hard line the book claims to run under (`risk-limits.json` calls constants divergence a P0).
5. **Strategic (pm agenda, not urgent-path)**: HyperGrowth/ETHUSDT no longer clears the charter KPI bar on honest full-year evidence (Sharpe 0.12 vs 0.5 minimum; MaxDD 21.84% vs <15% target). The bear-market-2026 workstream (#801-#807) and the kelly_momentum paper trial are the existing venues; this review adds urgency but proposes no strategy swap on a single-window basis.

Filed as: incident `2026-07-04-01-hypergrowth-drawdown-cap-breach.md` (P1), proposal `2026-07-04-02-hypergrowth-drawdown-containment.md`, GitHub issue (type:incident). Log entry appended.
