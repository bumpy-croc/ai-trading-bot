---
id: 2026-07-04-01-kelly-momentum-staging-paper-trial
from: quant-researcher
to: pm
status: open
risk_review_required: true
risk_verdict: approve-with-conditions   # null | approve | approve-with-conditions | reject
code_review_required: false
board_required: false      # staging paper only, per charter autonomy envelope; NOT a live-capital change
created: 2026-07-04T12:15:00Z
updated: 2026-07-04T12:15:00Z
---

## Ask

Run `kelly_momentum`/ETHUSDT/1h as a **staging paper session, in parallel with (never replacing) live HyperGrowth**, to build genuine Kelly-warm live trade history now that the sizer's wiring gap (#842) is fixed by #843.

## Context

PR #843 wired `KellyCriterionSizer.record_trade()` into both engines via the shared `Strategy.on_trade_closed` seam — the first time kelly_momentum's Kelly sizing has ever been able to activate (every prior backtest, including the 2026-07-03 tournament, ran in permanent cold-start fallback per risk-officer's independent finding). Issue #842 explicitly asked for a re-run once this fix landed. `docs/research/experiments/2026-07-04_kelly-active-evaluation.md` (issue #844) is that re-run.

## Proposed change

- Start a `kelly_momentum`/ETHUSDT/1h **paper** session on staging, sized identically to the backtest evaluation (`--risk-per-trade 0.02 --max-risk-per-trade 0.03 --max-position-size 0.20`, $85-equivalent paper balance).
- Run alongside live HyperGrowth — does not touch, pause, or resize the live HyperGrowth session in any way.
- No code changes required; strategy and sizer are already on `develop`.
- Minimum trial length: until 30+ closed trades accumulate live (est. ~5 months at the ~1-trade/5-days cadence observed in backtest) so Kelly's warm-state behavior can be observed outside backtest, not just inferred from replay.

## Evidence

- Backtest (365d, 90d, 30d, ETHUSDT/1h): `docs/research/experiments/2026-07-04_kelly-active-evaluation.md`
- Kelly wiring confirmed genuinely active via instrumented replay (not just code trace): `has_sufficient_history` flips `True` at trade 30 exactly, stable through trade 73.
- 365d: kelly_momentum -0.29% return / 0.44% MaxDD / Sharpe 0.0018, vs HyperGrowth -20.15% / 21.84% MaxDD / Sharpe 0.119 (HyperGrowth's own MaxDD here breaches the 20% portfolio hard cap — flagged separately, not caused by this proposal).
- kelly_momentum's better absolute/MaxDD numbers are an artifact of ~50x smaller realized position size (avg 0.25% vs HyperGrowth's 13.1% of balance) — this proposal does **not** claim kelly_momentum has superior risk-adjusted edge; HyperGrowth wins on Sharpe/Sortino once size is corrected for.
- No fabrication signatures found in any of the four runs (checked against the three signatures that flagged the pre-#838 tournament as fabricated).

## How this could lose money

1. **It largely can't, at the position sizes observed.** Realized backtest position sizes topped out at 1.57% of balance across 73 trades over a full year; even a full loss of every open position at that size would not threaten the 20% portfolio drawdown limit. The main "cost" of this trial is opportunity cost (capital sitting mostly idle) and inference/ops overhead, not capital loss — because it is a **paper** session, there is no capital at risk at all.
2. **Payoff-ratio risk if this ever escalates toward live.** avg_loss is consistently ~2-3x avg_win across every window tested; if win rate reverts toward 50% (already dropped from 73.3% pre-warmup to 44.2% post-warmup in-sample), expectancy goes solidly negative. This is a reason to NOT promote to live on this evidence alone, not a reason to avoid the paper trial.
3. **Long-only blind spot in a strategy marketed on "aggressive growth."** Zero shorts across 83 trades in a year that included a sustained ETHUSDT downtrend — if this reflects a structural signal-generator bias rather than this window's regime, it caps the strategy's addressable edge even if Kelly sizing is working correctly.
4. **False confidence from a short trial.** 30 trades is the bare minimum for Kelly's own warm-up threshold, not enough for a statistically solid live verdict. This proposal explicitly does NOT ask for a live-promotion decision at the end of a short trial — only for staging paper accumulation. A separate proposal, with a larger live sample, would be required before any live-capital change.

## What risk-officer should stress-test

- Independently verify the warm-up-boundary claim (ideally via a proper integration test asserting `has_sufficient_history` transitions correctly inside `Backtester.run()`, since none currently exists) rather than trusting this write-up's ad hoc instrumented replay.
- Correlation risk between the staging paper kelly_momentum session and live HyperGrowth — both trade ETHUSDT/1h; confirm the `max_correlated_exposure_pct=0.15` framing in `risk-limits.json` is meaningful across a live+paper pairing (it's arguably N/A since paper capital is not real, but flag if ops tooling conflates the two in dashboards/alerts).
- Confirm no dashboard/alerting path could mistake the paper session's numbers for live P&L given they'll be running concurrently on the same symbol.

## Recommendation

**Promising but not ready for live** — ready for a **staging paper trial only**. Per charter's "prefer multi-regime robustness over single-window wins," this experiment does not establish kelly_momentum has genuine edge over HyperGrowth; it establishes the sizing mechanism now works and is safe to observe live. Do not retire or resize HyperGrowth. Revisit with a live-promotion proposal only after a meaningfully larger live paper sample (target: 30+ live-warm trades, i.e., past the point where Kelly's own math is driving sizing on live-observed statistics, not backtest-replayed ones).

---

## Reviews

### risk-officer

**Verdict**: approve-with-conditions
**Confidence**: high

**Rationale**: This is a paper-only, staging-only session with zero capital at risk — it is inside the charter autonomy envelope and needs no board sign-off. Realized backtest sizing topped out at 1.57% of balance across 73 trades/year; there is no plausible path by which observing this in paper threatens capital. Value is real: it builds the Kelly-warm live history that both my 2026-07-03 verdict (b) and #842 asked for, and closes the wiring gap I flagged. I independently confirmed the mechanics that matter (Kelly formula source is correct; `record_trade` wiring is genuinely active per the researcher's instrumented replay, which I did not re-run but which is internally consistent with the trade-count/size data).

**Conditions**:
1. **Paper-isolation is the load-bearing safety property — verify it operationally before start.** kelly_momentum and live HyperGrowth both trade ETHUSDT/1h concurrently. Confirm the staging paper session writes to a distinct `trading_session`/paper account and that NO dashboard, alert, `account_history` heartbeat, or P&L aggregation path can conflate the paper session's numbers with live prod P&L. (The researcher flagged this too; it is a hard pre-condition, not a nice-to-have — a paper drawdown misread as live could trigger a spurious human halt of the real book.)
2. **`max_correlated_exposure_pct=0.15` is N/A here (paper ≠ real exposure) — but confirm ops tooling does not sum paper+live notional into a single correlation metric.** If it does, that is a monitoring bug to fix before this runs, not a reason to block the trial.
3. **No auto-escalation to a live-promotion decision.** A live-capital proposal must be filed separately, requires human sign-off per charter, and must NOT lean on "risk-per-trade" framing for kelly_momentum: I confirmed the CLI `--risk-per-trade`/`--max-risk-per-trade` flags do **not** reach this strategy's sizing (hardcoded `VolatilityRiskManager(base_risk=0.08)` at `kelly_momentum.py:37`; only `--max-position-size` and Kelly's internal `max_fraction=0.20` govern it). Any future risk-tightening on this strategy needs a code change, not a flag.
4. **Model-risk gate before any live consideration (not blocking paper).** File/track the integration test asserting `has_sufficient_history` transitions correctly inside a real `Backtester.run()` — none exists today. The current evidence is one instrumented replay; that is enough to start paper, not enough to promote.

**What I could not verify**: whether the staging environment can actually host a second concurrent paper session alongside paper HyperGrowth without session-collision (see the historical session-collision risk in balance recovery) — this is an ops feasibility question for live-ops, not a risk-envelope question. If staging cannot cleanly host both, the trial needs its own isolated vehicle (separate staging service / distinct session id); it must not share a session with any HyperGrowth paper session.

Ref: `docs/research/experiments/2026-07-04_kelly-active-evaluation.md`, full stress-test in `.claude/state/log.md` (2026-07-04 entry).
