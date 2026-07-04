---
id: 2026-07-04T1300-P1-hypergrowth-drawdown-cap-breach
opened_by: risk-officer
severity: P1
status: open
opened_at: 2026-07-04T13:00:00Z
mitigated_at: null
closed_at: null
human_paged: false   # no working page channel exists (observability audit P0); surfaced via GitHub issue + this file + session report to pm
affected_components: [live-engine, risk-management, hyper_growth-strategy]
affected_symbols: [ETHUSDT]
---

## What happened

The `risk-limits.json` portfolio hard cap (`max_drawdown_pct: 0.20`, breach_action `halt_new_entries_and_page_human`) was **breached in production and nothing fired**:

- Prod equity peaked at **$103.82 on 2026-04-22** (`account_history`, exchange-synced hourly heartbeat).
- Trough **$82.71 on 2026-06-06** → **20.33% peak-to-trough drawdown**. No halt, no alert, no incident was raised at the time.
- Current equity **$83.92** (2026-07-04 10:40 UTC) → **19.18% below the true peak**, i.e. inside the `critical_at_pct_of_limit: 0.80` band and **$0.86 above the 20% line ($83.06)**. The open SHORT position's stop-out (~$1.33) alone would re-breach the cap.

Independently, the first-ever honest full-year backtest of the incumbent live strategy (HyperGrowth/ETHUSDT/1h, current post-#835 config, engine `develop@e1d24239`) shows **-20.15% return / 21.84% MaxDD** — reproduced exactly in a fresh worktree this session. The breach is structural to the strategy configuration, not a backtest artifact and not bug-contingent.

## Detection

Surfaced sideways: the HyperGrowth 365d benchmark run inside the Kelly evaluation (issue #844, `docs/research/experiments/2026-07-04_kelly-active-evaluation.md`) showed the MaxDD breach; pm dispatched an independent quant-researcher/risk-officer review (this session). The retroactive live breach (2026-06-06) was found by this review's read-only `account_history` drawdown scan — **the system itself never detected it**, which is the point: four stacked control failures (see full analysis in `docs/research/experiments/2026-07-04_hypergrowth-365d-drawdown-stress-review.md`):

1. `PortfolioRiskManager.check_drawdown` (the 20% halt) has zero call sites in live — issue #749, open since 2026-06-10.
2. Backtest CLI `--max-drawdown` defaults to 0.5, so routine backtests don't enforce the 20% line either (constants say 0.20; risk-limits.json calls that divergence a P0).
3. HyperGrowth overrides the graduated breakers to `[0.15, 0.30, 0.45]`/`[0.8, 0.5, 0.2]` (`hyper_growth.py:294-300`) — second reduction tier sits *past* the kill line. Applies identically in both engines.
4. Live's drawdown input resets to ~0 on every restart (`PerformanceTracker.peak_balance = initial_balance`, no rehydration) — after the 2026-07-03 deploy, live currently perceives ~0.6% DD vs the true 19.18%.

Plus: prod `alert_webhook_url` unset — even firing events page nobody (2026-06-08 observability audit).

## Impact

Real money. Live account is at $83.92 from a $103.82 peak (-$19.90, -19.18%). Partially attributable to the June SL-fail cascade (fixed, #648/#653/#655), but the April→May leg was ordinary strategy trading and the current config's full-year backtest reproduces a >20% MaxDD on its own. No *new* capital loss occurred during this review; the incident is the unprotected state plus the already-realized breach.

## Timeline

```
2026-04-22 14:16 UTC — [peak] prod equity $103.82
2026-06-02..06     — [loss leg] SL-fail cascade + true-equity sync; 2026-06-06 18:14 trough $82.71 = 20.33% DD — hard cap breached, undetected
2026-06-10         — [prior signal] issue #749 opened (live check_drawdown dead code) — P2, not acted on
2026-07-03 21:36   — [deploy] #835 sizing config live; engine restart resets live's perceived peak to ~$84.44
2026-07-04 ~11:00  — [detection] #844 benchmark surfaces 21.84% backtest MaxDD; pm dispatches this review
2026-07-04 ~12:30  — [confirmation] exact reproduction + live account_history scan finds the 2026-06-06 retroactive breach
2026-07-04 ~13:00  — [escalation] this incident + GitHub issue (type:incident) + proposal 2026-07-04-02 filed
```

## Actions taken

- Reproduced the 365d backtest exactly (fresh disposable worktree at `develop@e1d24239`, same flags): -20.15% / 21.84% MaxDD / 104 trades / $67.38.
- Read-only prod DB analysis (equity curve, drawdown episodes, deposit/withdrawal exclusion, open-position exposure). No writes of any kind.
- Counterfactual stress runs quantifying each broken layer (see experiment doc §4).
- Filed proposal `2026-07-04-02-hypergrowth-drawdown-containment.md` with the tightening actions; GitHub issue with `type:incident`.
- **No production mutation performed** — entry-pause and code fixes are pm's call per the proposal.

## Current state

Bleeding not active this hour (position is stop-protected, sizes capped at 20%/2%-risk), but the account sits one ordinary stop-out from re-breaching the hard cap, with no automated layer that would halt or even notice. Charter's stated breach action (halt new entries + page human) is not currently achievable by the system itself.

## Post-mortem (filled after close)

### Root cause
### Contributing factors
### What went well
### What went poorly
### Action items (each links to a proposal or tracker)
