---
id: 2026-07-04T1300-P1-hypergrowth-drawdown-cap-breach
opened_by: risk-officer
severity: P1
status: mitigated
opened_at: 2026-07-04T13:00:00Z
mitigated_at: 2026-07-11T12:55:00Z
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

## Post-mortem (filled 2026-07-13 at mitigation; PM-directed follow-up to the 2026-07-13 weekly retro)

**Status rationale.** Set to `mitigated` (not `closed`): the acute protection gap that *defined*
this incident — no live drawdown halt, drawdown input resetting on every restart, and events paging
nobody — is closed and prod-verified (below). Final closure is gated on the risk-officer's
condition-3 **24–48h spurious-close-only watch** for #1001 (started 2026-07-12 17:35Z on
`[D-2026-07-12-04]`, i.e. through ~2026-07-14 17:35Z); once that window elapses clean, risk-officer/
PM flips to `closed`. `mitigated_at` = 2026-07-11 12:55Z, when the last acute-gap fix (guard seeded
from `account_history`, #851) was verified live.

### Root cause
Two independent things wearing one incident number:
1. **A functional protection gap** — the portfolio 20% max-drawdown hard cap
   (`risk-limits.json`) had *no live enforcement path at all*. Four stacked control failures (see
   Detection §1–4): the `PortfolioRiskManager.check_drawdown` halt was dead code in live (#749);
   the backtest CLI default (0.5) and HyperGrowth's breaker override (`[0.15,0.30,0.45]`, second
   tier past the kill line) meant even the offline signals didn't enforce the 20% line; and live's
   drawdown input reset to `initial_balance` on every restart with no rehydration, so the engine
   perceived ~0.6% DD against a true ~19%. Plus `alert_webhook_url` unset → firing events paged
   nobody.
2. **A data-quality artifact that inflated the original claim** — the "20.33% live breach off a
   $103.82 April peak" was computed over pre-#655 `account_history.balance`, which was a
   software-pinned `session_start` book value (May: one distinct value across 451 rows), not a live
   exchange read. Withdrawn same-day (see CORRECTION). No true-equity 20% breach was ever
   established.

So: the *unprotected state* was real; the *already-realized breach* was phantom.

### Contributing factors
- Phantom-balance failure mode (documented in `project_capital_erosion_postmortem`) claimed the
  review itself — treating `account_history.equity` as ground truth without a distinct-count sanity
  check on its balance base. Now distilled as `.claude/LESSONS.md` §5.6.
- No working operator page channel at detection time (2026-06-08 observability-audit finding) — the
  incident had to be surfaced via GitHub issue + this file + the PM session, not an alert.
- The strategy's *structural* full-year expectancy (365d honest backtest: −20.15% / 21.84% MaxDD,
  reproduced exactly) means the 20% cap is reachable by ordinary trading, not just by a bug — the
  guard contains the symptom; the expectancy is a separate strategic thread (returns-levers program
  + long-only proposal #1020).

### What went well
- Detection came from an *independent* honest-backtest surface (#844 Kelly eval), and the PM
  immediately dispatched a fresh quant-researcher/risk-officer review rather than acting on the
  first number.
- The phantom peak was caught and withdrawn **same-day**, with a surgical correction that named
  exactly which claims fell and which stood — the model the `decision-record` skill now cites.
- The review performed **zero** production mutations; all analysis was read-only.

### What went poorly
- A 20% portfolio hard cap shipped to live capital with **no enforcement wiring** and stayed that
  way undetected — #749 had flagged the dead code since 2026-06-10 (P2, unacted) three weeks before
  this incident surfaced it. A hard risk limit with no live call site is the failure that should
  never recur; a "limit exists in config" is not a control until a live path reads and enforces it.
- The drawdown input silently reset on every restart — a recovery path that re-initialized state
  without rehydrating it (the same class as LESSONS §1.3).

### Action items (each links to a proposal or tracker)
**Shipped & prod-verified (the acute gap — CLOSED):**
- New `src/engines/live/monitoring/drawdown_guard.py`: enforced **close-only halt** at the 20% hard
  cap with 10%/16% warning tiers (#848/#849) — the live enforcement path that #749's dead code
  never provided. Armed at every prod boot since (peak $84.42, hard cap 20.0%, session 20).
- Guard **seeds its peak from `account_history` true-equity**, not tracker book value / `initial_balance`
  (#850/#851, prod-verified 2026-07-11 12:55Z) — fixes the restart-reset (control failure #4).
- Same-iteration cap enforcement + dynamic-risk throttle anchored to the durable session peak
  (#1001, prod 2026-07-12) — closes an ordering gap where the cap enforced one iteration late.
- Operator alerts delivered via `$ALERT_WEBHOOK_URL` + explicit missing-channel flag (#855/#864) —
  fixes "pages nobody."
- Position/risk caps (20% max-position, 2% risk ceiling, `FEATURE_ENTRY_PAUSE`) (#835/#841); standup
  tripwires installed ($80.18 / $75.96 / $67.52).

**Tracked residuals (hardening/hygiene — do NOT block closure of the acute incident):**
- **#847** — durable *cross-session* peak anchor (guard currently seeds per-session from
  `account_history`; #847 makes the anchor durable across reconciled resets).
- **#986** — risk-ratification bundle (Board-owned): arm circuit breakers in prod, resolve the
  constants↔risk-limits 0.20/0.5 drift (control failure #2), and retire HyperGrowth's dead throttle
  tiers past the kill line (control failure #3).
- **#749** — remove the now-superseded dead `PortfolioRiskManager.check_drawdown` path (cleanup; the
  new guard supersedes it functionally).
- **Structural expectancy** — the 365d MaxDD breach is a strategy-expectancy problem, owned by the
  returns-levers research program and the HyperGrowth/ETHUSDT long-only proposal (#1020, board_required,
  awaiting Alex); the guard contains it but does not fix it.
- **Closure gate** — risk-officer's 24–48h spurious-close-only watch on #1001 completing clean
  (~2026-07-14 17:35Z), then flip `status: closed` with `closed_at`.

---

## CORRECTION — 2026-07-04 (same day, ledger-verified; supersedes the live-breach claim above)

The headline claim "the hard cap was breached in production (20.33%, 2026-06-06)" is **withdrawn**. pm challenged the $103.82 peak's provenance; independent re-verification against the ledger confirms the challenge:

- `account_history.balance` is software-pinned in the pre-sync era: Mar 2 distinct values, Apr 4, **May exactly 1 distinct value (99.9789) across 451 hourly rows**. The base was an optimistic `session_start` book value, not an exchange read.
- The April "$103.82 equity peak" = frozen ~$100 book + unrealized wiggle. True margin-equity reads begin 2026-06-03 (#655 sync, $84.14). **No true-equity 20% breach can be established**, and the "one stop-out from re-breach" imminence claim is likewise withdrawn.
- Adopted baseline policy (pm, 2026-07-04): drawdown peak = peak **true** equity since the last reconciled reset (2026-06-05 / session 20, ≈$84.40) → current live DD ≈ **0.6%**; standup tripwires ($80.18 / $75.96 / $67.52) stand.

**What still stands, unaffected**: the exact 365d backtest reproduction (-20.15% / 21.84% MaxDD — structural cap breach for the live config), all four control-layer failures, both counterfactuals, and the containment proposal's code/config steps (2-4). The proposal's step 1 (immediate entry-pause) is downgraded to "tripwires binding, no immediate pause" — see revised proposal.

**Severity**: recommend pm reclassify **P1 → P2** ("warning thresholds approached but not breached"): the unprotected-state finding is real but there is no active bleeding and no imminent breach under the corrected baseline. Reclassification is pm's call per incident README.

**Process note (blameless)**: the initial claim treated `account_history.equity` as ground truth without checking whether its balance base was a live read — the same phantom-balance failure mode documented in `project_capital_erosion_postmortem` claimed another victim: this review. Rule for future drawdown analysis: **verify the balance column varies like a market-tracking value (distinct-count sanity check) before treating any equity peak as real.**
