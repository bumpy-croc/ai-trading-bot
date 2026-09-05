---
id: 2026-08-29T0807-P0-close-only-latch-day5-unactioned
opened_by: daily-trading-standup
severity: P0
status: open        # open | mitigated | closed
opened_at: 2026-08-29T08:07:43Z
mitigated_at: null
closed_at: null
human_paged: true
affected_components: [live-engine, reconciliation, stop-loss-placement, close-only-latch]
affected_symbols: [ETHUSDT]
---

## UPDATE — 2026-09-05 08:05 UTC (Day 9, still unactioned)

Ninth daily-trading-standup sighting since GH #1121 was filed (2026-08-29). Newest `system_events`
row (2026-09-05 07:31:15.30 UTC) is still `CLOSE_ONLY_LATCHED`: *"STILL BLOCKED — DAY 8: Close-only
mode still active after 8d 22h..."* — elapsed from the 2026-08-27 08:35:28 UTC onset is now
**~8d 23.5h (~215.5h) against the 1h P0 SLA, i.e. ~215x over SLA**. Book still flat: 0 open
positions, equity $87.50216036 — unchanged to the cent for nine consecutive days (session peak
equity over the trailing 30d is $87.50798970, i.e. current drawdown ≈0.01%, not a capital-at-risk
condition). `FEATURE_ENTRY_PAUSE` confirmed `false`; no macro-event window covers now (next is CPI
2026-09-11). Decision loop is alive and generating real (non-zero) decisions at normal ~2min
cadence — this remains a control-plane halt, not a frozen process.

**Day 8 (2026-09-04) update was comment-only** — the prior run flagged PR #1129 as itself stalled
(CI-green, mergeable, zero reviews, open 3 days at the time) but did not add a file/log record.
This entry restores that continuity and folds in day 8's finding: **the durable-sink artifact for
this incident is now also part of the unactioned backlog**, unmerged for **4 days** as of today.

GH #1126 (proximate `-1111` mechanism) and #1127 (structural response-gap, Board decision) both
remain `OPEN`, untouched since 2026-08-31 — 5 days of no engagement on either. Cross-session sweep
this run found no daemon/PM session activity in the repo since the day-7 record; no restart, no
latch clear, no Board sitting on #1127.

**Escalating the non-response itself for the ninth consecutive day** (per the standup's own rule):
the two things that would change this outcome — a human running the documented restart playbook,
or the Board resolving #1127 — remain both untaken. This is now the longest-running unactioned P0
in this repo's incident history (exceeds the #1094 precedent of ~96h by more than double).

## UPDATE — 2026-09-03 08:01 UTC (Day 7, still unactioned)

Seventh daily-trading-standup sighting since GH #1121 was filed (2026-08-29). Newest `system_events`
row (2026-09-03 07:05:27.82 UTC) is still `CLOSE_ONLY_LATCHED`: *"STILL BLOCKED — DAY 6: Close-only
mode still active after 6d 22h..."* — elapsed now **~6d 23h (~167h) against the 1h P0 SLA, i.e.
~167x over SLA**. No change in root cause, no new CRITICAL `system_events` since the 2026-08-27
latch-confirmation entries, `FEATURE_ENTRY_PAUSE` still `false`, no macro-event window covers now
(`config/macro_events.json` next window is CPI 2026-09-11, not active). Book still flat: 0 open
positions, equity $87.50216036 — unchanged to the cent for the seventh consecutive day. GH #1126
(proximate mechanism) and #1127 (structural response-gap, Board decision) both remain `OPEN`,
untouched since 2026-08-31. PR #1129 — the durable-sink artifact for this very finding — remains
open, green CI, mergeable, unreviewed since it was opened six days ago for the day-5 record.

**Escalating the non-response itself, not re-describing the condition** (per the standup's own
rule for a repeat sighting): this is the **seventh** consecutive daily report of the same
unactioned P0. Cross-session sweep this run confirms no daemon/PM session has run in this repo
since the last dated log.md entries — no restart, no clear, no Board sitting on #1127 has
happened. The only two things that would change this outcome are still (a) a human running the
documented restart playbook, or (b) the Board resolving #1127's structural question. Both remain
open.

## UPDATE — 2026-09-02 08:00 UTC (Day 6, still unactioned)

Sixth daily-trading-standup sighting since GH #1121 was filed (2026-08-29). Newest `system_events`
row (2026-09-02 07:53:14 UTC) is still `CLOSE_ONLY_LATCHED`: *"STILL BLOCKED — DAY 5: Close-only
mode still active after 5d 23h..."* — elapsed now **~5d 23h18m (~143.3h) against the 1h P0 SLA**,
i.e. **143x over SLA**. No change in root cause, no new CRITICAL `system_events` since the
2026-08-27 latch-confirmation entries, `FEATURE_ENTRY_PAUSE` still `false` (not a contributing
lever). Book still flat: 0 open positions, equity $87.50216036 (unchanged to the cent since
2026-08-27, i.e. genuinely idle, not silently bleeding). This PR (#1129) — which itself carries
the incident-file + log.md record this run is updating — has sat open and unmerged for the same
window; the durable-sink artifact for the finding is itself part of the unactioned backlog.

**The finding as of today is the non-response, not the underlying condition** (per the standup's
own escalation rule): the same evidence has now been reported five times without a human clearing
the latch or an authorized agent restarting the service. Re-describing the condition further adds
no information; what would move this is either (a) a human running the documented restart
playbook, or (b) the Board deciding GH #1127's structural question (does any scheduled actor get
restart authority, or is human-only remediation an accepted opportunity-cost budget).

## What happened

Production latched **close-only mode** at **2026-08-27 08:35:28 UTC** after a stop-loss
re-placement failure on ETHUSDT, and — as of this record (2026-09-01 08:04 UTC) — has been
blocking all new entries for **~4d 23.5h (~119.5h)**, against the charter's **P0 SLA of 1
hour**. This is filed retroactively: the condition was detected and escalated on-time via GH
issue #1121 (filed 2026-08-29, day 1 of coverage), but per that issue's own 2026-08-31 comment,
no incident file and no `log.md` entry were ever created for it — only the GH issue existed.
This file and the accompanying `log.md` append close that specific gap, flagged by
`weekly-retro` on 2026-08-31 (see GH #1127) and carried forward unactioned into this run.

**This is the second occurrence of the same failure shape in eight days** — #1094 (2026-08-20
to 08-24, ~96h) was the first. The visibility fix shipped for #1094 (#1103, re-announce +
report-every-lever) worked exactly as designed here: the standup caught this recurrence on day
1 instead of not at all. **Detection is no longer the problem. Response is.**

## Detection

Caught by the daily-trading-standup's positive-state assertion (LESSONS §5.7: read the latch
state from `system_events`, don't infer it from a flat book). First escalated 2026-08-29
08:07 UTC via GH #1121. Confirmed present again on this run (2026-09-01) via the same query:
newest `system_events` row is `CLOSE_ONLY_LATCHED`, not `ENTRIES_ENABLED`.

## Impact

Live capital, ~$87.50. **No capital loss** — the halt is fail-safe by design (exits/stops stay
active, only new entries are blocked) and the book has been flat (0 open positions) for the
entire window. Pure opportunity cost: **~4d 23.5h of zero entries** on a bot whose decision
loop is demonstrably still generating real entry signals — prod logs 2026-09-01 06:55–06:59 UTC
show three consecutive `Decision: BUY | Size: 15.75` rows (a real, non-zero sizing decision,
not the pre-existing #1045/#700 near-zero-size pattern) that were silently blocked at the
`entry_coordinator` close-only early-return. This is the clearest evidence yet that the latch,
not signal quality, is the active constraint.

Weekly DD rate: equity 7 days ago (2026-08-25) $87.41 vs today $87.50 — flat, no drawdown-rate
concern. No unresolved capital-at-risk from the halt itself.

## Root cause

**Two independent issues, both already named in #1121:**

1. **Proximate trigger (2026-08-27 08:35–08:59 UTC):** ~24 consecutive `CLOSE_INVENTORY_LOCKED`
   abort cycles on ETHUSDT ("only 2.7–4.6% of intended base asset sellable"), followed by 5x
   `STOP_LOSS_PLACEMENT_FAILED` with Binance `-1111` ("price has too much precision" /
   PRICE_FILTER tickSize — LESSONS §1.1's bug class, previously hardened in #695/#699 for this
   exact code path). The position itself closed via its exchange-side stop-loss one minute
   later (trade id=19, +$0.248, 2026-08-27 08:59:05) — real exchange-side protection held; the
   failure was in our tracking/re-placement confirmation. Candidate mechanism filed separately:
   GH #1126 (tick-quantization sits inside `if symbol_info:` while the quantity cap twelve
   lines below runs unconditionally — a transient `get_symbol_info()` failure would silently
   skip quantization).

2. **Structural gap (the reason this has now sat unactioned for 5 days):**
   `_close_only_mode` is a plain in-process bool (`trading_engine.py:782`) with **no CLI/API to
   clear it remotely** and **no DB row at all** (`system_control_flags` confirmed empty,
   2026-08-31). The book resolved itself (flat since 08:59 UTC on day 1) but the latch did not
   — only a full restart of the "Trading Bot" Railway service clears it, and that action sits
   outside every scheduled agent's authorization envelope (`bot-monitor-live` is monitor-only by
   design; `live-ops` is explicitly barred from live-capital processes; the PM daemon has the
   envelope but runs only when a human starts a session — and none ran 2026-08-25→08-31). GH
   #1127 (filed by weekly-retro, 2026-08-31) names this precisely: **100% detection coverage,
   0% response coverage, and no instrument that can tell the difference.**

## Timeline

```
2026-08-27 08:35:28 UTC — close-only latch set (first CLOSE_ONLY_LATCHED-class event)
2026-08-27 08:57:48 UTC — CLOSE_INVENTORY_LOCKED, consecutive aborts: 24 (last of the series)
2026-08-27 08:57:50–58:03 — STOP_LOSS_PLACEMENT_FAILED x5, code=-1111 (PRICE_FILTER tickSize)
2026-08-27 08:58:03 UTC — RECONCILE_CRITICAL: ETHUSDT unprotected, SL re-placement failed
2026-08-27 08:59:05 UTC — trades.id=19 closes via exchange-side stop-loss, +$0.248 — book flat
2026-08-27 09:37 onward — CLOSE_ONLY_LATCHED fires hourly, elapsed climbs monotonically
                           (confirms a continuously-running process, not a crash-loop artifact)
2026-08-29 08:07:43 UTC — GH #1121 filed (day 1 of standup coverage) — no incident file, no log.md
2026-08-30 08:10 UTC     — standup comment on #1121: day 2, ~71h, still unactioned
2026-08-31 08:05 UTC     — standup comment on #1121: day 3, ~95.5h, reaches #1094 precedent scale
2026-08-31 09:38 UTC     — weekly-retro comments on #1121 (missing record noted); files GH #1126
                           (mechanism) and #1127 (structural: no scheduled actor can act)
2026-09-01 08:04 UTC     — standup writes this record + log.md entry (day 5), PR #1129 opened
2026-09-02 07:53 UTC     — newest system_events row: still CLOSE_ONLY_LATCHED, 5d23h elapsed
2026-09-02 08:00 UTC     — standup update (day 6): ~143.3h elapsed (143x the 1h P0 SLA), PR #1129
                           still open/unmerged; non-response is now the finding, not the condition
```

## Actions taken

Read-only forensic sweep only, from the daily-trading-standup scheduled task. No writes to prod
DB, no config/flag changes, no restart — all outside this agent's authorization for a
live-capital process. Queried `system_events`, `account_history`, `positions`, `trades`;
pulled `railway logs -e production -s "Trading Bot"` to confirm decision-loop liveness and the
blocked non-zero-size BUY signals; cross-checked GH #1121/#1126/#1127 and `log.md` for prior
escalation state before filing this record (per the standup's own non-duplication rule).

## Current state

**As of 2026-09-02 08:00 UTC (day 6):** close-only mode still latched, ~143.3h elapsed. 0 open
positions. Equity flat at $87.50216036 — unchanged since the halt began. `FEATURE_ENTRY_PAUSE`
confirmed `false` (not a contributing lever). No new CRITICAL `system_events` beyond the hourly
`CLOSE_ONLY_LATCHED` re-announcements. Nothing has changed mechanically since day 5 — the only
change is that the non-response window has grown by another 24h.

Prior (2026-09-01, day 5): close-only mode latched, 0 open positions, equity $87.50216036,
decision loop live with genuine non-zero-size BUY signals being blocked, not just near-zero noise.

## Recovery requirements (NOT executed — human/authorized-agent action required)

1. **Clear the latch**: human-authorized restart of the production "Trading Bot" Railway
   service (same playbook as #1094/#1121) — verify exchange-side ETHUSDT state independently
   first (no resting orders, no held/borrowed base asset) before restarting.
2. **Build the durable fix, not just the manual clear** (per #1121 item 2 and #1127): a safe
   auto-clear or remote-clear path once the triggering condition has resolved and the book has
   been flat for N cycles, so a third occurrence does not again default to "wait for a human
   session to start." This is the P0 work item GH #1127 asks the Board to choose between (build
   the clear path vs. a scheduled actor with restart authority vs. an explicit accepted
   opportunity-cost budget for human-only remediation).
3. **Verify #1126's mechanism** (tick-quantization conditional on `symbol_info`) before trusting
   the SL re-placement path again.

## Post-mortem (filled after close)

### Root cause
(see above — pending human sign-off / close; two independent issues, one proximate, one structural)
### Contributing factors
### What went well
Detection worked exactly as designed (#1103): caught on day 1, correctly distinguished from the
pre-existing #1045/#700 near-zero-sizing pattern, escalated (not re-reported) on days 2 and 3.
### What went poorly
Response, entirely. No scheduled actor in the fleet has authorization to clear a live-capital
process; the PM daemon that does only runs when a human starts a session, and none ran for six
days spanning this incident. Layer-2 recording (this file, the `log.md` entry) was also missed
until day 5, despite the gap being named explicitly on day 3 (#1121 weekly-retro comment).
### Action items (each links to a proposal or tracker)
GH #1121 (this incident, live), #1126 (proximate mechanism), #1127 (structural response gap,
Board decision needed).
