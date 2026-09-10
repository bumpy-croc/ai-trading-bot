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

## UPDATE — 2026-09-10 08:06 UTC (Day 14, still unactioned — PushNotification actually sent this run)

Newest `system_events` row (2026-09-10 07:39:46.9 UTC) is still `CLOSE_ONLY_LATCHED`: *"STILL
BLOCKED — DAY 13: Close-only mode still active after 13d 23h..."* — elapsed from the verified
2026-08-27 08:35:28.14 UTC onset to this run (~08:06 UTC) is **13d 23.5h (~335.5h) against the 1h
P0 SLA, i.e. ~335.5x over SLA**. This is the **fourteenth calendar day**.

No mechanical change in the condition: 0 open positions, equity unchanged to the cent at
$87.50216036 (session peak $87.50798970, DD ≈0.0067%, no capital at risk), `FEATURE_ENTRY_PAUSE`
reconfirmed `false` (no active macro-event window — next is CPI 2026-09-11, still >12h out),
decision loop alive (`Decision:` lines flowing at normal ~60-120s cadence in the prod log tail,
mix of BUY/HOLD at Size 0.00 — correctly attributed to the latch, not #1045/#700, since the
newest state row is `CLOSE_ONLY_LATCHED` not `ENTRIES_ENABLED`). One transient WS reconnect at
06:12 UTC today, self-healed within ~30s per the known #662/#663 pattern — not a new issue.

GH #1126 (proximate `-1111` mechanism): `OPEN`, zero comments, now 10 days untouched. GH #1127
(structural response-gap, Board decision): `OPEN`, one comment, now 3 days untouched. **PR #1129
(this incident's durable-sink record) remains `OPEN`, `MERGEABLE`, CI-green, zero human reviews,
now unmerged for 9 days** — the record of this incident is still not on `develop`.

**Escalation channel change, per this task's own third-escalation rule:** this is the 14th daily
sighting and the 13th GH comment; the GH-comment channel alone has not produced a human response
in 13 days. Unlike the 2026-09-09 run (which reported no `PushNotification` tool available) and
the 2026-09-08 run (which *claimed* a push notification without producing any file/log evidence
one was sent), **this run actually has the `PushNotification` tool and used it** — see the log.md
entry for confirmation of what was sent. The decision put to the human: **merge PR #1129, and
either authorize a restart to clear the latch or explicitly say the halt is accepted so the daily
re-filing stops.**

## UPDATE — 2026-09-09 08:10 UTC (Day 12/13, still unactioned — Day 12 GH comment was comment-only, same gap as Day 8)

Newest `system_events` row (2026-09-09 07:25:14.86 UTC) is still `CLOSE_ONLY_LATCHED`: *"STILL
BLOCKED — DAY 12: Close-only mode still active after 12d 22h..."* — elapsed from the verified
2026-08-27 08:35:28.14 UTC onset (re-queried directly from `system_events`, not cited from a prior
record — LESSONS §2.13) to this run (~08:10 UTC) is **~12d 23.5h (~311.6h) against the 1h P0 SLA,
i.e. ~311.6x over SLA**. This is the **thirteenth calendar day** of the incident.

**Day 12 (2026-09-08) was comment-only, same gap as Day 8**: that day's GH #1121 comment restamped
the issue title (it had been mislabeled P1-duration text despite carrying `priority:p0`) and stated
"switching channel: sending a direct push notification" — but no incident-file or `log.md` update
landed, and PR #1129's branch has no day-12 commit (last commit is the day-11 record, `da1ba65f`,
2026-09-07). Whether the claimed push notification actually reached the human cannot be verified
from this record; note it did not change the outcome (latch is still active two days later). This
run restores file/log continuity, matching the day-9 recovery of the day-8 gap.

**State, unchanged:** 0 open positions, equity $87.50216036 — unchanged to the cent for the entire
incident. Session peak equity (trailing 30d) $87.50798970 → drawdown ≈0.0067%, not a capital-at-risk
condition. `FEATURE_ENTRY_PAUSE` reconfirmed `false` via `railway variables` (not the cause).
Decision loop confirmed alive: 91 `Decision:` lines in the last ~200-line prod log window, latest
at 08:09:41 UTC — all SELL/HOLD at `Size: 0.00` this run (no non-zero blocked BUY observed today,
unlike day 11's example; this does not change the diagnosis — the newest state row is
`CLOSE_ONLY_LATCHED`, not `ENTRIES_ENABLED`, so flatness is attributed to the latch per the
standup's own rule, not to #1045/#700 signal drought). 7 CRITICAL `system_events` in the last 7
days, all the same hourly `CLOSE_ONLY_LATCHED` re-announcement — no new failure mode.

GH #1126 (proximate `-1111` mechanism): `OPEN`, **zero comments**, untouched since filing
(2026-08-31) — 9 days. GH #1127 (structural response-gap, Board decision): `OPEN`, one comment,
last activity 2026-09-07 — 2 days. **PR #1129 remains `OPEN`, `MERGEABLE`, CI-green, zero reviews,
now unmerged for 8 days** (opened 2026-09-01) — the durable-sink artifact for this very finding is
itself part of the unactioned backlog, per §2.9(f)'s "time-to-merge, not PR-opened" rule.

**This standup subagent has no PushNotification tool available** (a main-daemon-session capability
this run does not carry) — escalation channel for this run is limited to the GH comment + this file
+ `log.md`, the same channel used for 12 prior days. Per the task's own 3rd-escalation rule this
should already be a different channel/ask; that a claimed Day-12 push notification (if it happened)
produced no change reinforces treating the response gap itself, not the detection mechanism, as
what needs the human/Board decision on #1127.

**Escalating the non-response for a twelfth-plus consecutive day**: the two things that would
change this outcome — a human running the documented restart playbook, or the Board resolving
#1127's structural question — remain both untaken. Elapsed duration is now ~3.25x the #1094
precedent (~96h).

## UPDATE — 2026-09-07 08:06 UTC (Day 11, still unactioned)

Eleventh daily-trading-standup sighting since GH #1121 was filed (2026-08-29). Newest
`system_events` row (2026-09-07 07:59:17.5 UTC) is still `CLOSE_ONLY_LATCHED`: *"STILL BLOCKED —
DAY 10: Close-only mode still active after 10d 23h..."* — elapsed from the 2026-08-27 08:35:28 UTC
onset is now **~10d 23.5h (~263.5h) against the 1h P0 SLA, i.e. ~263x over SLA**. Book still flat:
0 open positions, equity $87.50216036 — unchanged to the cent for eleven consecutive days (session
peak equity since onset is $87.50798970, current drawdown ≈0.007%, not a capital-at-risk
condition). `FEATURE_ENTRY_PAUSE` confirmed `false`; no macro-event window covers now. Decision
loop confirmed alive with a genuine non-zero signal blocked today: `2026-09-07T07:55:22 Decision:
BUY | Size: 9.97 | Confidence: 0.05` — this is a real blocked entry, not the pre-existing
#1045/#700 near-zero-sizing pattern. No new CRITICAL `system_events` beyond the hourly
`CLOSE_ONLY_LATCHED` re-announcement.

GH #1126 (proximate `-1111` mechanism) and #1127 (structural response-gap, Board decision) both
remain `OPEN`, untouched since 2026-08-31 — 7 days of no engagement on either. PR #1129 (this
incident's durable-sink record) remains open, CI-green, `MERGEABLE`, zero reviews, now unmerged
for 6 days. Cross-session sweep this run found no daemon/PM session activity addressing this
incident since the day-10 record; no restart, no latch clear, no Board sitting on #1127. Also
flagged this run (not directly part of this incident but touching the same fleet): two long-running
`claude` processes observed at ~2d10.5h wall-clock (pids 34640/34641 and 39019/39027, both
`--resume`d sessions) — outside this incident's scope to investigate further, noted for the
cross-session sweep.

**Escalating the non-response itself for the eleventh consecutive day** (per the standup's own
rule): the two things that would change this outcome — a human running the documented restart
playbook, or the Board resolving #1127 — remain both untaken. Elapsed duration is now ~2.75x the
#1094 precedent (~96h).

## UPDATE — 2026-09-06 08:04 UTC (Day 10, still unactioned)

Tenth daily-trading-standup sighting since GH #1121 was filed (2026-08-29). Newest `system_events`
row (2026-09-06 07:44:17.6 UTC) is still `CLOSE_ONLY_LATCHED`: *"STILL BLOCKED — DAY 9: Close-only
mode still active after 9d 23h..."* — elapsed from the 2026-08-27 08:35:28.14 UTC onset is now
**~9d 23h9m (~239.2h) against the 1h P0 SLA, i.e. ~239x over SLA**. Book still flat: 0 open
positions, equity $87.50216036 — unchanged to the cent for ten consecutive days (session peak
equity over the trailing 30d is $87.50798970, current drawdown ≈0.007%, not a capital-at-risk
condition). `FEATURE_ENTRY_PAUSE` confirmed `false`; no macro-event window covers now. No new
CRITICAL `system_events` beyond the hourly `CLOSE_ONLY_LATCHED` re-announcement (last 7 days of
CRITICAL events are exclusively this same latch message, escalating DAY 3 → DAY 9).

**PR #1129 (this incident's durable-sink record) has now sat unmerged for 5 days** — CI-green,
`MERGEABLE`, zero reviews since it was opened 2026-09-01 for the day-5 record. This run pushes the
day-10 update onto the same branch rather than opening a new PR, consistent with the standup's
non-duplication rule.

GH #1126 (proximate `-1111` mechanism) and #1127 (structural response-gap, Board decision) both
remain `OPEN`, untouched since 2026-08-31 — 6 days of no engagement on either. Cross-session sweep
this run found no daemon/PM session activity addressing this incident since the day-9 record; no
restart, no latch clear, no Board sitting on #1127.

**Escalating the non-response itself for the tenth consecutive day** (per the standup's own rule):
the two things that would change this outcome — a human running the documented restart playbook,
or the Board resolving #1127 — remain both untaken. This is now more than 2.5x the #1094 precedent
(~96h) in unactioned duration.

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
2026-09-03 08:01 UTC     — standup update (day 7): ~167h elapsed (167x SLA), PR #1129 unmerged 2d
2026-09-04 08:07 UTC     — standup update (day 8, comment-only): ~191.5h elapsed, flags PR #1129
                           stalled 3 days but does not land a file/log record
2026-09-05 08:05 UTC     — standup update (day 9): ~215.5h elapsed (215x SLA), restores file/log
                           continuity; PR #1129 unmerged 4 days
2026-09-06 08:04 UTC     — standup update (day 10): ~239.2h elapsed (239x SLA), PR #1129 unmerged
                           5 days; GH #1126/#1127 untouched 6 days
2026-09-07 08:06 UTC     — standup update (day 11): ~263.5h elapsed (263x SLA), PR #1129 unmerged
                           6 days; genuine non-zero blocked BUY signal observed
2026-09-08 08:05 UTC     — standup comment-only (day 12): title restamped, claimed push
                           notification sent; no file/log update landed (same gap as day 8)
2026-09-09 08:10 UTC     — standup update (day 12/13): ~311.6h elapsed (311.6x SLA), restores
                           file/log continuity; PR #1129 unmerged 8 days; GH #1126 untouched 9 days
```

## Actions taken

Read-only forensic sweep only, from the daily-trading-standup scheduled task. No writes to prod
DB, no config/flag changes, no restart — all outside this agent's authorization for a
live-capital process. Queried `system_events`, `account_history`, `positions`, `trades`;
pulled `railway logs -e production -s "Trading Bot"` to confirm decision-loop liveness and the
blocked non-zero-size BUY signals; cross-checked GH #1121/#1126/#1127 and `log.md` for prior
escalation state before filing this record (per the standup's own non-duplication rule).

## Current state

**As of 2026-09-09 08:10 UTC (day 12/13):** close-only mode still latched, ~311.6h elapsed. 0 open
positions. Equity flat at $87.50216036 — unchanged since the halt began, thirteen consecutive
calendar days. `FEATURE_ENTRY_PAUSE` confirmed `false` (not a contributing lever). No new CRITICAL
`system_events` beyond the hourly `CLOSE_ONLY_LATCHED` re-announcements. Nothing has changed
mechanically since day 11 — the only change is that the non-response window has grown by ~2 more
days (including one comment-only day), and PR #1129 is now 8 days unmerged.

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
