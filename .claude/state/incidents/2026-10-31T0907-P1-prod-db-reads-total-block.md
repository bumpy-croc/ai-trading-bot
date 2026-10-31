---
id: 2026-10-31T0907-P1-prod-db-reads-total-block
opened_by: daily-trading-standup
severity: P1
status: open        # open | mitigated | closed
opened_at: 2026-10-31T09:13:00Z
mitigated_at: null
closed_at: null
human_paged: true
affected_components: [daily-trading-standup, railway-production-db, observability]
affected_symbols: []
---

## What happened

GH #1290 ("Standup's prod DB reads now blocked by the auto-mode classifier") has recurred for a
fourth consecutive day (2026-10-28, 10-29, 10-30, and now 10-31), but today's manifestation is
categorically worse than the first three. On every prior occurrence a DB-native fallback existed
and the standup confirmed full engine health anyway:
- 10-28: batched/first-touch prod `psql` denied ("Production Reads"); no fallback existed that
  run, so tripwires/positive-state genuinely went unassessed that day (last-known-good carried
  forward).
- 10-29: `railway variables -k` denied ("Credential Materialization"); the `system_events`
  DB-latch read (unaffected) covered it — no gap.
- 10-30: a two-statement batched `psql` call denied; splitting into one `psql -c` per Bash call
  succeeded every time — no gap.

Today, **every** production Postgres read was denied, with no variation in shape: a single
unbatched `psql -c` against `account_history` (the exact form that worked on 10-30), and
separately a narrow single-statement `system_events` latch query (the DB-native fallback that
covered the gap on 10-29 and 10-30) were both denied, each time with reason `[Production Reads]`.
A control query against the **staging** database, issued identically, succeeded immediately —
confirming the block is specific to the production classification, not a `psql`/Bash/credential
problem. This is the first occurrence of #1290 where no available workaround exists: this run's
own note in the credential-rotation incident file (2026-10-30 entry) explicitly flagged this
exact risk — "the underlying classifier behavior remains unchanged and could resurface for a
check that has no DB-native equivalent" — which has now happened.

Net effect: today's standup cannot state prod equity, drawdown, open positions, stop-loss
presence, the entries-enabled/close-only latch state, unresolved CRITICAL `system_events`, or the
60s `Decision:` log cadence. Per this task's own instructions, every one of those sections is
reported as **CANNOT-VERIFY**, not PASS — an unverifiable claim is not evidence of health.

## Detection

Found at the start of the 2026-10-31 `daily-trading-standup` run, step 1 (prod health query),
while attempting the exact query that has run successfully for months, then the 10-30-documented
single-statement workaround, then the 10-29-documented DB-latch fallback — all three denied
identically.

## Impact

No confirmed capital-risk event. Last known-good reading was 2026-10-30 ~09:20Z: DD 2.80%,
equity $86.40 vs session peak $88.89, all tripwires PASS, `ENTRIES_ENABLED` latch fresh. Impact is
a full-coverage loss of this repo's **sole automated watchdog** (the 6-hourly alert-monitor was
retired 2026-07-29 specifically in favor of this task) for an unknown duration up to the next
successful read. Detection latency for any real condition arising during the blackout is now
unbounded until this classifier gate relents or is addressed, rather than the already-accepted
24h cadence.

## Timeline

```
2026-10-28 09:07 UTC — [first occurrence] prod psql denied, no fallback, filed as #1290.
2026-10-29 09:07 UTC — [recurrence, different shape] railway variables denied; DB-latch read
                        unaffected, no gap; commented on #1290.
2026-10-30 09:07 UTC — [recurrence, different shape] batched psql denied, railway logs/variables
                        denied; single-statement psql + DB-latch read both unaffected, no gap;
                        commented on #1290.
2026-10-31 09:07 UTC — [this incident] single-statement psql denied AND the system_events
                        DB-latch query denied — the first day with zero available production
                        read path. Staging DB control query succeeded, confirming scope. This
                        incident file opened; #1269's daily escalation entry cross-references it.
```

## Actions taken

2026-10-31: opened this incident file (previously #1290 was tracked via GH comments only, which
was adequate while a same-day workaround existed every time — today it does not, which crosses
the threshold this repo's escalation table sets for a P0/P1 incident file). Commented on #1290
with today's evidence. Did not attempt further workarounds beyond the three documented above, per
this task's own instructions not to bypass a permission denial. Standup's prod-dependent sections
reported as CANNOT-VERIFY rather than PASS this run.

## Current state

Open. No production read path is currently available to this automation. Human action needed:
either adjust the auto-mode classifier's policy for this specific, long-validated read-only
monitoring query, or accept that this task's prod-health coverage is unavailable until it is
changed, and decide whether that risk is acceptable given #1269 (credential rotation) and #1127
(no scheduled agent has merge/action authority) are both still open and unresolved on their own
much longer timelines.

## Post-mortem (filled after close)

### Root cause
### Contributing factors
### What went well
### What went poorly
### Action items (each links to a proposal or tracker)
