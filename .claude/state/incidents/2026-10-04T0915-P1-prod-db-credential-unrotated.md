---
id: 2026-10-04T0915-P1-prod-db-credential-unrotated
opened_by: daily-trading-standup
severity: P1
status: open        # open | mitigated | closed
opened_at: 2026-10-04T08:15:00Z
mitigated_at: null
closed_at: null
human_paged: false
affected_components: [railway-production-db, secrets-handling]
affected_symbols: []
---

## What happened

The production Postgres connection string (`RAILWAY_PRODUCTION_DATABASE_URL`) was printed in
the output of the 2026-09-21 `daily-trading-standup` run. That run escalated it with "Rotate the
prod DB credential printed by the 09-21 standup (escalated 09-21)". As of 2026-10-04 (13 days
later), the credential has not been rotated, and the only durable record of the escalation is a
single line in the "Needs a human" section of PR
[#1266](https://github.com/bumpy-croc/ai-trading-bot/pull/1266)'s body (an unmerged,
docs-only weekly-retro PR, open since 2026-09-28). There is no entry in `.claude/state/log.md`,
no incident file, and no GitHub issue for it — confirmed via:
- `grep -i credential .claude/state/log.md` → no match for this item.
- `.claude/state/incidents/` → no prior file for it.
- `gh search issues 'repo:bumpy-croc/ai-trading-bot rotate credential in:body'` → only the PR
  #1266 body hit and an unrelated closed issue (#600).

This repo's own retro history shows exactly this failure mode before: PR #1026 (2026-07-13 retro)
was closed unmerged on 2026-07-21 and its distillate was lost until manually recovered three weeks
later. A credential-rotation action item living only inside an open PR body is one `gh pr close`
away from the same fate.

This standup run did **not** attempt to read, locate, or reproduce the credential value itself —
only to confirm the escalation exists and has gone unactioned. Per this agent's operating rules,
credential rotation is a human action (Railway dashboard: regenerate the Postgres credential /
`DATABASE_URL` for the production Postgres plugin, then update `RAILWAY_PRODUCTION_DATABASE_URL`
wherever it's consumed — `.env`, Railway service vars, any other reader).

## Detection

Found during the 2026-10-04 `daily-trading-standup` cross-session sweep, while investigating why
PR #1266 and #1263 (both weekly-retro, CI-green, no conflicts) have sat unmerged for ~6-13 days.
PR #1266's own body lists this as outstanding. This is a secondary/non-response finding, not a
newly observed leak.

## Impact

No evidence of unauthorized DB access found in this run (not specifically audited here — a
separate check of `pg_stat_activity` / connection logs for unrecognized clients is recommended).
Paper only in terms of immediate observed effect; the exposure itself is to the **production**
database, so a real secret has been outstanding and unrotated for 13 days. Treating this as
capital-risk-adjacent rather than cosmetic.

## Timeline

```
2026-09-21 ~08:xx UTC — [detection] prod DB credential printed in daily-trading-standup output;
                         escalated in that day's completion summary (per PR #1266 reference)
2026-09-28 08:10 UTC   — [non-response confirmed] PR #1266 ("needs a human" item 3) restates it
                         as still outstanding
2026-10-04 08:15 UTC   — [escalation] this incident file + GH issue opened; still unrotated,
                         13 days after first escalation
```

## Actions taken

None by this agent beyond recording the finding durably (this file, a `log.md` append, and a
GitHub issue) and opening a PR to land them on `develop`. No code changed, no DB accessed for
writes, no credential handled or displayed.

## Current state

Credential is presumed still live and unrotated. Awaiting human action (rotate in Railway, verify
the new value is picked up by the production service, confirm no unexpected clients have been
using the old one).

## Post-mortem (filled after close)

### Root cause
### Contributing factors
### What went well
### What went poorly
### Action items (each links to a proposal or tracker)
