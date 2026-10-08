---
id: 2026-10-04T0915-P1-prod-db-credential-unrotated
opened_by: daily-trading-standup
severity: P1
status: open        # open | mitigated | closed
opened_at: 2026-10-04T08:15:00Z
mitigated_at: null
closed_at: null
human_paged: true
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
database, so a real secret has been outstanding and unrotated for 17 days as of this update.
Treating this as capital-risk-adjacent rather than cosmetic.

## Timeline

```
2026-09-21 ~08:xx UTC — [detection] prod DB credential printed in daily-trading-standup output;
                         escalated in that day's completion summary (per PR #1266 reference)
2026-09-28 08:10 UTC   — [non-response confirmed] PR #1266 ("needs a human" item 3) restates it
                         as still outstanding
2026-10-04 08:15 UTC   — [escalation] this incident file + GH issue (#1269) opened; still
                         unrotated, 13 days after first escalation. PR #1268 opened to land the
                         record on develop.
2026-10-04 08:22 UTC   — [blocked] PR #1268 goes CONFLICTING against develop; a local rebase
                         (`rebase-1268` branch) is prepared same-day but is never pushed — the
                         host's Docker daemon is hung, and the pre-push hook's test run fails
                         against it (root cause tracked separately as GH #1273, opened 10-05).
2026-10-05 08:21 UTC   — [escalation #2] standup comment on #1269: still unrotated, PR #1268
                         still conflicting.
2026-10-05 08:28 UTC   — [escalation #3] second same-day standup comment, same content.
2026-10-06 08:35 UTC   — [escalation #4] standup comment on #1269: still unrotated.
2026-10-07 08:22 UTC   — [escalation #5] standup comment on #1269 + a push notification sent
                         (per that run's own report) — first channel change, still no human
                         response recorded on the issue.
2026-10-08 08:xx UTC   — [escalation #6, this update] still unrotated — **17 days** since the
                         09-21 escalation, **4 days** since the GH issue/PR/incident file were
                         opened. PR #1268 is UNCHANGED since 2026-10-04 08:22 (still CONFLICTING):
                         `git log origin/standup/1004-credential-rotation-record -1` is still
                         `09d19327` (the original incident-only commit). A local merge/rebase
                         attempt (`5f424e29`, merging develop in) and a separate `rebase-1268`
                         branch with the actual conflict fix (`ee483d28`) both exist only on
                         local disk — confirmed today via `git branch -r --contains ee483d28`
                         (empty) and `git ls-remote --heads origin rebase-1268` (empty). Root
                         cause is still GH #1273
                         (hung Docker / false pre-push failures on this host), now also
                         confirmed today to be blocking 17 other unrelated fix branches and to
                         have caused the 2026-10-05 `weekly-retro` run to fail mid-push with no
                         PR landed. This incident file itself is being recorded via a **separate,
                         fresh branch** (`standup/1008-credential-rotation-followup`) rather than
                         by amending PR #1268 or `rebase-1268`, specifically so it does not also
                         depend on #1273 being fixed to land.
2026-10-08 08:2x UTC   — [correction] that branch's own push just succeeded cleanly: `docker info`
                         responds, the fast unit suite ran and passed pre-push (2985 passed,
                         58s), and `git push` went through with no retry needed. GH #1273's
                         premise (Docker hung on this host) does not hold right now — whether it
                         recovered on its own or the 10-04/10-05 hang was transient, #1273 is not
                         currently blocking a push. That means `rebase-1268` (and the other 17
                         branches tracked by #1273) are pushable right now by anyone who runs it;
                         the 4-day gap since 10-04 looks like "nobody has tried since," not
                         necessarily "still broken." Not independently re-verified for the other
                         worktrees — flagged as the actionable next step, not an assumption that
                         the whole backlog is clear.
```

## Actions taken

2026-10-04: incident file + GH issue (#1269) opened; PR #1268 opened (CONFLICTING, unmerged).
2026-10-05 through 2026-10-07: three further standup-comment escalations on #1269, one including
a push notification. No credential rotation, no push of the prepared rebase, and no human comment
on #1269 recorded in any of them.
2026-10-08 (this update): recorded the non-response itself as the finding (6th escalation),
confirmed PR #1268 and the `rebase-1268` fix remain unpushed/unlanded 4 days on, and landed this
update plus a `log.md` append on a new branch independent of the blocked ones. No code changed,
no DB accessed for writes, no credential handled or displayed.

## Current state

Credential is presumed still live and unrotated as of 2026-10-08, **17 days** after the original
09-21 escalation and **4 days** after this incident was first formally opened, with zero recorded
human action across 6 escalations and one channel change (push notification, 10-07). PR #1268
remains CONFLICTING and unmerged; the fix for it (`rebase-1268`) exists only on local disk, unpushed,
same as 17 other finished fix branches — all blocked by GH #1273 (hung Docker / pre-push hook
false-failures on this host), still open at P1 since 2026-10-05. Two independent human actions are
outstanding: (1) rotate the credential in Railway, (2) unblock pushes on this host (or have someone
push `rebase-1268` from a different, healthy host) so PR #1268 can land. Per this task's own
escalation rules, a 6th identical ask through the same channel would itself be theatre — today's
GH comment changes the ask to a direct one-line decision request rather than repeating the status.

## Post-mortem (filled after close)

### Root cause
### Contributing factors
### What went well
### What went poorly
### Action items (each links to a proposal or tracker)
