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
2026-10-09 ~08:xx UTC  — [escalation #7, non-conforming] GH comment + 2nd push notification. No
                         branch/commit landed for this update — confirmed absent from
                         `.claude/worktrees/` and `git log --all` on 10-10/11. The exact
                         "comment-only, no durable sink" pattern this file exists to prevent.
2026-10-10/11 08:1x UTC — [escalation #8, 3rd push notification] still unrotated — ~20 days since
                         09-21, ~7 days since this incident opened. Confirmed `origin/develop` had
                         zero merges since 2026-10-04 (`812b5701`) — six days with #1270, #1271,
                         #1275 all green/mergeable and untouched. Title restamped on #1269 to drop
                         the baked-in day-count. Branch-commit discipline restored (PR #1278).
2026-10-12 08:2x UTC   — [escalation #9, 4th push notification] still unrotated — **21 days**
                         since 09-21, **8 days** since the incident was opened. `origin/develop`
                         STILL pinned at `812b5701` — confirmed via `git log origin/develop
                         --since 2026-10-04`: zero merges in 8 full days. A THIRD unmerged PR now
                         exists for this single incident — #1268 (CONFLICTING, 8d), #1275
                         (MERGEABLE, 4d), #1278 (MERGEABLE, 1d) — yesterday's own escalation PR
                         joined the backlog instead of clearing it; this update's PR will be a
                         fourth. 4th push notification attempt: `Mobile push not sent (Remote
                         Control inactive)` — identical failure to attempts #2/#3, confirming the
                         channel itself is non-functional on this host, not merely unanswered. No
                         human comment appears on #1269 across any of the 9 escalations.
```

## Actions taken

2026-10-04: incident file + GH issue (#1269) opened; PR #1268 opened (CONFLICTING, unmerged).
2026-10-05 through 2026-10-07: three further standup-comment escalations on #1269, one including
a push notification. No credential rotation, no push of the prepared rebase, and no human comment
on #1269 recorded in any of them.
2026-10-08 (prior update): recorded the non-response itself as the finding (6th escalation),
confirmed PR #1268 and the `rebase-1268` fix remain unpushed/unlanded 4 days on, and landed this
update plus a `log.md` append on a new branch independent of the blocked ones. No code changed,
no DB accessed for writes, no credential handled or displayed.
2026-10-09: 7th GH comment escalation on #1269, 2nd push notification sent. **Not recorded on a
branch** — that run landed only a GH comment, the exact "comment-only, no durable sink" anti-pattern
this incident's own record was created to stop (LESSONS §2.17). Confirmed today: no new worktree,
branch, or commit exists for a 10-09 update anywhere in `.claude/worktrees/` or `git log --all`.
2026-10-10/11: 8th escalation, 3rd push notification. Confirmed `origin/develop` had zero merges
since 2026-10-04 (`812b5701`) — six full days with PR #1270, #1271, and #1275 all green/mergeable
and untouched, not just this one incident's PR. Title restamped on #1269 to drop the baked-in
day-count. This update restored the branch-commit discipline the 10-09 run skipped.
2026-10-12 (this update): 9th escalation, 4th push notification (failed — confirmed non-functional
channel, not just unanswered). Verified `origin/develop` is still unchanged at `812b5701`, now 8
full days stalled, and that the PR queue for this incident alone has grown to three (#1268,
#1275, #1278), none merged, none reviewed as far as this run can tell. Restamped #1269's title
again to remove the escalation count so it stops going stale on every update. Landed this update
on its own fresh branch (`standup/1012-credential-rotation-followup3`) — the 10-09 lapse has not
recurred since.

## Current state

Credential is presumed still live and unrotated as of 2026-10-12, **21 days** after the original
09-21 escalation and **8 days** after this incident was formally opened, with **zero recorded
human action across 9 escalations and 4 push-notification attempts — all 4 of which failed to
deliver** (`Remote Control inactive`), confirming the channel itself is broken, not merely
unanswered. `origin/develop` has had **zero merges in 8 days** (still pinned at `812b5701`,
2026-10-04). The PR queue for this single incident has grown to **three** unmerged PRs — #1268
(CONFLICTING, 8 days old), #1275 (MERGEABLE, 4 days old), #1278 (MERGEABLE, 1 day old) — and this
update's own PR will be a fourth. There is no evidence any of them has been reviewed by a human.
Two independent human actions remain outstanding: (1) rotate the credential in Railway, (2) merge
any one of the open PRs to land a durable record on `develop` — right now `develop` carries zero
record of this P1, eight days after it was first opened. The ask is unchanged from prior runs
because it has never been answered; what has changed is that the push-notification channel is now
confirmed non-functional on this host, so the GH issue thread is the only channel left that this
task can use on its own.

## Post-mortem (filled after close)

### Root cause
### Contributing factors
### What went well
### What went poorly
### Action items (each links to a proposal or tracker)
