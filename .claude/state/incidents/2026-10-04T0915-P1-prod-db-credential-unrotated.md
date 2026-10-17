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
2026-10-13 ~08:xx UTC  — [escalation #10, 5th push notification] still unrotated — **22 days**
                         since 09-21, **9 days** since the incident was opened. `origin/develop`
                         remains pinned at `812b5701` (`git log origin/develop --since
                         2026-10-04` → zero commits): **9 full days** with no merge of any kind,
                         not just this incident's PRs. A FOURTH unmerged PR now exists for this
                         incident alone — #1268 (CONFLICTING, 9d), #1275 (MERGEABLE, 5d), #1278
                         (MERGEABLE, 2d), #1280 (MERGEABLE, 1d) — this update's PR will be the
                         fifth. All four are CI-green except #1268. 5th push notification attempt:
                         `Mobile push not sent (Remote Control inactive)` — the identical failure
                         message across all 5 attempts (10-07, 10-09, 10-10/11, 10-12, 10-13)
                         confirms this is a standing configuration state on this host (Remote
                         Control not connected/active), not a transient fault — repeating the
                         attempt without the human enabling Remote Control first is very unlikely
                         to ever succeed. The GH issue comment thread is therefore the only
                         channel this task can exercise on its own; it has now been used 10 times
                         with zero recorded replies.
2026-10-14 ~09:40 UTC  — [escalation #11, 6th push notification] still unrotated — **23 days**
                       since 09-21, **10 full days** since the incident was opened, and **10
                       full days with zero merges of any kind to `origin/develop`** (still
                       pinned at `812b5701`, confirmed via a fresh `git fetch` + `git rev-list
                       --count origin/develop..HEAD` style check). 6th push-notification attempt:
                       `Mobile push not sent (Remote Control inactive)` — identical to all 5
                       prior attempts spanning 7 days (10-07 through 10-14); the host-
                       configuration gap has not been addressed. A FIFTH unmerged PR exists for
                       this incident alone — #1268 (CONFLICTING, 10d), #1275 (MERGEABLE, 6d),
                       #1278 (MERGEABLE, 3d), #1280 (MERGEABLE, 2d), #1282 (MERGEABLE, 1d) —
                       this update's PR will be the sixth. All CI-green except #1268. Zero human
                       comments on #1269 across 11 escalations. Separately, today's run noted the
                       primary checkout's local `main` is 127 commits behind `origin/main`
                       (frozen since 2026-08-24, `8da478a3`) — the already-tracked drift
                       pattern (GH #1075); this task reads ratified config from `origin/main`
                       directly per that issue's own prescription, so it does not affect the
                       tripwire computation below.
2026-10-15 ~08:xx UTC  — [escalation #12, 7th push notification attempt] still unrotated — **24
                       days** since 09-21, **11 full days** since the incident was opened, and
                       **11 full days with zero merges of any kind to `origin/develop`**
                       (confirmed via fresh `git fetch origin` + `git log origin/develop -1`:
                       still `812b5701`, 2026-10-04). A SIXTH unmerged PR exists for this
                       incident alone — #1268 (CONFLICTING, 11d), #1275 (MERGEABLE, 7d), #1278
                       (MERGEABLE, 4d), #1280 (MERGEABLE, 3d), #1282 (MERGEABLE, 1d), #1283
                       (MERGEABLE, 1d) — this update's PR is the seventh. CI is green on **all
                       nine** open PRs, confirmed individually via `gh pr checks` on each; only
                       #1268's *mergeable* state (not its CI) is CONFLICTING. That means the
                       11-day develop freeze is now attributable entirely to absent human review/
                       merge action, not to CI or the Docker pre-push hook (#1273) — every blocked
                       PR has already cleared CI. Zero human comments on #1269 across 12
                       escalations. Separately, Railway CLI auth failed this run for BOTH `railway
                       logs` (production and staging) and `railway variables` — every call
                       returned `Unauthorized. Please run 'railway login' again.` — matching the
                       known recurring CLI-auth gap; this run substituted the DB-sourced
                       `system_events` latch row (`ENTRIES_ENABLED`, whose own message text
                       asserts "no ... entry pause in force") for the blocked `FEATURE_ENTRY_PAUSE`
                       variable read, but could NOT independently confirm the 60s `Decision:` log
                       cadence or staging `gate_reason`/governor lines this run — flagged as a
                       coverage gap below, not claimed as PASS.
2026-10-17 08:xx UTC   — [escalation #14, 9th push notification attempt] still unrotated — **26
                       days** since 09-21, **13 full days** since the incident was opened.
                       `origin/develop` confirmed still pinned at `812b5701` via a fresh `git
                       fetch origin` — **13 full days, zero merges of any kind**. Deliberate
                       change this run: did NOT open a ninth branch/PR. Today's cross-session
                       sweep counted 11 open PRs total, 8 of which are this incident's own daily
                       escalation branches (#1268, #1275, #1278, #1280, #1282, #1283, #1284,
                       #1285) — a ninth would only add to the pile the same sweep flags as a
                       contributing cause of a 28-worktree overrun (limit 6; #1273/#1277).
                       Instead this update lands as a new commit on the existing, still-open
                       `standup/1016-credential-rotation-followup7` branch (PR #1285). 9th push
                       notification: `Mobile push not sent (Remote Control inactive)` — identical
                       to all 8 prior attempts spanning 10-07 through 10-17 (10 days). Separately,
                       today's prod-health check found Railway CLI auth now failing for
                       **production** too (`railway logs`/`variables`/`whoami`), not just
                       staging, and the Railway MCP connector itself down (`CONNECTION_CLOSED`)
                       — a widening of already-tracked #1141 (previously staging-only);
                       commented there with today's evidence. DB-side evidence (fresh
                       `account_history`/`system_events` rows, ~48min old) independently
                       confirms the trading engine itself remains alive and undamaged by the
                       CLI/MCP outage. Zero human comments recorded on #1269 across 14
                       escalations.
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
2026-10-12: 9th escalation, 4th push notification (failed — confirmed non-functional channel, not
just unanswered). Verified `origin/develop` is still unchanged at `812b5701`, now 8 full days
stalled, and that the PR queue for this incident alone had grown to three (#1268, #1275, #1278),
none merged, none reviewed as far as that run could tell. Restamped #1269's title again to remove
the escalation count so it stops going stale on every update. Landed on its own fresh branch
(`standup/1012-credential-rotation-followup3`).
2026-10-13: 10th escalation, 5th push notification (failed again — identical
`Remote Control inactive` message across all 5 attempts spanning 6 days, confirming this is a
standing host-configuration gap, not a flaky channel). Verified `origin/develop` is STILL
unchanged at `812b5701` — 9 full days with zero merges of any kind. PR queue for this incident
has grown to four (#1268, #1275, #1278, #1280), none merged, none reviewed; this update's PR is
the fifth. Landed on its own fresh branch (`standup/1013-credential-rotation-followup4`).
2026-10-14: 11th escalation, 6th push notification (failed again, identical
`Remote Control inactive` message spanning 7 days). Verified `origin/develop` is STILL unchanged
at `812b5701` — 10 full days with zero merges of any kind. PR queue for this incident has grown
to five (#1268, #1275, #1278, #1280, #1282), none merged, none reviewed; this update's PR is the
sixth. Landed on its own fresh branch (`standup/1014-credential-rotation-followup5`).
2026-10-15 (prior update): 12th escalation, 7th push notification attempt. Verified `origin/develop`
is STILL unchanged at `812b5701` — 11 full days with zero merges of any kind. Confirmed via
`gh pr checks` on every open PR that CI is green across the board (including #1268); the freeze
is 100% a human review/merge gap, not a technical blocker. PR queue for this incident has grown
to six (#1268, #1275, #1278, #1280, #1282, #1283), none merged, none reviewed; this update's PR
is the seventh. Landed on its own fresh branch (`standup/1015-credential-rotation-followup6`).
Railway CLI auth failed for both `logs` and `variables` this run (production and staging) —
substituted the DB-sourced entries-latch row for the blocked `FEATURE_ENTRY_PAUSE` read, but
`Decision:`-cadence and staging governor/gate_reason log checks could not be performed this run.
2026-10-16 (this update): 13th escalation, 8th push notification attempt (failed identically —
`Remote Control inactive`). Verified `origin/develop` STILL unchanged at `812b5701` — 12 full
days with zero merges of any kind. PR queue for this incident has grown to seven (#1268, #1275,
#1278, #1280, #1282, #1283, #1284), none merged, none reviewed; this update's PR is the eighth.
Commented on #1127 (the systemic "no agent has merge authority" issue, open since 2026-08-31)
with today's evidence instead of filing a duplicate. Railway CLI auth again failed for both
`logs` and `variables` (production and staging) — same substitution as prior runs for the
entries-enabled/pause check; `Decision:`-cadence and staging governor/gate_reason log checks
again could not be performed. Landed on its own fresh branch
(`standup/1016-credential-rotation-followup7`).
2026-10-16 (this update): 13th escalation, 8th push notification attempt. **25 days** since the
09-21 escalation, **12 full days** since this incident was opened. `origin/develop` confirmed
still pinned at `812b5701` (fresh `git fetch origin` + `git log origin/main..origin/develop`
shows exactly that one commit) — **12 full days, zero merges of any kind**. PR queue for this
incident alone is now **seven**: #1268 (CONFLICTING, 12d), #1275 (MERGEABLE, 8d), #1278
(MERGEABLE, 5d), #1280 (MERGEABLE, 4d), #1282 (MERGEABLE, 3d), #1283 (MERGEABLE, 2d), #1284
(MERGEABLE, 1d) — this update's PR is the eighth. Two non-escalation fix PRs (#1270, #1271, both
CI-green, 12 days old) and the weekly-retro PR (#1281, 4 days old) are stuck in the same freeze.
Root-caused this run: every open PR is CI-green and `reviewDecision` is empty (no review
requested/blocking) on all of them — there is no technical or policy gate; nothing is merging
because no human or automated actor is acting on the merge button. Commented on the pre-existing
systemic issue for this (#1127, open since 2026-08-31) with today's evidence rather than filing a
new one. 8th push-notification attempt: `Mobile push not sent (Remote Control inactive)` —
identical to all 7 prior attempts spanning 10-07 through 10-16 (9 days); this is conclusively a
standing host-configuration gap, not a flaky channel, and is not expected to self-resolve.
2026-10-17 (this update): 14th escalation, 9th push notification (failed identically —
`Remote Control inactive`). Verified `origin/develop` STILL unchanged at `812b5701` — 13 full
days with zero merges of any kind. Deliberately did NOT open a ninth escalation PR — landed
this update as a new commit on the existing open `standup/1016-credential-rotation-followup7`
branch (PR #1285) instead, since 8 unmerged daily-escalation PRs already exist and today's
cross-session sweep flagged more of them as contributing to the 28-worktree overrun. Commented
on #1269 (status update) and #1141 (Railway CLI auth now also failing for the production
project, plus the Railway MCP connector down with CONNECTION_CLOSED — a widening of that
issue's previously staging-only scope). DB-side evidence confirms the trading engine itself
remains healthy throughout.

## Current state

Credential is presumed still live and unrotated as of 2026-10-17, **26 days** after the original
09-21 escalation and **13 full days** after this incident was formally opened, with **zero
recorded human action across 14 escalations and 9 push-notification attempts — all 9 confirmed
failed to deliver** (`Remote Control inactive`). `origin/develop` has had **zero merges in 13
full days** (still pinned at `812b5701`, 2026-10-04). The PR queue for this single incident holds
at **eight** unmerged PRs — #1268 (CONFLICTING, 13d), #1275 (MERGEABLE, 9d), #1278
(MERGEABLE, 6d), #1280 (MERGEABLE, 5d), #1282 (MERGEABLE, 4d), #1283 (MERGEABLE, 3d), #1284
(MERGEABLE, 2d), #1285 (MERGEABLE, 1d, updated by this entry) — this run deliberately did not add
a ninth. CI is green on every one of them, including #1268 (whose only problem is a merge
conflict against `develop`, not a test failure) — confirmed again this run that the freeze is
100% an absent-human-action gap: nobody is merging PRs that have nothing blocking them. There is
no evidence any of them has been reviewed by a human. Two independent human actions remain
outstanding: (1) rotate the credential in Railway, (2) merge any one of the open PRs to land a
durable record on `develop` — right now `develop` carries zero record of this P1, thirteen days
after it was first opened. A third, lower-cost action would unblock the automated escalation path
itself: enable/reconnect Remote Control on this host so push notifications can reach the human at
all. The ask is unchanged from prior runs because it has never been answered. At 14 escalations
and roughly 320x the charter's 1-hour P1 SLA, the automation has exhausted the actions available
to it on this host; the two things that would move this are both human-only (rotate the
credential, or merge a PR). This report's own final message puts a direct yes/no decision to the
human in a channel distinct from the GH issue and the push notification, per the three-channel-
change guidance for escalations past the third.

## Post-mortem (filled after close)

### Root cause
### Contributing factors
### What went well
### What went poorly
### Action items (each links to a proposal or tracker)
