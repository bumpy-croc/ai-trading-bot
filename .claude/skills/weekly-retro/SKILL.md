---
name: weekly-retro
description: The weekly consolidation pass — review the week's log, incidents, agent failures, and prediction-vs-outcome record, then distill them into LESSONS.md appends and skill amendments. Use on the weekly cadence (charter review rhythm) or after an unusually dense week. This is the ONLY routine editor of the distillate layer; output is concrete process diffs, not prose.
---

# Weekly Retro

Episodic → semantic. The week's events live in layer 2 (log.md, incidents, experiments); this
skill distills them into layer 3 (LESSONS.md, the skills themselves) — and it is the only skill
that routinely edits layer 3, which is what keeps the distillate curated instead of accreted
(`docs/architecture/memory_system.md`, discipline 4). A retro that produces "we should be more
careful" has failed; the unit of output is a DIFF — a LESSONS append, a skill amendment, a
checklist line, a new tripwire.

## Inputs (all layer 2 + working state)

0. **AGENDA.md (this skill's directory) — read FIRST.** The running agenda any agent or the PM
   appends to during the week the moment something retro-worthy happens. Every item must be
   actioned (LESSONS append, skill amendment, or GH issue) or given an explicit written
   disposition in the retro PR — never silently dropped. After actioning, **clear the file back
   to its header template in the same PR**. An empty agenda is not a skipped step: still sweep
   inputs 1–7 below; the agenda supplements the sweep, it does not replace it.
0b. **The previous retro's own PR — verify it MERGED, and that anything it deferred landed.**
   `gh pr list --state merged --search "retro"` / check the branch is on `develop`. If last week's
   retro deferred distillate to another PR, open that PR: **merged → fine; closed or still open →
   its distillate is not on `develop` and recovering it is this retro's first job** (diff its
   files against `develop`, re-land the distillate-only subset). Also re-check the issues the last
   retro filed — an issue that has sat untouched for a week is itself a finding. Earned: #1026
   closed unmerged 2026-07-21 after the 07-20 retro deliberately did not reproduce it; LESSONS §2.9.
   **If that PR is still open: `git reset --hard origin/<its-branch>` and build this retro on top of
   it** — this PR then supersedes it and merges whether or not the old one lands, instead of adding a
   second stranded PR to the queue. And **if this is the second consecutive stranded retro PR, the
   completion summary must lead with it, addressed to the human** — the retro cannot merge its own
   output, so an unmerged queue is invisible everywhere else (LESSONS §2.9 rule (d); #1047 sat
   `CLEAN` and unmerged for 14 days, taking §2.9/§2.10 off `develop` with it).
1. **log.md** — the week's entries end to end, not just the tail. **A week with no entries is a
   finding, not a quiet week** — cross-check against the scheduled-task traces (input 6): if the
   monitors ran and the log is empty, findings were surfaced and dropped (LESSONS §2.10).
2. **Incidents + corrections** — anything opened/closed/corrected. Corrections are retro gold:
   the phantom-peak withdrawal (2026-07-04 13:55) became LESSONS §5.6 (the distinct-count check)
   — that's the pipeline working as designed.
3. **Agent failures** — stalls, wake-losses, kills, review collisions, fabricated/wrong relayed
   claims (`agent-fleet-health` findings; the 2026-07-05 ml-engineer session-integrity note).
4. **Prediction-vs-outcome record** — every dated prediction made in the log (deploy will be
   healthy, model will beat incumbent, "no midnight heroics" calls) vs what happened. Agents'
   calibration is a charter review item (monthly: "review calibration of each agent"); the
   risk-officer's own calibration note (2026-07-04: "consistent, not a reversal") is the format.
5. **Experiments** — verdicts vs their preregistered thresholds; any p-hacking drift.
6. **Scheduled tasks** — audit against the **registry**, not the directory. Call
   `mcp__scheduled-tasks__list_scheduled_tasks` and **diff it both ways** against
   `ls ~/.claude/scheduled-tasks`: a directory with no registry entry is **NOT a live task**, and its
   `SKILL.md` survives deregistration so it looks installed forever. (`ls` alone reported "no task
   missed its schedule" for three consecutive retros while six directories were unregistered —
   LESSONS §3. As of 2026-08-17: 19 directories, 13 registered, **4 enabled**.)
   Then check each live task's `enabled` / `lastRunAt` **and its expected trace**: date sessions by
   the first internal `"timestamp"` in `~/.claude/projects/<slug>/*.jsonl`, **not** by file mtime
   (claude-mem rewrites mtimes). Failure modes, all silent:
   - **didn't fire** — app closed (tasks run only while it's open);
   - **deregistered** — absent from the registry entirely. **This proves deregistration, not
     failure**: on 2026-08-13 Alex confirmed all six unregistered directories were deliberate
     retirements, after the 08-10 retro reported them as a 12-day monitoring outage. Ask before
     calling a missing registry entry an outage (LESSONS §3);
   - **fired and died on turn 1** — grep transcripts for BOTH `may not exist or you may not have
     access` (stale model-provider selection, #1051) and `hit your weekly limit` / `hit your usage
     limit` (quota exhaustion — killed the 2026-08-15 standup, and the 08-13 prod-promote agent
     mid-deploy). `lastRunAt` records the *attempt*, so these look healthy; **treat any ~20-line
     transcript as failed until proven otherwise**;
   - **fired late as a catch-up, masking a missed slot** — on app reopen every overdue task fires at
     once. Several tasks sharing a `lastRunAt` to the second (2026-08-17: three within 32ms) is a
     catch-up batch, not punctuality. Check each `lastRunAt` against its `cronExpression` to find the
     slot actually missed, and note that those runs execute concurrently against one repo.
   A task that ran but produced no artifact is a finding of the same weight as one that never fired.
7. **Model scoreboard** — new rows this week; stale `latest` claims.

## The distillation pass

For each notable event, ask: **which layer-3 artifact would have prevented it or captured it?**

| Event class | Distills into |
|---|---|
| New bug class / trap → rule | LESSONS.md append (its "the trap → the rule" format, newest at section bottom) |
| A skill's procedure was wrong or incomplete | Amend the SKILL.md (PR like any code change) |
| A monitoring signature learned | LESSONS §5 (NOT the bot-monitor-live skill — method vs specifics, one-definition rule) |
| A process gap with no owner skill | Proposal for a new skill or checklist (Board-visible) |
| One-off with no generalizable rule | Nothing — resist the urge; a lessons file full of episodes is an unread lessons file |

Rules for editing layer 3:
- Every new rule cites its earned event (issue #, log date, incident id) — untraceable rules
  rot into superstition; the existing LESSONS/skills style is "each one paid for."
- Prefer amending an existing rule over adding a near-duplicate (consolidation, not accretion).
- Deleting a rule is allowed when its precondition died (e.g. a fixed bug's workaround) — note
  the removal in the retro log entry so the record explains the edit.
- Mid-week emergency LESSONS appends are legal; the retro reconciles/merges them.

## Scoreboard + tracker updates

Append a `docs/research/model-promotions.md` row for any `latest` symlink change this week
(the append-only promotion log — with eval numbers; there is no separate `model-scoreboard.md`,
the retrain task writes here too). Verify the deployed model matches reality
(`atb live-control list-models` / registry symlinks). No promotion this week → nothing to append.
Check the standup tripwire table is still the ratified one.

## Output (the definition of done)

A log.md entry (kind `note`, via `decision-record`) listing, as diffs:
- LESSONS.md sections appended/amended (with the earning event for each);
- skills amended (PR links);
- calibration verdicts per agent (over/under-confident, with the week's examples);
- scheduled-task audit result (fired/missed);
- process changes proposed to the Board (anything touching layer 1 → `risk-ratification`).

Zero diffs is a legitimate outcome for a quiet week — say so explicitly rather than
manufacturing lessons.

## Red flags

- A retro entry that is narrative summary without a single artifact diff.
- Editing log.md or closed incidents "while we're at it" — layer 2 is append-only, always.
- A lesson written into two places (skill + LESSONS) — pick the right layer, link the other.
- Skipping the prediction-vs-outcome pass because the week "went fine" — calibration drift is
  invisible exactly when things go fine.
- **Deferring this week's distillate to a PR you don't control.** "Recorded the per-item map, the
  other PR carries the artifacts" is how a week of lessons dies (#1026). Re-land the distillate-only
  subset yourself and accept the duplicate-append risk — it is strictly cheaper (LESSONS §2.9).
- **Counting a filed issue as a disposition.** An issue with no assignee and no dispatched agent is
  a note to yourself (LESSONS §2.11) — 5/5 issues from the 07-27 retro had zero activity 14 days on.
  Filing is still the right action for code changes; just don't report it as resolved, and escalate
  the *queue* (N unowned, M days) as a single item once it repeats.
- **Reporting a green scheduled-task audit from the wrong instrument** — see input 6. "All tasks
  fired" is only sayable from the registry plus per-task artifact evidence.
- **Bundling a log-consolidation or human-directed log/incident rewrite into the retro PR.** The
  retro PR ships **distillate only** (LESSONS/skill diffs + AGENDA clear + the append-only retro
  log entry). A destructive log rewrite conflicts on every new log append and strands the whole
  distillate behind a merge conflict — the 2026-07-13 retro (#1026) sat CI-green but unmerged for
  7 days because a 52-line PM-directed log-consolidation was bundled in. Log-consolidations and
  incident post-mortems are separate PRs, reviewed on their own merits.
