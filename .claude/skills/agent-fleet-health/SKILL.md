---
name: agent-fleet-health
description: The stall sweep for the agent fleet — find silent/stuck agents, salvage completed work from wake-loss, prune orphaned worktrees, catch dangling PRs, and guard the main checkout's branch. Use on a periodic cadence, when a dispatched agent has gone quiet, or when worktrees/PRs are accumulating.
---

# Agent Fleet Health

Dispatched agents fail in two opposite ways: they die silently (work lost), or they finish and
the wake-up link dies (work COMPLETED but never collected — the 2026-07-04 exit-sweep agent had
finished all 18 runs; only the notification broke; results were salvaged from its scratchpad).
And sometimes they neither die nor finish: a session that shipped #855 kept churning ~13% CPU
for 7h after its purpose completed and had to be killed (PID hunt, 2026-07-04 ops entry).
This sweep tells the three apart. Operates on layer 4 (working state); material actions
(kills, salvages) log to layer 2. See `docs/architecture/memory_system.md`.

## The sweep

**1. Inventory the fleet (three sources, cross-referenced):**
```bash
ps aux | grep -E "claude" | grep -v grep                     # owning processes
git -C /Users/alex/Sites/ai-trading-bot worktree list        # workspaces
ls -lt /private/tmp/claude-501/*/*/scratchpad/ 2>/dev/null   # state files, by mtime
```
Plus session tooling if available (`ccd_session_mgmt` list_sessions), and the log.md tail for
what was dispatched and why.

**2. Classify each lane:**

| Signal | Meaning | Action |
|---|---|---|
| Process alive, state file mtime fresh | working | leave alone |
| Process alive, silent >90min, state stale, no recent tool activity | STALLED or purpose-complete churn | read its state file + transcript; if purpose complete (its PR merged, its report filed) → kill it; else resume with a state recap |
| No process, state file shows unfinished stage | wake-loss OR crash | check for completed outputs FIRST (the exit-sweep lesson: the background job may have finished) — salvage results before re-running anything |
| No process, outputs complete, never reported | wake-loss after success | collect results, log the salvage, close the lane |

**Resume-with-state:** when resuming/re-dispatching, feed the agent its own crash-safe state
JSON + a recap of stage/next-action (the `delegation-protocol` contract requires agents to
maintain these precisely so this step works statelessly).

**3. Orphaned worktrees.** For each worktree: branch merged or PR closed → remove
(`git worktree remove <path>` + `git worktree prune`; the `clean_gone` command handles [gone]
branches). Worktrees on UNMERGED branches are someone's active workspace — never delete without
matching them to a lane (pm.md gate l). Tournament/smoke worktrees also need artifact hygiene:
no stray model dirs, no `latest` symlink moved (`model-tournament` deliverable 3).

**4. Dangling PRs.** `gh pr list --state open` — every open PR maps to a live lane, a log.md
entry, or gets flagged. A PR whose branch's worktree is gone and whose author-session is dead is
a decision for the PM: adopt, close, or re-dispatch. Check `mergeable_state` — a conflicted PR
runs NO CI and looks deceptively "pending forever" (LESSONS §2.6).

**5. Main-checkout guard.** `git -C /Users/alex/Sites/ai-trading-bot branch --show-current`
MUST print `main`. A parallel session once switched it (the weekend that earned pm.md gate l);
the main checkout is the production reference — flag loudly, restore only if its tree is clean,
and never run git mutations there (LESSONS §3).

## The wake-loss reality (why backstops, not trust)

Wake-ups are lossy: scheduled tasks run only while the app is open (flagged to the Board
2026-07-03), background-completion notifications can break, and compaction can eat a
coordinator's context. Backstop pattern for every long-running dispatch:
- the WORKER runs long steps as a single background process and END TURNS (never polls), with a
  crash-safe state JSON updated after every stage;
- the PM arms a **backstop watcher** — `run_in_background` + notify — on every long dispatch,
  so a lost wake-up degrades to a late collection, not lost work;
- **never detached/nohup processes** — they survive their owner with no collection path and
  become the 7h-churn class.

## Record

Sweep summary (lanes found, classifications, kills, salvages, worktrees pruned) — one log.md
entry if anything material happened; routine clean sweeps don't need one. Update
`.claude/state/handover.md` if lane states changed (`session-handover`). Kills of live-capital-
relevant agents are decisions (`decision-record`).

## Red flags

- Killing a lane without reading its state file first (may be mid-critical-section on prod).
- Re-running an expensive job before checking whether the "dead" agent already finished it.
- A worktree removal that takes uncommitted work with it — `git -C <wt> status` first.
- "The agent said it's done" without artifacts — verify relayed claims (LESSONS §2.5).
