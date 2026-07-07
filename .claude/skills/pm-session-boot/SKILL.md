---
name: pm-session-boot
description: Deterministic PM session initialization — the pm.md hard-gate reads (a–l) plus scheduled-task inventory plus in-flight-work recovery, producing a standard state-of-the-world summary BEFORE any decision. Use at the start of every daemon/PM session, after compaction, after a crash/restart, or whenever the current picture feels stale.
---

# PM Session Boot

Same inputs → same picture, any session, post-compaction included. The PM is memoryless between
sessions; everything it needs is in files, and this skill is the deterministic order to read
them. Boot is READ-ONLY across layers 1, 2, 4, and 5 (`docs/architecture/memory_system.md`) —
the first write of a session is a decision, and decisions come AFTER the picture, via
`decision-record`.

## The gate: nothing material before all boxes tick

Run the pm.md required reads a–l — they are a HARD GATE, "not optional, not time-boxable, not
satisfiable by 'I already have the context'" — plus two boot-specific additions (m, n).
Parallelize into one batch:

**Identity (layer 1):**
- a. `.claude/state/charter.md` — unfilled TODO in mission/autonomy/escalation → STOP, page human.
- b. `.claude/state/risk-limits.json` — the hard lines. Missing/invalid → refuse material decisions.

**Record (layer 2):**
- c. Tail `.claude/state/log.md` (~50 lines) — recent decisions, open threads, corrections.
- d. `proposals/` + `incidents/` filtered `status: open`. Any P0 → the session scopes to it.
- e. `.claude/state/wakeups.jsonl` — lines with `wake_at <= now` are this tick's top priority.

**Backlog + code momentum:**
- f. `gh issue list --state open --label state:researching,state:proposed,state:building,state:paper`
- g. `gh issue list --state open --label needs:human-approval` — blocked-on-human set.
- h. `gh pr list --state open --json number,title,isDraft,statusCheckRollup` (mind
  `mergeable_state` — conflicted PRs run no CI, LESSONS §2.6).
- i. `git log --oneline -20` on develop. j. `docs/project_status.md` + changelog top.
- k. CODE.md / CLAUDE.md task matrix (coverage gates).

**Retrieval (layer 5):**
- k2. Explicit mem-search on the session's topic — auto-loaded context does NOT satisfy this
  (pm.md loophole rule). Empty result still counts, but say so.

**Parallel-session sweep (earned by the weekend of conflicting PRs + a session moving the main
checkout off `main`):**
- l. RUNNING sessions (if session tooling available) + always: open PRs from branches not in
  log.md; `git -C /Users/alex/Sites/ai-trading-bot branch --show-current` MUST print `main`;
  `git -C /Users/alex/Sites/ai-trading-bot worktree list` — unmerged-branch worktrees are
  someone's active workspace.

**Boot additions:**
- m. **Scheduled-task inventory**: `ls ~/.claude/scheduled-tasks` — know what's armed
  (standup, alert-monitor, event-window pause flips, retrains) and check each expected firing
  actually fired (tasks run only while the app is open — the known silent-miss mode). A pause
  flag that should have flipped and didn't is an immediate action item.
- n. **In-flight recovery (layer 4)**: read `.claude/state/handover.md` (a `session-handover`
  snapshot, if present), then scan scratchpad state JSONs + background-task outputs for
  interrupted lanes. Layer 4 is a HINT — verify each claimed lane against ground truth
  (ps/worktrees/gh) before adopting it; the exit-sweep job once completed while its handover
  state said "running." Full triage of stalls → `agent-fleet-health`.

## Output: the state-of-the-world summary

Standard block, before any recommendation or dispatch:

```
Sources checked: charter ✓ | risk-limits ✓ | log ✓ | proposals/incidents ✓ | wakeups ✓ |
  gh issues ✓ | PRs ✓ | git log ✓ | status/changelog ✓ | CODE/CLAUDE ✓ | memory ✓ |
  cross-session ✓ | scheduled-tasks ✓ | in-flight ✓        (✗ = say why; ✗ on a/b = STOP)
POSTURE: live equity + open position + guard state + active pauses/flags (one line)
DUE NOW: expired wakeups, missed scheduled firings, open P0/P1
IN FLIGHT: lanes (agent/worktree/PR/stage) adopted from recovery, each marked verified/stale
BLOCKED ON HUMAN: the needs:human-approval set + unratified layer-1 items
NEXT: the single top action and why (rubric via decision-record if it's material)
```

## Red flags

- Making any dispatch/merge/deploy before the summary exists.
- Trusting handover.md lanes without verification (layer 4 is never authoritative).
- A boot that skips m because "the tasks always fire" — the 2026-07-03 Board flag says otherwise.
- Two boots in one session producing different pictures from the same files — that's a skill
  bug; fix it in `weekly-retro`.
