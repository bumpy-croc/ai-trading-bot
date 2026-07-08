---
name: session-handover
description: Serialize in-flight state before context loss — write a snapshot of every live lane (agent, worktree, stage, state file, next action), armed watchers, and pending Board items to .claude/state/handover.md. Use before ending a session with work in flight, when compaction looms, before a risky long-running step, or on request ("hand over", "checkpoint the session").
---

# Session Handover

Context is the most perishable asset in this system: compaction, crashes, and app closes eat it
without warning, and wake-ups are lossy (the exit-sweep agent finished all 18 runs while its
wake-link was dead — only its scratchpad state made the results salvageable, 2026-07-04).
This skill serializes the CURRENT session's picture so `pm-session-boot` step (n) can rebuild
it in seconds.

**Layer discipline:** this skill writes ONLY layer 4 — `.claude/state/handover.md` is an
overwrite-ok SNAPSHOT, not a log (`docs/architecture/memory_system.md`). It is never
authoritative: boot verifies every lane against ground truth. Anything that must survive
authoritatively (decisions, verdicts, incident state) belongs in layer 2 via `decision-record`
— write those FIRST, then snapshot. A handover is not a substitute for logging.

## When to write

- Before deliberately ending a session with unfinished lanes.
- When context pressure is visible (long session, big tool outputs) — don't wait for compaction.
- Immediately after dispatching a long-running background job (the snapshot is the backstop's
  map if the wake-up never arrives).
- After any lane changes state materially (stage advanced, PR opened, agent died).

Overwrite the whole file each time — stale partial snapshots are worse than none.

## Format: `.claude/state/handover.md`

```markdown
# Session Handover — <UTC timestamp> — session <id/description>
STALENESS WARNING: snapshot, not truth. Verify every lane before acting (pm-session-boot n).

## Live lanes
### <lane name> (one block per lane)
- owner: <agent id/name or "this session">
- worktree/branch: <path> / <branch>          # or "none"
- PR/issue: #NNN (state)
- stage: <last KNOWN completed stage — completed, not attempted>
- state file: <absolute path to its crash-safe JSON in scratchpad>
- next action: <the single next step, executable without this session's context>
- verify by: <ground-truth check: file that must exist, PR state, process name>

## Armed watchers / background jobs
- <job>: started <time>, expected done <time>, output lands at <path>, backstop: <what/when>

## Pending Board items
- <item> — waiting since <date>, ref <issue/log entry>

## Scheduled expectations
- <task>: should fire at <time UTC>; if missed → <action>

## Do-NOT list (session-specific live hazards)
- e.g. "worktree X is mid-rebase — don't prune"; "staging flag Y deliberately ON for trial Z"
```

Rules for content:
- **Stage = last verifiable checkpoint**, phrased against artifacts ("bundle synced, model.onnx
  verified") not intentions ("training almost done"). The reader must be able to verify without
  trusting you — relayed claims have been wrong before (LESSONS §2.5, the 2026-07-05
  session-integrity note).
- **Next action must be self-contained**: paths absolute, commands concrete, no "continue as
  discussed." Assume the reader has zero conversational context — same standard as the
  `delegation-protocol` dispatch prompt.
- Point at each lane's own crash-safe state JSON rather than duplicating its contents — one
  definition per fact; the JSON is maintained by the lane owner after every stage
  (`delegation-protocol` contract).
- Include the Do-NOT list: the most expensive handover failures are a successor "cleaning up"
  something deliberately in flight (a reviewer mid-read worktree — the #843 review-collision
  class).

## The contract with boot

`pm-session-boot` (n) reads this file, then VERIFIES each lane (process alive? artifact exists?
PR state matches?) before adopting it — because between snapshot and boot, lanes finish, die,
or get collected by watchers. A handover lane that fails verification goes to
`agent-fleet-health` for triage, not straight to re-dispatch (the work may already be done).

## Red flags

- Recording a decision or verdict ONLY here — layer 4 evaporates by design; log.md or it
  didn't happen.
- A lane with no state file and no verify-by — unrecoverable by construction; fix the lane's
  discipline, not just the snapshot.
- Appending history to handover.md — it's a snapshot; history lives in log.md.
- Handing over an unpushed branch as a lane — push first; a worktree-only branch dies with
  the machine.
