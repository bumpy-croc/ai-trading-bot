---
name: delegation-protocol
description: The dispatch contract between the PM and specialist agents — what every dispatch prompt must contain (isolation, compute discipline, crash-safe state, reporting) and what the PM owes every dispatch (backstop watchers, review gauntlet for money-path code, consolidated fix rounds, resume-with-state). Use when dispatching any agent, writing a dispatch prompt, or diagnosing why a delegation failed.
---

# Delegation Protocol

Every clause below was earned by a real delegation failure: a session moved the main checkout
off `main`; parallel heavy backtests cooked the machine and broke determinism; a 7h agent
churned CPU after finishing; wake-loss stranded completed work; a reviewer read a worktree
mid-fix and filed phantom findings; relayed claims turned out fabricated or wrong. State JSONs
are layer 4, dispatch decisions layer 2 (`docs/architecture/memory_system.md`).

## The dispatch prompt — mandatory clauses (copy into every long/consequential dispatch)

1. **Isolation.** Work in a disposable worktree from `origin/develop`
   (`git worktree add .claude/worktrees/<name> origin/develop --detach` or `-b <branch>`).
   NEVER touch `/Users/alex/Sites/ai-trading-bot` (the main checkout IS the prod reference —
   a cherry-pick scare and a branch-switch incident earned this), never staging/prod, never a
   shared registry's `latest` symlink. The prompt must be self-contained: paths absolute,
   context included — the agent has none of yours.
2. **Compute discipline.** Heavy jobs (training, backtests) STRICTLY SEQUENTIAL — one at a
   time machine-wide (thermal + the 0.1s-timeout non-determinism under CPU contention, #913;
   also the standing "run backtests sequentially" feedback). Expect 1.5–4x nominal durations
   under load. Cloud (SageMaker) jobs may parallelize.
3. **Long steps: background + END TURN.** One background process per long step, then end the
   turn — never poll, never `sleep`-loop, NEVER detached/nohup (no collection path — the
   7h-churn class).
4. **Crash-safe state.** Maintain an incremental state JSON in the scratchpad, updated after
   EVERY stage (stage completed, artifact paths, next action). Wake-ups are lossy; this file
   is how `agent-fleet-health` / `pm-session-boot` salvage or resume the lane statelessly —
   it saved the 18-run exit sweep.
5. **Continuation recap every turn.** Each turn ends with a one-block recap (done / doing /
   next / state-file path) so any resumer — including you after a wake-loss — picks up cold.
6. **No chips / no scope-spawning.** Out-of-scope findings go in the report to the PM, who
   decides; don't spawn side-tasks from inside a dispatch.
7. **Report to PM, claims with evidence.** Final message = what was done + artifact paths +
   verification performed. Never relay another agent's claim as fact — verify against
   filesystem/logs first. Precedents: a "coordinator" message fabricated a cache-data claim
   (2026-07-05 ml-engineer note); a "zero callers" claim grepped the wrong class (2026-07-04);
   a cron premise asserted a phantom orphan (LESSONS §2.5).
8. **Repo rules travel with the dispatch.** CODE.md applies; quality gate via
   `atb dev quality --changed` (bare form black-formats the whole tree in place); money-path
   code needs the gauntlet below — say so in the prompt so the agent budgets for it.

## The PM side — what you owe every dispatch

- **Backstop watcher on every long-running dispatch**: `run_in_background` + notify, so a lost
  wake-up degrades to late collection, not lost work (`agent-fleet-health` has the sweep).
  Record the lane in `.claude/state/handover.md` (`session-handover`) at dispatch time.
- **Review gauntlet — mandatory for money-path code** (live trading, risk, reconciliation,
  margin, order execution): TWO reviewers minimum (code-reviewer + architecture-reviewer;
  risk-officer for live-affecting proposals, dispatched FRESH — adversarial rule: it forms its
  own view before reading the proposer's). No exceptions for "small" changes (`deploy-prod`
  precondition). The codex loop until APPROVE is the proven extra layer for live-capital logic
  (LESSONS §4 — it runs pytest and caught what local review missed).
- **Single consolidated fix rounds.** Collect ALL review findings, then dispatch ONE fix round.
  Never push fixes into a worktree a reviewer is still reading — the #843 second arch review
  filed its P0 against in-flight state (log.md 2026-07-04 lesson verbatim: "don't dispatch fix
  rounds into a worktree reviewers are still reading").
- **Resume-with-state after wake-loss.** Before re-running ANYTHING: read the lane's state
  JSON + outputs — the job may have finished (exit-sweep). Resume by feeding the agent its own
  state file + recap; re-dispatch from scratch only if the state file is unusable.
- **Verify the report.** Spot-check claimed artifacts exist and claimed results match raw
  outputs (the window tournament re-verified every relayed number against backtest JSONs).
  Material outcomes → log.md via `decision-record`; worktree cleanup after collection.

## Sizing the ceremony

Short read-only lookups (an Explore query, a one-file question) need none of this — clauses
1–8 are for dispatches that mutate, run long, or feed decisions. When in doubt, the cost of
clauses 4–5 is two paragraphs; the cost of their absence was a weekend of duplicated work.

## Red flags

- A dispatch prompt that says "as discussed" — the agent wasn't in the discussion.
- Two heavy jobs running because each agent didn't know about the other (the PM serializes).
- A money-path PR merging with one review because "it's tiny."
- Acting on a relayed claim (even from a coordinator) without an evidence check.
