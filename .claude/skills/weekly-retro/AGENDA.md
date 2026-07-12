# Weekly Retro — Agenda

Running agenda for the next weekly retro. **Any agent or the PM appends items here the moment
they're noticed** (process failures, near-misses, corrections, patterns worth codifying) instead of
carrying them in memory. One bullet per item: date, what happened, and the rule/change it suggests.

The retro reads this file FIRST, actions every item (LESSONS.md append, skill amendment, or GH
issue — or an explicit "no action, reasoning" disposition in the retro PR), then **clears the file
back to this header** in the same PR. Items must never be silently dropped: everything below either
becomes a diff or gets a written disposition.

---

## Items

- **2026-07-07/10 — wake-loss pattern (6+ occurrences).** Agents end turns on background
  waits/monitors that never fire (worst during laptop-lid sleep). Proposed rule for
  delegation-protocol: never end a turn on a background wait you can complete synchronously
  in-turn; if you must wait, the PM arms the backstop. Structural backstop now exists:
  `pm-fleet-watchdog` scheduled task (hourly, fires on app relaunch) — reference it in the rule.
- **2026-07-10 — "component-complete ≠ runnable" (PR #948 → halt → PR #950).** Well-tested
  components that couldn't run end-to-end from the CLI halted tournament Phase 3. Rule: multi-piece
  scaffolding's definition-of-done includes per-consumer end-to-end dry-run acceptance tests through
  the REAL CLI entrypoints. Evidence of value: those tests then caught 4 further real bugs.
- **2026-07-10 — claude-review bot findings ignored by process.** A real #838-class units bug sat
  as an inline bot comment on #948 while both dispatched reviewers missed it; merge flow only read
  the bot's pass/fail status. Rule: every merge flow harvests
  `gh api repos/OWNER/REPO/pulls/N/comments` and dispositions each bot finding explicitly.
- **2026-07-10 — credential written to disk.** An agent saved a live ECR authorization token to a
  plaintext scratchpad file (PM caught + deleted). LESSONS rule: never write credentials to files;
  pipe `aws ecr get-login-password | docker login --password-stdin` in one command.
- **2026-07-10 — nightly pruner deleted a live agent's worktree mid-tournament.** Partially fixed:
  `eod-worktree-prune` now hard-skips `.agent-active` sentinels + 48h age floor. Remaining: codify
  the sentinel convention in delegation-protocol (every agent worktree gets `touch .agent-active`
  at creation).
- **2026-07-10 — shared-venv `atb` staleness trap.** The console script is an editable install
  pinned to ONE worktree; bare `atb` from any other worktree silently executes that worktree's
  code. Workaround in use: `PYTHONPATH=<worktree> python3 -m cli.__main__`. LESSONS entry + consider
  a GH issue for a real fix (warn when cwd repo-root ≠ installed source root).
- **2026-07-11 — cwd-relative model registry path.** `DEFAULT_MODEL_REGISTRY_PATH = "src/ml/models"`
  resolves against process cwd → produced a silent all-HOLD/0-trade exam result from the wrong
  directory. LESSONS entry; consider GH issue to anchor it to module/repo root.
- **2026-07-10 — review-summary completeness.** Reviewer agents must enumerate EVERY P-level
  finding from their findings file in the summary returned to the PM — a finding that only lives in
  the file effectively doesn't exist for the fix round.
- **2026-07-07/10 — transient 401 auth failures killed agents mid-dispatch (2×).**
  Resume-with-state recovers cleanly; note as a known transient in delegation-protocol's backstop
  guidance (retry the resume once before diagnosing deeper).
- **2026-07-12 — promote boot-check "alembic 0 pending" criterion can't literally pass on prod.**
  0712 promote shipped migration 0013 (#968); prod's redundancy guard skipped it (schema already
  matched models) leaving the alembic stamp stale at 0012 while staging ran it to 0013. Boot check
  reworded ad hoc to PASS-WITH-NOTE. Decide: stamp prod in a maintenance window + amend deploy-prod
  skill's check to "schema matches models AND (0 pending OR skip-guard fired with schema-match)".
