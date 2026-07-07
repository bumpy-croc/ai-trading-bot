---
name: deploy-prod
description: Deploy to PRODUCTION (live capital) — default is the parity promote (main becomes a byte-identical copy of develop via an ours-merge history tie); also covers the prod-first hotfix path (cherry-pick or direct fix on main with mandatory back-propagation). Use for "deploy to production", "promote to prod", "sync main", or "hotfix production".
---

# Deploy to Production

Production trades **live capital** (Railway env `production`, branch `main`). Every
step below was exercised on real deploys (#841, #849, #851, #864, #905, #911). Railway
auto-deploys `main`.

## Preconditions (both modes)

- Content is CI-green and, for money-path code, has passed the two-reviewer gauntlet
  (code-reviewer + architecture-reviewer). No exceptions for "small" changes.
- Timing per charter: avoid Fridays after 18:00 UTC and windows immediately before
  macro events (FOMC/CPI) — unless the Board has explicitly waived it.
- Know the rollback before you deploy (see bottom).

## Default: parity promote (main := develop, byte-identical)

```bash
git fetch origin
git worktree add .claude/worktrees/promote-<name> -b promote/<name> origin/develop
cd .claude/worktrees/promote-<name>
git merge -s ours origin/main -m "Tie main history for parity promote (tree = develop exactly)"
# PROVE the tree before pushing — non-negotiable:
[ "$(git rev-parse HEAD^{tree})" = "$(git rev-parse origin/develop^{tree})" ] \
  && echo TREE-IDENTICAL || echo "STOP - trees differ"
git push -u origin promote/<name>
gh pr create --base main --head promote/<name> --title "Promote to production: <what>" --body "<delta audit>"
# CI green → merge with a MERGE COMMIT (squash breaks the history tie that keeps future syncs clean):
gh pr merge <N> --merge
```

The PR body must carry a **delta audit**: what main gains, which parts are flag-gated
OFF, and any judgment calls flagged to the Board.

## Hotfix mode (prod-first, the exception path)

1. Branch **from main**: `git worktree add ... -b hotfix/<name> origin/main`.
2. Either cherry-pick the reviewed squash commit from develop and **verify patch-id**:
   `git show <sha> | git patch-id --stable` must match on both sides — proof the
   applied diff is byte-identical to what was reviewed. Or commit a novel fix directly
   (then it needs the review gauntlet on this branch).
3. **Run the FULL unit suite ON the main-based branch** — develop's green CI does not
   transfer (main may lack develop-only siblings; cherry-picks can conflict with
   unpromoted work — resolve surgically and re-verify per-file patch-ids where possible).
4. PR → main (squash OK for a single fix). CI green → merge.
5. **Back-propagate to develop immediately** (PR or the next parity sync) — main must
   never silently hold changes develop lacks. Reference: the #831 pattern.

## Post-deploy verification (non-negotiable checklist)

```bash
railway deployment list -e production -s "Trading Bot" --json | jq -r '.[0].status'  # SUCCESS
railway logs -e production -s "Trading Bot" | grep -iE \
  "Trading loop started|guard armed|Max Position|cross.symbol|alert channel|Recovered position|ERROR"
```

- Trading loop started; **drawdown guard armed at the correct session peak** (session-
  scoped, phantom-era excluded — see drawdown_guard.py docstring);
- Max Position banner matches config (20.0%);
- The intended model resolves natively — **zero cross-symbol/mismatch warnings**;
- Open positions re-adopted with their stop-loss orders tracked;
- Alert channel line (webhook configured, or the loud unset warning if expected).

**Gotcha:** prod REUSES its active session row across restarts — do NOT wait for a new
`trading_sessions` row as a health signal; watch the startup banner, status ticks, and
the hourly `account_history` heartbeat instead. Read-only DB ground truth:
`RAILWAY_PRODUCTION_DATABASE_URL` in the main checkout's `.env`, first statement
`SET default_transaction_read_only = on;`.

## Rollback

- Railway: redeploy the previous SUCCESS deployment (dashboard or CLI), or revert-merge
  the promote PR and let auto-deploy ship the revert.
- Plain restart without a code change: env-var nudge —
  `railway variables --set "RESTART_NUDGE=<timestamp>" -e production -s "Trading Bot"`.
- Recovery is restart-safe by design (session reuse, position re-adoption, guard peak
  recompute) — a restart mid-position is proven safe; never manually flatten to "help".
