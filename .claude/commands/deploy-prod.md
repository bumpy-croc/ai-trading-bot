# Deploy to Production

Promote the current release line to the **production** Railway environment
(`main`) via a **"Promote to production" pull request** — never a force-push.

**⚠️ CRITICAL: This deploys a real-capital trading bot. Soak on staging first
(`/deploy-staging`) and only promote a green, reviewed `develop`.**

> **Why not `git reset --hard` + force-push?**
> `main` carries its own history that does **not** exist on `develop`: every
> prior production cutover is a "Promote to production: …" merge/squash commit
> made directly on `main`. Resetting `main` to another branch and force-pushing
> would **rewrite production history and discard those promotion commits** — a
> destructive operation on the production branch, which `CLAUDE.md` forbids.
> Promotion is therefore done with an additive merge commit via a PR.

## Instructions

1. **Verify Clean Working Tree & Fetch**
   ```bash
   git status --porcelain
   git fetch origin develop main
   ```
   - If there are uncommitted changes, warn the user and stop.

2. **Confirm the release point is ready**
   - `develop` (or `staging`) is the source of truth. Ideally it has already
     been soaked via `/deploy-staging`.
   - Sanity check what will ship:
     ```bash
     git log --oneline origin/main..origin/develop
     ```

3. **Open the "Promote to production" PR**
   Create a PR with **base `main`** and **head `develop`** (prefer the GitHub
   MCP server; otherwise `gh pr create --base main --head develop`). Title it:
   ```
   Promote to production: <one-line summary of what's shipping>
   ```
   Body should list the PRs/changes included and the testing performed.

4. **Reconcile conflicts additively (do NOT force-push `main`)**
   The PR is often `dirty` because `main` and `develop` both touched
   `docs/changelog.md` (and occasionally code). Resolve by **merging `main`
   into the head branch**, never by resetting `main`:
   ```bash
   git checkout develop          # or a dedicated release branch
   git merge origin/main         # resolve conflicts (usually just the changelog: keep both sides)
   git push origin develop
   ```
   This makes the PR mergeable while preserving `main`'s promotion history.

5. **Wait for green CI**
   All required checks (unit-tests shards, integration-tests, claude-review)
   must pass on the PR before merging. Do not merge a red or in-progress PR to
   production.

6. **Merge the PR with a merge commit**
   Use a **merge commit** (not squash/rebase) so `main` keeps the promotion
   marker and full history. Via GitHub MCP `merge_pull_request`
   (`merge_method: "merge"`) or `gh pr merge <n> --merge`.

7. **Verify Deployment**
   - Confirm `main` advanced and now contains the release:
     ```bash
     git fetch origin main
     git rev-list --count origin/main..origin/develop   # expect 0
     ```
   - Confirm Railway picked up the deploy on the **main / production**
     environment (Railway MCP / `railway` CLI / dashboard) and that `/health`
     is green.
   - Watch the production bot's liveness (account_history freshness /
     heartbeat) for a few cycles after cutover, since this ships live-engine
     changes. Do **not** perform destructive operations against production.

## Output

Report:
- The promotion PR number and the merge commit SHA now on `main` (production).
- Confirmation that `main..develop` is empty (fully promoted).
- Railway production deploy status (and `/health`) if available.
- Any warnings or errors encountered.
