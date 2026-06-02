# Deploy to Production

Promote **`staging` → `main`** (the production Railway environment) via a
**"Promote to production" pull request** — never a force-push.

**⚠️ CRITICAL: This deploys a real-capital trading bot. Soak on staging first
(`/deploy-staging`) and only promote a green, reviewed `staging`.**

> **Environment chain:** `develop` → `staging` → `main`, mapping to Railway
> `development` → `staging` → `production`. Production is always promoted from
> **`staging`**, not directly from `develop`.
>
> **Why not `git reset --hard` + force-push?**
> `main` carries its own history that does **not** exist on `staging`/`develop`:
> every prior production cutover is a "Promote to production: …" merge/squash
> commit made on `main`. Resetting `main` to another branch and force-pushing
> would **rewrite production history and discard those promotion commits** — a
> destructive operation on the production branch, which `CLAUDE.md` forbids.
> Promotion is therefore an additive merge commit via a PR.
>
> **`staging` is a protected long-running branch** (`allow_deletions: false`),
> so merging a `staging`→`main` PR will **not** delete it (the repo has
> `delete_branch_on_merge: true`, which skips protected branches). Do not
> recreate `staging` after a promotion — it persists by design.

## Instructions

1. **Verify Clean Working Tree & Fetch**
   ```bash
   git status --porcelain
   git fetch origin develop staging main
   ```
   - If there are uncommitted changes, warn the user and stop.

2. **Ensure `staging` is up to date and soaked**
   - Run `/deploy-staging` first if `staging` is behind `develop`.
   - Sanity check what will ship to production:
     ```bash
     git log --oneline origin/main..origin/staging
     ```

3. **Open the "Promote to production" PR**
   Create a PR with **base `main`** and **head `staging`** (prefer the GitHub
   MCP server; otherwise `gh pr create --base main --head staging`). Title it:
   ```
   Promote to production: <one-line summary of what's shipping>
   ```
   Body should list the PRs/changes included and the testing performed.

4. **Reconcile conflicts additively (do NOT force-push `main`)**
   The PR is often `dirty` because `main` and the release line both touched
   `docs/changelog.md` (and occasionally code). Resolve by **merging `main`
   into `staging`**, never by resetting `main`:
   ```bash
   git checkout staging
   git merge origin/main         # resolve conflicts (usually just the changelog: keep both sides)
   git push origin staging
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
   `staging` is protected and will survive the merge.

7. **Verify Deployment**
   - Confirm `main` advanced and now contains the release:
     ```bash
     git fetch origin main staging
     git rev-list --count origin/main..origin/staging   # expect 0
     ```
   - Confirm `staging` still exists (it must, being protected):
     ```bash
     git ls-remote --heads origin staging
     ```
   - Confirm Railway picked up the deploy on the **production** environment
     (Railway MCP / `railway` CLI / dashboard) and that `/health` is green.
   - Watch the production bot's liveness (account_history freshness /
     heartbeat) for a few cycles after cutover, since this ships live-engine
     changes. Do **not** perform destructive operations against production.

## Output

Report:
- The promotion PR number and the merge commit SHA now on `main` (production).
- Confirmation that `main..staging` is empty (fully promoted) and `staging`
  still exists.
- Railway production deploy status (and `/health`) if available.
- Any warnings or errors encountered.
