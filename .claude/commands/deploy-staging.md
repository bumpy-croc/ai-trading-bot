# Deploy to Staging

Deploy the current `develop` line to the **staging** Railway environment by
syncing the long-running `staging` branch to `develop`.

> **Environment chain:** `develop` → `staging` → `main`, mapping to Railway
> `development` → `staging` → `production`.
>
> **`staging` is a protected, long-running branch.** It maps 1:1 to the Railway
> staging environment and must **never be deleted**. It is protected
> (`allow_deletions: false`), so the repo-wide `delete_branch_on_merge: true`
> will skip it — merging a `staging`→`main` promotion PR will not remove it.
> Never delete `staging`; this command only fast-forwards/resets its **content**
> to `develop` (force-push is allowed on the branch, the ref itself persists).
>
> Production (`main`) is **different**: it carries its own "Promote to
> production" merge history and must NEVER be force-pushed. See `/deploy-prod`.

## Instructions

1. **Verify Clean Working Tree**
   ```bash
   git status --porcelain
   ```
   - If there are uncommitted changes, warn the user and stop.

2. **Fetch Latest from Remote**
   ```bash
   git fetch origin develop staging
   ```
   - Confirm `staging` exists on the remote:
     ```bash
     git ls-remote --heads origin staging
     ```
     It always should (it is protected). If it is somehow missing, STOP and
     alert the user — branch protection may have been removed — rather than
     silently recreating it.

3. **Sync `staging`'s content to `develop`**
   ```bash
   git checkout -B staging origin/develop
   ```
   - This points the local `staging` at `develop`'s tree. The remote branch
     ref is never deleted; only its content is updated.

4. **Push to Remote**
   ```bash
   git push origin staging --force-with-lease
   ```
   - `staging` only ever mirrors the release line, so a fast-forward/reset push
     is expected. `--force-with-lease` guards against a surprise concurrent push.

5. **Return to `develop`**
   ```bash
   git checkout develop
   ```

6. **Verify Deployment**
   - Confirm the push succeeded and report the commit SHA now on `staging`.
   - Confirm `staging` still exists on the remote.
   - Confirm Railway has picked up the deploy on the **staging** environment
     (Railway MCP server / `railway` CLI / dashboard) and that the
     `/health` check is green.

## Output

Report:
- The commit SHA deployed to staging.
- Confirmation that `staging` now matches `develop` and still exists.
- Railway staging deploy status (and `/health`) if available.
- Any warnings or errors encountered.

## Notes

- This is the recommended pre-production soak step: deploy here first, let the
  staging bot run, then run `/deploy-prod` to promote `staging` → `main`.
- Staging and production use separate databases
  (`RAILWAY_STAGING_DATABASE_URL` / `RAILWAY_PRODUCTION_DATABASE_URL`).
