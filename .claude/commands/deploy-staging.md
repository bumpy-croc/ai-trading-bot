# Deploy to Staging

Deploy the current `develop` line to the **staging** Railway environment by
updating the `staging` branch to match `develop`.

> **Branch model (read first).** PRs squash-merge into `develop` (integration).
> `staging` is a **disposable** environment branch that simply tracks whatever
> `develop` currently is — it never carries unique commits of its own and may be
> auto-deleted after a promotion, so recreating it from `develop` is always safe.
> Production (`main`) is **different**: it carries its own "Promote to production"
> merge history and must NEVER be force-pushed. See `/deploy-prod`.

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

3. **Point `staging` at the current `develop`**
   `staging` is disposable, so reset it to `develop` (create it if it does not
   exist — it is frequently cleaned up after a promotion):
   ```bash
   git checkout -B staging origin/develop
   ```

4. **Push to Remote**
   ```bash
   git push origin staging --force-with-lease
   ```
   - `staging` carries no unique history, so a reset-and-push is safe here.
     `--force-with-lease` still guards against a surprise concurrent push.
   - If the remote `staging` was deleted, this recreates it
     (`git push -u origin staging`).

5. **Return to `develop`**
   ```bash
   git checkout develop
   ```

6. **Verify Deployment**
   - Confirm the push succeeded and report the commit SHA now on `staging`.
   - Confirm Railway has picked up the deploy on the **staging** environment
     (Railway MCP server / `railway` CLI / dashboard) and that the
     `/health` check is green.

## Output

Report:
- The commit SHA deployed to staging.
- Confirmation that `staging` now matches `develop`.
- Railway staging deploy status (and `/health`) if available.
- Any warnings or errors encountered.

## Notes

- This is the recommended pre-production soak step: deploy here first, let the
  staging bot run, then run `/deploy-prod` to promote.
- Staging and production use separate databases
  (`RAILWAY_STAGING_DATABASE_URL` / `RAILWAY_PRODUCTION_DATABASE_URL`).
