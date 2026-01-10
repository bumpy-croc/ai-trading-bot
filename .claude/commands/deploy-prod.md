# Deploy to Production

Merge the `staging` branch into `main` with a clean history, then push to remote.

**⚠️ CRITICAL: This deploys to production. Ensure staging has been tested.**

## Instructions

1. **Verify Clean Working Tree**
   ```bash
   git status --porcelain
   ```
   - If there are uncommitted changes, warn the user and stop

2. **Fetch Latest from Remote**
   ```bash
   git fetch origin staging main
   ```

3. **Checkout and Reset Main to Match Staging**
   ```bash
   git checkout main
   git reset --hard origin/staging
   ```
   - This ensures main has identical history to staging

4. **Push to Remote**
   ```bash
   git push origin main --force-with-lease
   ```
   - Uses `--force-with-lease` for safety (fails if remote has unexpected changes)

5. **Return to Original Branch**
   ```bash
   git checkout develop
   ```

6. **Verify Deployment**
   - Confirm the push succeeded
   - Report the commit SHA now on main (production)

## Output

Report:
- The commit SHA deployed to production
- Confirmation that main now matches staging
- Any warnings or errors encountered
