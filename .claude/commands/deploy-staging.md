# Deploy to Staging

Merge the `develop` branch into `staging` with a clean history, then push to remote.

## Instructions

1. **Verify Clean Working Tree**
   ```bash
   git status --porcelain
   ```
   - If there are uncommitted changes, warn the user and stop

2. **Fetch Latest from Remote**
   ```bash
   git fetch origin develop staging
   ```

3. **Checkout and Reset Staging to Match Develop**
   ```bash
   git checkout staging
   git reset --hard origin/develop
   ```
   - This ensures staging has identical history to develop

4. **Push to Remote**
   ```bash
   git push origin staging --force-with-lease
   ```
   - Uses `--force-with-lease` for safety (fails if remote has unexpected changes)

5. **Return to Original Branch**
   ```bash
   git checkout develop
   ```

6. **Verify Deployment**
   - Confirm the push succeeded
   - Report the commit SHA now on staging

## Output

Report:
- The commit SHA deployed to staging
- Confirmation that staging now matches develop
- Any warnings or errors encountered
