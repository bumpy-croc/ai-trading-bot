---
name: deploy-staging
description: Deploy to the staging environment (paper trading) — default is syncing the staging branch to develop via a merge-commit PR; also covers staging env-flag changes and boot verification. Use for "deploy to staging", "sync staging", enabling/validating feature cohorts on paper, or staging a model/strategy trial.
---

# Deploy to Staging

Staging = the **paper-trading validation environment** (Railway env `staging`, branch
`staging`, ~$1,000 paper account). It is where live-affecting changes earn their trial
before production. Railway auto-deploys the `staging` branch.

## Default: sync staging to develop

```bash
git fetch origin
gh pr create --repo bumpy-croc/ai-trading-bot --base staging --head develop \
  --title "Sync staging to develop (<what this carries>)" --body "<summary>"
gh pr merge <N> --repo bumpy-croc/ai-trading-bot --merge   # MERGE COMMIT, never squash
```

- Merge commit preserves the sync-merge history (`staging` is not fast-forwardable
  from develop). CI on the sync PR is advisory — develop content is already CI-green.
- Direct `git push origin develop:staging` is rejected (non-FF/protected); always PR.

## Env flags (cohort control)

Staging feature cohorts are env vars on the staging "Trading Bot" service. Resolution:
`FEATURE_<UPPER_SNAKE(flag_key)>` env → `FEATURE_FLAGS_OVERRIDES` JSON → repo
`feature_flags.json` defaults. Batch every change into ONE command (each `--set`
triggers a redeploy):

```bash
railway variables --set "FEATURE_X=true" --set "FEATURE_Y=dry_run" -e staging -s "Trading Bot"
```

## Boot verification (always, after any deploy/redeploy)

```bash
railway deployment list -e staging -s "Trading Bot" --json | jq -r '.[0].status'  # wait SUCCESS
railway logs -e staging -s "Trading Bot" | grep -iE "Trading loop started|guard armed|model|ERROR"
```

Checklist: trading loop started; expected subsystems/flags armed (their boot lines);
the intended model resolves (no cross-symbol warnings); no new ERRORs. Paper ground
truth via read-only DB: `RAILWAY_STAGING_DATABASE_URL` in the main checkout's `.env` →
`psql` with `SET default_transaction_read_only = on;` first.

## Gotchas (earned)

- A clean restart may open a NEW paper session (fresh $1,000 baseline) — fine for
  trials; don't read it as lost money. Stale OPEN position rows from dead sessions may
  linger in the staging DB; the engine ignores them.
- If `railway` CLI says "Environment is deleted" or similar after infra changes:
  re-link with `railway link --project innovative-transformation --environment staging`.
- Trial durations: model/strategy trials need ≥48h paper before any promotion talk
  (see `docs/architecture/model_evaluation_system.md` L3a).
