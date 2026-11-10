# Development Scripts

Helper scripts for development and quality assurance.

## Scripts
- `run_mypy.py`: Type checks `src/` and `cli/` with the project's `mypy.ini` (also the CI `type-check` job); `--targets` limits it to named files

Usage:
```bash
python bin/run_mypy.py
```

## Railway Deployment

For Railway deployment, use the Railway CLI directly:

```bash
# Install Railway CLI
npm install -g @railway/cli

# Initialize project
railway init

# Add database
railway add postgresql

# Deploy
railway up
```

See `docs/development.md#railway-deployment-quick-start` for detailed deployment and environment automation guidance.
