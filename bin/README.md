# Development Scripts

Helper scripts for development and quality assurance.

## Scripts
- `run_mypy.py`: Type checking script with project-specific configuration

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

See `docs/development.md` for detailed deployment and environment automation guidance.
