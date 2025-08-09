# Railway Deployment Scripts

Helper scripts for Railway setup and deployment.

## Scripts
- `railway-setup.sh`: Installs CLI, configures project, deploys services

Usage:
```bash
./bin/railway-setup.sh
```

Features:
- Error handling and timestamped logging
- CLI installation and project linking
- Deployment and health verification
- Detailed diagnostics on failure

Security:
- Scripts download pinned commits
- Non-root execution where possible
- Logged operations for audit 