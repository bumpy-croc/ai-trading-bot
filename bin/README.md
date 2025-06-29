# Deployment Scripts

This directory contains the deployment scripts used by GitHub Actions workflows.

## Scripts

### `deploy-staging.sh`
- Deploys the application to the staging environment
- Downloads deployment package from S3
- Sets up Python environment and dependencies
- Configures systemd service for staging
- Performs health checks

**Usage**: `./deploy-staging.sh <commit_sha> <s3_bucket>`

### `deploy-production.sh`
- Deploys the application to the production environment
- More conservative settings and additional security measures
- Longer timeouts and more thorough verification
- Enhanced logging and monitoring

**Usage**: `./deploy-production.sh <commit_sha> <s3_bucket>`

## Features

Both scripts include:
- ✅ Comprehensive error handling with line-by-line error reporting
- ✅ Timestamped logging for better debugging
- ✅ Automatic backup creation before deployment
- ✅ Prerequisites checking (AWS CLI, credentials, directories)
- ✅ Python virtual environment management
- ✅ Service health verification
- ✅ Detailed diagnostics on failure

## Security

- Scripts are downloaded directly from GitHub using the specific commit SHA
- All operations run with appropriate user permissions
- Production script includes additional security settings
- Comprehensive logging for audit trails 