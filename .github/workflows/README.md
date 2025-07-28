# GitHub Actions Deployment Workflow

This repository uses a three-environment deployment strategy:

## Environments

### 1. Development (Automatic)
- **Trigger**: Every push to `main` branch
- **Purpose**: Continuous integration and testing
- **Platform**: Railway automatic deployment
- **Railway Environment**: `development`

### 2. Staging (Manual)
- **Trigger**: Manual workflow dispatch
- **Purpose**: Paper trading and pre-production testing
- **Workflow**: `deploy-staging.yml`
- **Railway Environment**: `staging`

### 3. Production (Manual)
- **Trigger**: Manual workflow dispatch
- **Purpose**: Live trading
- **Workflow**: `deploy-production.yml`
- **Railway Environment**: `production`

## Manual Deployment Process

### How to Deploy Manually

1. **Navigate to GitHub Actions**:
   - Go to your repository on GitHub
   - Click on the "Actions" tab
   - Select the workflow you want to run (e.g., "Deploy to Staging" or "Deploy to Production")

2. **Trigger Manual Deployment**:
   - Click the "Run workflow" button (blue button on the right)
   - Select the branch (usually `main`)
   - Fill in optional parameters:
     - **Service ID**: Override the default Railway service ID (optional)
     - **Commit SHA**: Deploy a specific commit (optional, defaults to latest)
   - Click "Run workflow"

### Deployment Parameters

#### Service ID
- **Purpose**: Override the default Railway service ID
- **When to use**: If you have multiple services in your Railway project
- **Format**: Railway service ID (e.g., `abc123-def456-ghi789`)

#### Commit SHA
- **Purpose**: Deploy a specific commit instead of the latest
- **When to use**: 
  - Rollback to a previous version
  - Deploy a specific feature branch commit
  - Ensure staging and production use the same code
- **Format**: Full commit SHA (e.g., `a1b2c3d4e5f6...`)

## Typical Workflow

1. **Development**: Code is automatically deployed by Railway on every push to `main`
2. **Staging**: When ready for testing, manually deploy to staging for paper trading
3. **Production**: After successful staging validation, manually deploy to production

## Required Secrets

Ensure these secrets are configured in your GitHub repository:

- `RAILWAY_TOKEN`: Your Railway authentication token
- `RAILWAY_PROJECT_ID`: Your Railway project ID

## Environment Protection

Each environment can have protection rules configured:
- **Required reviewers**: Require approval before deployment
- **Wait timer**: Delay deployment for a specified time
- **Deployment branches**: Restrict which branches can deploy

To configure protection rules:
1. Go to repository Settings → Environments
2. Select the environment (staging/production)
3. Configure protection rules as needed 