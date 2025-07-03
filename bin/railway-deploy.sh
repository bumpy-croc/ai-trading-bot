#!/bin/bash
set -e

# Function to log with timestamps
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to handle errors
handle_error() {
    log "❌ Error occurred at line $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Default values
ENVIRONMENT="staging"
PROJECT_NAME=""
SERVICE_NAME="ai-trading-bot"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -p|--project)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -s|--service)
            SERVICE_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -e, --environment    Environment to deploy to (staging|production) [default: staging]"
            echo "  -p, --project        Railway project name (required)"
            echo "  -s, --service        Service name [default: ai-trading-bot]"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$PROJECT_NAME" ]; then
    log "❌ Project name is required. Use -p or --project to specify."
    exit 1
fi

if [ "$ENVIRONMENT" != "staging" ] && [ "$ENVIRONMENT" != "production" ]; then
    log "❌ Environment must be 'staging' or 'production'"
    exit 1
fi

log "🚀 Starting Railway deployment..."
log "📋 Environment: $ENVIRONMENT"
log "📋 Project: $PROJECT_NAME"
log "📋 Service: $SERVICE_NAME"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    log "❌ Railway CLI not found. Installing..."
    
    # Install Railway CLI
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://railway.app/install.sh | sh
        export PATH="$HOME/.railway/bin:$PATH"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install railway
        else
            curl -fsSL https://railway.app/install.sh | sh
            export PATH="$HOME/.railway/bin:$PATH"
        fi
    else
        log "❌ Unsupported operating system. Please install Railway CLI manually."
        exit 1
    fi
fi

# Verify Railway CLI installation
if ! command -v railway &> /dev/null; then
    log "❌ Railway CLI installation failed"
    exit 1
fi

log "✅ Railway CLI found"

# Check if we're logged in
log "🔐 Checking Railway authentication..."
if ! railway whoami &> /dev/null; then
    log "❌ Not logged in to Railway. Please run 'railway login' first."
    exit 1
fi

log "✅ Railway authentication verified"

# Link to the project
log "🔗 Linking to Railway project..."
railway link "$PROJECT_NAME" || {
    log "❌ Failed to link to project '$PROJECT_NAME'"
    log "📋 Make sure the project exists and you have access to it"
    exit 1
}

# Set environment variables based on deployment environment
log "⚙️ Setting up environment variables..."

case $ENVIRONMENT in
    "production")
        railway variables set ENVIRONMENT=production
        railway variables set PYTHONPATH=/app
        railway variables set PYTHONUNBUFFERED=1
        log "🎯 Configured for PRODUCTION deployment"
        ;;
    "staging")
        railway variables set ENVIRONMENT=staging
        railway variables set PYTHONPATH=/app
        railway variables set PYTHONUNBUFFERED=1
        log "🧪 Configured for STAGING deployment"
        ;;
esac

# Pre-deployment checks
log "🔍 Running pre-deployment checks..."

# Check if required environment variables are set
if [ -z "$RAILWAY_API_KEY" ]; then
    log "⚠️ RAILWAY_API_KEY not found in environment"
    log "📋 Make sure to set it in your Railway project settings"
fi

# Validate Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    log "❌ Dockerfile not found in current directory"
    exit 1
fi

# Validate railway.json exists
if [ ! -f "railway.json" ]; then
    log "❌ railway.json not found in current directory"
    exit 1
fi

log "✅ Pre-deployment checks passed"

# Deploy to Railway
log "🚀 Deploying to Railway..."

if [ "$ENVIRONMENT" = "production" ]; then
    log "⚠️  This is a PRODUCTION deployment - proceeding with extra caution"
    read -p "Are you sure you want to deploy to PRODUCTION? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        log "❌ Production deployment cancelled"
        exit 1
    fi
fi

# Start the deployment
railway up --detach || {
    log "❌ Railway deployment failed"
    exit 1
}

log "⏳ Waiting for deployment to complete..."

# Wait for deployment to finish (check status every 10 seconds for up to 5 minutes)
for i in {1..30}; do
    if railway status | grep -q "HEALTHY\|SUCCESS"; then
        log "✅ Deployment completed successfully!"
        break
    elif railway status | grep -q "CRASHED\|FAILED\|ERROR"; then
        log "❌ Deployment failed!"
        log "📋 Recent logs:"
        railway logs --limit 20
        exit 1
    fi
    
    if [ $i -eq 30 ]; then
        log "⚠️ Deployment is taking longer than expected..."
        log "📋 Current status:"
        railway status
    fi
    
    sleep 10
done

# Get deployment info
log "📊 Deployment Information:"
railway status
railway domain

log "🎉 Railway deployment completed!"
log "📋 Environment: $ENVIRONMENT"
log "📋 You can view logs with: railway logs"
log "📋 You can check status with: railway status"