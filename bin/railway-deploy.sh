#!/bin/bash
set -e

# Function to log with timestamps
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Default values
ENVIRONMENT="staging"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -e, --environment    Set ENVIRONMENT variable (staging|production) [default: staging]"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Note: Make sure you've already linked to your Railway project with 'railway link'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h for help"
            exit 1
            ;;
    esac
done

log "🚀 Starting Railway deployment..."
log "📋 Environment: $ENVIRONMENT"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    log "❌ Railway CLI not found. Please install it first:"
    log "   curl -fsSL https://railway.app/install.sh | sh"
    exit 1
fi

# Check if we're logged in
if ! railway whoami &> /dev/null; then
    log "❌ Not logged in to Railway. Please run 'railway login' first."
    exit 1
fi

# Check if project is linked
if ! railway status &> /dev/null; then
    log "❌ No Railway project linked. Please run 'railway link' first."
    exit 1
fi

# Set environment variable
log "⚙️ Setting environment variable..."
railway variables set ENVIRONMENT="$ENVIRONMENT"

# Production confirmation
if [ "$ENVIRONMENT" = "production" ]; then
    log "⚠️  This is a PRODUCTION deployment"
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        log "❌ Deployment cancelled"
        exit 1
    fi
fi

# Deploy
log "🚀 Deploying to Railway..."
railway up

log "✅ Deployment completed!"
log "📋 View logs: railway logs"
log "📋 Check status: railway status"
log "📋 View app: railway domain"