#!/bin/bash
set -e

# Function to log with timestamps
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log "🚄 Railway Setup Script for AI Trading Bot"
log "=========================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    log "❌ Railway CLI not found. Please install it first:"
    log "   curl -fsSL https://railway.app/install.sh | sh"
    exit 1
fi

# Check authentication
if ! railway whoami &> /dev/null; then
    log "❌ Not logged in to Railway. Please run 'railway login' first."
    exit 1
fi

log "✅ Railway CLI ready"

# Initialize or link project
if railway status &> /dev/null; then
    log "✅ Already linked to a Railway project"
else
    log "🔗 Setting up Railway project..."
    read -p "Create new project or link existing? (new/link): " CHOICE
    
    if [ "$CHOICE" = "new" ]; then
        read -p "Enter project name: " PROJECT_NAME
        if [ -z "$PROJECT_NAME" ]; then
            log "❌ Project name is required"
            exit 1
        fi
        railway project new "$PROJECT_NAME"
    else
        railway link
    fi
fi

# Set up core environment variables
log "⚙️ Setting up core environment variables..."
railway variables set ENVIRONMENT="staging"
railway variables set PYTHONPATH="/app"
railway variables set PYTHONUNBUFFERED="1"

# Set up database
read -p "Add PostgreSQL database? (y/n) [default: y]: " ADD_DB
ADD_DB=${ADD_DB:-y}

if [ "$ADD_DB" = "y" ] || [ "$ADD_DB" = "Y" ]; then
    log "🗄️ Adding PostgreSQL database..."
    railway add postgresql || log "⚠️ Database setup failed (you can add it manually later)"
fi

log "✅ Railway setup completed!"
log ""
log "📋 Next Steps:"
log "1. Set your API keys in Railway dashboard:"
log "   railway open  # Opens dashboard"
log "   - Go to Variables tab"
log "   - Add BINANCE_API_KEY and BINANCE_SECRET_KEY"
log ""
log "2. Deploy your application:"
log "   ./bin/railway-deploy.sh"
log ""
log "3. Monitor your deployment:"
log "   railway logs"
log "   railway status"
log ""
log "🎉 Happy trading on Railway!"