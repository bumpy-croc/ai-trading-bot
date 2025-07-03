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
else
    log "✅ Railway CLI found"
fi

# Check authentication
log "🔐 Checking Railway authentication..."
if ! railway whoami &> /dev/null; then
    log "❌ Not logged in to Railway. Please run 'railway login' first."
    exit 1
fi

log "✅ Railway authentication verified"

# Get user input for project configuration
read -p "Enter your Railway project name (or press Enter to create new): " PROJECT_NAME
read -p "Enter environment (staging/production) [default: staging]: " ENVIRONMENT
ENVIRONMENT=${ENVIRONMENT:-staging}

if [ -z "$PROJECT_NAME" ]; then
    read -p "Enter new project name: " PROJECT_NAME
    if [ -z "$PROJECT_NAME" ]; then
        log "❌ Project name is required"
        exit 1
    fi
    
    log "🆕 Creating new Railway project: $PROJECT_NAME"
    railway project new "$PROJECT_NAME" || {
        log "❌ Failed to create project"
        exit 1
    }
else
    log "🔗 Linking to existing project: $PROJECT_NAME"
    railway link "$PROJECT_NAME" || {
        log "❌ Failed to link to project"
        exit 1
    }
fi

# Set up environment variables
log "⚙️ Setting up environment variables..."

# Core environment variables
railway variables set ENVIRONMENT="$ENVIRONMENT"
railway variables set PYTHONPATH="/app"
railway variables set PYTHONUNBUFFERED="1"
railway variables set HEALTH_CHECK_PORT="8000"

# API keys and secrets (user will need to set these manually)
log "📋 Setting up environment variable placeholders..."
log "   You'll need to set these manually in Railway dashboard:"

required_vars=(
    "BINANCE_API_KEY"
    "BINANCE_SECRET_KEY"
    "RAILWAY_API_KEY"
)

optional_vars=(
    "SENTICRYPT_API_KEY"
    "CRYPTOCOMPARE_API_KEY"
    "AUGMENTO_API_KEY"
    "DATABASE_URL"
)

for var in "${required_vars[@]}"; do
    log "   ⚠️  REQUIRED: $var"
    railway variables set "$var" "YOUR_${var}_HERE" || true
done

for var in "${optional_vars[@]}"; do
    log "   ℹ️  OPTIONAL: $var"
    railway variables set "$var" "" || true
done

# Set up database if needed
read -p "Do you want to add a PostgreSQL database? (y/n) [default: y]: " ADD_DATABASE
ADD_DATABASE=${ADD_DATABASE:-y}

if [ "$ADD_DATABASE" = "y" ] || [ "$ADD_DATABASE" = "Y" ]; then
    log "🗄️ Adding PostgreSQL database..."
    railway add --database postgresql || {
        log "⚠️ Database setup failed, you can add it manually later"
    }
fi

# Set up Redis if needed
read -p "Do you want to add a Redis cache? (y/n) [default: n]: " ADD_REDIS
ADD_REDIS=${ADD_REDIS:-n}

if [ "$ADD_REDIS" = "y" ] || [ "$ADD_REDIS" = "Y" ]; then
    log "🔴 Adding Redis cache..."
    railway add --database redis || {
        log "⚠️ Redis setup failed, you can add it manually later"
    }
fi

# Create .railway directory for local config
mkdir -p .railway
echo "service: ai-trading-bot" > .railway/config.json

log "✅ Railway setup completed!"
log ""
log "📋 Next Steps:"
log "1. Set your API keys in Railway dashboard:"
log "   - Visit: https://railway.app/project/$PROJECT_NAME"
log "   - Go to Variables tab"
log "   - Update the placeholder values"
log ""
log "2. Deploy your application:"
log "   ./bin/railway-deploy.sh -p $PROJECT_NAME -e $ENVIRONMENT"
log ""
log "3. Monitor your deployment:"
log "   railway logs"
log "   railway status"
log ""
log "🎉 Happy trading on Railway!"