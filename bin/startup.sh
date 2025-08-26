#!/bin/bash
set -e

echo "🚀 Starting AI Trading Bot..."

# Check if DATABASE_URL is available
if [ -z "$DATABASE_URL" ]; then
    echo "❌ DATABASE_URL environment variable not found"
    echo "   Please ensure your Railway PostgreSQL service is properly configured"
    exit 1
fi

echo "✅ Database URL found: ${DATABASE_URL:0:20}..."

# Run database migrations
echo "🔄 Running database migrations..."
atb db migrate

if [ $? -eq 0 ]; then
    echo "✅ Migrations completed successfully"
else
    echo "❌ Migrations failed, exiting..."
    exit 1
fi

# Start the application
echo "🚀 Starting application..."
exec atb live-health ml_basic
