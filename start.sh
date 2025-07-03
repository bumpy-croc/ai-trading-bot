#!/bin/bash
set -e

# Start health check server in background
echo "Starting health check server..."
python health_check.py &
HEALTH_PID=$!

# Wait a moment for health check server to start
sleep 2

# Function to cleanup on exit
cleanup() {
    echo "Shutting down..."
    kill $HEALTH_PID 2>/dev/null || true
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Start the main application
echo "Starting main application: $@"
exec "$@"