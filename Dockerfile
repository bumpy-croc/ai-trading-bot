# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

# Install system dependencies with Railway-compatible cache mounts (service-id-prefixed)
RUN --mount=type=cache,id=f032a62c-d98d-4fa7-9302-359249be154b-apt-cache,target=/var/cache/apt \
    --mount=type=cache,id=f032a62c-d98d-4fa7-9302-359249be154b-apt-lib,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

# Install Python dependencies with cached pip wheels
RUN --mount=type=cache,id=f032a62c-d98d-4fa7-9302-359249be154b-pip-cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
# Removed: data directory moved to src/data, logs and ml already exist

# Make scripts executable
RUN chmod +x scripts/health_check.py scripts/run_live_trading_with_health.py

# Expose port for health checks
EXPOSE 8000

# Health check endpoint (Railway format)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD /bin/sh -c 'curl -f http://localhost:${PORT:-8000}/health || exit 1'

# Default command - use the combined runner
CMD ["python", "scripts/run_live_trading_with_health.py", "ml_basic"]