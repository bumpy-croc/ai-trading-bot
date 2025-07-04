# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

# Install system dependencies with build cache for APT
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt \
    --mount=type=cache,id=apt-lib-cache,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

# Leverage BuildKit cache for pip wheels – this dramatically speeds up rebuilds
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
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