# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

# Install system dependencies (no cache mounts for now)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl make && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set Python path and runtime envs
ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy production requirements and install dependencies
COPY requirements-server.txt ./requirements-server.txt

# Install Python dependencies (production only, without heavy training libs)
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r requirements-server.txt

# Copy application code
COPY . .

# Create necessary directories
# Removed: data directory moved to src/data, logs and ml already exist

# Create non-root user and adjust ownership
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port for health checks
EXPOSE 8000

# Health check endpoint (Railway format)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD /bin/sh -c 'curl -f http://localhost:${PORT:-8000}/health || exit 1'

# Default command - use the combined runner
CMD ["make", "live-health", "ml_basic"]
