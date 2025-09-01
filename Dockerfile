# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl make gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set Python path and runtime envs
ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# Copy production requirements and install dependencies
COPY requirements-server.txt ./requirements-server.txt

# Install Python dependencies globally (production standard)
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r requirements-server.txt

# Copy application code
COPY . .

# Install the package globally (not editable for production)
RUN pip install .

# Create necessary directories
RUN mkdir -p logs artifacts

# Create non-root user and adjust ownership
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port for health checks
EXPOSE 8000

# Health check endpoint (Railway format)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD /bin/sh -c 'curl -f http://localhost:${PORT:-8000}/health || exit 1'

# Default command - run migrations then start application
CMD ["atb db verify --apply-migrations && atb live-health ml_basic"]
