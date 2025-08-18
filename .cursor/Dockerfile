# Background Agent Dockerfile - Self-contained with PostgreSQL
FROM python:3.11-slim

# Set environment variables to prevent interactive prompts and time issues
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV TERM=xterm

# Install system dependencies including PostgreSQL with proper time handling
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        wget \
        git \
        build-essential \
        libpq-dev \
        postgresql \
        postgresql-contrib \
        supervisor \
        ca-certificates \
        gnupg \
        lsb-release \
        xvfb \
        xauth \
        x11-utils && \
    # Fix potential time-related issues with package verification
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Create necessary directories
RUN mkdir -p logs ml data /var/lib/postgresql/data /var/run/postgresql

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=postgresql://postgres:password@localhost:5432/trading_bot
ENV RAILWAY_STAGING_DATABASE_URL=postgresql://postgres:password@localhost:5432/trading_bot
ENV RAILWAY_PRODUCTION_DATABASE_URL=postgresql://postgres:password@localhost:5432/trading_bot
ENV PGDATA=/var/lib/postgresql/data

# Expose PostgreSQL and dashboard ports
EXPOSE 5432 5000

# Start supervisor to manage PostgreSQL, dashboard, and the application
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"] 