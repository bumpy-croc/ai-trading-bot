# Background Agent Dockerfile - Self-contained with PostgreSQL
FROM python:3.11-slim

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies including PostgreSQL
RUN apt-get update && apt-get clean && \
    apt-get install -y --no-install-recommends \
        debian-archive-keyring \
        curl \
        git \
        build-essential \
        libpq-dev \
        postgresql \
        postgresql-contrib \
        supervisor \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Create necessary directories
RUN mkdir -p logs ml data /var/lib/postgresql/data /var/run/postgresql

# Create supervisord configuration
COPY <<EOF /etc/supervisor/conf.d/supervisord.conf
[supervisord]
nodaemon=true
user=root

[program:postgres]
command=/usr/lib/postgresql/17/bin/postgres -D /var/lib/postgresql/data
user=postgres
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stderr_logfile=/dev/stderr
environment=PGDATA="/var/lib/postgresql/data"
priority=100

[program:initdb]
command=/bin/bash -c "sleep 5 && psql -U postgres -d trading_bot -f /tmp/init.sql"
user=postgres
autostart=true
autorestart=false
startsecs=0
stdout_logfile=/dev/stdout
stderr_logfile=/dev/stderr
priority=200
startretries=0
exitcodes=0,1

[program:dashboard]
command=python scripts/start_dashboard.py
directory=/workspace
user=root
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stderr_logfile=/dev/stderr
environment=PYTHONPATH="/workspace",PYTHONUNBUFFERED="1",FLASK_ENV="production",PORT="5000"
priority=250

[program:app]
command=python scripts/run_live_trading_with_health.py ml_basic
directory=/workspace
user=root
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stderr_logfile=/dev/stderr
environment=PYTHONPATH="/workspace",PYTHONUNBUFFERED="1"
priority=300
EOF

# Set environment variables
ENV PYTHONPATH=/workspace/src
ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=postgresql://postgres:password@localhost:5432/trading_bot
ENV RAILWAY_STAGING_DATABASE_URL=postgresql://postgres:password@localhost:5432/trading_bot
ENV RAILWAY_PRODUCTION_DATABASE_URL=postgresql://postgres:password@localhost:5432/trading_bot
ENV PGDATA=/var/lib/postgresql/data

# Expose PostgreSQL and dashboard ports
EXPOSE 5432 5000

# Start supervisor to manage PostgreSQL, dashboard, and the application
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"] 