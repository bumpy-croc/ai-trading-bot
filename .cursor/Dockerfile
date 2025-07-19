# Background Agent Dockerfile - Self-contained with PostgreSQL
FROM python:3.11-slim

# Install system dependencies including PostgreSQL
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        build-essential \
        libpq-dev \
        postgresql \
        postgresql-contrib \
        supervisor \
        && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Create necessary directories
RUN mkdir -p logs ml data /var/lib/postgresql/data /var/run/postgresql

# Set up PostgreSQL
RUN chown -R postgres:postgres /var/lib/postgresql /var/run/postgresql && \
    su - postgres -c "initdb -D /var/lib/postgresql/data" && \
    echo "host all all 0.0.0.0/0 md5" >> /var/lib/postgresql/data/pg_hba.conf && \
    echo "listen_addresses='*'" >> /var/lib/postgresql/data/postgresql.conf

# Create database and user
RUN echo "CREATE DATABASE trading_bot;" > /tmp/init.sql && \
    echo "CREATE USER postgres WITH PASSWORD 'password';" >> /tmp/init.sql && \
    echo "GRANT ALL PRIVILEGES ON DATABASE trading_bot TO postgres;" >> /tmp/init.sql && \
    echo "ALTER USER postgres CREATEDB;" >> /tmp/init.sql

# Create supervisord configuration
RUN cat > /etc/supervisor/conf.d/supervisord.conf << 'EOF'
[supervisord]
nodaemon=true
user=root

[program:postgres]
command=/usr/lib/postgresql/15/bin/postgres -D /var/lib/postgresql/data
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
ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=postgresql://postgres:password@localhost:5432/trading_bot
ENV RAILWAY_STAGING_DATABASE_URL=postgresql://postgres:password@localhost:5432/trading_bot
ENV RAILWAY_PRODUCTION_DATABASE_URL=postgresql://postgres:password@localhost:5432/trading_bot
ENV PGDATA=/var/lib/postgresql/data

# Expose PostgreSQL and dashboard ports
EXPOSE 5432 5000

# Start supervisor to manage PostgreSQL, dashboard, and the application
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"] 