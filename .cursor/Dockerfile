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

# Copy requirements files
COPY requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt

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

# Copy supervisor configuration
COPY .cursor/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

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