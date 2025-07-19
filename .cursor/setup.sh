#!/bin/bash

# Install Python dependencies
pip install --upgrade pip --no-cache-dir
pip install --no-cache-dir -r requirements.txt

# Set up PostgreSQL
chown -R postgres:postgres /var/lib/postgresql /var/run/postgresql
su - postgres -c '/usr/lib/postgresql/15/bin/initdb -D /var/lib/postgresql/data'

# Configure PostgreSQL
echo "host all all 0.0.0.0/0 md5" >> /var/lib/postgresql/data/pg_hba.conf
echo "listen_addresses='*'" >> /var/lib/postgresql/data/postgresql.conf

# Create database initialization script
cat > /tmp/init.sql << 'EOF'
CREATE DATABASE trading_bot;
CREATE USER postgres WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE trading_bot TO postgres;
ALTER USER postgres CREATEDB;
EOF

# Create supervisord configuration
cat > /etc/supervisor/conf.d/supervisord.conf << 'EOF'
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

echo "Setup completed successfully!" 