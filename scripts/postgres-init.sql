-- PostgreSQL initialization script for local development
-- This runs automatically when the PostgreSQL container starts for the first time

-- Create database (already created by POSTGRES_DB env var, but ensuring it exists)
-- CREATE DATABASE ai_trading_bot;

-- Create user (already created by POSTGRES_USER env var, but ensuring it exists)
-- CREATE USER trading_bot WITH PASSWORD 'dev_password_123';

-- Grant all privileges on database to user
GRANT ALL PRIVILEGES ON DATABASE ai_trading_bot TO trading_bot;

-- Connect to the ai_trading_bot database
\c ai_trading_bot;

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO trading_bot;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_bot;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_bot;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO trading_bot;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO trading_bot;

-- Create extensions that might be useful for trading bot
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Log successful initialization
INSERT INTO pg_stat_statements_info DEFAULT VALUES ON CONFLICT DO NOTHING;
