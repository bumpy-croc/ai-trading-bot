# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

# Install system dependencies and PostgreSQL
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        postgresql \
        postgresql-contrib \
        supervisor \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -ms /bin/bash ubuntu

# Set working directory
WORKDIR /home/ubuntu

# Copy requirements and install dependencies (dev + prod)
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt

# Do NOT copy application code from parent directory (agent will clone code at runtime)

# Create PostgreSQL data directory
RUN mkdir -p /var/lib/postgresql/data

# Add supervisord config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose app and database ports
EXPOSE 8000 5432

# Switch to non-root user
USER ubuntu

# Default command: run supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"] 