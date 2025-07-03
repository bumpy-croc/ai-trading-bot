FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs ml

# Make scripts executable
RUN chmod +x health_check.py run_live_trading_with_health.py

# Expose port for health checks
EXPOSE 8000

# Health check endpoint (Railway format)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - use the combined runner
CMD ["python", "run_live_trading_with_health.py", "adaptive"]