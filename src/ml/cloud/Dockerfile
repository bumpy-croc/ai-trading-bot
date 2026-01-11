# SageMaker Training Container for AI Trading Bot
# Uses official TensorFlow GPU image as base

# Base image: TensorFlow 2.15 CPU with Python 3.11
# CPU version is more stable for SageMaker training
FROM tensorflow/tensorflow:2.15.0

# Metadata
LABEL maintainer="AI Trading Bot"
LABEL description="SageMaker training container with trading bot ML pipeline"

# Set working directory
WORKDIR /opt/ml/code

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY src/ml/cloud/requirements-cloud.txt /opt/ml/code/
# Ignore blinker (pre-installed in TensorFlow image as distutils)
RUN pip install --no-cache-dir --ignore-installed blinker -r requirements-cloud.txt

# Copy source code
# We copy the entire src/ directory to maintain imports
COPY src/ /opt/ml/code/src/

# Copy CLI directory (needed by training pipeline)
COPY cli/ /opt/ml/code/cli/
COPY pyproject.toml /opt/ml/code/

# NOTE: Skip installing ai-trading-bot package to avoid numpy ABI incompatibility
# The source code is available via PYTHONPATH
# RUN pip install --no-cache-dir --ignore-installed blinker .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/ml/code:/opt/ml/code/cli \
    # Disable TensorFlow warnings
    TF_CPP_MIN_LOG_LEVEL=2 \
    # Enable GPU memory growth
    TF_FORCE_GPU_ALLOW_GROWTH=true

# SageMaker expects the entrypoint to be at /opt/ml/code/train
# We create a wrapper script that calls our entrypoint
RUN echo '#!/usr/bin/env python3\nimport sys\nsys.path.insert(0, "/opt/ml/code")\nfrom src.ml.cloud.entrypoint import main\nsys.exit(main())' > /opt/ml/code/train && \
    chmod +x /opt/ml/code/train

# Set the entrypoint (SageMaker will call this)
ENTRYPOINT ["python3", "/opt/ml/code/train"]
