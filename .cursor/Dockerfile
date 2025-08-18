FROM debian:bookworm-slim

ARG DEBIAN_FRONTEND=noninteractive

# Base packages and GitHub CLI installation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        gnupg && \
    rm -rf /var/lib/apt/lists/* && \
    # Install GitHub CLI
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \
        tee /etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null && \
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | \
        tee /etc/apt/sources.list.d/github-cli.list >/dev/null && \
    apt-get update && \
    apt-get install -y --no-install-recommends gh && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for safety
RUN useradd -m cursor && chown -R cursor:cursor /home/cursor
USER cursor

CMD ["bash"] 